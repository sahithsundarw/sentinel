"""
Local training script for Sentinel — RTX 4060 (8GB VRAM)
Uses standard transformers + PEFT + bitsandbytes (no Unsloth).
Usage:
  python scripts/train_local.py --phase sft
  python scripts/train_local.py --phase ppo
  python scripts/train_local.py --phase all   <- runs SFT then PPO back to back
"""
import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import requests
import torch

ENV_URL = "https://varunventra-guardrail-arena.hf.space"
CHECKPOINT_DIR = Path("checkpoints/llama-sft")
PPO_CHECKPOINT_DIR = Path("checkpoints/llama-ppo")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Unsloth pre-quantized mirrors — ungated, no license approval needed
MODEL_CONFIGS = [
    {
        "name": "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        "label": "Llama-3.1-8B",
        "max_seq_length": 1024,
        "vram_required_gb": 7.0,
    },
    {
        "name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "label": "Llama-3.2-3B",
        "max_seq_length": 1024,
        "vram_required_gb": 4.0,
    },
]

ZERO_SHOT_BASELINES = {
    "Llama-3.1-8B": 0.5428,
    "Llama-3.2-3B": None,
}

VALID_ACTIONS = ("allow", "refuse", "modify", "escalate")

PROMPT_TEMPLATE = (
    "### SYSTEM\n"
    "You are a content safety moderator. Given a prompt and context, "
    "respond with exactly one word: allow, refuse, modify, or escalate.\n\n"
    "### USER\n"
    "Application context: {application_context}\n"
    "Risk level: {risk_level}\n"
    "Turn: {turn_number}\n"
    "Prompt: {user_prompt}\n\n"
    "### ASSISTANT\n"
    "{correct_action}"
)

INFERENCE_TEMPLATE = (
    "### SYSTEM\n"
    "You are a content safety moderator. Respond with exactly one word: "
    "allow, refuse, modify, or escalate.\n\n"
    "### USER\n"
    "Application context: {application_context}\n"
    "Risk level: {risk_level}\n"
    "Prompt: {user_prompt}\n\n"
    "### ASSISTANT\n"
)


def pick_model_config():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Detected VRAM: {vram_gb:.1f} GB")
    for cfg in MODEL_CONFIGS:
        if vram_gb >= cfg["vram_required_gb"]:
            print(f"Selected: {cfg['label']} ({cfg['name']})")
            return cfg
    return MODEL_CONFIGS[-1]


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    return model, tokenizer


def attach_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def generate_synthetic_data(env_url, n_examples=150):
    """Generate training data by stepping through tasks and recording ground-truth actions."""
    examples = []
    task_ids = ["basic_threat_detection", "context_aware_policy"]

    for task_id in task_ids:
        reset_resp = requests.post(f"{env_url}/reset", json={"task_id": task_id})
        if reset_resp.status_code != 200:
            print(f"  WARNING: could not reset {task_id}: {reset_resp.status_code}")
            continue
        reset_data = reset_resp.json()
        session_id = reset_data.get("session_id", "")
        obs = reset_data.get("observation", reset_data)
        done = False

        while not done and len(examples) < n_examples:
            step_resp = requests.post(
                f"{env_url}/step",
                params={"session_id": session_id},
                json={"prompt_id": obs.get("prompt_id", ""),
                      "action_type": "allow",
                      "reason": "collecting training data"},
            )
            step_data = step_resp.json()
            info = step_data.get("info", {})
            correct_action = info.get("ground_truth_action", "refuse")

            examples.append({"observation": obs, "correct_action": correct_action})
            done = step_data.get("done", False)
            obs = step_data.get("observation") or obs
            time.sleep(0.05)

    print(f"Generated {len(examples)} synthetic training examples")
    return examples


def fetch_training_data(env_url):
    resp = requests.get(
        f"{env_url}/training_data",
        params={"format": "sft", "task_id": "basic_threat_detection"},
    )
    resp.raise_for_status()
    raw = resp.json()

    # Endpoint may return a list or a dict wrapping a list
    print(f"Raw response type: {type(raw)}")
    if isinstance(raw, dict):
        print(f"Dict keys: {list(raw.keys())}")
        # Try common wrapper keys, then flatten all list values
        data = (raw.get("data") or raw.get("prompts") or raw.get("examples")
                or raw.get("items") or raw.get("training_data") or [])
        if not data:
            # Fall back: concatenate all list values in the dict
            for v in raw.values():
                if isinstance(v, list):
                    data = v
                    break
    else:
        data = raw

    print(f"First item: {str(data[0])[:200]}" if data else "No items found")
    print(f"Total items fetched: {len(data)}")

    if len(data) < 20:
        print(f"Only {len(data)} examples from endpoint — falling back to synthetic generation...")
        data = generate_synthetic_data(env_url)

    return data


def build_dataset(raw_data):
    texts = []

    for item in raw_data:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                obs = parsed.get("observation", parsed)
                correct_action = parsed.get("correct_action", "refuse")
            except (json.JSONDecodeError, AttributeError):
                continue
        elif isinstance(item, dict):
            obs = item.get("observation", item)
            correct_action = item.get("action", item.get("correct_action", "refuse"))
        else:
            continue

        if isinstance(obs, str):
            user_prompt = obs
            app_context = "general"
            risk_level = "medium"
            turn_number = 1
        else:
            user_prompt = obs.get("user_prompt", obs.get("prompt", ""))
            app_context = obs.get("application_context", "general")
            risk_level = obs.get("risk_level", "medium")
            turn_number = obs.get("turn_number", 1)

        if not user_prompt:
            continue

        if correct_action not in VALID_ACTIONS:
            correct_action = "refuse"

        text = PROMPT_TEMPLATE.format(
            application_context=app_context,
            risk_level=risk_level,
            turn_number=turn_number,
            user_prompt=user_prompt,
            correct_action=correct_action,
        )
        texts.append({"text": text})

    if len(texts) < 10:
        print(f"WARNING: only {len(texts)} valid examples after parsing — falling back to synthetic data...")
        synthetic = generate_synthetic_data(ENV_URL)
        return build_dataset(synthetic)

    from datasets import Dataset
    print(f"Built dataset with {len(texts)} examples")
    return Dataset.from_list(texts)


def _extract_action(raw_response):
    """Scan first 5 words of model output for a valid action word."""
    words = raw_response.strip().lower().split()
    for word in words[:5]:
        clean = word.strip(".,!?:;\"'")
        if clean in VALID_ACTIONS:
            return clean
    return "refuse"


def evaluate_model(model, tokenizer, env_url, max_steps=67):
    model.eval()
    reset_resp = requests.post(f"{env_url}/reset",
                               json={"task_id": "basic_threat_detection"})
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    session_id = reset_data.get("session_id", "")
    obs = reset_data.get("observation", reset_data)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        prompt = INFERENCE_TEMPLATE.format(
            application_context=obs.get("application_context", "general"),
            risk_level=obs.get("risk_level", "medium"),
            user_prompt=obs.get("user_prompt", obs.get("prompt", "")),
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        del inputs, outputs
        action = _extract_action(raw_response)

        step_resp = requests.post(
            f"{env_url}/step",
            params={"session_id": session_id},
            json={
                "prompt_id": obs.get("prompt_id", ""),
                "action_type": action,
                "reason": f"content safety decision: {action}",
            },
        )
        step_data = step_resp.json()
        reward_raw = step_data.get("reward", 0)
        reward = reward_raw.get("score", 0) if isinstance(reward_raw, dict) else float(reward_raw or 0)
        total_reward += reward
        steps += 1
        done = step_data.get("done", False)
        obs = step_data.get("observation") or obs

        if steps % 10 == 0:
            print(f"  Step {steps}: '{raw_response.strip()[:40]}' → {action} | reward: {reward:.3f}")

        time.sleep(0.05)

    avg = round(total_reward / max(steps, 1), 4)
    print(f"  Eval complete: {steps} steps, avg reward: {avg:.4f}")
    return avg, steps


def run_sft_phase(config, env_url):
    from trl import SFTTrainer, SFTConfig

    baseline_score = ZERO_SHOT_BASELINES.get(config["label"])

    raw = fetch_training_data(env_url)
    model, tokenizer = load_model_4bit(config["name"])

    if baseline_score is None:
        print("Running zero-shot eval before SFT...")
        baseline_score, _ = evaluate_model(model, tokenizer, env_url)
        print(f"Zero-shot baseline: {baseline_score:.4f}")

    model = attach_lora(model)
    dataset = build_dataset(raw)
    print(f"Training dataset: {len(dataset)} examples")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="paged_adamw_8bit",
            output_dir=str(CHECKPOINT_DIR),
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print("Starting SFT training...")
    trainer.train()
    model.save_pretrained(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    print(f"Checkpoint saved to {CHECKPOINT_DIR}")

    print("Evaluating fine-tuned model...")
    post_sft_score, _ = evaluate_model(model, tokenizer, env_url)

    sft_results = {
        "model": config["label"],
        "model_id": config["name"],
        "task": "basic_threat_detection",
        "baseline_score": baseline_score,
        "post_sft_score": post_sft_score,
        "improvement": round(post_sft_score - baseline_score, 4),
        "improvement_pct": round(
            (post_sft_score - baseline_score) / max(baseline_score, 0.001) * 100, 1),
        "training_examples": len(dataset),
        "epochs": 3,
        "checkpoint_path": str(CHECKPOINT_DIR),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "llama_sft_scores.json", "w") as f:
        json.dump(sft_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SFT RESULTS — {config['label']}")
    print(f"Before: {baseline_score:.4f}")
    print(f"After:  {post_sft_score:.4f}")
    print(f"Delta:  +{sft_results['improvement']:.4f} ({sft_results['improvement_pct']}%)")
    print(f"{'='*50}")
    return sft_results


def _post_with_retry(url, data, retries=4, delay=10, params=None):
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=data, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Network retry {attempt+1}/{retries}: {e}")
            time.sleep(delay)


def run_ppo_phase(config, checkpoint_dir, env_url, episodes=20):
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    import bitsandbytes as bnb

    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nLoading SFT checkpoint from {checkpoint_dir}...")
    base_model, tokenizer = load_model_4bit(config["name"])

    # prepare_model_for_kbit_training must run before PeftModel.from_pretrained
    # so gradient checkpointing hooks are installed on the base model first.
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True)

    model = PeftModel.from_pretrained(
        base_model, str(checkpoint_dir), is_trainable=True)

    # Belt-and-suspenders: explicitly unfreeze every LoRA parameter.
    for name, param in model.named_parameters():
        param.requires_grad_(True if "lora_" in name else False)

    # Diagnosis — fail fast rather than silently train nothing.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters — LoRA weights are all frozen. "
            "Verify the SFT checkpoint at " + str(checkpoint_dir))

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01,
    )

    ACTIONS = list(VALID_ACTIONS)  # allow, refuse, modify, escalate

    # Pre-compute first token id for each action word once.
    action_token_ids = [
        tokenizer.encode(" " + a, add_special_tokens=False)[0]
        for a in ACTIONS
    ]

    def get_action_and_logprob(obs):
        """Single forward pass with grad enabled; sample from action-token logits."""
        prompt = INFERENCE_TEMPLATE.format(
            application_context=obs.get("application_context", "general"),
            risk_level=obs.get("risk_level", "medium"),
            user_prompt=obs.get("user_prompt", obs.get("prompt", "")),
        )
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=256).to("cuda")
        with torch.enable_grad():
            logits = model(**enc).logits[0, -1, :]  # next-token logits
        del enc

        action_logits  = logits[action_token_ids]
        action_logprobs = F.log_softmax(action_logits, dim=0)
        action_probs    = F.softmax(action_logits, dim=0).detach()

        action_idx    = torch.multinomial(action_probs, 1).item()
        chosen_action = ACTIONS[action_idx]
        chosen_logprob = action_logprobs[action_idx]  # retains grad graph
        return chosen_action, chosen_logprob

    episode_rewards = []
    ema_baseline = 0.0  # exponential moving average reward for variance reduction

    for episode in range(episodes):
        model.train()  # train mode — dropout active, grad graph built in forward pass
        print(f"\nRL Episode {episode+1}/{episodes}")

        reset_resp = _post_with_retry(f"{env_url}/reset",
                                      {"task_id": "basic_threat_detection"})
        reset_data = reset_resp.json()
        session_id = reset_data.get("session_id", "")
        obs        = reset_data.get("observation", reset_data)

        trajectory   = []  # (logprob_tensor, reward_float)
        episode_reward = 0.0
        steps = 0
        done  = False
        action_counts = {a: 0 for a in ACTIONS}

        while not done and steps < 100:
            action, logprob = get_action_and_logprob(obs)
            action_counts[action] += 1

            step_resp = _post_with_retry(
                f"{env_url}/step",
                {"prompt_id": obs.get("prompt_id", ""),
                 "action_type": action,
                 "reason": f"content safety decision: {action}"},
                params={"session_id": session_id},
            )
            step_data  = step_resp.json()
            reward_raw = step_data.get("reward", 0)
            reward     = (reward_raw.get("score", 0)
                          if isinstance(reward_raw, dict)
                          else float(reward_raw or 0))

            trajectory.append((logprob, reward))
            episode_reward += reward
            steps += 1
            done = step_data.get("done", False)
            obs  = step_data.get("observation") or obs

        avg_reward   = episode_reward / max(steps, 1)
        episode_rewards.append(round(avg_reward, 4))

        # REINFORCE update: loss = -logprob * advantage, averaged over trajectory.
        ema_baseline = 0.9 * ema_baseline + 0.1 * avg_reward
        optimizer.zero_grad()
        total_loss = sum(
            -lp * (r - ema_baseline) for lp, r in trajectory
        ) / len(trajectory)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(action_counts.items()))
        print(f"  Reward: {avg_reward:.4f}  [{dist_str}]  "
              f"Loss: {total_loss.item():.4f}  Grad norm: {avg_grad:.6f}")
        if avg_grad == 0.0:
            print("  WARNING: Zero gradients — weights not updating")

        try:
            requests.post(f"{env_url}/training_log", json={
                "episode": episode + 1,
                "task": "basic_threat_detection",
                "agent_type": f"reinforce_{config['label']}",
                "reward": avg_reward,
                "phase": "reinforce",
            }, timeout=10)
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    PPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(PPO_CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(PPO_CHECKPOINT_DIR))

    print("Evaluating post-RL model...")
    model.eval()
    post_rl_score, _ = evaluate_model(model, tokenizer, env_url)

    sft_path = RESULTS_DIR / "llama_sft_scores.json"
    baseline_score = ZERO_SHOT_BASELINES.get(config["label"], 0.5428)
    post_sft_score = None
    if sft_path.exists():
        with open(sft_path) as f:
            sft_data = json.load(f)
        baseline_score = sft_data.get("baseline_score", baseline_score)
        post_sft_score = sft_data.get("post_sft_score")

    results = {
        "model": config["label"],
        "algorithm": "REINFORCE",
        "task": "basic_threat_detection",
        "episodes": episodes,
        "episode_rewards": episode_rewards,
        "baseline_score": baseline_score,
        "post_sft_score": post_sft_score,
        "post_ppo_score": post_rl_score,  # key kept for chart compatibility
        "improvement_over_baseline": round(post_rl_score - baseline_score, 4),
        "checkpoint_path": str(PPO_CHECKPOINT_DIR),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "llama_ppo_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"REINFORCE RESULTS — {config['label']}")
    print(f"Zero-shot baseline:  {baseline_score:.4f}")
    if post_sft_score is not None:
        print(f"After SFT:           {post_sft_score:.4f}")
    print(f"After RL:            {post_rl_score:.4f}")
    print(f"Improvement:         +{results['improvement_over_baseline']:.4f}")
    print(f"{'='*50}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["sft", "ppo", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-url", default=ENV_URL)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. This script requires a CUDA GPU.")

    import transformers
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    # Load .env if present
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    config = pick_model_config()

    if args.phase in ("sft", "all"):
        run_sft_phase(config, args.env_url)
        subprocess.run([sys.executable, "generate_charts.py"], check=False)

    if args.phase in ("ppo", "all"):
        run_ppo_phase(config, CHECKPOINT_DIR, args.env_url, args.episodes)
        subprocess.run([sys.executable, "generate_charts.py"], check=False)

    print("\nAll training complete. Charts updated.")
