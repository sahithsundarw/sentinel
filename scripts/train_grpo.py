"""
GRPO training script for Sentinel — RTX 4060 (8 GB VRAM)
Group Relative Policy Optimization on Tasks 1, 2, and 3 sequentially.
Task 4 is handled by the Tabular Q-Learner (train_agent.py).

Usage:
  python scripts/train_grpo.py                         # default: 20 eps, K=8, all 3 tasks
  python scripts/train_grpo.py --episodes 30 --k 4     # smaller group size
  python scripts/train_grpo.py --tasks 1               # single task dry-run
  python scripts/train_grpo.py --resume task2          # resume after crash
  python scripts/train_grpo.py --kl-beta 0.02          # KL penalty weight (default 0.02)
  python scripts/train_grpo.py --free-form             # emit JSON instead of constrained token
  python scripts/train_grpo.py --reward-mode grader    # use grader-aligned reward signal

Algorithm:
  At each episode, run K full rollouts from the same start state.
  Compute episode-level group advantage: adv_k = (r_k - mean(r)) / (std(r) + eps)
  Loss = -mean_k[ adv_k * mean_t(log_pi(a_t | s_t)) ] + kl_beta * KL(pi_theta || pi_ref)
  KL penalty is computed per-step against the frozen reference model (LoRA base).
  Same VRAM footprint as REINFORCE since rollouts are sequential, not batched.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import requests
import torch
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL          = "https://varunventra-guardrail-arena.hf.space"
RESULTS_DIR      = Path("results")
CHECKPOINT_BASE  = Path("checkpoints")
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_BASE.mkdir(exist_ok=True)

TASK_IDS = {
    1: "basic_threat_detection",
    2: "context_aware_policy",
    3: "multiturn_adversarial",
}
MAX_STEPS_PER_EP = 150  # hard cap per rollout

VALID_ACTIONS     = ("allow", "refuse", "modify", "escalate")
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

# ── Shared helpers (mirrors train_local.py) ───────────────────────────────────
class _SessionExpired(Exception):
    pass


def _post_with_retry(url, data, retries=8, delay=15, params=None):
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=data, params=params, timeout=60)
            if resp.status_code == 410:
                raise _SessionExpired("Session expired (410)")
            if resp.status_code in (502, 503, 504):
                # HF Space restarting — wait longer on each retry
                wait = delay * (2 ** min(attempt, 4))  # 15, 30, 60, 120, 120 ...
                print(f"  Network retry {attempt+1}/{retries}: {resp.status_code} (Space waking up, waiting {wait}s...)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except _SessionExpired:
            raise
        except requests.exceptions.Timeout:
            wait = delay * (attempt + 1)
            if attempt == retries - 1:
                raise
            print(f"  Network retry {attempt+1}/{retries}: timeout (waiting {wait}s...)")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (attempt + 1)
            print(f"  Network retry {attempt+1}/{retries}: {e} (waiting {wait}s...)")
            time.sleep(wait)


def _extract_action(raw):
    words = raw.strip().lower().split()
    for w in words[:5]:
        c = w.strip(".,!?:;\"'")
        if c in VALID_ACTIONS:
            return c
    return "refuse"


def _parse_action_freeform(output: str) -> str:
    """Parse a free-form JSON or keyword response into an action type.

    Uses last-occurrence keyword scan as fallback (CoT models put final
    decision last; first-occurrence misclassifies negated phrases).
    Returns "refuse" if no valid action is found.
    """
    import json as _json
    try:
        start = output.find("{")
        end   = output.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = _json.loads(output[start:end])
            at = parsed.get("action_type", "").lower().strip()
            if at in VALID_ACTIONS:
                return at
    except Exception:
        pass
    lower = output.lower()
    last_pos, chosen = -1, None
    for kw in VALID_ACTIONS:
        pos = lower.rfind(kw)
        if pos > last_pos:
            last_pos, chosen = pos, kw
    return chosen if chosen else "refuse"


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Loading {model_name}... (free VRAM: {free:.1f} GB)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        token=os.environ.get("HF_TOKEN"), low_cpu_mem_usage=True,
    )
    print("  Model loaded.", flush=True)
    return model, tokenizer


def attach_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ── GRPO core ─────────────────────────────────────────────────────────────────
def _build_prompt(obs):
    return INFERENCE_TEMPLATE.format(
        application_context=obs.get("application_context", "general_assistant"),
        risk_level=obs.get("user_risk_score", 0.5),
        user_prompt=obs.get("user_prompt", obs.get("prompt", "")),
    )


def _forward_action(model, tokenizer, obs, action_token_ids, free_form=False):
    """Single forward pass; returns (chosen_action, log_prob_tensor, action_logprobs_all).

    action_logprobs_all is the full log-softmax distribution over the 4 action tokens,
    used for KL divergence computation against the reference model.
    """
    prompt = _build_prompt(obs)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")

    if free_form:
        with torch.enable_grad():
            logits_all = model(**enc).logits[0, -1, :]
        del enc
        a_logprobs_all = F.log_softmax(logits_all[action_token_ids], dim=0)
        a_probs = F.softmax(logits_all[action_token_ids], dim=0).detach()
        idx = torch.multinomial(a_probs, 1).item()
        chosen_action = VALID_ACTIONS[idx]
        return chosen_action, a_logprobs_all[idx], a_logprobs_all
    else:
        with torch.enable_grad():
            logits = model(**enc).logits[0, -1, :]
        del enc
        a_logprobs = F.log_softmax(logits[action_token_ids], dim=0)
        a_probs    = F.softmax(logits[action_token_ids], dim=0).detach()
        idx        = torch.multinomial(a_probs, 1).item()
        return VALID_ACTIONS[idx], a_logprobs[idx], a_logprobs


@torch.no_grad()
def _ref_logprobs(ref_model, tokenizer, obs, action_token_ids):
    """Compute reference model log-probs over 4 action tokens (no grad)."""
    prompt = _build_prompt(obs)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    logits = ref_model(**enc).logits[0, -1, :]
    del enc
    return F.log_softmax(logits[action_token_ids], dim=0)


def _run_episode(
    model, tokenizer, env_url, task_id, action_token_ids,
    seed=None, ref_model=None, free_form=False, reward_mode="step",
    session_id_reuse=None,
):
    """Run one full episode; return (logprobs_list, kl_terms_list, total_reward, action_counts).

    kl_terms_list contains per-step KL(pi_theta || pi_ref) tensors (empty if no ref_model).
    session_id_reuse: if provided, reuse this session_id instead of /reset (cross-episode chaining).
    """
    reset_body = {"task_id": task_id}
    if seed is not None:
        reset_body["seed"] = seed
    if reward_mode != "step":
        reset_body["reward_mode"] = reward_mode

    if session_id_reuse:
        session_id = session_id_reuse
        reset_resp = _post_with_retry(f"{env_url}/reset", reset_body, params={"session_id": session_id})
    else:
        reset_resp = _post_with_retry(f"{env_url}/reset", reset_body)

    data       = reset_resp.json()
    session_id = data.get("session_id", session_id_reuse or "")
    obs        = data.get("observation", data)

    logprobs      = []
    kl_terms      = []
    total_reward  = 0.0
    steps         = 0
    done          = False
    action_counts = {a: 0 for a in VALID_ACTIONS}

    while not done and steps < MAX_STEPS_PER_EP:
        model.train()
        action, logprob, logprobs_all = _forward_action(
            model, tokenizer, obs, action_token_ids, free_form=free_form)
        action_counts[action] += 1

        if ref_model is not None:
            ref_lps = _ref_logprobs(ref_model, tokenizer, obs, action_token_ids)
            # KL per step: sum_a pi(a) * (log pi(a) - log pi_ref(a))
            pi = logprobs_all.exp()
            kl = (pi * (logprobs_all - ref_lps)).sum()
            kl_terms.append(kl)

        try:
            step_resp = _post_with_retry(
                f"{env_url}/step",
                {"prompt_id": obs.get("prompt_id", ""),
                 "action_type": action,
                 "reason": f"grpo safety decision: {action}"},
                params={"session_id": session_id},
            )
        except _SessionExpired:
            break

        step_data  = step_resp.json()
        reward_raw = step_data.get("reward", 0)
        reward     = (reward_raw.get("score", 0) if isinstance(reward_raw, dict)
                      else float(reward_raw or 0))
        logprobs.append(logprob)
        total_reward += reward
        steps += 1
        done = step_data.get("done", False)
        obs  = step_data.get("observation") or obs

    avg_reward = total_reward / max(steps, 1)
    return logprobs, kl_terms, avg_reward, action_counts, session_id


def _get_grader_score(env_url, task_id, model, tokenizer, action_token_ids, n_eval=3):
    """Run n_eval episodes and call /grader for classification accuracy."""
    scores = []
    for _ in range(n_eval):
        reset_resp = requests.post(
            f"{env_url}/reset", json={"task_id": task_id}, timeout=30)
        reset_resp.raise_for_status()
        data       = reset_resp.json()
        session_id = data["session_id"]
        obs        = data.get("observation", data)
        done       = False
        steps      = 0

        model.eval()
        while not done and steps < MAX_STEPS_PER_EP:
            action, *_ = _forward_action(model, tokenizer, obs, action_token_ids)
            try:
                step_resp = requests.post(
                    f"{env_url}/step",
                    params={"session_id": session_id},
                    json={"prompt_id": obs.get("prompt_id", ""),
                          "action_type": action,
                          "reason": f"eval: {action}"},
                    timeout=30,
                )
                if step_resp.status_code == 410:
                    break
                step_resp.raise_for_status()
            except Exception:
                break
            step_data = step_resp.json()
            done      = step_data.get("done", False)
            obs       = step_data.get("observation") or obs
            steps    += 1

        try:
            for _g in range(5):
                grader_resp = requests.get(
                    f"{env_url}/grader", params={"session_id": session_id}, timeout=60)
                if grader_resp.status_code in (502, 503, 504):
                    time.sleep(30 * (2 ** min(_g, 2)))
                    continue
                grader_resp.raise_for_status()
                scores.append(float(grader_resp.json().get("score", 0.0)))
                break
        except Exception:
            pass

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── Per-task training loop ────────────────────────────────────────────────────
def train_task(
    task_num, task_id, model_name, env_url, episodes, K,
    resume_ep=0, kl_beta=0.02, free_form=False, reward_mode="step",
):
    import bitsandbytes as bnb

    print(f"\n{'='*60}")
    print(f"GRPO Training  |  Task {task_num}: {task_id}")
    print(f"Episodes: {episodes}  |  K (group size): {K}  |  KL β: {kl_beta}")
    print(f"Free-form: {free_form}  |  Reward mode: {reward_mode}")
    print(f"{'='*60}")

    ckpt_dir = CHECKPOINT_BASE / f"llama-grpo-task{task_num}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_model_4bit(model_name)
    # Freeze a copy of the base model to use as the KL reference policy.
    # The reference model has no LoRA adapters — it is the frozen pre-training checkpoint.
    ref_model = None
    if kl_beta > 0:
        print("Loading frozen reference model for KL penalty...")
        ref_model, _ = load_model_4bit(model_name)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    model = attach_lora(model)

    # Resume from mid-run checkpoint if requested
    if resume_ep > 0:
        resume_ckpt = ckpt_dir / f"ep{resume_ep}"
        if resume_ckpt.exists():
            from peft import PeftModel
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = PeftModel.from_pretrained(model, str(resume_ckpt), is_trainable=True)
            for n, p in model.named_parameters():
                p.requires_grad_("lora_" in n)
            print(f"Resumed from checkpoint: {resume_ckpt}")
        else:
            print(f"Resume checkpoint not found: {resume_ckpt} — starting fresh")

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=0.01,
    )

    ACTIONS = list(VALID_ACTIONS)
    action_token_ids = [
        tokenizer.encode(" " + a, add_special_tokens=False)[0] for a in ACTIONS
    ]

    # Pre-training grader score
    print("\nPre-training grader eval...")
    pre_score = _get_grader_score(env_url, task_id, model, tokenizer, action_token_ids)
    print(f"  Pre-training grader score: {pre_score:.4f}")

    episode_rewards   = []
    action_dist_ep1   = None
    action_dist_final = None
    episode_log       = []  # full per-episode detail for judges

    for ep in range(resume_ep, episodes):
        print(f"\nGRPO Episode {ep+1}/{episodes} (Task {task_num})")

        # -- Collect K rollouts --
        rollouts = []  # list of (logprobs, kl_terms, avg_reward, action_counts)
        for k in range(K):
            try:
                lps, kls, avg_r, counts, _ = _run_episode(
                    model, tokenizer, env_url, task_id, action_token_ids,
                    seed=ep * 100 + k,
                    ref_model=ref_model,
                    free_form=free_form,
                    reward_mode=reward_mode,
                )
            except (requests.exceptions.ConnectionError, _SessionExpired) as e:
                print(f"  Rollout {k+1} failed: {e} — skipping episode")
                break
            rollouts.append((lps, kls, avg_r, counts))
            print(f"  Rollout {k+1}/{K}: avg_reward={avg_r:.4f}  "
                  f"actions={' '.join(f'{a}:{c}' for a,c in counts.items())}")

        if len(rollouts) < 2:
            print("  Not enough rollouts — skipping update")
            continue

        rewards = [r[2] for r in rollouts]
        avg_ep  = sum(rewards) / len(rewards)
        episode_rewards.append(round(avg_ep, 4))

        if ep == 0:
            action_dist_ep1 = {k: v for k, v in rollouts[0][3].items()}

        # -- Collapse detection --
        total_actions = sum(sum(r[3].values()) for r in rollouts)
        action_totals = {a: sum(r[3].get(a, 0) for r in rollouts) for a in VALID_ACTIONS}
        dominant = max(action_totals, key=action_totals.get)
        dominant_pct = action_totals[dominant] / max(total_actions, 1)
        if dominant_pct > 0.65:
            print(f"  WARNING: collapse detected — {dominant} is {dominant_pct:.0%} of actions")

        # -- GRPO advantage --
        mean_r = avg_ep
        std_r  = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

        if std_r < 0.001:
            print(f"  Advantage std={std_r:.5f} too small — skipping update (all rollouts identical)")
            continue

        optimizer.zero_grad()
        losses = []
        for logprobs, kl_terms, reward, _ in rollouts:
            if not logprobs:
                continue
            advantage = (reward - mean_r) / (std_r + 1e-6)
            policy_loss = -advantage * torch.stack(logprobs).mean()
            # KL penalty: pushes policy toward frozen reference (prevents forgetting)
            kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0)
            ep_loss = policy_loss + kl_beta * kl_loss
            losses.append(ep_loss)

        if not losses:
            continue

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.requires_grad and p.grad is not None]
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        print(f"  Group rewards: {[round(r,4) for r in rewards]}  "
              f"Advantage std: {std_r:.4f}  Loss: {total_loss.item():.4f}  "
              f"Grad: {avg_grad:.6f}")
        if avg_grad == 0.0:
            print("  WARNING: zero gradients — no weight update")

        episode_log.append({
            "episode":       ep + 1,
            "rollout_rewards": [round(r, 4) for r in rewards],
            "avg_reward":    round(avg_ep, 4),
            "advantage_std": round(std_r, 4),
            "loss":          round(total_loss.item(), 4),
            "grad_norm":     round(avg_grad, 6),
            "action_dist":   action_totals,
            "dominant_action": dominant,
            "dominant_pct":  round(dominant_pct, 3),
            "collapse":      dominant_pct > 0.65,
        })

        # Save partial log after every episode so a crash doesn't lose everything
        _partial = {
            "task_id":            task_id,
            "pre_score":          pre_score,
            "post_score":         None,
            "k":                  K,
            "episodes_completed": ep + 1,
            "episodes_target":    episodes,
            "reward_mode":        reward_mode,
            "timestamp":          datetime.utcnow().isoformat() + "Z",
            "log":                episode_log,
        }
        with open(RESULTS_DIR / f"grpo_training_log_task{task_num}.json", "w") as _f:
            json.dump(_partial, _f, indent=2)

        # -- Post to training_log --
        try:
            requests.post(f"{env_url}/training_log", json={
                "episode": ep + 1,
                "task_id": task_id,
                "agent_name": f"grpo_llama3_task{task_num}",
                "grader_score": avg_ep,
                "cumulative_reward": avg_ep,
            }, timeout=10)
        except Exception:
            pass

        # -- Checkpoint every 5 episodes --
        if (ep + 1) % 5 == 0:
            ep_ckpt = ckpt_dir / f"ep{ep+1}"
            ep_ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ep_ckpt))
            print(f"  Checkpoint: {ep_ckpt}")

        gc.collect()
        torch.cuda.empty_cache()

    # -- Final checkpoint + eval --
    final_ckpt = ckpt_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))
    print(f"\nFinal checkpoint: {final_ckpt}")

    if episode_rewards:
        action_dist_final = {k: v for k, v in rollouts[-1][3].items()}

    print("Post-training grader eval...")
    post_score = _get_grader_score(env_url, task_id, model, tokenizer, action_token_ids)
    print(f"  Post-training grader score: {post_score:.4f}")

    result = {
        "task_id":       task_id,
        "task_num":      task_num,
        "pre_score":     pre_score,
        "post_score":    post_score,
        "improvement":   round(post_score - pre_score, 4),
        "episodes":      len(episode_rewards),
        "episode_rewards": episode_rewards,
        "action_dist_ep1":   action_dist_ep1,
        "action_dist_final": action_dist_final,
        "checkpoint":    str(final_ckpt),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "episode_log":   episode_log,
    }

    log_path = RESULTS_DIR / f"grpo_training_log_task{task_num}.json"
    with open(log_path, "w") as f:
        json.dump({
            "task_id":     task_id,
            "pre_score":   pre_score,
            "post_score":  post_score,
            "k":           K,
            "episodes":    len(episode_rewards),
            "reward_mode": "grader",
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "log":         episode_log,
        }, f, indent=2)
    print(f"  Training log saved: {log_path}")

    # -- Post final score to leaderboard --
    try:
        requests.post(f"{env_url}/training_log", json={
            "episode":        episodes,
            "task_id":        task_id,
            "agent_name":     "Llama-3.1-8B-GRPO-20ep",
            "grader_score":   post_score,
            "cumulative_reward": post_score,
        }, timeout=10)
    except Exception:
        pass

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GRPO training on Sentinel Tasks 1-3")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per task (default: 20)")
    parser.add_argument("--k", type=int, default=8,
                        help="Group size K for GRPO (default: 8; must be ≥2)")
    parser.add_argument("--tasks", type=str, default="1,2,3",
                        help="Comma-separated task numbers to train (default: 1,2,3)")
    parser.add_argument("--model", type=str,
                        default="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
                        help="HuggingFace model name")
    parser.add_argument("--env-url", type=str, default=ENV_URL)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from mid-run: e.g. 'task2' resumes Task 2 from latest checkpoint")
    parser.add_argument("--kl-beta", type=float, default=0.02,
                        help="KL penalty weight against frozen reference policy (default: 0.02; 0 to disable)")
    parser.add_argument("--free-form", action="store_true",
                        help="Emit JSON output instead of constrained single-token sampling")
    parser.add_argument("--reward-mode", type=str, default="grader",
                        choices=["step", "grader"],
                        help="Reward signal: 'grader' (default, leaderboard-aligned) or 'step' (shaped RL reward)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF repo to push checkpoints to after each task (e.g. myuser/sentinel-grpo)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. This script requires a CUDA GPU.")

    K = max(args.k, 2)  # GRPO needs at least 2 samples in a group

    import transformers
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    task_nums = [int(t.strip()) for t in args.tasks.split(",")]
    for t in task_nums:
        if t not in TASK_IDS:
            raise SystemExit(f"Unknown task number: {t}. Valid: 1, 2, 3")

    # Health check
    try:
        r = requests.get(f"{args.env_url}/health", timeout=30)
        assert r.status_code == 200
        print(f"Environment healthy: {args.env_url}")
    except Exception as e:
        raise SystemExit(f"Cannot connect to environment: {e}")

    all_results = {}
    start_time  = time.time()

    for task_num in task_nums:
        task_id   = TASK_IDS[task_num]
        resume_ep = 0

        if args.resume:
            # Find latest checkpoint for this task
            ckpt_dir = CHECKPOINT_BASE / f"llama-grpo-task{task_num}"
            if ckpt_dir.exists():
                ep_dirs = sorted(
                    [int(d.name[2:]) for d in ckpt_dir.iterdir()
                     if d.is_dir() and d.name.startswith("ep")],
                    reverse=True,
                )
                if ep_dirs:
                    resume_ep = ep_dirs[0]
                    print(f"Resuming Task {task_num} from episode {resume_ep}")

        result = train_task(
            task_num=task_num,
            task_id=task_id,
            model_name=args.model,
            env_url=args.env_url,
            episodes=args.episodes,
            K=K,
            resume_ep=resume_ep,
            kl_beta=args.kl_beta,
            free_form=args.free_form,
            reward_mode=args.reward_mode,
        )
        all_results[f"task{task_num}"] = result

        print(f"\nTask {task_num} complete: {result['pre_score']:.4f} -> "
              f"{result['post_score']:.4f}  (+{result['improvement']:.4f})")

        if args.push_to_hub:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                token = os.environ.get("HF_TOKEN")
                ckpt_path = str(CHECKPOINT_BASE / f"llama-grpo-task{task_num}" / "final")
                print(f"  Pushing task{task_num} checkpoint + log to {args.push_to_hub}...")
                api.upload_folder(
                    folder_path=ckpt_path,
                    repo_id=args.push_to_hub,
                    path_in_repo=f"task{task_num}",
                    token=token,
                    repo_type="model",
                )
                log_path_upload = str(RESULTS_DIR / f"grpo_training_log_task{task_num}.json")
                api.upload_file(
                    path_or_fileobj=log_path_upload,
                    path_in_repo=f"logs/grpo_training_log_task{task_num}.json",
                    repo_id=args.push_to_hub,
                    token=token,
                    repo_type="model",
                )
                print(f"  Pushed checkpoint + training log to {args.push_to_hub}")
            except Exception as e:
                print(f"  Push failed: {e} — files still saved locally")

    elapsed = round((time.time() - start_time) / 3600, 2)

    # -- Write aggregated results --
    task1 = all_results.get("task1", {})
    grpo_results = {
        "model":            "Llama-3.1-8B-Instruct",
        "algorithm":        "GRPO",
        "k_group_size":     K,
        "kl_beta":          args.kl_beta,
        "free_form":        args.free_form,
        "reward_mode":      args.reward_mode,
        "episodes_per_task": args.episodes,
        "env_url":          args.env_url,
        "elapsed_hours":    elapsed,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "tasks":            all_results,
        # Keys expected by generate_charts.py chart_multi_model_comparison:
        "post_rl_score":    task1.get("post_score", 0.0),
        "baseline_score":   task1.get("pre_score", 0.0),
        "episode_rewards":  task1.get("episode_rewards", []),
        "action_distribution_ep1":    task1.get("action_dist_ep1"),
        "action_distribution_ep20":   task1.get("action_dist_final"),
    }

    out_path = RESULTS_DIR / "llama_grpo_scores.json"
    with open(out_path, "w") as f:
        json.dump(grpo_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # -- Print summary table --
    print(f"\n{'='*60}")
    print("GRPO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"{'Task':<40}  {'Before':>7}  {'After':>7}  {'Delta':>7}")
    print(f"{'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}")
    for task_num in task_nums:
        r = all_results.get(f"task{task_num}", {})
        tid = TASK_IDS[task_num]
        pre  = r.get("pre_score", 0)
        post = r.get("post_score", 0)
        print(f"{tid:<40}  {pre:>7.4f}  {post:>7.4f}  {post-pre:>+7.4f}")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.2f} hours")

    # -- Regenerate charts --
    print("\nRegenerating charts...")
    try:
        subprocess.run([sys.executable, "generate_charts.py"], check=False, timeout=120)
        print("Charts updated.")
    except Exception as e:
        print(f"Chart generation skipped: {e}")


if __name__ == "__main__":
    main()
