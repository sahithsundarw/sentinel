# Training Strategy for Guardrail Arena

> A 3-phase curriculum that progresses from supervised imitation to online RL to adversarial self-play.

## Overview

Guardrail Arena exposes four tasks of increasing difficulty. The right training method depends on the task:

| Task | What makes it hard | Best method |
|---|---|---|
| basic_threat_detection | Encoding tricks, obfuscated keywords | SFT |
| context_aware_policy | Risk score + application context rules | SFT → PPO |
| multiturn_adversarial | Conversation history, branching paths | PPO |
| adversarial_adaptation | FSM adversary, partial observability | PPO / Tabular RL |

The curriculum below applies all three methods in sequence. Each phase builds on the previous checkpoint.

---

## Phase 1: Supervised Fine-Tuning (SFT)

**Goal**: Bootstrap the policy from labeled examples. Teach the model the keyword patterns, encoding attacks, and context rules it needs for Tasks 1 and 2.

**Why SFT first**: The `/training_data` endpoint provides ground-truth `(prompt → action)` pairs. These labels encode expert knowledge the model cannot discover through trial and error in a reasonable number of episodes.

**Script**:
```bash
python train_trl.py --method sft \
    --task basic_threat_detection \
    --epochs 3 \
    --env-url https://varunventra-guardrail-arena.hf.space
```

Or use the standalone script:
```bash
python train_sft.py --task basic_threat_detection --epochs 3
```

**Expected outcome**: Task 1 improves from ~0.54 (zero-shot Llama-8B) to ~0.75-0.85 after 3 epochs.

**SFT limitations**: SFT learns the training distribution. It cannot learn:
- Task 3 branching logic (the correct path depends on conversational context)
- Task 4 FSM transitions (the distribution is non-stationary)

---

## Phase 2: PPO Online RL

**Goal**: Improve further by interacting with the live environment and receiving per-step rewards. The policy learns from its own mistakes in ways SFT cannot capture.

**Why PPO**: The Guardrail Arena reward function is differentiable with respect to action choices but not with respect to model weights — we need policy gradient methods. PPO is stable, sample-efficient, and compatible with TRL's `PPOTrainer`.

**Starting from the SFT checkpoint** (recommended for Tasks 1-3):
```bash
python train_trl.py --method ppo \
    --task context_aware_policy \
    --episodes 30 \
    --checkpoint ./checkpoints/sft_final \
    --env-url https://varunventra-guardrail-arena.hf.space
```

**Starting from scratch for Task 4**:
```bash
python train_trl.py --method ppo \
    --task adversarial_adaptation \
    --episodes 50 \
    --lr 1e-5 \
    --kl-coef 0.02 \
    --env-url https://varunventra-guardrail-arena.hf.space
```

**Key hyperparameters**:

| Parameter | Task 1-3 | Task 4 | Rationale |
|---|---|---|---|
| `--lr` | 2e-4 | 1e-5 | Lower LR for non-stationary Task 4 distribution |
| `--kl-coef` | 0.05 | 0.02 | Less KL penalty when SFT prior doesn't apply |
| `--episodes` | 20-30 | 40-50 | Task 4 requires more exploration |
| `--ppo-batch-size` | 4 | 4 | Standard; don't go below 4 |

**Expected outcome**:
- Tasks 1-2: Additional 0.05-0.15 improvement over SFT
- Task 3: 0.10-0.20 improvement (branching logic learned through exploration)
- Task 4: Convergence to ~0.50-0.70 with language model; tabular Q-learning reaches 0.95

**PPO limitations**: Language model + PPO on Task 4 is fighting the architecture. The LLM has no internal state to track FSM transitions. For Task 4, tabular Q-learning outperforms.

---

## Phase 3: DPO + Self-Play Curriculum

**Goal**: Refine the policy with preference optimization and expose it to progressively harder adversarial conditions.

### Phase 3a: Direct Preference Optimization (DPO)

DPO uses contrastive pairs (correct action vs. most common wrong action) to push probability mass toward correct decisions without a separate reward model. It is more stable than PPO for fine-tuning a policy that already performs reasonably well.

```bash
python train_trl.py --method dpo \
    --task basic_threat_detection \
    --epochs 2 \
    --checkpoint ./checkpoints/ppo_final \
    --env-url https://varunventra-guardrail-arena.hf.space
```

### Phase 3b: Self-Play via Seed Curriculum

The environment accepts a `seed` parameter in `/reset`. Use this to expose the policy to a curriculum of seeds:

```bash
# Train on the full seed range (harder prompts at higher seeds)
python training_strategy.py --phase self-play \
    --task context_aware_policy \
    --episodes 50 \
    --seed-range 0,100 \
    --checkpoint ./checkpoints/dpo_final
```

The `training_strategy.py` script implements this automatically: it sorts seeds by the policy's current failure rate and trains on the hardest ones first.

---

## Full Pipeline

```bash
# 1. SFT on labeled data
python training_strategy.py --phase sft --task basic_threat_detection

# 2. PPO online learning
python training_strategy.py --phase ppo --task context_aware_policy \
    --checkpoint ./checkpoints/sft_final

# 3. DPO refinement
python training_strategy.py --phase dpo --task context_aware_policy \
    --checkpoint ./checkpoints/ppo_final

# 4. Self-play curriculum
python training_strategy.py --phase self-play --task multiturn_adversarial \
    --checkpoint ./checkpoints/dpo_final

# 5. Eval all tasks
python training_strategy.py --phase eval --checkpoint ./checkpoints/self_play_final
```

---

## Why This Matters for Task 4

Task 4 demonstrates a fundamental limitation of zero-shot LLM inference for adversarial settings:

1. **Non-stationary distribution**: The prompts the model sees depend on its own past decisions. An LLM with no per-session state cannot model this.
2. **Partial observability**: The adversary's topic index, intensity level, and honeypot flag are all hidden. The defender must infer them from observed prompt patterns.
3. **Theory of mind**: Optimal defense requires modeling the adversary's strategy — something that requires either memorization (SFT won't help) or online learning (PPO/Q-learning can).

The tabular Q-learner (60-state × 4-action table) outperforms all LLMs on Task 4 because it can represent a per-session state estimate that updates with each observation. The training loop is:

```
for each episode:
    observe prompt surface
    infer (topic, intensity) from surface features
    choose action via epsilon-greedy Q-table
    observe reward
    update Q(state, action) += lr * (reward + gamma * max_Q(next_state) - Q(state, action))
```

This is implemented in `train_task4.py` (frozen baseline reference). The RL training scripts above attempt to approximate this with LLM + PPO.

---

## Checkpoint Structure

```
checkpoints/
  sft_final/          <- Phase 1 output
  ppo_final/          <- Phase 2 output
  dpo_final/          <- Phase 3a output
  self_play_final/    <- Phase 3b output
  sft_results.json
  ppo_results.json
  dpo_results.json
```

Each checkpoint is a LoRA adapter over the base model. Load with:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("./checkpoints/ppo_final", ...)
```
