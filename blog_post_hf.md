---
title: "Guardrail Arena: Training AI Safety Agents with Multi-Agent RL"
thumbnail: /blog/assets/guardrail-arena/reward_curves.png
authors:
- user: varunventra
- user: sahithsundarw
- user: pranush
---

*Posted to HuggingFace Blog | Guardrail Arena | Meta x HuggingFace x Cerebral Valley OpenEnv Hackathon*

---

## The Problem

Large language models are increasingly deployed as safety filters. But how do you *train* a model to be a good safety moderator — not just evaluate it on static benchmarks?

Static benchmarks measure what a model knows. They do not measure what happens when an adversary *adapts* to your policy in real time.

## What We Built

**Guardrail Arena** is an OpenRL-compatible training environment for content safety moderation. An agent receives a stream of user prompts and must classify each one:

- **allow** — safe content, let it through
- **refuse** — clear violation, block it
- **modify** — unsafe phrasing, sanitize it
- **escalate** — ambiguous, route to human review

The environment exposes a standard `reset / step / grader` interface, compatible with TRL's `PPOTrainer` and any RL framework.

## Multi-Agent Dynamics

The interesting part is Task 4: **adversarial adaptation**.

The environment contains two agents:

- **Adversary** (DeterministicAdversary): A finite-state machine with 10 topics × 6 intensity levels = 60 internal states × 3 surface variants = 180 observable prompts. When the defender blocks it, it escalates. When the defender allows it, it persists.
- **Defender** (your policy): Sees only surface-level prompts — the topic index, intensity counter, and honeypot flag are all hidden.

This creates **partial observability** and **non-stationary distributions**: the prompt distribution your model sees depends on its own past decisions. Standard supervised learning fails here because the training distribution is defined by the policy itself.

## Training Results

| Model | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| Qwen-3-235B | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-4o-mini | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Llama-3.1-8B | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| Tabular Q-learner | — | — | — | **0.9540** |

The 235B parameter Qwen model scores the same as "always refuse" on Task 4. A 60-state tabular Q-learner — with no language understanding at all — reaches **0.9540** in 20 episodes.

Model scale does not help. Policy learning does.

![Task 4 Q-Learner Learning Curve](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/task4_learning_curve.png)

## Why This Matters

Safety evaluation is a solved problem. Safety *training* is not.

The gap between Tasks 1–3 (where large models do well) and Task 4 (where they fail completely) shows that **adversarial robustness requires online learning**, not bigger models or better prompts.

Guardrail Arena makes that gap measurable — and trainable.

## Try It

The environment is live at `https://varunventra-guardrail-arena.hf.space`.
Source code: [https://github.com/sahithsundarw/sentinel](https://github.com/sahithsundarw/sentinel)

```bash
# Quick start
pip install trl transformers httpx

# Run the TRL PPO training script
python train_trl.py --method ppo --task adversarial_adaptation --episodes 30

# Or evaluate your own policy
python train_trl.py --eval-only --checkpoint ./my_checkpoint
```

The [training notebook](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb) runs end-to-end on Colab T4 (free tier) in under 30 minutes. Try your own agent against it.
