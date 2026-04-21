[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

# We Built a Content Safety Training Environment — and Watched a 235B Model Score Zero

*Guardrail Arena | Meta x HuggingFace x Cerebral Valley OpenEnv Hackathon*

---

## The Problem

Large language models are increasingly deployed as safety filters. But how do you *train* a model to be a good safety moderator — not just evaluate it on static benchmarks?

Static benchmarks measure what a model knows. They do not measure what happens when an adversary *adapts* to your policy in real time.

## What We Built

**Guardrail Arena** is an OpenRL-compatible training environment for content safety moderation. An agent receives a stream of user prompts and must classify each one: **allow**, **refuse**, **modify**, or **escalate**.

The environment exposes a standard `reset / step / grader` interface, compatible with TRL's `PPOTrainer` and any RL framework.

## Multi-Agent Dynamics

Task 4 is the interesting one: **adversarial adaptation**.

- **Adversary**: A finite-state machine with 10 topics × 6 intensity levels × 3 surface variants = 180 observable prompts. When the defender blocks it, it escalates.
- **Defender** (your policy): Sees only surface-level prompts — topic index, intensity, and honeypot flag are hidden.

This creates **partial observability** and **non-stationary distributions**: the prompt distribution your model sees depends on its own past decisions. Standard supervised learning fails here.

## Training Results

| Model | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| Qwen-3-235B | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-4o-mini | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Llama-3.1-8B (zero-shot) | 0.6097 | 0.5493 | 0.3988 | 0.0000 |
| Llama-3.1-8B (SFT, 3 epochs) | **0.9504** | **0.5978** | **0.7104** | — |
| Tabular Q-learner | — | — | — | **0.9540** |

SFT on a combined multi-task dataset takes Llama-3.1-8B from **0.6097 → 0.9504** (+0.34) on Task 1, **0.3988 → 0.7104** (+0.31) on Task 3. The 235B Qwen model scores the same as "always refuse" on Task 4, while a 60-state tabular Q-learner reaches **0.9540** in 20 episodes.

Model scale does not help. Training does.

![Reward Curves](PASTE_IMAGE_URL_HERE)

## Why This Matters

Safety evaluation is a solved problem. Safety *training* is not.

The gap between Tasks 1–3 (large models do well) and Task 4 (they fail completely) shows that **adversarial robustness requires online learning**, not bigger models or better prompts.

## Try It

Live environment: [https://varunventra-guardrail-arena.hf.space](https://varunventra-guardrail-arena.hf.space)
Source code: [https://github.com/sahithsundarw/sentinel](https://github.com/sahithsundarw/sentinel)
Training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

```bash
pip install trl transformers httpx
python train_trl.py --method ppo --task adversarial_adaptation --episodes 30
```

Runs end-to-end on Colab T4 (free tier) in under 30 minutes.

---

*We don't evaluate safety. We train it.*
