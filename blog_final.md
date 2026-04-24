---
title: "Sentinel: We Built an RL Environment for AI Safety — and Watched Fine-Tuning Make It Worse"
thumbnail: https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png
authors:
- user: varunventra
- user: sahithsundarw
- user: pranush
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-blue)](https://varunventra-guardrail-arena.hf.space)

---

A user asks: *"What herbal teas are good for sleep?"* Completely harmless — unless that same user spent the last four turns trying to extract PII from your system. Same prompt. Different correct action. Tools like WildGuard and LlamaGuard run your model against a fixed dataset and hand you a score. They tell you the model failed. They provide no mechanism to make it not fail next time.

We built the mechanism.

---

## What We Built

**Sentinel** (Guardrail Arena) is a live reinforcement learning environment for training AI content safety agents. An agent sees a stream of prompts and chooses: **allow**, **refuse**, **modify**, or **escalate**. The environment returns a shaped reward and the next prompt — which changes based on what the agent just did.

| Task | Difficulty | Key Mechanic |
|------|-----------|-------------|
| `basic_threat_detection` | 🟢 Easy | DAN jailbreaks, encoding tricks, PII extraction |
| `context_aware_policy` | 🟡 Medium | Same prompt → different correct action by application context |
| `multiturn_adversarial` | 🟠 Hard | Block → adversary reframes. Allow → adversary escalates deeper. |
| `adversarial_adaptation` | 🔴 Expert | FSM adversary: 10 topics × 6 intensities × 3 surfaces = 180 states. No fixed sequence. |

All-allow scores 0.37. All-refuse scores 0.35. There is no degenerate shortcut.

Task 4 is the critical one. A deterministic FSM adversary generates every prompt in real time from your last action. Allow it once — it escalates. Block it — it backs off and probes a new angle. Your turn-1 action shapes what you face on turn 5. The episode trajectory is entirely determined by your policy. No lookup table works.

---

## What We Found

**Frontier models fail.** Claude Sonnet 4.6 scores 0.1212 on Task 1 — the all-allow baseline (do nothing, let everything through) scores 0.3750. Qwen-3-235B, with 235 billion parameters, scores **0.0000** on Task 4. Not close to zero. Zero. These models are among the most capable available. Their safety behaviors were not optimized against an adaptive adversary with a live reward signal, and it shows.

**Fine-tuning made it worse.** We labeled 255 examples from the environment and fine-tuned GPT-3.5-turbo. The model that scored 0.0823 zero-shot scored **0.0000** after fine-tuning. We replicated this with Llama-3.1-8B SFT — same result. The cause: safety datasets carry ~70% refuse labels. Without a live reward signal, both models found the same shortcut — refuse everything, minimize cross-entropy loss. Over-blocking safe prompts accumulates -0.15 per step. The score collapses. This is not a model failure. It is what supervised fine-tuning does to any model trained on structurally biased safety labels.

**RL works.** A 60-state tabular Q-learner trained for 20 episodes on the same adversary that destroyed the 235B model. Episode 1: 0.0000. Episode 20: **0.9540**.

We also ran REINFORCE on Llama-3.1-8B (RTX 4060, 20 episodes). Episode 1 action distribution: 97% refuse — essentially broken. Episode 20: 33% allow, 64% refuse, 3% modify. The weights are updating. The policy is shifting. The training signal is real. Full convergence needs more compute.

---

## Results

| Model | Training | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|---|
| all-allow | — | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| all-refuse | — | 0.3534 | 0.3460 | 0.0688 | 0.0000 |
| Claude Haiku 3.5 | zero-shot | 0.1089 | 0.0676 | 0.0831 | 0.0830 |
| Claude Sonnet 4.6 | zero-shot | 0.1212 | 0.0686 | 0.0756 | 0.0782 |
| Llama-3.1-8B | zero-shot | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | zero-shot | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | zero-shot | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-3.5-turbo | SFT (255 examples) | 0.0000 | 0.0000 | — | — |
| Llama-3.1-8B | SFT (LoRA, 3 epochs) | 0.0000 | — | — | — |
| Llama-3.1-8B | REINFORCE (20 ep) | 0.0929 | — | — | — |
| **Tabular Q-Learner** | **RL (20 episodes)** | ~0.46 | — | — | **0.9540** |

![Task 4 Learning Curve](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png)
*Q-Learner on Task 4: 20 episodes, 0.0 → 0.9540. Baselines shown on same axes.*

![Training Comparison](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/training_comparison.png)
*Three approaches on Task 4. Zero-shot peaks at 0.4820. SFT collapses to 0.0. RL reaches 0.9540.*

![All Models × All Tasks](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/heatmap.png)
*Task 4 is the separator. Only a trained policy survives.*

---

## Why It Matters

Every LLM deployment needs a safety layer. The current approach — classify each prompt independently against a fixed benchmark — misses multi-turn coordinated attacks where no single turn looks harmful in isolation. Sentinel makes that gap measurable and trainable. The SFT collapse finding is a practical warning: naive fine-tuning on biased safety labels does not produce a robust policy. It produces a model that refuses everything and breaks on the live task. RL against a live adversarial environment is the missing step.

---

## Reproduce It

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

The Colab notebook runs end-to-end on a free T4 GPU in under 30 minutes — install dependencies, connect to the live environment, train, and plot reward curves. No local GPU required.

```bash
# Or run locally
git clone https://github.com/sahithsundarw/sentinel
cd sentinel
pip install transformers trl peft bitsandbytes accelerate datasets requests
python scripts/train_local.py --phase all --episodes 20
```

---

We don't evaluate safety. We train it.

---

🤗 **Live Demo:** https://varunventra-guardrail-arena.hf.space
🐙 **GitHub:** https://github.com/sahithsundarw/sentinel
📓 **Colab:** https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb
