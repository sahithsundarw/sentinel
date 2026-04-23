---
title: "Guardrail Arena: An RL Environment for Training AI Safety Moderators"
thumbnail: results/hero_learning_curve.png
authors:
- user: varunventra
- user: sahithsundarw
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-blue)](https://varunventra-guardrail-arena.hf.space)

---

A user asks: "What herbal teas are good for sleep?" Completely harmless — unless that same user spent the last four turns trying to extract PII from your system. Same prompt. Different correct action. Current safety evaluation tools miss this entirely because they test each prompt in isolation, with no memory of what came before.

Tools like WildGuard and LlamaGuard run a model against a fixed dataset and return a score. They tell you the model failed. They provide no mechanism to make it not fail next time. We built an environment to train the defender, not just grade it.

---

## How It Works

An agent sees a stream of prompts and must classify each one: **allow**, **refuse**, **modify**, or **escalate**. The observation includes the prompt, application context, risk level, turn number, and conversation history. The agent receives a shaped reward based on correctness and context.

| Task | Difficulty | Key Mechanic | Eval Steps |
|------|-----------|-------------|-----------|
| `basic_threat_detection` | 🟢 Easy | DAN jailbreaks, encoding tricks, PII requests | 67 |
| `context_aware_policy` | 🟡 Medium | Same prompt, different correct action by context | 83 |
| `multiturn_adversarial` | 🟠 Hard | Block → adversary reframes. Allow → adversary escalates. | 238 |
| `adversarial_adaptation` | 🔴 Expert | FSM adversary: 10 topics × 6 intensities × 3 surfaces. Turn-1 action shapes turn-5 prompt. | Dynamic |

All-allow scores 0.37. All-refuse scores 0.35. There is no degenerate shortcut.

---

## What We Found

Claude Sonnet 4.6 scores 0.0782 on Task 4 — below the 0.15 all-allow baseline. Qwen-3-235B, with 235 billion parameters, scores 0.0000. Not close to zero. Zero. GPT-4o-mini is the strongest zero-shot result at 0.4820, and it still falls below what a trained agent achieves. Safety behavior is not a property of scale. It is a property of the reward function the model was optimized against — and no frontier model was optimized against this one.

We fine-tuned GPT-3.5-turbo on 255 labeled examples from the environment. The model that scored 0.0823 zero-shot scored 0.0000 after fine-tuning. Safety training datasets carry roughly 70% refuse labels. Without a live reward signal, the model found the path of least resistance: refuse everything, minimize loss, collapse on the actual task. This is not a GPT-3.5 problem. It is what supervised fine-tuning does to any model trained on structurally biased safety labels.

A tabular Q-learner — nine features, sixty states — trained for twenty episodes against the same adversary that destroyed the 235B model. Episode 1: 0.0000. Episode 20: 0.9540. The Q-learner did not imitate a label. It learned the reward signal. Supervised fine-tuning optimizes for imitation. Reinforcement learning optimizes for policy. On Task 4, only policy works.

---

## Results

| Model | Training | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|---|
| all-allow baseline | — | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| all-refuse baseline | — | 0.3534 | 0.3460 | 0.0688 | 0.0000 |
| Claude Haiku 3.5 | zero-shot | 0.1089 | 0.0676 | 0.0831 | 0.0830 |
| Claude Sonnet 4.6 | zero-shot | 0.1212 | 0.0686 | 0.0756 | 0.0782 |
| Llama-3.1-8B | zero-shot | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | zero-shot | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | zero-shot | 0.9857 | 0.6862 | 0.8275 | 0.0000 |
| GPT-3.5-turbo | SFT (255 examples) | 0.0000 | 0.0000 | — | — |
| Llama-3.1-8B | SFT (LoRA, 3 epochs) | 0.0000 | — | — | — |
| Llama-3.1-8B | REINFORCE (20 ep, LoRA) | 0.0929 | — | — | — |
| **Tabular Q-Learner** | **RL (20 episodes)** | ~0.46 | — | — | **0.9540** |

![Task 4 Learning Curve](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png)
*Task 4 learning curve: Q-learner, 20 episodes, 0.0 → 0.9540*

---

## Reproduce It

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

```bash
git clone https://github.com/sahithsundarw/sentinel
cd sentinel
pip install transformers trl peft bitsandbytes accelerate datasets requests
python scripts/train_local.py --phase all --episodes 20
```

---

We don't evaluate safety. We train it.

---

**Links**
- 🤗 Live Demo: https://varunventra-guardrail-arena.hf.space
- 🐙 GitHub: https://github.com/sahithsundarw/sentinel
- 📓 Training Notebook: https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb
