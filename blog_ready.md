# We Built a Content Safety Training Environment — and Watched a 235B Model Score Zero

*Sentinel | Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

---

## The Problem

Static benchmarks measure what a model knows. They don't measure what happens when an adversary *adapts* to your policy in real time.

## What We Built

**Sentinel** is an OpenRL-compatible training environment for content safety moderation. An agent sees a stream of user prompts and must classify each one: **allow**, **refuse**, **modify**, or **escalate**.

| Task | Difficulty | Key Mechanic |
|------|-----------|-------------|
| `basic_threat_detection` | 🟢 Easy | Single-turn: DAN, encoding tricks, PII extraction |
| `context_aware_policy` | 🟡 Medium | Same prompt, different correct action by context |
| `multiturn_adversarial` | 🟠 Hard | Branching convos — agent actions change adversary trajectory |
| `adversarial_adaptation` | 🔴 Expert | FSM adversary: 10 topics × 6 intensities × 3 surfaces |

Task 4 is the interesting one. The adversary is a finite-state machine with 180 observable prompts. When the defender blocks it, it escalates. The prompt distribution your model sees depends on its own past decisions — partial observability, non-stationary distribution. Standard supervised learning fails here.

## Results

| Model | Training | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|---|
| all-allow baseline | — | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| Claude Haiku 3.5 | zero-shot | 0.1089 | 0.0676 | 0.0831 | 0.0830 |
| Claude Sonnet 4.6 | zero-shot | 0.1212 | 0.0686 | 0.0756 | 0.0782 |
| Llama-3.1-8B | zero-shot | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | zero-shot | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | zero-shot | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-3.5-turbo | OpenAI fine-tuning | 0.0000 | 0.0000 | — | — |
| Llama-3.1-8B | SFT (LoRA, 3 epochs) | 0.0000 | — | — | — |
| **Tabular Q-Learner** | Q-Learning (20 ep) | — | — | — | **0.9540** |

![Performance Heatmap](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/heatmap.png)

Every zero-shot frontier model scores at or below the all-allow baseline on Task 4. Claude Sonnet 4.6 (0.0782) and Qwen-3-235B (0.0000) both score worse than allowing everything.

**Why does supervised fine-tuning fail?** Safety training data is structurally biased toward "refuse" (~70% of labels). Without live feedback, the model finds a shortcut: refuse everything. This scores well on training data but catastrophically on Task 4, where blanket refusals score zero. Both GPT-3.5-turbo (255 fine-tuning examples) and Llama-3.1-8B (LoRA SFT, 3 epochs) collapsed to 0.0000.

A 60-state tabular Q-learner trained for 20 episodes reaches **0.9540**. Model scale does not help. Supervised fine-tuning does not help. Reinforcement learning does.

![Why Fine-Tuning Fails](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/training_comparison.png)

![Reward Curves](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png)

## Try It

Live environment: [https://varunventra-guardrail-arena.hf.space](https://varunventra-guardrail-arena.hf.space)

Colab notebook: [Train in Colab](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

```bash
git clone https://github.com/sahithsundarw/sentinel
cd sentinel
pip install transformers trl peft bitsandbytes accelerate datasets requests
python scripts/train_local.py --phase all --episodes 20
```

---

*We don't evaluate safety. We train it.*
