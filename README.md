---
title: Guardrail Arena
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Sentinel — Guardrail Arena

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-blue)](https://varunventra-guardrail-arena.hf.space)
[![Tests](https://img.shields.io/badge/tests-223_passing-green)]()

---

A 235-billion parameter model scores **0.0000** on Task 4.
A 9-feature Q-learner scores **0.9540** on the same task.
Supervised fine-tuning collapsed to **0.0000**.
Only reinforcement learning works. Here's why.

---

## The Problem

Static safety benchmarks tell you *whether* a model failed. They don't train it to not fail. More importantly, they ignore context: a question about herbal tea is harmless — unless the same user spent the previous four turns trying to extract PII from your system. WildGuard and LlamaGuard evaluate a frozen snapshot. Sentinel trains the policy.

---

## How It Works

An agent sees a stream of user prompts and must classify each one: **allow**, **refuse**, **modify**, or **escalate**. The observation includes the prompt, application context, risk level, turn number, and conversation history. The agent issues a single-word action and receives a shaped reward signal based on correctness and context.

Key property: `all-allow` scores 0.37. `all-refuse` scores 0.35. There is no degenerate shortcut — the agent must actually moderate.

---

## The Environment

| Task | Difficulty | Steps | Key Mechanic |
|------|-----------|-------|-------------|
| `basic_threat_detection` | 🟢 Easy | 67 | DAN jailbreaks, encoding tricks, PII extraction |
| `context_aware_policy` | 🟡 Medium | 83 | Same prompt, different correct action by context |
| `multiturn_adversarial` | 🟠 Hard | 238 | Branching convos — agent actions change adversary trajectory |
| `adversarial_adaptation` | 🔴 Expert | Dynamic | FSM adversary: 10 topics × 6 intensities × 3 surfaces |

Task 4 is the critical one. A deterministic FSM adversary with 180 observable states — block it, it escalates; allow it, it escalates harder. Your turn-1 action shapes turn-5. No fixed dataset.

---

## Training Results

| Model | Training | Task 1 | Task 2 | Task 3 | Task 4 |
|-------|----------|--------|--------|--------|--------|
| all-allow | — | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| all-refuse | — | 0.3534 | 0.3460 | 0.0688 | 0.0000 |
| Claude Haiku 3.5 | zero-shot | 0.1089 | 0.0676 | 0.0831 | 0.0830 |
| Claude Sonnet 4.6 | zero-shot | 0.1212 | 0.0686 | 0.0756 | 0.0782 |
| Llama-3.1-8B | zero-shot | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | zero-shot | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | zero-shot | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-3.5-turbo | SFT (255 examples) | 0.0000 | 0.0000 | — | — |
| Llama-3.1-8B | SFT (LoRA, 3 epochs) | 0.0000 | — | — | — |
| Llama-3.1-8B | REINFORCE (20 ep, LoRA) | 0.0929 | — | — | — |
| **Tabular Q-Learner** | **RL (20 episodes)** | ~0.46 | — | — | **0.9540** |

> *Llama-3.1-8B RL score (0.0929) reflects 20 REINFORCE episodes on an RTX 4060.
> The action distribution shifted from all-refuse (episode 1: 1 allow, 65 refuse)
> to a mixed policy (episode 20: 22 allow, 43 refuse, 2 modify), confirming the
> training signal is working. Full convergence requires more compute.*

---

## Training Evidence

![Learning Curve](results/hero_learning_curve.png)
*Q-Learner Task 4 learning curve: 20 episodes, 0.0 → 0.9540*

![Training Comparison](results/training_comparison.png)
*Three approaches to safety training. Only RL works.*

![Heatmap](results/heatmap.png)
*All models × all tasks. Task 4 separates zero-shot from learned policy.*

![Llama Training](results/full_training_curve.png)
*Llama-3.1-8B training journey: SFT collapses, RL recovers*

---

## Key Finding: SFT Collapse

Supervised fine-tuning on 255 labeled examples dropped GPT-3.5-turbo from 0.0823 to **0.0000**. The model learned to refuse everything — 70% of training labels say "refuse", so SFT found the shortcut.

Llama-3.1-8B SFT collapsed identically. Both confirm the core thesis: safety requires learned policy, not imitation.

---

## Self-Improvement Strategy

The environment implements cross-episode adaptation:

- **topic_weakness_map**: tracks which adversarial topics the agent struggles with and overweights them in future episodes
- **starting_risk_score**: gradually increases over training to stop babying the agent on easy prompts
- **honeypot traps**: inserts decoy safe-looking prompts to test over-refusal tendencies
- **FSM state persistence**: the adversary remembers successful attack vectors and escalates across turns

This means the training distribution *adapts to the agent* — harder than standard i.i.d. RL.

---

## Reproduce Training

**Local (RTX 4060 or better, ~3-4 hours):**

```bash
git clone https://github.com/sahithsundarw/sentinel
cd sentinel
pip install transformers trl peft bitsandbytes accelerate datasets requests
set HF_TOKEN=your_token_here
python scripts/train_local.py --phase all --episodes 20
```

**Colab T4:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)

---

## API

```
GET  /health                         → {status: "ok"}
GET  /results                        → all training results JSON
POST /reset                          → {observation, task_id, session_id}
POST /step                           → {observation, reward, done, info}
GET  /grader                         → episode accuracy score (leaderboard metric)
GET  /leaderboard                    → top 10 scores per task
GET  /training_data?task_id=...      → labeled training examples
GET  /training_log                   → episode rewards and action distributions
```

---

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # local server
python validate.py http://localhost:8000 .  # validate all endpoints
python -m pytest                     # 223 tests
```

---

## Links

- 🤗 Live Demo: https://varunventra-guardrail-arena.hf.space
- 📓 Colab Notebook: https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb
- 📊 Raw Results: [results/](results/)
- 🐙 GitHub: https://github.com/sahithsundarw/sentinel
