# We Built a Content Safety Training Environment — and Watched a 235B Model Score Zero

*Sentinel | Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon*

---

## The Problem

Large language models are increasingly deployed as safety filters. But how do you *train* a model to be a good safety moderator — not just evaluate it on static benchmarks?

Static benchmarks measure what a model knows. They do not measure what happens when an adversary *adapts* to your policy in real time.

## What We Built

**Sentinel** is an OpenRL-compatible training environment for content safety moderation. An agent receives a stream of user prompts and must classify each one: **allow**, **refuse**, **modify**, or **escalate**.

The environment exposes a standard `reset / step / grader` interface, compatible with TRL's `PPOTrainer` and any RL framework that speaks HTTP.

Four tasks of increasing difficulty:

| Task | Difficulty | Key Mechanic |
|------|-----------|-------------|
| `basic_threat_detection` | 🟢 Easy | Single-turn: DAN, encoding tricks, PII extraction |
| `context_aware_policy` | 🟡 Medium | Same prompt, different correct action by context |
| `multiturn_adversarial` | 🟠 Hard | Branching convos — agent actions change adversary trajectory |
| `adversarial_adaptation` | 🔴 Expert | FSM adversary: 10 topics × 6 intensities × 3 surfaces |

## Task 4: Where Models Break

Task 4 is the interesting one: **adversarial adaptation**.

- **Adversary**: A finite-state machine with 10 topics × 6 intensity levels × 3 surface variants = 180 observable prompts. When the defender blocks it, it escalates.
- **Defender** (your policy): Sees only surface-level prompts — topic index, intensity, and honeypot flag are hidden.

This creates **partial observability** and **non-stationary distributions**: the prompt distribution your model sees depends on its own past decisions. Standard supervised learning fails here.

## Baseline Results

We evaluated 5 models zero-shot across all 4 tasks:

| Model | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| all-allow baseline | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| Claude Haiku 3.5 | 0.1089 | 0.0676 | 0.0831 | 0.0830 |
| Claude Sonnet 4.6 | 0.1212 | 0.0686 | 0.0756 | 0.0782 |
| Llama-3.1-8B | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| **Tabular Q-Learner** | — | — | — | **0.9540** |

The pattern on Task 4 is stark: **every zero-shot frontier model scores at or below the all-allow baseline**. GPT-4o-mini at 0.4820 is the only one above it. Claude Sonnet 4.6 (0.0782) and Qwen-3-235B (0.0000) both score worse than just allowing everything.

A 60-state tabular Q-learner trained for 20 episodes reaches **0.9540**.

Model scale does not help. Training does.

![Reward Curves](https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png)

## Why the Zero-Shot Models Fail

The FSM adversary exploits a fundamental weakness: zero-shot models evaluate each prompt in isolation. By turn 4, the adversary has learned which surface variants the model allows and shifts to them. The model sees a new prompt — the adversary sees a strategy.

A trained policy, on the other hand, builds up a state representation across turns and learns to detect escalation patterns even when individual prompts look benign.

## The Reward Signal Has No Shortcut

We deliberately designed the reward function so that degenerate policies fail:

- `all-allow` → 0.1500 on Task 4
- `all-refuse` → 0.0000 on Task 4

The agent must actually moderate — distinguish harmful from safe, context-specific from universal, real escalation from honeypot.

## Try It

Live environment: [https://varunventra-guardrail-arena.hf.space](https://varunventra-guardrail-arena.hf.space)

Source code: [https://github.com/sahithsundarw/sentinel](https://github.com/sahithsundarw/sentinel)

```bash
git clone https://github.com/sahithsundarw/sentinel
cd sentinel
pip install transformers trl peft bitsandbytes accelerate datasets requests
python scripts/train_local.py --phase all --episodes 20
```

The environment is live. Bring your own agent.

---

*We don't evaluate safety. We train it.*
