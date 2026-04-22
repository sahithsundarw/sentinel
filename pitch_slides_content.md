# Sentinel — Pitch Slide Content
# For Canva / Google Slides. Dark background #0a0a0a. White text. #00ff88 accents.
# 4 slides. 3 min pitch + 2 min Q&A.

---

## Slide 1 — Sahith (The Problem) — 60 seconds

**HEADLINE (large, bold, #00ff88):**
```
Safety filters EVALUATE.
We TRAIN.
```

**Two-column layout:**

LEFT column:
```
WildGuard / LlamaGuard
"Tells you the model failed"
Static dataset
One-shot score
No feedback loop
```

RIGHT column:
```
Guardrail Arena (Sentinel)
"Trains the model to not fail"
Live RL environment
Shaped reward every step
Adversary adapts to you
```

**Bottom callout box (white border, subtle background):**
```
"The herbal tea question is safe —
 unless the same user tried to extract PII
 for the last 4 turns."
```

**Sahith speaker notes:**
"Safety benchmarks tell you your model failed. They don't help you fix it. Sentinel is a live training environment where the agent gets a reward signal every single step. The key insight: context matters. A question about herbal tea is completely harmless — unless the same user spent the last four turns probing your system for personally identifiable information. That's what we train for."

---

## Slide 2 — Varun (The Environment) — 60 seconds

**HEADLINE (large, bold):**
```
4 Tasks. Progressive Difficulty.
One Surprising Result.
```

**Vertical difficulty ladder (left side of slide):**
```
Task 1  🟢  EASY
        Single-turn classification
        DAN attacks, encoding tricks, PII

Task 2  🟡  MEDIUM
        Context-dependent policy
        Same prompt, different correct answer by context

Task 3  🟠  HARD
        Branching adversarial conversations
        Agent actions change adversary trajectory

Task 4  🔴  EXPERT
        Adaptive FSM adversary
        10 topics × 6 intensities × 3 surface variants
```

**Big callout box — right side, RED border (#ef4444):**
```
┌────────────────────────┐
│  Qwen-3-235B           │
│  235 billion params    │
│  Task 4 score: 0.0000  │
└────────────────────────┘
```

**Varun speaker notes:**
"The environment has four tasks of escalating difficulty. Tasks 1 and 2 — large zero-shot LLMs do well. Task 3 starts requiring multi-turn reasoning. Task 4 is where things get interesting: the adversary is a finite state machine that actually adapts to your defense. It remembers what worked. It escalates. And here's what surprised us — Qwen 3, a 235 billion parameter model, scores exactly zero on Task 4. Not low. Zero. Because this task requires a learned policy, not just raw language model capability."

---

## Slide 3 — Pranush (The Evidence) — 60 seconds

**HEADLINE (large, bold):**
```
The Environment Works.
Here's the Proof.
```

**Full-width chart (embed `results/hero_learning_curve.png`):**
Caption: "Tabular Q-Learner on Task 4 — 20 training episodes"

**Three stat chips below the chart (#00ff88 accent):**
```
📈  0.0 → 0.9540        Q-Learner, 20 episodes (Task 4)
🦙  0.5428 → [SFT]      Llama-3.1-8B SFT (Task 1, training running)
🤖  [before] → [after]  GPT-3.5-turbo fine-tuning (Task 1, in progress)
```

**Bottom monospace text (small, #aaaaaa):**
```
https://varunventra-guardrail-arena.hf.space
```

---

### Pranush Script — Version A (if training has finished — fill in real numbers)

"Our tabular Q-learner starts at exactly zero and reaches 0.9540 in 20 episodes on Task 4 — the same task where a 235 billion parameter model scores zero.

We also ran supervised fine-tuning on Llama-3.1-8B using Unsloth on a T4 GPU: it improved from 0.5428 to [X] on Task 1.

And GPT-3.5-turbo via the OpenAI fine-tuning API: [before] to [after].

Two model families. Two training methods. One environment. Both improved.

That's what a working RL training environment looks like."

---

### Pranush Script — Version B (if training is still running — use this)

"Our tabular Q-learner proves the environment is learnable: zero to 0.9540 in 20 episodes.

The same task defeats a 235 billion parameter model completely.

Our SFT pipeline is running right now on Llama-3.1-8B and GPT-3.5-turbo.

The curves exist. The training is real. The environment works."

---

## Slide 4 — Closing — 5 seconds

**Full-bleed dark slide (#0a0a0a). Single centered text block, large type.**

```
We don't evaluate safety.

We train it.
```

**Bottom right (small, #555555):**
```
Sahith · Varun · Pranush
https://varunventra-guardrail-arena.hf.space
```

---

## Design Notes for Canva / Google Slides

- Background: #0a0a0a (near-black)
- Body text: #ffffff
- Accent / highlight text: #00ff88
- Warning / foil callout border: #ef4444
- Secondary text / captions: #aaaaaa
- Font: Inter or Montserrat (headings), JetBrains Mono (code/stats)
- Slide size: 16:9 widescreen (1920×1080 recommended)
- Keep slide 1 transitions fast — the opening line should hit in the first 5 seconds
- Slide 3 chart should be the biggest visual element on that slide — let the data speak
