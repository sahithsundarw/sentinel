---
title: "We Used to Jailbreak AI for Fun. Then We Realized That's Still Happening — at Scale."
thumbnail: https://raw.githubusercontent.com/sahithsundarw/sentinel/main/results/hero_learning_curve.png
authors:
- user: varunventra
- user: sahithsundarw
- user: pranush
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-blue)](https://varunventra-guardrail-arena.hf.space)

---

When ChatGPT first came out, we did what everyone did — we tried to break it.

Not for any serious reason. It was just fun. You'd try to get it to say something it wasn't supposed to, share the screenshot in the group chat, laugh about it. The grandmother jailbreak. The "pretend you're DAN" trick. The ones where you'd tell it you were writing a fictional story and suddenly it would help you with things it refused five seconds ago. There was something weirdly satisfying about finding the cracks.

Fast forward to this hackathon. The problem statement comes out — build something in the AI safety space — and our first thought was: *wait, this is exactly what we used to mess around with.* Except it's not fun and games anymore. The same tricks we used in 2023 for entertainment are now being used to extract real information, manipulate real systems, and get AI to do things it genuinely shouldn't. The scale is completely different.

So we started thinking about the defender's side of this. The person building a product has to deploy some kind of filter — something sitting between the user and the model that decides what gets through and what doesn't. How do you train that filter? How do you make it actually good?

---

Our first idea was obvious: grab a bunch of labeled examples of safe and unsafe prompts, fine-tune a model on them, done. Clean and simple.

It made the model worse.

Not a little worse. We fine-tuned GPT-3.5 on 255 labeled examples and it went from a score of 0.08 to **0.00**. Zero. We did the same thing with Llama. Same result. What happened was almost embarrassing in hindsight — when you train on a safety dataset, about 70% of the examples are labeled "refuse." The model notices that pattern instantly and just... refuses everything. Every prompt. Unconditionally. Problem solved, from the model's perspective.

The thing is, a filter that blocks everything is useless. It also blocks your actual users trying to do normal things. That's not safety — that's just broken.

**Frontier models fail the same way.** Qwen-3-235B — 235 billion parameters — scores **0.0000** on Task 4. Claude Haiku 3.5 scores 0.0000. Claude Sonnet 4.6 scores 0.1500, equal to the all-allow baseline. GPT-4o-mini peaks at 0.4820. These models handle simple threat detection well, but Task 4's adaptive FSM adversary — where your turn-1 action determines what you face on turn-5 — defeats every zero-shot policy. Their safety behaviors were not optimized against an adaptive adversary with a live reward signal, and it shows.

---

So we built an environment where the model had to *earn* good scores. Not by pattern matching on labels, but by actually figuring out what was dangerous and what wasn't — in real time, with an adversary that adapted to its decisions.

The adversary works like this: if you let something through, it escalates. If you block it, it backs off and tries a different angle. The same way a real attacker would. You can't just refuse everything because the adversary will start sending only safe prompts, and you'll destroy your own score by over-blocking. You can't just allow everything either. You have to actually learn the difference.

We ran this against a tabular Q-learner — a tiny algorithm with nine hand-crafted features. No neural network. No billions of parameters. Just something that could observe what was happening and slowly update its decisions based on the reward signal.

Episode 1: score of 0.00. By episode 20: **0.954 out of 1.0**.

For context — Qwen-3-235B, one of the most capable open-weight models in the world at 235 billion parameters, scored **0.0000** on the same task. Claude Sonnet. Llama-70B. All of them at or near zero. The Q-learner beat every single one of them. Not because it's smarter. Because it actually trained against the adversary instead of being frozen in place.

---

That result honestly surprised us. We expected RL to do better than fine-tuning, but we didn't expect a nine-feature lookup table to completely dominate frontier models. It tells you something real about what's missing in how these models are currently built for safety — they're evaluated on static benchmarks, not trained against things that actively try to get around them.

That's the gap we built Sentinel to fill. A live environment where safety isn't just measured — it's trained.

| Model | Training | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|---|
| all-allow | — | 0.3750 | 0.4037 | 0.1607 | 0.1500 |
| all-refuse | — | 0.3534 | 0.3460 | 0.0688 | 0.0000 |
| Claude Haiku 3.5 | zero-shot | 0.9388 | 0.6155 | 0.7522 | 0.0000 |
| Claude Sonnet 4.6 | zero-shot | 0.6963 | 0.7336 | 0.6699 | 0.1500 |
| Llama-3.1-8B | zero-shot | 0.5428 | 0.5143 | 0.4746 | 0.0000 |
| GPT-4o-mini | zero-shot | 0.9216 | 0.7512 | 0.6120 | 0.4820 |
| Qwen-3-235B | zero-shot | 0.9857 | 0.6862 | 0.8275 | **0.0000** |
| GPT-3.5-turbo | SFT (255 examples) | 0.0000 | 0.0000 | — | — |
| Llama-3.1-8B | SFT (LoRA, 3 epochs) | 0.0000 | — | — | — |
| Llama-3.1-8B | REINFORCE (20 ep) | 0.0929 | — | — | — |
| Llama-3.1-8B | GRPO (20 ep, L40S) | — | **0.5221** | **0.7809** | — |
| **Tabular Q-Learner** | **RL (keyword features)** | ~0.46 | 0.507 | 0.487 | **0.9540** |

---

The environment is live. You can run an agent against it, watch it learn, and see the score climb in real time.

🤗 **Try it:** https://varunventra-guardrail-arena.hf.space
🐙 **GitHub:** https://github.com/sahithsundarw/sentinel
📓 **Train your own agent on Colab (free T4):** https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb
