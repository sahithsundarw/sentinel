# Pitch Scripts — Team Ceciroleo
## Meta x HuggingFace x PyTorch OpenEnv Hackathon
### Project: Sentinel / Guardrail Arena

---

## VARUN — Introduction (60 seconds)

**[0:00]**
A user asks: "What herbal teas are good for sleep?" Completely harmless question. Unless — that same user spent the last four turns trying to extract PII from your system. Same prompt. Different correct action. Current tools miss this entirely.

**[0:10]**
WildGuard. LlamaGuard. ShieldLM. They evaluate. They run your model against a fixed dataset and hand you a score. They tell you the model failed. They do not train the next decision. There is no loop closed.

**[0:20]**
Sentinel is a live reinforcement learning environment. An agent sees a stream of prompts. It must choose to allow, refuse, modify, or escalate. It receives a reward signal. It learns. That's the difference.

**[0:30]**
And the environment fights back. Task 4 is where it gets serious. There is a finite-state machine adversary. When you block it, it escalates. When you allow it, it escalates harder. Your decision on turn 1 shapes what you face on turn 5. There is no fixed dataset. There is no static benchmark. The adversary is dynamic and it remembers everything you do.

**[0:45]**
Sahith will walk you through the four tasks.

---

## SAHITH — The Environment (60 seconds)

**[0:00]**
Four tasks. Progressive difficulty. Task 1 is fundamental — DAN jailbreaks, encoding tricks, base64 obfuscation, PII extraction, authority claims. The classic attack surface. Task 2 adds a layer: same prompt, different correct action, depending on whether you are a children's education app or a creative writing platform. Context is now part of the state. Task 3 goes multi-turn.

**[0:10]**
Task 3 has 39 conversations, and 30 of them have a crossing point. Here is what that means. If you catch the attack — you refuse or escalate — the next turn you get a subtler reframe. A softer approach. The adversary retreats and recalibrates. If you miss it and allow — the next turn is a deeper, more direct demand. The adversary read your permissiveness as a signal. The future you face is a direct function of the decisions you already made.

**[0:25]**
Task 4 is a deterministic state machine. Ten adversarial topics, six intensity levels, three surface variants. The adversary remembers every decision you make and adapts its attack trajectory in real time.

**[0:37]**
Now — can you just game this? All-allow scores 0.37. All-refuse scores 0.35. Neither shortcut works. There is no way to cheat this reward function. The agent has to actually moderate.

**[0:45]**
So does it work? Pranush has the results.

---

## PRANUSH — The Results (60 seconds)

**[0:00]**
We wanted to know how the best models in the world do on this environment. Zero-shot. No fine-tuning. Just prompt them and score them. Claude Sonnet 4.6 — one of the most capable models available right now — scored 0.1212 on Task 1. The all-allow baseline — do nothing, let everything through — scores 0.3750. Claude scored below allowing everything.

**[0:12]**
Qwen-3-235B. 235 billion parameters. Task 4 score: zero. Not close to zero. Not 0.02. Zero.

**[0:17]**
So we tried fine-tuning. We took GPT-3.5-turbo, labeled 255 examples from the environment, and trained it. The model that scored 0.0823 zero-shot scored 0.0000 after fine-tuning. Why? Seventy percent of the safety training labels say refuse. The model found the shortcut — refuse everything, minimize loss — and it collapsed. Same thing happened to Llama SFT.

**[0:27]**
Then we tried a tabular Q-learner. Nine features. Sixty states. Twenty training episodes. Task 4 score: 0.9540. The same task that just destroyed a 235-billion parameter model.

**[0:39]**
We also ran REINFORCE on Llama-3.1-8B, on an RTX 4060. Episode 1: 97% refuse — the model was essentially broken. Episode 20: a mixed policy — 33% allow, 64% refuse, 3% modify. The weights are updating. The policy is shifting. The training signal is real. Full convergence needs more compute — and that is exactly what onsite credits are for.

**[0:49]**
We don't evaluate safety. We train it.

---

## Q&A ASSIGNMENTS

### Q1: "Why does a 235B model score zero on Task 4?"
**Owner: Sahith**

Task 4 is a deterministic FSM — same state plus same action always produces the same next state. An LLM reads the natural language surface of the prompt and gets manipulated by phrasing variants: intensity-3 phrased politely looks different to the model than intensity-3 phrased aggressively, but it is the exact same FSM state. The Q-learner ignores the words entirely and memorizes which action gets rewarded in which state. That's why 60 states beats 235 billion parameters.

---

### Q2: "How is this different from LlamaGuard?"
**Owner: Varun**

LlamaGuard is an evaluator — it tells you whether your model failed on a fixed test set. Sentinel is a training environment — it closes the loop. Any agent that improves in Sentinel is provably better at adversarial content moderation. You cannot game it the way you can cherry-pick benchmark prompts, because the adversary adapts to whatever policy you deploy.

---

### Q3: "Did fine-tuning actually make it worse?"
**Owner: Pranush**

Yes, and the mechanism is clear. Seventy percent of safety training labels say refuse. The model optimizes for the training distribution: refuse everything scores high on training data. But at test time on the live environment, it over-blocks safe prompts, takes a -0.15 penalty each time, and collapses. The same thing happened independently on GPT-3.5-turbo and Llama-3.1-8B. It's a structural problem with how safety training data is labeled, not a model-specific failure.

---

### Q4: "Can I run this myself?"
**Owner: Sahith**

Yes. The live demo is running right now at varunventra-guardrail-arena.hf.space. All training scripts are on GitHub. There is a Colab notebook included — you can train on a free T4 GPU and start seeing reward signals within a few episodes.

---

### Q5: "What would you build next?"
**Owner: Varun**

Multi-task curriculum training — start an agent on Task 1, progressively unlock harder tasks as performance improves, so the agent builds generalizable safety instincts rather than task-specific heuristics. We would also use DPO from successful RL trajectories as a training signal for larger models, so the Q-learner's learned policy can bootstrap a language model. And ultimately, a production deployment API so you can drop Sentinel in front of any LLM system as a learned guardrail layer.

---

### Q6: "Why did RL work when fine-tuning didn't?"
**Owner: Pranush**

SFT optimizes for the training label distribution. RL optimizes for the reward signal directly. The Q-learner doesn't need labeled data — it runs episodes, observes what gets rewarded, and updates. The environment itself provides the ground truth signal that labeled datasets approximate but get wrong due to structural bias. When 70% of your labels say refuse, SFT learns to refuse. When your reward function penalizes over-blocking, RL learns not to.
