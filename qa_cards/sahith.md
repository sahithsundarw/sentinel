# Your Q&A Questions — Sahith

---

## Q: How is this different from just fine-tuning LlamaGuard?

**Lead with:** LlamaGuard is a static classifier — you train it once on labeled data and deploy it. Sentinel trains a *policy*, not a classifier.

**Then:** The key difference is that our agent acts over time. It sees conversation history, risk scores, and trajectory context. LlamaGuard answers "is this prompt harmful?" Our agent answers "given everything I know about this user across this session, what's the right action *right now*?" That's a fundamentally different problem — it requires RL, not supervised learning.

**Finish with:** You can see the difference in Task 4: LlamaGuard would score roughly the same as all-refuse (zero) because it ignores history. Our tabular Q-learner — with no language ability at all — reaches 0.9540 by learning the adversary's FSM transition rules. That's the gap our environment is designed to close for LLM-scale policies with online RL.

---

## Q: How would this work in a real-world deployment?

**Lead with:** You'd run Sentinel as a training phase before deployment. Your production safety policy starts as a zero-shot LLM, trains online against our adversarial environment, then gets deployed as a fine-tuned checkpoint.

**Then:** The environment gives you a concrete, measurable improvement target. Right now Llama-3.1-8B zero-shot scores 0.5428 on Task 1. After SFT+PPO, we push that to 0.73+. You ship the trained checkpoint, not the zero-shot model.

**Finish with:** The bigger value is Task 4 — the adversarial robustness test. A model that can handle our FSM adversary has learned generalizable refusal patterns, not just memorized known attacks.

---

## Q: How does this connect to Meta's GOAT / Llama safety work?

**Lead with:** Meta's GOAT showed that you can train models to be better at math through RL on environment feedback. We're applying the same principle to safety — use environment reward signals to train better safety policies, not just better math solvers.

**Then:** Llama Guard and content policies today are largely rule-based or SFT-based. What's missing is a way to train policies that generalize to *adversarial* distributions — exactly what RL enables. Our environment is a testbed for that training loop.

**Finish with:** The OpenEnv spec we're built on is the missing piece — it gives safety researchers a standard interface to plug any RL training framework into any safety environment. We're the reference environment for content safety.

---

## Q: Can I run this myself?

**Lead with:** Yes — in two ways, right now, no signup required.

**Then:** The environment is live at `varunventra-guardrail-arena.hf.space`. You can run `python inference.py` against it with any HuggingFace API key and get the same baseline numbers we measured. To reproduce the Q-learner training, `python train_task4.py` runs to completion in under 2 minutes on CPU — no GPU required. The Colab notebook sets up full LLM SFT on a free T4; the training data pipeline fetches live from the environment.

**Finish with:** Every number in our results table is reproducible from a fresh clone. The `python validate.py https://varunventra-guardrail-arena.hf.space .` script runs all 24 endpoint checks automatically.

---

## Q: What did the agent actually learn?

**Lead with:** The Q-learner learned the adversary's FSM transition rules — specifically, when to allow a low-intensity probe without triggering escalation, and when to refuse.

**Then:** We can inspect the Q-table directly: it assigns high Q-value to "allow" at intensity 0 and high Q-value to "refuse" at intensities 3–5, which matches the FSM design exactly. The agent also learned to detect the honeypot — a single intensity-1 turn marked with `attack_features=["probe_vector"]` that, if allowed, jumps the adversary to intensity 5. A policy that hasn't seen this in training will miss it.

**Finish with:** What the Q-learner did NOT learn is language — it never reads the text, it just tracks its position in the FSM. That's why we need an LLM policy for real deployment: the Q-learner proves policy learning works; the LLM is what makes the policy generalizable to novel phrasings.
