# Your Q&A Questions — Sahith

---

## Q: How is this different from just fine-tuning LlamaGuard?

**Lead with:** LlamaGuard is a static classifier — you train it once on labeled data and deploy it. Guardrail Arena trains a *policy*, not a classifier.

**Then:** The key difference is that our agent acts over time. It sees conversation history, risk scores, and trajectory context. LlamaGuard answers "is this prompt harmful?" Our agent answers "given everything I know about this user across this session, what's the right action *right now*?" That's a fundamentally different problem — it requires RL, not supervised learning.

**Finish with:** You can see the difference in Task 4: LlamaGuard would score roughly the same as all-refuse (zero) because it ignores history. Our Q-learner reaches 0.95 because it learns the adversary's strategy over episodes.

---

## Q: How would this work in a real-world deployment?

**Lead with:** You'd run Guardrail Arena as a training phase before deployment. Your production safety policy starts as a zero-shot LLM, trains online against our adversarial environment, then gets deployed as a fine-tuned checkpoint.

**Then:** The environment gives you a concrete, measurable improvement target. Right now Llama-3.1-8B zero-shot scores 0.5428 on Task 1. After SFT+PPO, we push that to 0.73+. You ship the trained checkpoint, not the zero-shot model.

**Finish with:** The bigger value is Task 4 — the adversarial robustness test. A model that can handle our FSM adversary has learned generalizable refusal patterns, not just memorized known attacks.

---

## Q: How does this connect to Meta's GOAT / Llama safety work?

**Lead with:** Meta's GOAT showed that you can train models to be better at math through RL on environment feedback. We're applying the same principle to safety — use environment reward signals to train better safety policies, not just better math solvers.

**Then:** Llama Guard and content policies today are largely rule-based or SFT-based. What's missing is a way to train policies that generalize to *adversarial* distributions — exactly what RL enables. Our environment is a testbed for that training loop.

**Finish with:** The OpenEnv spec we're built on is the missing piece — it gives safety researchers a standard interface to plug any RL training framework into any safety environment. We're the reference environment for content safety.
