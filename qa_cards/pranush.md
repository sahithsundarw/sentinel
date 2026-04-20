# Your Q&A Questions — Pranush

---

## Q: How many episodes does it take to see improvement?

**Lead with:** For the tabular Q-learner on Task 4: significant improvement by episode 10, convergence at 0.9540 by episode 15.

**Then:** For LLM fine-tuning on Task 1: SFT alone moves Llama-3.1-8B from 0.5428 to around 0.68 in 3 epochs. PPO online learning on top of that pushes it toward 0.73+ over 20 episodes. The SFT phase is fast — it's just supervised classification on labeled data. The PPO phase is slower but gives the most gains on the harder tasks because it's actually interacting with the environment.

**Finish with:** The key insight is that Task 4 *requires* online learning — no amount of SFT on static data will break 0.20 on Task 4 because the training data doesn't contain the adversary's adaptive behavior. You need to run episodes to see the adversary in action.

---

## Q: What would you build next if you had more time?

**Lead with:** The highest-value next step is multi-task training — a single agent that learns to handle all four tasks simultaneously, with curriculum progression from Task 1 to Task 4.

**Then:** Right now you train separately per task. With curriculum learning — start on Task 1 until you hit 0.80, then unlock Task 2, and so on — you'd get a single checkpoint that's robust across the full difficulty spectrum. The second thing I'd build is a real-time visualization of the adversary's internal state during an episode, so you can watch the FSM transitions as the agent plays. That would be a compelling live demo.

**Finish with:** Longer term, the environment is general enough to host different adversary types — not just the deterministic FSM but learned adversaries, red-team LLMs, even human red-teamers. We designed the interface to be extensible.
