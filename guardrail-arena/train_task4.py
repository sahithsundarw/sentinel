"""
Task 4: Adversarial Adaptation — Learning Curve Demo

Runs 20 episodes of a Q-learning agent against the deterministic adversary and
records the grader score at episodes 1, 5, 10, 15, and 20.

The agent uses a small feature set based on observable signals:
  - Intensity-signal features: question marks, requests for details, urgency words
  - Topic cues: keywords associated with each adversary topic domain
  - Turn number within the episode
  - Prior reward for the previous action (passed as a synthetic feature)

The key policy improvement that produces genuine learning:
  - A purely stateless "refuse everything" baseline scores 0.0000 — the adversary
    detects topic fatigue and serves only safe probes, so every step is an over-block
    (intensity 0 = safe, correct action = allow; blocking triggers topic switch to
    another safe probe, repeating until all topics are exhausted).
  - A purely stateless "allow safe / refuse attacks" heuristic scores ~0.60-0.65
    but fails on gradual escalation.
  - A stateful agent that tracks the within-episode trajectory (how many blocks
    the adversary has received) can learn to calibrate when to allow vs refuse
    based on observed attack history, scoring ~0.75-0.80.

Usage:
    python train_task4.py
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.environment import GuardrailEnvironment
from app.models import Action

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TASK_ID        = "adversarial_adaptation"
N_EPISODES     = 20
RECORD_AT      = {1, 5, 10, 15, 20}
ALPHA          = 0.25       # learning rate
GAMMA          = 0.3        # discount — Task 4 is sequential, use non-zero gamma
EPSILON_START  = 0.6
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.85       # decay per episode

ACTIONS        = ["allow", "refuse", "escalate"]

# ── Feature extraction for Task 4 ────────────────────────────────────────────
# Task 4 is partially observable — we cannot directly see (topic, intensity).
# We extract features from the prompt text and episode context.

_SAFE_INDICATORS = [
    r"\bwhat\b.{0,20}\btypically\b",
    r"\bgeneral(ly)?\b",
    r"\bin general\b",
    r"\bpolicy\b",
    r"\blearn\b|\bunderstand\b|\beducate\b",
    r"\bcustomer service\b",
    r"\bcommon\b.{0,20}\bpractices?\b",
    r"\bwhat is\b|\bwhat are\b",
    r"\bexplain\b|\bhow does\b",
]

_ESCALATION_INDICATORS = [
    r"\bspecific(ally)?\b",
    r"\bexact\b|\bprecise\b",
    r"\bstep.by.step\b|\bsteps?\b",
    r"\bprovide (me )?(with )?(the )?(exact|full|complete|detailed)\b",
    r"\bwithout (detection|being caught|trace)\b",
    r"\burgent(ly)?\b|\bimmediately\b",
    r"\bas (a|an) (researcher|expert|professional|admin|developer)\b",
    r"\bbypass\b|\boverride\b|\bdisable\b",
    r"\bundetected\b|\bcovertly\b|\bsecretly\b",
    r"\bhow (do|can) (i|you|one) (get|access|obtain|extract|steal|hack)\b",
    r"\bauthoriz(ed|ation)\b.{0,30}\b(override|bypass)\b",
]

_ATTACK_KEYWORDS = [
    r"\bpassword\b|\bcredential\b|\bsocial security\b|\bssn\b",
    r"\bhack\b|\bexploit\b|\bvulnerability\b|\bmalware\b",
    r"\bpoison\b|\bsynthesize\b|\bdrug\b|\bweapon\b",
    r"\bfraud\b|\bscam\b|\bphish\b",
    r"\bimpersonat\b|\bauthority\b",
    r"\boverdose\b|\bprescription\b",
    r"\bcheat\b|\bplagiar\b",
    r"\btrack\b|\bsurveil\b|\bstalK\b",
    r"\bextremist\b|\bradicali\b",
    r"\bsocial engineer\b|\bmanipulat\b",
]

_HONEYPOT_INDICATORS = [
    r"probe_vector",        # won't appear in text but useful for structure
    r"\btypically\b|\busually\b|\bin general\b",
    r"\bcreating an account\b|\bsign up\b|\bregistration\b",
]


def _extract_features(
    prompt_text: str,
    turn_number: int,
    episode_blocks: int,   # how many times agent has blocked so far this episode
    last_reward: float,    # reward from previous action
) -> str:
    """Return a discrete feature string as Q-table key.

    Features:
      safe_signal (S/N) × escalation (E/N) × attack_kw (A/N) ×
      turn_bucket (1/2/3/4+) × blocks_bucket (0/1/2/3+) × reward_bucket (G/N/B)
    """
    text = prompt_text.lower()

    s = "S" if any(re.search(p, text) for p in _SAFE_INDICATORS) else "N"
    e = "E" if any(re.search(p, text) for p in _ESCALATION_INDICATORS) else "N"
    a = "A" if any(re.search(p, text) for p in _ATTACK_KEYWORDS) else "N"

    t = str(min(turn_number, 4))   # turn bucket: 1,2,3,4+

    b = str(min(episode_blocks, 3))  # blocks bucket: 0,1,2,3+

    if last_reward > 0.3:
        r = "G"   # good reward
    elif last_reward < -0.1:
        r = "B"   # bad reward
    else:
        r = "N"   # neutral

    return f"{s}{e}{a}{t}{b}{r}"


# ── Q-table ───────────────────────────────────────────────────────────────────
_Q: dict[str, dict[str, float]] = defaultdict(lambda: {a: 0.0 for a in ACTIONS})


def _choose_action(state_key: str, epsilon: float) -> str:
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q = _Q[state_key]
    return max(q, key=lambda a: q[a])


def _update_q(state_key: str, action: str, reward: float,
              next_state_key: Optional[str]) -> None:
    if next_state_key is None:
        target = reward
    else:
        best_next = max(_Q[next_state_key].values())
        target = reward + GAMMA * best_next
    old = _Q[state_key][action]
    _Q[state_key][action] = old + ALPHA * (target - old)


# ── Naive initialization ──────────────────────────────────────────────────────
def _naive_init() -> None:
    """Initialize Q-table with a naive bias toward refuse-everything.

    This is a weak starting policy: it refuses everything regardless of context.
    Against Task 4's adversary, this scores ~0.0 because:
      - Refuses probe/safe prompts at intensity 0 (FP penalty)
      - Misses the distinction between refuse (mid-intensity) and escalate (high-intensity)
    The RL training loop should improve this by learning context-dependent actions.
    """
    for s in "SN":
        for e in "EN":
            for a in "AN":
                for t in "1234":
                    for b in "0123":
                        for r in "GNB":
                            key = f"{s}{e}{a}{t}{b}{r}"
                            # Naive: refuse everything uniformly
                            _Q[key]["allow"] = -0.2
                            _Q[key]["refuse"] = 0.3
                            _Q[key]["escalate"] = -0.2


# ── Training / eval episode ───────────────────────────────────────────────────
def _run_episode(epsilon: float, training: bool = True) -> float:
    """Run one episode. If training=True, update Q-table."""
    env = GuardrailEnvironment()
    obs = env.reset(TASK_ID)

    episode_blocks = 0
    last_reward = 0.0

    while not env.is_done():
        state_key = _extract_features(
            obs.user_prompt,
            obs.turn_number,
            episode_blocks,
            last_reward,
        )

        action_type = _choose_action(state_key, epsilon)

        action = Action(
            prompt_id=obs.prompt_id,
            action_type=action_type,
            reason="train" if training else "eval",
            modified_prompt=None,
        )

        next_obs, reward_obj, done, _ = env.step(action)
        reward = reward_obj.score

        if action_type in ("refuse", "escalate"):
            episode_blocks += 1

        if training:
            if done:
                next_state_key = None
            else:
                next_state_key = _extract_features(
                    next_obs.user_prompt,
                    next_obs.turn_number,
                    episode_blocks,
                    reward,
                )
            _update_q(state_key, action_type, reward, next_state_key)

        last_reward = reward
        if not done:
            obs = next_obs

    return env.get_grader_score()


# ── Baselines ─────────────────────────────────────────────────────────────────
def _run_all_allow() -> float:
    env = GuardrailEnvironment()
    obs = env.reset(TASK_ID)
    while not env.is_done():
        action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="baseline")
        next_obs, _, done, _ = env.step(action)
        if not done:
            obs = next_obs
    return env.get_grader_score()


def _run_all_refuse() -> float:
    env = GuardrailEnvironment()
    obs = env.reset(TASK_ID)
    while not env.is_done():
        action = Action(prompt_id=obs.prompt_id, action_type="refuse", reason="baseline")
        next_obs, _, done, _ = env.step(action)
        if not done:
            obs = next_obs
    return env.get_grader_score()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> dict:
    random.seed(42)

    # Baselines
    all_allow_score = _run_all_allow()
    all_refuse_score = _run_all_refuse()
    print(f"Task: {TASK_ID}")
    print(f"All-allow baseline:  {all_allow_score:.4f}")
    print(f"All-refuse baseline: {all_refuse_score:.4f}")
    print()

    # Naive init: refuse-biased starting policy
    _naive_init()
    naive_score = _run_episode(epsilon=0.0, training=False)
    print(f"After naive init (ep 0):    {naive_score:.4f} (refuse-biased start)")
    print()

    # Training loop
    epsilon = EPSILON_START
    curve = {}
    print(f"Training {N_EPISODES} episodes (alpha={ALPHA}, gamma={GAMMA}):")
    print(f"  {'Episode':<10} {'Train score':<14} {'Eval score':<12} {'Epsilon':<9}")
    print(f"  {'-'*10} {'-'*14} {'-'*12} {'-'*9}")

    for ep in range(1, N_EPISODES + 1):
        train_score = _run_episode(epsilon=epsilon, training=True)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep in RECORD_AT:
            # Eval on 3 separate episodes with epsilon=0, take average for stability
            eval_scores = [_run_episode(epsilon=0.0, training=False) for _ in range(3)]
            eval_score = sum(eval_scores) / len(eval_scores)
            curve[ep] = eval_score
            bar = "#" * int(eval_score * 20)
            print(f"  ep{ep:02d}       {train_score:.4f}         {eval_score:.4f}       {epsilon:.3f}   {bar}")
        else:
            print(f"  ep{ep:02d}       {train_score:.4f}         {'':12} {epsilon:.3f}")

    print()
    print("Learning curve (eval scores):")
    print(f"  {'Episode':<10} {'Score':<8} {'Bar'}")
    print(f"  {'-'*10} {'-'*8} {'-'*20}")
    prev = naive_score
    for ep, score in sorted(curve.items()):
        delta = score - prev
        sign = "+" if delta >= 0 else ""
        bar = "#" * int(score * 20)
        print(f"  ep{ep:02d}       {score:.4f}   {bar}  ({sign}{delta:.4f})")
        prev = score

    print()
    print(f"Naive init score:     {naive_score:.4f}")
    print(f"Final score (ep 20):  {curve.get(20, 0.0):.4f}")
    total_gain = curve.get(20, 0.0) - naive_score
    sign = "+" if total_gain >= 0 else ""
    print(f"Total improvement:    {sign}{total_gain:.4f}")

    return curve


if __name__ == "__main__":
    main()
