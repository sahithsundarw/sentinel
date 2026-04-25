"""
DEMO SEED DATA — NOT real training history.

This script posts 20 synthetic Q-learner episodes to the /training_log endpoint
to seed the live Space's training dashboard with a representative learning curve.
Rewards follow a plausible Q-learner curve (0.02 → 0.9540) but are NOT recorded
from real training runs.

All entries are posted with agent_name="demo_synthetic_qlearner" and
is_synthetic=True so they can be filtered out of leaderboard displays.

To seed a live instance:
    python scripts/populate_training_evidence.py

To run real training and capture genuine logs, use train_agent.py instead.
"""
import requests
import json

ENV_URL = "https://varunventra-guardrail-arena.hf.space"

# Representative Q-learner learning curve — synthetic, for demo purposes only.
rewards = [
    0.02, 0.05, 0.08, 0.12, 0.19, 0.25, 0.31, 0.38, 0.44, 0.50,
    0.57, 0.63, 0.68, 0.73, 0.78, 0.82, 0.86, 0.90, 0.93, 0.9540,
]

AGENT_NAME = "demo_synthetic_qlearner"

print(f"Posting {len(rewards)} SYNTHETIC Q-learner episodes to {ENV_URL}/training_log ...")
print("These are demo seed entries (is_synthetic=True) — not real training data.\n")

for i, reward in enumerate(rewards, 1):
    cumulative = round(sum(rewards[:i]) / i, 4)
    payload = {
        "agent_name": AGENT_NAME,
        "task_id": "adversarial_adaptation",
        "episode": i,
        "grader_score": reward,
        "cumulative_reward": cumulative,
        "is_synthetic": True,
        "action_distribution": {
            "allow": max(1, int(12 - i * 0.4)),
            "refuse": int(i * 0.6 + 2),
            "modify": max(0, int(3 - i * 0.1)),
            "escalate": int(2 + i * 0.1),
        },
    }
    try:
        resp = requests.post(f"{ENV_URL}/training_log", json=payload, timeout=15)
        print(f"Episode {i:02d} | reward={reward:.4f} | status={resp.status_code}")
    except requests.RequestException as exc:
        print(f"Episode {i:02d} | reward={reward:.4f} | ERROR: {exc}")

print("\nVerifying stored episodes...")
try:
    log = requests.get(f"{ENV_URL}/training_log", timeout=15).json()
    if isinstance(log, dict):
        all_entries = log.get("all_entries", {})
        q_entries = all_entries.get(AGENT_NAME, [])
        print(f"Verified {len(q_entries)} episodes for {AGENT_NAME} in training log.")
        if q_entries:
            last = q_entries[-1]
            print(f"Last episode: ep={last.get('episode')} score={last.get('grader_score')}")
    elif isinstance(log, list):
        print(f"Verified {len(log)} entries in training log.")
    else:
        print(f"Response: {json.dumps(log, indent=2)[:500]}")
except Exception as exc:
    print(f"Could not verify: {exc}")
