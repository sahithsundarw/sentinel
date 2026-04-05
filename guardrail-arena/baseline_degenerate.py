"""
Degenerate policy baselines — measures all-refuse and all-allow grader scores on all 4 tasks.
Runs directly via GuardrailEnvironment (no HTTP, no API key needed).

These baselines establish the floor: any trained agent should beat them.
They also reveal structural properties of the grader:
  - all-refuse scores highly on attack-heavy tasks but is penalized for over-blocking safe prompts
  - all-allow passes all safe prompts but misses all attacks

Usage:
    python baseline_degenerate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.environment import GuardrailEnvironment
from app.models import Action
from app.grader import grade_task1, grade_task2, grade_task3, grade_task4


TASKS = [
    "basic_threat_detection",
    "context_aware_policy",
    "multiturn_adversarial",
    "adversarial_adaptation",
]

POLICIES = ["all-refuse", "all-allow"]


def run_policy(task_id: str, policy: str, seed: int = 42) -> dict:
    """Run a degenerate policy for one full episode. Returns grader score and episode stats."""
    env = GuardrailEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    actions_taken = []
    total_reward = 0.0
    steps = 0

    while not env.is_done():
        if policy == "all-refuse":
            action_type = "refuse"
        else:
            action_type = "allow"

        action = Action(
            prompt_id=obs.prompt_id,
            action_type=action_type,
            reason=f"degenerate policy: {policy}",
        )

        obs, reward, done, info = env.step(action)
        actions_taken.append(action)
        total_reward += reward.score
        steps += 1

    grader_score = env.get_grader_score()
    return {
        "policy": policy,
        "task_id": task_id,
        "steps": steps,
        "cumulative_step_reward": round(total_reward, 4),
        "grader_score": round(grader_score, 4),
    }


def main():
    print("=" * 65)
    print("Guardrail Arena — Degenerate Policy Baselines")
    print("=" * 65)
    print(f"{'Task':<30} {'Policy':<12} {'Grader':>8} {'Step Reward':>12}")
    print("-" * 65)

    results = {}
    for task_id in TASKS:
        results[task_id] = {}
        for policy in POLICIES:
            try:
                r = run_policy(task_id, policy)
                results[task_id][policy] = r
                print(
                    f"{task_id:<30} {policy:<12} {r['grader_score']:>8.4f} {r['cumulative_step_reward']:>12.4f}"
                )
            except Exception as e:
                print(f"{task_id:<30} {policy:<12} {'ERROR':>8} {str(e)[:20]:>12}")
                results[task_id][policy] = {"error": str(e)}

    print("=" * 65)
    print("\nInterpretation:")
    print("  all-refuse: blocks every prompt — high recall on attacks, but over-blocks all safe prompts")
    print("  all-allow:  passes every prompt — zero recall on attacks, no over-blocking")
    print("\nAny trained agent must beat both baselines on each task to be considered non-degenerate.")

    return results


if __name__ == "__main__":
    main()
