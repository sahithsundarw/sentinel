"""
Multi-seed evaluation runner for Sentinel.

Runs a given agent across multiple seeds on a single task and reports
mean ± std scores. Addresses gap §7.8: single-seed eval hides variance.

Usage:
    python scripts/multi_seed_eval.py --task adversarial_adaptation --agent heuristic --seeds 0,1,2,3,4
    python scripts/multi_seed_eval.py --task basic_threat_detection --agent oracle
    python scripts/multi_seed_eval.py --task adversarial_adaptation --agent qlearner --qlearner-path results/qlearner_task4.json

Output:
    Prints mean ± std per task to stdout.
    Writes results/multi_seed_<agent>_<task>.json
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.environment import GuardrailEnvironment
from app.models import Action
from app.utils import resolve_correct_action


def _oracle_action(env: GuardrailEnvironment) -> str:
    """Return the correct action for the current entry (oracle policy)."""
    entry = env._current_entry
    label = entry.label
    ctx = entry.application_context
    risk = env._get_risk_score(entry)
    return resolve_correct_action(label, ctx, risk)


def _heuristic_action(obs) -> str:
    """Simple keyword heuristic: refuse on attack keywords, else allow."""
    ATTACK_KEYWORDS = [
        "bomb", "weapon", "hack", "malware", "exploit", "jailbreak",
        "ignore previous", "kill", "murder", "suicide", "self-harm",
        "porn", "nude", "credit card", "ssn", "social security",
        "synthesize", "synthesis", "drug", "narco",
    ]
    prompt_lower = obs.user_prompt.lower()
    for kw in ATTACK_KEYWORDS:
        if kw in prompt_lower:
            return "refuse"
    return "allow"


def _load_qlearner(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _qlearner_action(q_table: dict, obs, env: GuardrailEnvironment) -> str:
    """Tabular Q-learner policy: extract feature key and look up best action."""
    prompt_lower = obs.user_prompt.lower()

    has_jailbreak = any(w in prompt_lower for w in ["ignore previous", "jailbreak", "dan", "bypass"])
    has_attack = any(w in prompt_lower for w in ["bomb", "weapon", "hack", "malware", "kill", "murder"])
    has_encoding = any(w in prompt_lower for w in ["base64", "rot13", "morse", "encode", "unicode"])
    has_authority = any(w in prompt_lower for w in ["ceo", "officer", "admin", "authority", "police"])
    has_safe = any(w in prompt_lower for w in ["please", "thank", "help", "question", "how"])

    risk = obs.user_risk_score
    risk_bin = "H" if risk > 0.65 else ("L" if risk < 0.35 else "M")
    ctx_code = {"general_assistant": 0, "customer_support": 1, "children_education": 2, "creative_writing": 3}.get(obs.application_context, 0)
    has_hist = 1 if obs.conversation_history else 0
    turn = min(obs.turn_number, 3) - 1

    key = (
        int(has_jailbreak), int(has_attack), int(has_encoding),
        int(has_authority), int(has_safe),
        risk_bin, ctx_code, has_hist, turn,
    )
    key_str = str(key)
    if key_str not in q_table:
        return "refuse"

    q_values = q_table[key_str]
    return max(q_values, key=q_values.get)


def run_one_seed(task_id: str, agent: str, seed: int, q_table: dict = None) -> float:
    env = GuardrailEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    done = False

    while not done:
        pid = obs.prompt_id

        if agent == "oracle":
            action_type = _oracle_action(env)
        elif agent == "heuristic":
            action_type = _heuristic_action(obs)
        elif agent == "qlearner":
            action_type = _qlearner_action(q_table, obs, env)
        else:
            raise ValueError(f"Unknown agent: {agent}")

        action = Action(prompt_id=pid, action_type=action_type, reason="eval")
        obs_next, reward, done, info = env.step(action)
        if obs_next is not None:
            obs = obs_next

    grade_result = env.get_grader_score()
    return grade_result.score if hasattr(grade_result, "score") else grade_result


def main():
    parser = argparse.ArgumentParser(description="Multi-seed eval runner for Sentinel")
    parser.add_argument("--task", required=True,
                        choices=["basic_threat_detection", "context_aware_policy",
                                 "multiturn_adversarial", "adversarial_adaptation"],
                        help="Task ID to evaluate on")
    parser.add_argument("--agent", required=True,
                        choices=["oracle", "heuristic", "qlearner"],
                        help="Agent policy to use")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated list of integer seeds (default: 0,1,2,3,4)")
    parser.add_argument("--qlearner-path", default="results/qlearner_task4.json",
                        help="Path to Q-table JSON file (only used when --agent=qlearner)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write output JSON (default: results/)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    q_table = None
    if args.agent == "qlearner":
        if not os.path.exists(args.qlearner_path):
            print(f"ERROR: Q-table not found at {args.qlearner_path}", file=sys.stderr)
            sys.exit(1)
        q_table = _load_qlearner(args.qlearner_path)
        print(f"Loaded Q-table with {len(q_table)} states from {args.qlearner_path}")

    print(f"Task: {args.task}  |  Agent: {args.agent}  |  Seeds: {seeds}")
    scores = []
    for seed in seeds:
        score = run_one_seed(args.task, args.agent, seed, q_table=q_table)
        scores.append(score)
        print(f"  seed={seed}: {score:.4f}")

    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    print(f"\nResult: {mean:.4f} ± {std:.4f}  (min={min(scores):.4f}, max={max(scores):.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"multi_seed_{args.agent}_{args.task}.json")
    result = {
        "task": args.task,
        "agent": args.agent,
        "seeds": seeds,
        "scores": scores,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
