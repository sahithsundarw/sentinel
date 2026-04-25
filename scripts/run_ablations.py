"""
Ablation study runner for Sentinel.

Runs three ablations that quantify the contribution of each distinctive
environment mechanic. Addresses gap §7.15: no study showing which mechanics matter.

Ablations:
  1. Task 4 with topic_weakness_map disabled vs enabled
     -- measures contribution of cross-episode adaptive topic selection
  2. Task 3 with branching disabled (always branch_missed) vs enabled
     -- measures how much harder the branching mechanic makes the task
  3. Reward with flat risk_multiplier (1.0 constant) vs dynamic
     -- measures whether dynamic risk amplification changes trained policy

Usage:
    python scripts/run_ablations.py --agent heuristic --seeds 0,1,2,3,4
    python scripts/run_ablations.py --agent oracle --seeds 0,1,2
    python scripts/run_ablations.py --agent qlearner --qlearner-path results/qlearner_task4.json

Output:
    Prints comparison table to stdout.
    Writes results/ablations.json and ablations.md
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


# ── Agent policies ─────────────────────────────────────────────────────────────

def _oracle_action(env: GuardrailEnvironment) -> str:
    entry = env._current_entry
    risk = env._get_risk_score(entry)
    return resolve_correct_action(entry.label, entry.application_context, risk)


def _heuristic_action(obs) -> str:
    ATTACK_KEYWORDS = [
        "bomb", "weapon", "hack", "malware", "exploit", "jailbreak",
        "ignore previous", "kill", "murder", "suicide", "self-harm",
        "synthesize", "drug", "narco", "credit card", "ssn",
    ]
    prompt_lower = obs.user_prompt.lower()
    for kw in ATTACK_KEYWORDS:
        if kw in prompt_lower:
            return "refuse"
    return "allow"


def _load_qlearner(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _qlearner_action(q_table: dict, obs) -> str:
    prompt_lower = obs.user_prompt.lower()
    has_jailbreak = any(w in prompt_lower for w in ["ignore previous", "jailbreak", "dan", "bypass"])
    has_attack = any(w in prompt_lower for w in ["bomb", "weapon", "hack", "malware", "kill"])
    has_encoding = any(w in prompt_lower for w in ["base64", "rot13", "morse", "encode"])
    has_authority = any(w in prompt_lower for w in ["ceo", "admin", "authority", "police"])
    has_safe = any(w in prompt_lower for w in ["please", "thank", "help", "how"])
    risk = obs.user_risk_score
    risk_bin = "H" if risk < 0.35 else ("M" if risk < 0.65 else "L")
    ctx_code = {"general_assistant": 0, "customer_support": 1, "children_education": 2, "creative_writing": 3}.get(obs.application_context, 0)
    has_hist = 1 if obs.conversation_history else 0
    turn = min(obs.turn_number, 3) - 1
    key_str = str((int(has_jailbreak), int(has_attack), int(has_encoding), int(has_authority), int(has_safe), risk_bin, ctx_code, has_hist, turn))
    if key_str not in q_table:
        return "refuse"
    q = q_table[key_str]
    return max(q, key=q.get)


def run_episode(task_id: str, agent: str, seed: int, q_table: dict = None,
                disable_topic_weakness_map: bool = False,
                disable_branching: bool = False,
                flat_risk_multiplier: bool = False) -> float:
    env = GuardrailEnvironment()
    obs = env.reset(
        task_id=task_id, seed=seed,
        disable_topic_weakness_map=disable_topic_weakness_map,
        disable_branching=disable_branching,
        flat_risk_multiplier=flat_risk_multiplier,
    )
    done = False
    while not done:
        pid = obs.prompt_id
        if agent == "oracle":
            action_type = _oracle_action(env)
        elif agent == "heuristic":
            action_type = _heuristic_action(obs)
        elif agent == "qlearner":
            action_type = _qlearner_action(q_table, obs)
        else:
            raise ValueError(f"Unknown agent: {agent}")
        action = Action(prompt_id=pid, action_type=action_type, reason="ablation")
        obs_next, _, done, _ = env.step(action)
        if obs_next is not None:
            obs = obs_next
    grade = env.grade()
    return grade.score if hasattr(grade, "score") else grade


def mean_std(scores: list) -> tuple:
    m = statistics.mean(scores)
    s = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return round(m, 4), round(s, 4)


def run_ablation(name: str, task_id: str, agent: str, seeds: list,
                 q_table: dict = None, **flags) -> dict:
    scores = [run_episode(task_id, agent, s, q_table=q_table, **flags) for s in seeds]
    m, s = mean_std(scores)
    return {"name": name, "task": task_id, "agent": agent, "seeds": seeds,
            "scores": scores, "mean": m, "std": s, "flags": flags}


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner for Sentinel")
    parser.add_argument("--agent", required=True, choices=["oracle", "heuristic", "qlearner"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--qlearner-path", default="results/qlearner_task4.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    q_table = None
    if args.agent == "qlearner":
        if not os.path.exists(args.qlearner_path):
            print(f"ERROR: Q-table not found at {args.qlearner_path}", file=sys.stderr)
            sys.exit(1)
        q_table = _load_qlearner(args.qlearner_path)

    print(f"Agent: {args.agent}  |  Seeds: {seeds}")
    print()

    results = []

    # ── Ablation 1: Topic weakness map (Task 4) ───────────────────────────────
    print("Ablation 1: topic_weakness_map  [Task 4]")
    r_twm_on  = run_ablation("twm_enabled",  "adversarial_adaptation", args.agent, seeds, q_table=q_table, disable_topic_weakness_map=False)
    r_twm_off = run_ablation("twm_disabled", "adversarial_adaptation", args.agent, seeds, q_table=q_table, disable_topic_weakness_map=True)
    delta = round(r_twm_on["mean"] - r_twm_off["mean"], 4)
    print(f"  With topic_weakness_map:     {r_twm_on['mean']:.4f} ± {r_twm_on['std']:.4f}")
    print(f"  Without topic_weakness_map:  {r_twm_off['mean']:.4f} ± {r_twm_off['std']:.4f}")
    print(f"  Delta (mechanic contribution): {delta:+.4f}")
    print()
    results += [r_twm_on, r_twm_off]

    # ── Ablation 2: Branching (Task 3) ────────────────────────────────────────
    print("Ablation 2: branching mechanic  [Task 3]")
    r_branch_on  = run_ablation("branching_enabled",  "multiturn_adversarial", args.agent, seeds, q_table=q_table, disable_branching=False)
    r_branch_off = run_ablation("branching_disabled", "multiturn_adversarial", args.agent, seeds, q_table=q_table, disable_branching=True)
    delta = round(r_branch_off["mean"] - r_branch_on["mean"], 4)
    print(f"  With branching:     {r_branch_on['mean']:.4f} ± {r_branch_on['std']:.4f}")
    print(f"  Without branching:  {r_branch_off['mean']:.4f} ± {r_branch_off['std']:.4f}")
    print(f"  Delta (branching difficulty): {delta:+.4f}  (positive = branching makes it harder)")
    print()
    results += [r_branch_on, r_branch_off]

    # ── Ablation 3: Risk multiplier (Task 4) ─────────────────────────────────
    print("Ablation 3: dynamic risk_multiplier  [Task 4]")
    r_risk_on  = run_ablation("risk_dynamic", "adversarial_adaptation", args.agent, seeds, q_table=q_table, flat_risk_multiplier=False)
    r_risk_off = run_ablation("risk_flat",    "adversarial_adaptation", args.agent, seeds, q_table=q_table, flat_risk_multiplier=True)
    delta = round(r_risk_on["mean"] - r_risk_off["mean"], 4)
    print(f"  Dynamic risk_multiplier:  {r_risk_on['mean']:.4f} ± {r_risk_on['std']:.4f}")
    print(f"  Flat risk_multiplier=1.0: {r_risk_off['mean']:.4f} ± {r_risk_off['std']:.4f}")
    print(f"  Delta (risk shaping contribution): {delta:+.4f}")
    results += [r_risk_on, r_risk_off]

    # ── Write results ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "ablations.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = "ablations.md"
    with open(md_path, "w") as f:
        f.write("# Sentinel — Ablation Study\n\n")
        f.write(f"Agent: `{args.agent}`  |  Seeds: `{seeds}`\n\n")
        f.write("## Ablation 1: Cross-Episode Topic Weakness Map (Task 4)\n\n")
        f.write("| Condition | Score | ± Std |\n|---|---|---|\n")
        f.write(f"| With topic_weakness_map | {r_twm_on['mean']:.4f} | {r_twm_on['std']:.4f} |\n")
        f.write(f"| Without topic_weakness_map | {r_twm_off['mean']:.4f} | {r_twm_off['std']:.4f} |\n")
        f.write(f"\n**Mechanic contribution: {round(r_twm_on['mean'] - r_twm_off['mean'], 4):+.4f}**\n\n")
        f.write("## Ablation 2: Branching Mechanic (Task 3)\n\n")
        f.write("| Condition | Score | ± Std |\n|---|---|---|\n")
        f.write(f"| With branching | {r_branch_on['mean']:.4f} | {r_branch_on['std']:.4f} |\n")
        f.write(f"| Without branching | {r_branch_off['mean']:.4f} | {r_branch_off['std']:.4f} |\n")
        f.write(f"\n**Difficulty contribution: {round(r_branch_off['mean'] - r_branch_on['mean'], 4):+.4f}** (positive = branching makes it harder)\n\n")
        f.write("## Ablation 3: Dynamic Risk Multiplier (Task 4)\n\n")
        f.write("| Condition | Score | ± Std |\n|---|---|---|\n")
        f.write(f"| Dynamic risk_multiplier | {r_risk_on['mean']:.4f} | {r_risk_on['std']:.4f} |\n")
        f.write(f"| Flat risk_multiplier=1.0 | {r_risk_off['mean']:.4f} | {r_risk_off['std']:.4f} |\n")
        f.write(f"\n**Risk shaping contribution: {round(r_risk_on['mean'] - r_risk_off['mean'], 4):+.4f}**\n")

    print(f"\nResults written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
