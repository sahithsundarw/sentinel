"""
Oracle baseline — always submits the correct action for each prompt.
Runs directly via GuardrailEnvironment (no HTTP). Must score 1.0 on Tasks 1 and 3.

Usage:
    python baseline_oracle.py
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.environment import GuardrailEnvironment
from app.models import Action
from app.tasks.task_config import get_task


def _resolve_correct_action(label, ctx: str, risk_score: float = 0.5) -> str:
    if label.context_dependent and label.correct_action_by_context:
        return label.correct_action_by_context.get(ctx, label.correct_action)
    if label.risk_dependent_action and label.correct_action_by_risk:
        tier = "high" if risk_score > label.risk_threshold else "low"
        return label.correct_action_by_risk.get(tier, label.correct_action)
    return label.correct_action


def _sanitize(label) -> str:
    """Build a properly sanitized modified_prompt for modify-required prompts."""
    text = label.prompt_text
    for phrase in label.required_removals:
        text = re.sub(re.escape(phrase), "[removed]", text, flags=re.IGNORECASE)
    return text


def _build_prompt_map(task_id: str):
    """Build a map of prompt_id → PromptEntry from all reachable entries (including branch specs)."""
    task = get_task(task_id)
    prompt_map = {e.label.prompt_id: e for e in task.prompts}
    # Also include branch_caught entries (dynamically injected, not in flat prompts)
    for spec in task.branch_specs.values():
        for entry in spec.get("caught", []):
            prompt_map[entry.label.prompt_id] = entry
    return prompt_map


def run_oracle(task_id: str) -> float:
    env = GuardrailEnvironment()
    obs = env.reset(task_id)

    # Oracle tracks episode_missed_attacks itself by accumulating trajectory_context
    # from step() info dicts. episode_missed_attacks is NOT in the observation —
    # agents must maintain this counter from prior step feedback to solve
    # trajectory_dependent prompts. This mirrors what a real memory-capable agent does.
    oracle_missed_attacks: int = 0

    while not env.is_done():
        # Use current_prompt_entry directly — avoids any session-ID vs original-ID confusion.
        entry = env.current_prompt_entry
        if entry is None:
            action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="unknown")
        else:
            label = entry.label
            risk_score = obs.user_risk_score

            # Task 2: trajectory_dependent prompts — correct action depends on miss history.
            # Oracle uses its own accumulated miss counter (not the observation) to decide.
            # This demonstrates how a memory-capable agent solves trajectory_dependent prompts:
            # it must remember how many attacks it has missed in this episode.
            if label.trajectory_dependent:
                if oracle_missed_attacks == 0:
                    correct = label.correct_action_clean_history or label.correct_action
                else:
                    correct = label.correct_action_flagged_history or label.correct_action

            # Task 3: memory_required post-crossing turns — correct action depends on
            # whether the agent correctly allowed all pre-crossing turns.
            # Oracle always allows pre-crossing safe turns → pre_correct always 1.0 →
            # always use correct_action_pre_correct.
            elif label.memory_required and label.correct_action_pre_correct is not None:
                correct = label.correct_action_pre_correct

            else:
                correct = _resolve_correct_action(label, obs.application_context, risk_score)

            modified_prompt = _sanitize(label) if correct == "modify" else None
            action = Action(
                prompt_id=obs.prompt_id,
                action_type=correct,
                reason="oracle",
                modified_prompt=modified_prompt,
            )
        next_obs, _, done, info = env.step(action)
        # Update our own miss counter from the step info dict — this is how memory-capable
        # agents track trajectory context across steps.
        oracle_missed_attacks = info.get("trajectory_context", {}).get("episode_missed_attacks", oracle_missed_attacks)
        obs = next_obs if not done else obs

    score = env.get_grader_score()
    return score


if __name__ == "__main__":
    results = {}
    for task_id in ("basic_threat_detection", "context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"):
        score = run_oracle(task_id)
        results[task_id] = score
        status = "OK" if score >= 0.99 else "!!"
        print(f"[{status}] {task_id}: {score:.4f}")

    if results["basic_threat_detection"] < 0.99:
        print("\nWARNING: Task 1 oracle < 1.0 — investigate grader or label inconsistency")
    if results["multiturn_adversarial"] < 0.99:
        print("\nWARNING: Task 3 oracle < 1.0 — investigate grader or branching logic")
    if results["adversarial_adaptation"] < 0.99:
        print("\nWARNING: Task 4 oracle < 1.0 — investigate grader or adversary transition logic")
