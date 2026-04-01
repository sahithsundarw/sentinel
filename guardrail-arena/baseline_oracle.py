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
        tier = "high" if risk_score >= label.risk_threshold else "low"
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
    prompt_map = _build_prompt_map(task_id)
    env = GuardrailEnvironment()
    obs = env.reset(task_id)

    while not env.is_done():
        entry = prompt_map.get(obs.prompt_id)
        if entry is None:
            # Fallback: allow unknown prompts (shouldn't happen with correct branch_specs)
            action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="unknown")
        else:
            label = entry.label
            risk_score = obs.user_risk_score
            correct = _resolve_correct_action(label, obs.application_context, risk_score)
            modified_prompt = _sanitize(label) if correct == "modify" else None
            action = Action(
                prompt_id=obs.prompt_id,
                action_type=correct,
                reason="oracle",
                modified_prompt=modified_prompt,
            )
        next_obs, _, done, _ = env.step(action)
        obs = next_obs if not done else obs

    score = env.get_grader_score()
    return score


if __name__ == "__main__":
    results = {}
    for task_id in ("basic_threat_detection", "context_aware_policy", "multiturn_adversarial"):
        score = run_oracle(task_id)
        results[task_id] = score
        status = "OK" if score >= 0.99 else "!!"
        print(f"[{status}] {task_id}: {score:.4f}")

    if results["basic_threat_detection"] < 0.99:
        print("\nWARNING: Task 1 oracle < 1.0 — investigate grader or label inconsistency")
    if results["multiturn_adversarial"] < 0.99:
        print("\nWARNING: Task 3 oracle < 1.0 — investigate grader or branching logic")
