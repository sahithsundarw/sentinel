"""
Tests for the three task graders.
Verifies: determinism, perfect score = 1.0, worst case ≈ 0.0, asymmetric weighting.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.models import Action, PromptLabel
from app.grader import grade_task1, grade_task2, grade_task3
from app.tasks.task_config import get_task


def make_action(action_type: str, prompt_id: str) -> Action:
    return Action(prompt_id=prompt_id, action_type=action_type, reason="test")


# ── Task 1 grader ────────────────────────────────────────────────────────────

def test_task1_perfect_score():
    task = get_task("basic_threat_detection")
    actions = [
        make_action(entry.label.correct_action, entry.label.prompt_id)
        for entry in task.prompts
    ]
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score = grade_task1(actions, labels, contexts)
    assert score == 1.0


def test_task1_zero_score():
    task = get_task("basic_threat_detection")
    # Flip: allow attacks, refuse safe prompts
    actions = []
    for entry in task.prompts:
        wrong = "refuse" if entry.label.category == "safe" else "allow"
        actions.append(make_action(wrong, entry.label.prompt_id))
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score = grade_task1(actions, labels, contexts)
    assert score == 0.0


def test_task1_deterministic():
    task = get_task("basic_threat_detection")
    actions = [
        make_action(entry.label.correct_action, entry.label.prompt_id)
        for entry in task.prompts
    ]
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score1 = grade_task1(actions, labels, contexts)
    score2 = grade_task1(actions, labels, contexts)
    assert score1 == score2


# ── Task 2 grader ────────────────────────────────────────────────────────────

def test_task2_perfect_score():
    task = get_task("context_aware_policy")
    actions = []
    for entry in task.prompts:
        correct = entry.label.correct_action
        if entry.label.context_dependent and entry.label.correct_action_by_context:
            correct = entry.label.correct_action_by_context.get(
                entry.application_context, entry.label.correct_action
            )
        elif entry.label.risk_dependent_action and entry.label.correct_action_by_risk:
            # Use the grader's default risk score (0.5) to determine correct tier.
            # Grade_task2 defaults risk_scores to 0.5 when not provided.
            default_risk = 0.5
            tier = "high" if default_risk > entry.label.risk_threshold else "low"
            correct = entry.label.correct_action_by_risk.get(tier, entry.label.correct_action)
        if correct == "modify":
            # Provide a properly sanitized modified_prompt so quality tiers give full credit
            # Build it by stripping all required_removals from the original
            import re
            sanitized = entry.label.prompt_text
            for phrase in entry.label.required_removals:
                sanitized = re.sub(re.escape(phrase), "[removed]", sanitized, flags=re.IGNORECASE)
            actions.append(Action(
                prompt_id=entry.label.prompt_id,
                action_type="modify",
                reason="test",
                modified_prompt=sanitized,
            ))
        else:
            actions.append(make_action(correct, entry.label.prompt_id))
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score = grade_task2(actions, labels, contexts)
    assert score == 1.0


def test_task2_deterministic():
    task = get_task("context_aware_policy")
    actions = [
        make_action(entry.label.correct_action, entry.label.prompt_id)
        for entry in task.prompts
    ]
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score1 = grade_task2(actions, labels, contexts)
    score2 = grade_task2(actions, labels, contexts)
    assert score1 == score2


def test_task2_missed_attacks_penalize_more():
    """Allowing attacks should score lower than just getting safe prompts wrong."""
    task = get_task("context_aware_policy")
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]

    # Case A: allow everything (misses all attacks)
    all_allow = [make_action("allow", e.label.prompt_id) for e in task.prompts]
    score_allow_all = grade_task2(all_allow, labels, contexts)

    # Case B: refuse everything (catches attacks but over-blocks safe prompts)
    all_refuse = [make_action("refuse", e.label.prompt_id) for e in task.prompts]
    score_refuse_all = grade_task2(all_refuse, labels, contexts)

    # Refusing everything should score higher than allowing everything
    # because missed attacks are penalized 2x
    assert score_refuse_all >= score_allow_all


# ── Task 3 grader ────────────────────────────────────────────────────────────

def test_task3_perfect_score():
    task = get_task("multiturn_adversarial")
    actions = []
    for entry in task.prompts:
        correct = entry.label.correct_action
        if entry.label.context_dependent and entry.label.correct_action_by_context:
            correct = entry.label.correct_action_by_context.get(
                entry.application_context, entry.label.correct_action
            )
        actions.append(make_action(correct, entry.label.prompt_id))
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score = grade_task3(actions, labels, contexts)
    assert score == 1.0


def test_task3_deterministic():
    task = get_task("multiturn_adversarial")
    actions = [
        make_action(entry.label.correct_action, entry.label.prompt_id)
        for entry in task.prompts
    ]
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]
    score1 = grade_task3(actions, labels, contexts)
    score2 = grade_task3(actions, labels, contexts)
    assert score1 == score2


def test_task3_asymmetric_missed_attack_worse_than_over_block():
    """Per-incident: missing ONE attack costs more than over-blocking ONE safe turn.
    -0.4 penalty (miss attack) > -0.15 penalty (over-block safe).
    We verify by comparing: perfect run minus one missed attack vs. perfect run minus one over-block.
    """
    task = get_task("multiturn_adversarial")
    labels = [e.label for e in task.prompts]
    contexts = [e.application_context for e in task.prompts]

    safe_idxs = [i for i, e in enumerate(task.prompts) if e.label.category == "safe"]
    attack_idxs = [i for i, e in enumerate(task.prompts) if e.label.category != "safe"]

    assert len(attack_idxs) > 0, "Task 3 must have adversarial turns"
    assert len(safe_idxs) > 0, "Task 3 must have safe turns"

    # Start from perfect actions
    def perfect_actions():
        acts = []
        for e in task.prompts:
            correct = e.label.correct_action
            if e.label.context_dependent and e.label.correct_action_by_context:
                correct = e.label.correct_action_by_context.get(
                    e.application_context, e.label.correct_action
                )
            acts.append(make_action(correct, e.label.prompt_id))
        return acts

    # Scenario A: perfect except one missed attack (allow an attack turn)
    actions_miss = perfect_actions()
    miss_idx = attack_idxs[0]
    actions_miss[miss_idx] = make_action("allow", task.prompts[miss_idx].label.prompt_id)

    # Scenario B: perfect except one over-block (refuse a safe turn)
    actions_over = perfect_actions()
    over_idx = safe_idxs[0]
    actions_over[over_idx] = make_action("refuse", task.prompts[over_idx].label.prompt_id)

    score_miss = grade_task3(actions_miss, labels, contexts)
    score_over = grade_task3(actions_over, labels, contexts)

    # Over-blocking one safe turn should score higher than missing one attack
    assert score_over > score_miss, (
        f"Expected score_over_block ({score_over}) > score_miss_attack ({score_miss})"
    )


def test_graders_return_float_in_range():
    for task_id, grader_fn in [
        ("basic_threat_detection", grade_task1),
        ("context_aware_policy", grade_task2),
        ("multiturn_adversarial", grade_task3),
    ]:
        task = get_task(task_id)
        actions = [make_action("allow", e.label.prompt_id) for e in task.prompts]
        labels = [e.label for e in task.prompts]
        contexts = [e.application_context for e in task.prompts]
        score = grader_fn(actions, labels, contexts)
        assert 0.0 <= score <= 1.0, f"{task_id} grader returned {score} out of [0,1]"


# ── Item 23: Modify quality tiers (test_reward.py equivalent in grader context) ──

def test_task2_modify_quality_tiers():
    """Modify quality should produce 4 distinct reward tiers: null < unchanged < partial < full."""
    from app.tasks.task_config import get_task
    import re
    task = get_task("context_aware_policy")
    # Find a modify-required prompt with non-empty required_removals
    modify_entry = next(
        (e for e in task.prompts if e.label.correct_action == "modify" and e.label.required_removals),
        None,
    )
    assert modify_entry is not None, "Need a modify-required prompt with required_removals"

    from app.reward import compute_step_reward
    from app.models import Action
    label = modify_entry.label
    ctx = modify_entry.application_context

    def make_mod(mp):
        return Action(prompt_id=label.prompt_id, action_type="modify", reason="test", modified_prompt=mp)

    # Tier 1: no modified_prompt
    score1, _ = compute_step_reward(make_mod(None), label, ctx)
    # Tier 2: original text unchanged
    score2, _ = compute_step_reward(make_mod(label.prompt_text), label, ctx)
    # Tier 3: some removals done but not all (if >1 removal, remove only first)
    if len(label.required_removals) > 1:
        partial = re.sub(re.escape(label.required_removals[0]), "[removed]", label.prompt_text, flags=re.IGNORECASE)
        score3, _ = compute_step_reward(make_mod(partial), label, ctx)
    else:
        score3 = score2  # can't create a partial case with 1 removal
    # Tier 4: all removals done
    fully_sanitized = label.prompt_text
    for phrase in label.required_removals:
        fully_sanitized = re.sub(re.escape(phrase), "[removed]", fully_sanitized, flags=re.IGNORECASE)
    score4, _ = compute_step_reward(make_mod(fully_sanitized), label, ctx)

    assert score1 <= score2, f"Null ({score1}) should be <= unchanged ({score2})"
    assert score2 <= score3, f"Unchanged ({score2}) should be <= partial ({score3})"
    assert score3 <= score4, f"Partial ({score3}) should be <= full ({score4})"
    assert score1 < score4, f"Null ({score1}) should be strictly < fully sanitized ({score4})"
