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
