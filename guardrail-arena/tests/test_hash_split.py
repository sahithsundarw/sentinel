"""
Tests for hash-based train/eval split determinism.

Verifies that _hash_split produces identical results on repeated calls
(no randomness or ordering dependency).
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from app.tasks.task_config import _hash_split, _TASK1, _TASK2, _TASK3


def test_hash_split_deterministic():
    """Same prompt_id always returns the same split."""
    sample_ids = [
        "b1a2c3d4-e5f6-7890-abcd-ef1234567890",
        "d5c6e7f8-a9b0-1234-8901-234567890124",
        "d0aa0001-d0aa-4000-8000-aa0000000001",
        "f7e8a9b0-c1d2-3456-0123-456789012346",
    ]
    for pid in sample_ids:
        first = _hash_split(pid)
        second = _hash_split(pid)
        assert first == second, f"Non-deterministic split for prompt_id {pid}: {first} != {second}"


def test_hash_split_values_are_valid():
    """_hash_split only returns 'eval' or 'train'."""
    sample_ids = [
        "b1a2c3d4-e5f6-7890-abcd-ef1234567890",
        "test-id-1",
        "test-id-2",
        "another-prompt-xyz",
    ]
    for pid in sample_ids:
        result = _hash_split(pid)
        assert result in ("eval", "train"), f"Invalid split value '{result}' for {pid}"


def test_task1_split_is_deterministic():
    """Task 1 eval/train sets are stable across two imports of task_config."""
    eval_ids_1 = {e.label.prompt_id for e in _TASK1.prompts}
    train_ids_1 = {e.label.prompt_id for e in _TASK1.train_prompts}

    # Re-derive from _hash_split directly
    all_entries = list(_TASK1.prompts) + list(_TASK1.train_prompts)
    eval_ids_2 = {e.label.prompt_id for e in all_entries if _hash_split(e.label.prompt_id) == "eval"}
    train_ids_2 = {e.label.prompt_id for e in all_entries if _hash_split(e.label.prompt_id) == "train"}

    assert eval_ids_1 == eval_ids_2, "Task 1 eval split is not deterministic"
    assert train_ids_1 == train_ids_2, "Task 1 train split is not deterministic"


def test_task2_split_is_deterministic():
    """Task 2 eval/train sets are stable."""
    all_entries = list(_TASK2.prompts) + list(_TASK2.train_prompts)
    eval_ids_from_config = {e.label.prompt_id for e in _TASK2.prompts}
    eval_ids_from_hash = {e.label.prompt_id for e in all_entries if _hash_split(e.label.prompt_id) == "eval"}
    assert eval_ids_from_config == eval_ids_from_hash, "Task 2 eval split is not deterministic"


def test_task3_split_is_deterministic():
    """Task 3 eval/train sets are stable (conversation-level hash)."""
    eval_ids_from_config = {e.label.prompt_id for e in _TASK3.prompts}
    # All Task 3 prompts with same conversation_id share a split — verify consistency
    from collections import defaultdict
    conv_splits: dict = defaultdict(set)
    for e in list(_TASK3.prompts) + list(_TASK3.train_prompts):
        cid = e.conversation_id or "__no_conv__"
        conv_splits[cid].add(e.label.split)
    for cid, splits in conv_splits.items():
        assert len(splits) == 1, f"Conversation {cid} has inconsistent splits: {splits}"


def test_eval_split_roughly_30_percent():
    """Eval split should be approximately 30% of total for Tasks 1 and 2."""
    for task, name in [(_TASK1, "Task1"), (_TASK2, "Task2")]:
        total = len(task.prompts) + len(task.train_prompts)
        eval_frac = len(task.prompts) / total
        assert 0.20 <= eval_frac <= 0.45, (
            f"{name} eval fraction {eval_frac:.2%} is outside expected [20%, 45%] range"
        )
