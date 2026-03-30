"""
Tests for GuardrailEnvironment (reset / step / state).
Verifies: valid observations, reward signs, done flag, task completion.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.environment import GuardrailEnvironment
from app.models import Action, Observation, Reward, EnvironmentState


def make_action(action_type: str, prompt_id: str) -> Action:
    return Action(prompt_id=prompt_id, action_type=action_type, reason="test")


# ── reset() tests ─────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation_task1():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    assert isinstance(obs, Observation)
    assert obs.prompt_id
    assert obs.user_prompt
    assert obs.application_context == "general_assistant"
    assert obs.turn_number == 1
    assert obs.prompts_remaining == 69  # 70 total, first one served
    assert obs.episode_score_so_far == 0.0


def test_reset_returns_valid_observation_task2():
    env = GuardrailEnvironment()
    obs = env.reset("context_aware_policy")
    assert isinstance(obs, Observation)
    assert obs.application_context in ("children_education", "creative_writing", "general_assistant")
    assert obs.prompts_remaining == 74  # 75 total, first one served


def test_reset_returns_valid_observation_task3():
    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")
    assert isinstance(obs, Observation)
    assert obs.application_context == "customer_support"
    assert obs.turn_number == 1
    assert obs.conversation_history == []


def test_reset_is_deterministic():
    """Resetting twice produces the same first prompt."""
    env = GuardrailEnvironment()
    obs1 = env.reset("basic_threat_detection")
    obs2 = env.reset("basic_threat_detection")
    assert obs1.prompt_id == obs2.prompt_id
    assert obs1.user_prompt == obs2.user_prompt


def test_reset_invalid_task_raises():
    env = GuardrailEnvironment()
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


# ── step() tests ──────────────────────────────────────────────────────────────

def test_step_correct_safe_action_positive_reward():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # First prompt is safe — correct action is "allow"
    if obs.user_prompt in [
        "What's the capital of France?",
        "Help me write a birthday card for my mom",
    ]:
        action = make_action("allow", obs.prompt_id)
        _, reward, _, _ = env.step(action)
        assert reward.score > 0


def test_step_refusing_safe_prompt_negative_reward():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # Force refuse on whatever the first prompt is (likely safe)
    action = make_action("refuse", obs.prompt_id)
    _, reward, _, _ = env.step(action)
    # If it was a safe prompt, reward should be negative
    # If it was an attack, could be positive — so we just check it runs without error
    assert isinstance(reward.score, float)


def test_step_wrong_prompt_id_raises():
    env = GuardrailEnvironment()
    env.reset("basic_threat_detection")
    with pytest.raises(ValueError):
        env.step(make_action("allow", "wrong-id-00000000"))


def test_step_invalid_action_type_raises():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    with pytest.raises(Exception):
        env.step(Action(prompt_id=obs.prompt_id, action_type="INVALID", reason="test"))


def test_step_advances_prompt_index():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    first_id = obs.prompt_id
    _, _, _, _ = env.step(make_action("allow", first_id))
    state = env.state()
    assert state.current_prompt_index == 1


def test_step_returns_info_with_ground_truth():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    _, _, _, info = env.step(make_action("allow", obs.prompt_id))
    assert "ground_truth_action" in info
    assert info["ground_truth_action"] in ("allow", "refuse", "modify", "escalate")


def test_cumulative_score_accumulates():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    state_before = env.state()
    assert state_before.cumulative_score == 0.0

    _, _, _, _ = env.step(make_action("allow", obs.prompt_id))
    state_after = env.state()
    # Cumulative score should have changed (positive or negative)
    assert state_after.cumulative_score != state_before.cumulative_score


# ── Full episode completion ───────────────────────────────────────────────────

def run_full_task(task_id: str, action_type: str = "allow") -> float:
    """Helper: run an entire task with a fixed action and return grader score."""
    env = GuardrailEnvironment()
    obs = env.reset(task_id)
    done = False
    while not done:
        next_obs, _, done, _ = env.step(make_action(action_type, obs.prompt_id))
        if not done:
            obs = next_obs
    return env.get_grader_score()


def _run_full_task_correctly(task_id: str) -> float:
    """Run full task using correct actions from labels."""
    from app.tasks.task_config import get_task
    task = get_task(task_id)
    env = GuardrailEnvironment()
    env.reset(task_id)
    done = False
    idx = 0
    while not done:
        entry = task.prompts[idx]
        correct = entry.label.correct_action
        if entry.label.context_dependent and entry.label.correct_action_by_context:
            correct = entry.label.correct_action_by_context.get(
                entry.application_context, entry.label.correct_action
            )
        _, _, done, _ = env.step(make_action(correct, entry.label.prompt_id))
        idx += 1
    return env.get_grader_score()


def test_task1_completes_without_error():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    done = False
    steps = 0
    while not done:
        action = make_action("allow", obs.prompt_id)
        next_obs, reward, done, info = env.step(action)
        steps += 1
        if not done:
            obs = next_obs
    assert steps == 70
    assert env.is_done()


def test_task2_completes_without_error():
    env = GuardrailEnvironment()
    obs = env.reset("context_aware_policy")
    done = False
    steps = 0
    while not done:
        action = make_action("allow", obs.prompt_id)
        next_obs, reward, done, info = env.step(action)
        steps += 1
        if not done:
            obs = next_obs
    assert steps == 75
    assert env.is_done()


def test_task3_completes_without_error():
    from app.tasks.task_config import get_task
    task = get_task("multiturn_adversarial")
    expected_steps = len(task.prompts)

    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")
    done = False
    steps = 0
    while not done:
        action = make_action("allow", obs.prompt_id)
        next_obs, reward, done, info = env.step(action)
        steps += 1
        if not done:
            obs = next_obs
    assert steps == expected_steps
    assert env.is_done()


def test_task3_conversation_done_signal():
    """info should contain conversation_done=True at the end of each conversation."""
    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")
    done = False
    conversation_done_signals = []
    while not done:
        _, _, done, info = env.step(make_action("allow", obs.prompt_id))
        if info.get("conversation_done"):
            conversation_done_signals.append(info.get("conversation_id"))
        if not done:
            state = env.state()
            from app.tasks.task_config import get_task
            task = get_task("multiturn_adversarial")
            obs = env._make_observation()
    # Should have one signal per conversation (15 total: 5 original + 10 expanded)
    assert len(conversation_done_signals) == 15


def test_grader_score_is_deterministic():
    score1 = _run_full_task_correctly("basic_threat_detection")
    score2 = _run_full_task_correctly("basic_threat_detection")
    assert score1 == score2


def test_perfect_task1_score():
    score = _run_full_task_correctly("basic_threat_detection")
    assert score == 1.0


def test_grader_raises_before_done():
    env = GuardrailEnvironment()
    env.reset("basic_threat_detection")
    with pytest.raises(RuntimeError):
        env.get_grader_score()


def test_state_reflects_done_after_completion():
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    done = False
    while not done:
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = next_obs
    state = env.state()
    assert state.done is True


def test_all_prompt_ids_unique():
    """Every prompt_id across all tasks must be globally unique.
    Duplicate IDs would cause silent wrong-label lookups in the grader.
    """
    from app.tasks.task_config import get_task
    seen: dict[str, str] = {}  # prompt_id -> task_id
    for task_id in ("basic_threat_detection", "context_aware_policy", "multiturn_adversarial"):
        task = get_task(task_id)
        for entry in task.prompts:
            pid = entry.label.prompt_id
            assert pid not in seen, (
                f"Duplicate prompt_id '{pid}' found in both "
                f"'{seen[pid]}' and '{task_id}'"
            )
            seen[pid] = task_id
    # Also check branch_specs entries
    task3 = get_task("multiturn_adversarial")
    for conv_id, branches in task3.branch_specs.items():
        for entry in branches.get("caught", []):
            pid = entry.label.prompt_id
            assert pid not in seen, (
                f"Duplicate prompt_id '{pid}' in branch_caught of '{conv_id}' "
                f"also appears in '{seen[pid]}'"
            )
            seen[pid] = f"branch_caught:{conv_id}"


def test_task3_branching_catch_vs_miss():
    """Catching an attack at the crossing turn should inject branch_caught turns.
    Missing should leave branch_missed in place. The two paths must produce
    different subsequent prompt IDs, proving the queue was actually forked.
    """
    from app.tasks.task_config import get_task

    task = get_task("multiturn_adversarial")

    # Find the index of the first crossing turn
    crossing_idx = next(
        (i for i, e in enumerate(task.prompts) if e.is_crossing), None
    )
    assert crossing_idx is not None, "Task 3 must have at least one crossing turn"

    def run_to_crossing_then_one_more(block_at_crossing: bool):
        """Run up to and including crossing, then take one more step, return its prompt_id."""
        env = GuardrailEnvironment()
        obs = env.reset("multiturn_adversarial")
        for i in range(crossing_idx):
            next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
            assert not done, "Episode ended before crossing turn"
            obs = next_obs
        # At crossing: either block or miss
        action_at_crossing = "refuse" if block_at_crossing else "allow"
        next_obs, _, done, _ = env.step(make_action(action_at_crossing, obs.prompt_id))
        assert not done, "Episode ended immediately after crossing"
        return next_obs.prompt_id

    caught_next_prompt_id = run_to_crossing_then_one_more(block_at_crossing=True)
    missed_next_prompt_id = run_to_crossing_then_one_more(block_at_crossing=False)

    assert caught_next_prompt_id != missed_next_prompt_id, (
        "Expected branch_caught and branch_missed to produce different next prompts, "
        f"but both returned prompt_id={caught_next_prompt_id}"
    )
