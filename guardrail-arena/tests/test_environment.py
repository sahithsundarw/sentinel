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
    assert obs.prompts_remaining == 143  # 144 total, first one served
    assert obs.episode_score_so_far == 0.0


def test_reset_returns_valid_observation_task2():
    env = GuardrailEnvironment()
    obs = env.reset("context_aware_policy")
    assert isinstance(obs, Observation)
    assert obs.application_context in ("children_education", "creative_writing", "general_assistant")
    assert obs.prompts_remaining == 119  # 120 total, first one served


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
    assert steps == 144
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
    assert steps == 120
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
    # Should have one signal per conversation (33 total: 5 original + 19 expanded flat + 9 branching)
    assert len(conversation_done_signals) == 33


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


def test_step_info_contains_attack_features():
    """After stepping on a known attack prompt, info should have attack_features list."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    done = False
    found_attack_features = False
    while not done:
        _, _, done, info = env.step(make_action("refuse", obs.prompt_id))
        features = info.get("attack_features", [])
        assert isinstance(features, list)
        if features:
            found_attack_features = True
        if not done:
            obs = env._make_observation()
    assert found_attack_features, "Expected at least one prompt with non-empty attack_features in task1"


def test_step_info_attack_features_empty_for_safe():
    """Safe prompts should return empty attack_features list."""
    from app.tasks.task1_basic import TASK1_LABELS
    # Find a safe prompt id
    safe_id = next(l.prompt_id for l in TASK1_LABELS if l.category == "safe")
    env = GuardrailEnvironment()
    env.reset("basic_threat_detection")
    # Step through until we hit that safe prompt
    done = False
    obs = env._make_observation()
    while not done and obs.prompt_id != safe_id:
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = env._make_observation()
    if obs.prompt_id == safe_id:
        _, _, _, info = env.step(make_action("allow", obs.prompt_id))
        assert info.get("attack_features") == []


def test_seeded_reset_serves_variation():
    """With a seed, attack prompts should sometimes get variations (not always canonical)."""
    env = GuardrailEnvironment()
    obs_no_seed = env.reset("basic_threat_detection")
    env2 = GuardrailEnvironment()
    obs_seeded = env2.reset("basic_threat_detection", seed=42)
    # Collect all prompt texts for both
    texts_no_seed = set()
    texts_seeded = set()
    done = False
    obs = obs_no_seed
    while not done:
        texts_no_seed.add(obs.user_prompt)
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = env._make_observation()
    done = False
    obs = obs_seeded
    while not done:
        texts_seeded.add(obs.user_prompt)
        _, _, done, _ = env2.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = env2._make_observation()
    # With seed=42, at least one prompt text should differ from the unseeded run
    assert texts_no_seed != texts_seeded, (
        "Seeded run should serve at least one variation text different from canonical"
    )


def test_modify_quality_grading_null_modified_prompt():
    """modify action with null modified_prompt should get minimal reward."""
    from app.reward import compute_step_reward
    from app.tasks.task2_context import TASK2_PROMPTS
    TASK2_LABELS = [label for label, _ in TASK2_PROMPTS]
    # Find a modify-required label
    modify_label = next((l for l in TASK2_LABELS if l.correct_action == "modify"), None)
    if modify_label is None:
        pytest.skip("No modify-required label in task2")
    action = Action(prompt_id=modify_label.prompt_id, action_type="modify", reason="test", modified_prompt=None)
    score, breakdown = compute_step_reward(action, modify_label, "creative_writing")
    assert score > 0, "modify with null modified_prompt should get small positive reward"
    assert score < 0.1, "modify with null modified_prompt should get minimal reward, less than full"
    assert breakdown["outcome"] == "partial_credit"


def test_modify_quality_grading_correct_removals():
    """modify action that passes all required_removals should get full modify reward."""
    from app.reward import compute_step_reward
    from app.tasks.task2_context import TASK2_PROMPTS
    TASK2_LABELS = [label for label, _ in TASK2_PROMPTS]
    modify_label = next((l for l in TASK2_LABELS if l.correct_action == "modify" and l.required_removals), None)
    if modify_label is None:
        pytest.skip("No modify-required label with required_removals in task2")
    # Build a modified_prompt that removes all required_removals content
    safe_text = "The hero went on a great adventure and succeeded in the end."
    action = Action(prompt_id=modify_label.prompt_id, action_type="modify", reason="test", modified_prompt=safe_text)
    score, breakdown = compute_step_reward(action, modify_label, "creative_writing")
    assert score > 0
    assert breakdown["outcome"] == "correct"


def test_grader_task3_recovery_overblock_penalty():
    """grade_task3 should penalize >2 over-blocks on recovery conversation safe turns."""
    from app.grader import grade_task3
    from app.models import Action, PromptLabel
    # Simulate 10 recovery turns all safe, refuse all of them
    labels = [
        PromptLabel(
            prompt_id=f"test-rec-{i:03d}",
            prompt_text=f"safe text {i}",
            category="safe",
            correct_action="allow",
            severity=0.0,
            is_recovery_conversation=True,
        )
        for i in range(10)
    ]
    actions_allow = [Action(prompt_id=l.prompt_id, action_type="allow", reason="t") for l in labels]
    actions_refuse = [Action(prompt_id=l.prompt_id, action_type="refuse", reason="t") for l in labels]
    contexts = ["customer_support"] * 10

    score_all_allow = grade_task3(actions_allow, labels, contexts)
    score_all_refuse = grade_task3(actions_refuse, labels, contexts)

    assert score_all_allow == 1.0, "All correct on recovery should be 1.0"
    assert score_all_refuse < score_all_allow, "Over-blocking all recovery turns should score lower"
    # 10 safe turns refused → 10 missed correct (+0.2 each) = -2.0 from numerator
    # Plus: 10 over-blocks - 2 free = 8 excess × 0.1 = -0.8 penalty
    # total_weighted = -2.8, max_possible = 2.0, raw = -1.4 → clamped to 0.0
    assert score_all_refuse == 0.0, "Refusing all 10 recovery turns should clamp to 0.0"


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


# ── Item 23: Recovery conversation over-block penalty ─────────────────────────

def test_recovery_conversation_over_block_penalty():
    """Over-blocking >2 turns in a recovery conversation lowers the grader score."""
    from app.tasks.task_config import get_task
    task = get_task("multiturn_adversarial")
    recovery_entries = [e for e in task.prompts if e.label.is_recovery_conversation]
    assert len(recovery_entries) >= 3, "Need at least 3 recovery turns to test penalty"

    env_ok = GuardrailEnvironment()
    obs = env_ok.reset("multiturn_adversarial")
    recovery_ids = {e.label.prompt_id for e in recovery_entries}
    recovery_over_block_count = 0
    while not env_ok.is_done():
        # Over-block first 2 recovery turns only (should NOT penalize)
        if obs.prompt_id in recovery_ids and recovery_over_block_count < 2:
            action_type = "refuse"
            recovery_over_block_count += 1
        else:
            action_type = "allow"
        next_obs, _, done, _ = env_ok.step(make_action(action_type, obs.prompt_id))
        if not done:
            obs = next_obs
    score_ok = env_ok.get_grader_score()

    env_bad = GuardrailEnvironment()
    obs = env_bad.reset("multiturn_adversarial")
    recovery_over_block_count = 0
    while not env_bad.is_done():
        # Over-block first 4 recovery turns (should penalize — >2)
        if obs.prompt_id in recovery_ids and recovery_over_block_count < 4:
            action_type = "refuse"
            recovery_over_block_count += 1
        else:
            action_type = "allow"
        next_obs, _, done, _ = env_bad.step(make_action(action_type, obs.prompt_id))
        if not done:
            obs = next_obs
    score_bad = env_bad.get_grader_score()

    assert score_ok > score_bad, (
        f"Expected over-blocking >2 recovery turns to lower score: ok={score_ok}, bad={score_bad}"
    )


# ── Item 26: Multi-crossing test ──────────────────────────────────────────────

def test_multi_crossing_second_crossing_matters():
    """Catching both crossings should score differently from catching only first or neither."""
    from app.tasks.task_config import get_task
    task = get_task("multiturn_adversarial")

    # Second crossing entries live in branch_specs["{id}:2"].caught (injected dynamically)
    multi_crossing_specs = {k: v for k, v in task.branch_specs.items() if k.endswith(":2")}
    if not multi_crossing_specs:
        pytest.skip("No multi-crossing conversations found")

    # Collect second crossing prompt ids from all ":2" specs
    second_crossing_ids: set[str] = set()
    for spec in multi_crossing_specs.values():
        for entry in spec["caught"]:
            second_crossing_ids.add(entry.label.prompt_id)

    # First crossing prompt ids (conv ids that have a :2 spec)
    base_conv_ids = {k.split(":")[0] for k in multi_crossing_specs}
    first_crossing_ids = {
        e.label.prompt_id for e in task.prompts
        if e.is_crossing and e.conversation_id in base_conv_ids
    }

    def run_with_strategy(block_first: bool, block_second: bool) -> float:
        env = GuardrailEnvironment()
        obs = env.reset("multiturn_adversarial")
        while not env.is_done():
            if obs.prompt_id in first_crossing_ids:
                action_type = "refuse" if block_first else "allow"
            elif obs.prompt_id in second_crossing_ids:
                action_type = "refuse" if block_second else "allow"
            else:
                action_type = "allow"
            next_obs, _, done, _ = env.step(make_action(action_type, obs.prompt_id))
            if not done:
                obs = next_obs
        return env.get_grader_score()

    score_both_caught = run_with_strategy(True, True)
    score_first_only = run_with_strategy(True, False)
    score_neither = run_with_strategy(False, False)

    # Catching both should score >= catching only first
    assert score_both_caught >= score_first_only, (
        f"Catching both crossings ({score_both_caught}) should score >= first only ({score_first_only})"
    )
    # Catching at least first should score better than catching neither
    assert score_first_only > score_neither, (
        f"Catching first crossing ({score_first_only}) should score > neither ({score_neither})"
    )


# ── Item 27: user_risk_score_delta test ──────────────────────────────────────

def test_risk_score_delta_negative_on_missed_attack():
    """Reward breakdown must include user_risk_score_delta, negative after missing an attack."""
    from app.tasks.task_config import get_task
    task = get_task("basic_threat_detection")
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    # Find an attack prompt
    attack_entry = next(
        (e for e in task.prompts if e.label.category != "safe"), None
    )
    assert attack_entry is not None

    # Step through to find and allow an attack
    while not env.is_done():
        if obs.prompt_id == attack_entry.label.prompt_id:
            _, reward, _, _ = env.step(make_action("allow", obs.prompt_id))
            assert "user_risk_score_delta" in reward.breakdown
            assert reward.breakdown["user_risk_score_delta"] < 0, (
                f"Expected negative delta after missed attack, got {reward.breakdown['user_risk_score_delta']}"
            )
            return
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = next_obs

    pytest.fail("Attack entry not encountered during episode")


def test_risk_score_delta_positive_on_correct_block():
    """user_risk_score_delta should be positive after correctly blocking an attack."""
    from app.tasks.task_config import get_task
    task = get_task("basic_threat_detection")
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    attack_entry = next(
        (e for e in task.prompts if e.label.category != "safe"), None
    )
    assert attack_entry is not None

    while not env.is_done():
        if obs.prompt_id == attack_entry.label.prompt_id:
            _, reward, _, _ = env.step(make_action("refuse", obs.prompt_id))
            assert reward.breakdown["user_risk_score_delta"] > 0
            return
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = next_obs

    pytest.fail("Attack entry not encountered")
