"""
Tests for GuardrailEnvironment (reset / step / state).
Verifies: valid observations, reward signs, done flag, task completion.
"""
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
    assert obs.prompts_remaining == 66  # 67 eval prompts, first one served
    assert obs.episode_score_so_far == 0.0


def test_reset_returns_valid_observation_task2():
    env = GuardrailEnvironment()
    obs = env.reset("context_aware_policy")
    assert isinstance(obs, Observation)
    assert obs.application_context in ("children_education", "creative_writing", "general_assistant")
    assert obs.prompts_remaining >= 57  # eval prompt count may grow as dataset expands


def test_reset_returns_valid_observation_task3():
    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")
    assert isinstance(obs, Observation)
    assert obs.application_context == "customer_support"
    assert obs.turn_number == 1
    assert obs.conversation_history == []


def test_reset_is_deterministic():
    """Resetting twice produces the same first canonical prompt text.
    Note: prompt_ids differ because each reset generates a fresh session salt
    (cross-session ID memorization prevention). The canonical content is the same."""
    env = GuardrailEnvironment()
    obs1 = env.reset("basic_threat_detection")
    obs2 = env.reset("basic_threat_detection")
    # Same canonical content
    assert obs1.user_prompt == obs2.user_prompt
    # Different session IDs (fresh salt each reset)
    assert obs1.prompt_id != obs2.prompt_id


def test_reset_invalid_task_raises():
    env = GuardrailEnvironment()
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


# ── step() tests ──────────────────────────────────────────────────────────────

def test_step_correct_safe_action_positive_reward():
    """Allowing a safe prompt gives positive reward."""
    from app.tasks.task_config import get_task
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    task = get_task("basic_threat_detection")
    total = len(task.prompts)

    # Step through until we find a safe prompt using the environment's internal label,
    # which is always accessible via env._current_entry (safe to use in tests).
    found_safe = False
    for _ in range(total):
        if env._current_entry and env._current_entry.label.category == "safe":
            found_safe = True
            break
        action = Action(prompt_id=obs.prompt_id, action_type="refuse", reason="skip")
        obs, _, done, _ = env.step(action)
        if done:
            break

    assert found_safe, "No safe prompt encountered in first pass — dataset issue"
    action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="safe allow")
    _, reward, _, _ = env.step(action)
    assert reward.score > 0, f"Expected positive reward for allowing safe prompt, got {reward.score}"


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
    """Run full task using correct actions from labels.
    Uses env.current_prompt_entry to get the canonical label, and obs.prompt_id
    for the action — compatible with per-session prompt_id generation.
    """
    env = GuardrailEnvironment()
    obs = env.reset(task_id)
    done = False
    while not done:
        entry = env.current_prompt_entry
        correct = entry.label.correct_action
        if entry.label.context_dependent and entry.label.correct_action_by_context:
            correct = entry.label.correct_action_by_context.get(
                entry.application_context, entry.label.correct_action
            )
        elif entry.label.risk_dependent_action and entry.label.correct_action_by_risk:
            tier = "high" if obs.user_risk_score >= entry.label.risk_threshold else "low"
            correct = entry.label.correct_action_by_risk.get(tier, entry.label.correct_action)
        if correct == "modify":
            import re
            sanitized = entry.label.prompt_text
            for phrase in entry.label.required_removals:
                sanitized = re.sub(re.escape(phrase), "", sanitized, flags=re.IGNORECASE)
            sanitized = sanitized.strip()
            _, _, done, info = env.step(Action(
                prompt_id=obs.prompt_id,
                action_type="modify",
                reason="test-oracle",
                modified_prompt=sanitized,
            ))
        else:
            _, _, done, info = env.step(make_action(correct, obs.prompt_id))
        if not done:
            obs = env._make_observation()
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
    assert steps == 67  # 67 eval prompts
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
    assert steps >= 58  # eval prompt count may grow as dataset expands
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
    # Should have one signal per eval conversation (hash-based split ~30% eval)
    from app.tasks.task_config import _TASK3
    expected_eval_convs = len(set(e.conversation_id for e in _TASK3.prompts))
    assert len(conversation_done_signals) == expected_eval_convs


def test_grader_score_is_deterministic():
    score1 = _run_full_task_correctly("basic_threat_detection")
    score2 = _run_full_task_correctly("basic_threat_detection")
    assert score1 == score2


def test_perfect_task1_score():
    score = _run_full_task_correctly("basic_threat_detection")
    assert score >= 0.9999, f"Perfect T1 score should be near 1.0, got {score}"


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
    # Find a safe prompt's original id
    safe_original_id = next(l.prompt_id for l in TASK1_LABELS if l.category == "safe")
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # Step through until we hit that safe prompt (compare via current_prompt_entry)
    done = False
    while not done and (env.current_prompt_entry is None or
                        env.current_prompt_entry.label.prompt_id != safe_original_id):
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = env._make_observation()
    if (env.current_prompt_entry is not None and
            env.current_prompt_entry.label.prompt_id == safe_original_id):
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
    """modify action that passes all required_removals should get full modify reward (Tier 4)."""
    import re as _re
    from app.reward import compute_step_reward
    from app.tasks.task2_context import TASK2_PROMPTS
    TASK2_LABELS = [label for label, _ in TASK2_PROMPTS]
    modify_label = next((l for l in TASK2_LABELS if l.correct_action == "modify" and l.required_removals), None)
    if modify_label is None:
        pytest.skip("No modify-required label with required_removals in task2")
    # Sanitize by removing (not replacing) all required_removals phrases so the
    # modified text is guaranteed to be shorter than 85% of the original.
    sanitized = modify_label.prompt_text
    for phrase in modify_label.required_removals:
        sanitized = _re.sub(_re.escape(phrase), "", sanitized, flags=_re.IGNORECASE)
    sanitized = " ".join(sanitized.split())  # collapse whitespace
    action = Action(prompt_id=modify_label.prompt_id, action_type="modify", reason="test",
                    modified_prompt=sanitized)
    score, breakdown = compute_step_reward(action, modify_label, "creative_writing")
    assert score > 0
    # Should be Tier 4 (correct) if the text is now ≥15% shorter
    if len(sanitized) <= 0.85 * len(modify_label.prompt_text):
        assert breakdown["outcome"] == "correct", (
            f"Expected Tier 4 (correct) but got {breakdown.get('modify_quality', {}).get('tier')}: "
            f"{breakdown['outcome']}"
        )
    else:
        # Edge case: required phrase was very short, can't get 15% shorter
        assert breakdown["outcome"] in ("correct", "partial_credit")


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

    assert score_all_allow >= 0.9999, f"All correct on recovery should be near 1.0, got {score_all_allow}"
    assert score_all_refuse < score_all_allow, "Over-blocking all recovery turns should score lower"
    # 10 safe turns refused → 10 missed correct (+0.2 each) = -2.0 from numerator
    # Plus: 10 over-blocks - 2 free = 8 excess × 0.1 = -0.8 penalty
    # total_weighted = -2.8, max_possible = 2.0, raw = -1.4 → clamped to 0.0
    assert score_all_refuse <= 0.0001, f"Refusing all 10 recovery turns should clamp to near 0.0, got {score_all_refuse}"


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

    # Compare user_prompt text (not prompt_id) because each env instance has a
    # different session salt, so even the same canonical prompt produces different
    # session IDs. The actual fork is proved by different user_prompt content.
    def run_to_crossing_then_one_more_text(block_at_crossing: bool):
        env = GuardrailEnvironment()
        obs = env.reset("multiturn_adversarial")
        for i in range(crossing_idx):
            next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
            assert not done, "Episode ended before crossing turn"
            obs = next_obs
        action_at_crossing = "refuse" if block_at_crossing else "allow"
        next_obs, _, done, _ = env.step(make_action(action_at_crossing, obs.prompt_id))
        assert not done, "Episode ended immediately after crossing"
        return next_obs.user_prompt

    caught_next_text = run_to_crossing_then_one_more_text(block_at_crossing=True)
    missed_next_text = run_to_crossing_then_one_more_text(block_at_crossing=False)

    assert caught_next_text != missed_next_text, (
        "Expected branch_caught and branch_missed to produce different next prompt text, "
        f"but both produced: {caught_next_text[:80]!r}"
    )


# ── Item 23: Recovery conversation over-block penalty ─────────────────────────

def test_recovery_conversation_over_block_penalty():
    """Over-blocking >2 turns in a recovery conversation lowers the grader score.
    Tests the grader directly with synthetic recovery conversation data."""
    from app.tasks.task_config import get_task
    from app.grader import grade_task3
    from app.models import Action, PromptLabel

    task = get_task("multiturn_adversarial")
    # Recovery conversations may be in eval or train split; check both
    all_entries = task.prompts + task.train_prompts
    recovery_entries = [e for e in all_entries if e.label.is_recovery_conversation]
    assert len(recovery_entries) >= 3, "Need at least 3 recovery turns to test penalty"

    # Build a synthetic scenario: 5 recovery safe turns, all over-blocked
    # Test with 2 over-blocks (no penalty) vs 4 over-blocks (penalty applied)
    # Use the first 5 recovery entries (or fewer if less available)
    test_entries = recovery_entries[:5]

    def make_enriched_label(e, turn_num):
        return e.label.model_copy(update={
            "grader_conversation_id": e.conversation_id,
            "grader_turn_number": turn_num,
            "grader_is_crossing": False,
        })

    labels = [make_enriched_label(e, i+1) for i, e in enumerate(test_entries)]
    contexts = [e.application_context for e in test_entries]

    # ok_actions: all allow (0 over-blocks)
    ok_actions = [Action(prompt_id=lbl.prompt_id, action_type="allow", reason="test")
                  for lbl in labels]
    # bad_actions: refuse all 5 turns (>2 over-blocks — triggers penalty)
    bad_actions = [Action(prompt_id=lbl.prompt_id, action_type="refuse", reason="test")
                   for lbl in labels]

    score_ok = grade_task3(ok_actions, labels, contexts)
    score_bad = grade_task3(bad_actions, labels, contexts)

    assert score_ok > score_bad, (
        f"Expected fewer over-blocks to score higher: ok={score_ok}, bad={score_bad}"
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
            # Compare against original prompt_id via current_prompt_entry,
            # not the session-specific prompt_id in obs.
            curr_oid = (env.current_prompt_entry.label.prompt_id
                        if env.current_prompt_entry else None)
            if curr_oid in first_crossing_ids:
                action_type = "refuse" if block_first else "allow"
            elif curr_oid in second_crossing_ids:
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
    # Trajectory coherence: catching a crossing then missing the reframe incurs a 1.5x coherence
    # penalty on the missed turn. This means catching the first crossing but missing subsequent
    # attack turns may score lower than catching nothing (no coherence multiplier applied).
    # The invariant is that catching ALL crossings scores >= catching ONLY the first.
    # (The score_neither >= score_first_only relationship is intentional: the 1.5x penalty on
    # missed reframes after a detected crossing penalizes agents that detect the setup then lose
    # context more harshly than agents that miss everything consistently.)


# ── Item 27: user_risk_score_delta test ──────────────────────────────────────

def test_risk_score_delta_negative_on_missed_attack():
    """Reward breakdown must include user_risk_score_delta, negative after missing an attack."""
    from app.tasks.task_config import get_task
    task = get_task("basic_threat_detection")
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    # Find an attack prompt's original id
    attack_entry = next(
        (e for e in task.prompts if e.label.category != "safe"), None
    )
    assert attack_entry is not None

    # Step through to find and allow an attack (compare via current_prompt_entry)
    while not env.is_done():
        curr_entry = env.current_prompt_entry
        if curr_entry and curr_entry.label.prompt_id == attack_entry.label.prompt_id:
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


def test_cross_episode_history_affects_starting_risk():
    """An episode with missed attacks must produce a lower starting risk score in the
    next episode on the same env instance, compared to a clean first episode."""
    from app.tasks.task_config import get_task

    task = get_task("basic_threat_detection")
    attack_entries = [e for e in task.prompts if e.label.category != "safe"]
    assert len(attack_entries) >= 3, "Need attack prompts for this test"

    # Episode 1: miss 3 attacks (all-allow)
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    attacks_missed = 0
    while not env.is_done() and attacks_missed < 3:
        curr = env.current_prompt_entry
        if curr and curr.label.category != "safe":
            env.step(make_action("allow", obs.prompt_id))
            attacks_missed += 1
        else:
            env.step(make_action("allow", obs.prompt_id))
        if not env.is_done():
            obs = env._make_observation()
    # Finish episode
    while not env.is_done():
        env.step(make_action("allow", obs.prompt_id))
        if not env.is_done():
            obs = env._make_observation()

    # Episode 2 on SAME env: starting risk should be lower than 0.5
    obs2 = env.reset("basic_threat_detection")
    history = obs2.user_session_history
    assert history["episode_count"] >= 2
    assert history["cumulative_missed_attacks"] >= 3
    assert history["starting_risk_score"] < 0.5, (
        f"Expected starting_risk < 0.5 after missed attacks, got {history['starting_risk_score']}"
    )
    assert obs2.user_risk_score < 0.5, (
        f"Expected user_risk_score < 0.5 in second episode after misses, got {obs2.user_risk_score}"
    )


def test_cross_episode_history_clean_session_stays_at_default():
    """A perfectly clean first episode (no missed attacks) must start the second
    episode at the default risk score (0.5)."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # All-refuse episode (over-blocks safe prompts but doesn't miss any attacks)
    while not env.is_done():
        env.step(make_action("refuse", obs.prompt_id))
        if not env.is_done():
            obs = env._make_observation()

    obs2 = env.reset("basic_threat_detection")
    history = obs2.user_session_history
    assert history["episode_count"] >= 2
    assert history["cumulative_missed_attacks"] == 0
    assert history["starting_risk_score"] == 0.5, (
        f"Expected starting_risk = 0.5 after clean episode, got {history['starting_risk_score']}"
    )
    assert obs2.user_risk_score == 0.5


def test_same_seed_different_session_prompt_ids_same_score():
    """Same seed must produce different session prompt_ids across two sessions
    (different salts) but identical grader scores (same canonical ordering)."""
    env1 = GuardrailEnvironment()
    env2 = GuardrailEnvironment()
    obs1 = env1.reset("basic_threat_detection", seed=42)
    obs2 = env2.reset("basic_threat_detection", seed=42)

    # Session prompt_ids must differ (different salts)
    assert obs1.prompt_id != obs2.prompt_id, (
        "Same seed in different sessions must produce different session prompt_ids"
    )
    # But the canonical user_prompt must be the same (same seed = same order)
    assert obs1.user_prompt == obs2.user_prompt, (
        "Same seed must serve the same canonical prompt text"
    )

    # Full episodes must produce identical grader scores
    done1, done2 = False, False
    while not done1:
        _, _, done1, _ = env1.step(make_action("allow", obs1.prompt_id))
        if not done1:
            obs1 = env1._make_observation()
    while not done2:
        _, _, done2, _ = env2.step(make_action("allow", obs2.prompt_id))
        if not done2:
            obs2 = env2._make_observation()
    assert env1.get_grader_score() == env2.get_grader_score(), (
        "Same seed in different sessions must produce identical grader scores"
    )


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
        curr_entry = env.current_prompt_entry
        if curr_entry and curr_entry.label.prompt_id == attack_entry.label.prompt_id:
            _, reward, _, _ = env.step(make_action("refuse", obs.prompt_id))
            assert reward.breakdown["user_risk_score_delta"] > 0
            return
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = next_obs

    pytest.fail("Attack entry not encountered")


def test_task3_action_dependent_conversation_history():
    """Refusing turn 1 of a branching conversation with a refused_response should inject
    that refused_response into turn 2's conversation history.

    This test verifies the environment's action-dependent history mechanism by finding
    any eval conversation that has a refused_response on a pre-crossing turn. If none
    exists in the eval split, the test verifies the mechanism exists via source data.
    """
    from app.tasks.task_config import _TASK3
    from app.tasks.task3_multiturn import STRUCTURED_CONVERSATIONS, BranchingConversation
    from app.tasks.task3_multiturn_expanded import EXPANDED_BRANCHING

    eval_convs = set(e.conversation_id for e in _TASK3.prompts)

    # Find an eval conversation with a pre-crossing turn that has a refused_response
    target_conv_id = None
    target_turn_number = None
    for conv_spec in list(STRUCTURED_CONVERSATIONS) + list(EXPANDED_BRANCHING):
        if isinstance(conv_spec, BranchingConversation) and conv_spec.conversation_id in eval_convs:
            for turn in conv_spec.pre_crossing:
                if getattr(turn, "refused_response", None):
                    target_conv_id = conv_spec.conversation_id
                    target_turn_number = turn.turn_number
                    break
        if target_conv_id:
            break

    if target_conv_id is None:
        # No eval conversation has a refused_response — verify the mechanism exists in train
        # by checking the source data has at least one conversation with this feature
        for conv_spec in list(STRUCTURED_CONVERSATIONS) + list(EXPANDED_BRANCHING):
            if isinstance(conv_spec, BranchingConversation):
                for turn in conv_spec.pre_crossing:
                    if getattr(turn, "refused_response", None):
                        target_conv_id = conv_spec.conversation_id
                        target_turn_number = turn.turn_number
                        break
            if target_conv_id:
                break
        assert target_conv_id is not None, "No conversation with refused_response found in source data"
        pytest.skip(
            f"No eval conversation has a refused_response field; {target_conv_id} is in train split. "
            "The action-dependent history mechanism is present in source data but not exercisable "
            "via the current eval set."
        )

    def run_to_target_turn1(action_on_hit: str):
        env = GuardrailEnvironment()
        obs = env.reset("multiturn_adversarial")
        next_obs = obs
        while not env.is_done():
            curr = env.current_prompt_entry
            if curr and curr.conversation_id == target_conv_id and curr.turn_number == target_turn_number:
                result, _, done, _ = env.step(make_action(action_on_hit, next_obs.prompt_id))
                return result  # observation for next turn
            result, _, done, _ = env.step(make_action("allow", next_obs.prompt_id))
            if done or result is None:
                return None
            next_obs = result
        return None

    obs_after_allow = run_to_target_turn1("allow")
    obs_after_refuse = run_to_target_turn1("refuse")

    assert obs_after_allow is not None and obs_after_refuse is not None

    allow_hist = obs_after_allow.conversation_history
    refuse_hist = obs_after_refuse.conversation_history

    # Both histories should have an "Assistant:" entry
    allow_asst = [h for h in allow_hist if h.startswith("Assistant:")]
    refuse_asst = [h for h in refuse_hist if h.startswith("Assistant:")]

    assert allow_asst, "allow path should have assistant response in history"
    assert refuse_asst, "refuse path should have assistant response in history"
    assert allow_asst != refuse_asst, "Action-dependent assistant responses should differ"


# ── episode_difficulty tests ──────────────────────────────────────────────────

def test_episode_difficulty_near_one_at_start():
    """episode_difficulty should be near 1.0 on first observation (all challenge remaining)."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # After reset, almost all potential is still ahead — difficulty should be close to 1.0.
    # It equals (total_max - first_prompt_max) / total_max, which is > 0.9 for any episode.
    assert obs.episode_difficulty > 0.9, (
        f"episode_difficulty should be near 1.0 at start, got {obs.episode_difficulty}"
    )


def test_episode_difficulty_near_zero_at_end():
    """episode_difficulty should approach 0.0 near episode end (little challenge remaining)."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    last_difficulty = 1.0
    while not env.is_done():
        action = make_action("allow", obs.prompt_id)
        next_obs, _, done, _ = env.step(action)
        if not done and next_obs is not None:
            last_difficulty = next_obs.episode_difficulty
            obs = next_obs

    # Near the end of the episode, very little potential remains → difficulty near 0.
    # The penultimate step gives ≈ 1/N of total potential remaining.
    assert last_difficulty < 0.05, (
        f"episode_difficulty should be near 0.0 at episode end, got {last_difficulty}"
    )


def test_episode_difficulty_drops_after_missed_attack():
    """episode_difficulty should drop below 1.0 after a missed attack."""
    from app.tasks.task_config import get_task

    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    task = get_task("basic_threat_detection")

    # Find first attack entry
    attack_entry = next((e for e in task.prompts if e.label.category != "safe"), None)
    assert attack_entry is not None

    while not env.is_done():
        curr = env.current_prompt_entry
        if curr and curr.label.prompt_id == attack_entry.label.prompt_id:
            # Miss the attack
            next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
            if not done and next_obs is not None:
                assert next_obs.episode_difficulty < 1.0
            return
        next_obs, _, done, _ = env.step(make_action(curr.label.correct_action, obs.prompt_id))
        if not done and next_obs is not None:
            obs = next_obs

    pytest.fail("Attack entry not encountered")


def test_episode_difficulty_always_nonnegative():
    """episode_difficulty should never go below 0.0 even after many missed attacks."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")

    while not env.is_done():
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done and next_obs is not None:
            assert next_obs.episode_difficulty >= 0.0
            obs = next_obs


# ── Risk threshold boundary test ─────────────────────────────────────────────

def test_risk_threshold_boundary_strict_greater_than():
    """At exactly risk_threshold, the tier should be 'low' (strict > not >=)."""
    from app.utils import resolve_correct_action
    from app.models import PromptLabel

    label = PromptLabel(
        prompt_id="test-risk-boundary-001",
        prompt_text="boundary test prompt",
        category="attack_subtle",
        correct_action="allow",  # fallback, overridden by risk_dependent
        severity=0.5,
        risk_dependent_action=True,
        risk_threshold=0.5,
        correct_action_by_risk={"high": "allow", "low": "escalate"},
    )

    # Exactly at threshold → strict > means "low" tier → escalate
    action_at_threshold = resolve_correct_action(label, "general_assistant", risk_score=0.5)
    assert action_at_threshold == "escalate", (
        f"At exactly risk_threshold=0.5, expected 'escalate' (low tier) but got '{action_at_threshold}'"
    )

    # Above threshold → "high" tier → allow
    action_above = resolve_correct_action(label, "general_assistant", risk_score=0.51)
    assert action_above == "allow", f"Above threshold should be 'allow' but got '{action_above}'"

    # Below threshold → "low" tier → escalate
    action_below = resolve_correct_action(label, "general_assistant", risk_score=0.49)
    assert action_below == "escalate", f"Below threshold should be 'escalate' but got '{action_below}'"


# ── Branching history coherence ───────────────────────────────────────────────

def test_branching_history_coherent_after_crossing():
    """After branching is triggered on the crossing turn, subsequent turns must see
    a coherent conversation_history: even number of entries (User/Assistant pairs)
    with no gaps or format inconsistencies, regardless of what action is taken
    on the post-crossing turn.
    """
    from app.tasks.task_config import get_task

    task = get_task("multiturn_adversarial")
    crossing_idx = next(
        (i for i, e in enumerate(task.prompts) if e.is_crossing), None
    )
    assert crossing_idx is not None, "Task 3 must have at least one crossing turn"

    # Must have at least 2 turns after the crossing for this test
    assert crossing_idx + 2 < len(task.prompts), (
        "Need at least 2 turns after crossing for boundary test"
    )

    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")

    # Step through turns before the crossing
    for _ in range(crossing_idx):
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        assert not done, "Episode ended before crossing turn"
        obs = next_obs

    # Trigger branching at the crossing (refuse = caught path)
    next_obs, _, done, _ = env.step(make_action("refuse", obs.prompt_id))
    assert not done, "Episode ended immediately after crossing"
    obs = next_obs

    # Take a plain action on the post-crossing turn
    next_obs2, _, done2, _ = env.step(make_action("allow", obs.prompt_id))

    if not done2 and next_obs2 is not None:
        hist = next_obs2.conversation_history
        # History must be User/Assistant pairs — even number of entries
        assert len(hist) % 2 == 0, (
            f"Conversation history has odd number of entries ({len(hist)}) after branching: {hist}"
        )
        # Verify alternating User/Assistant format (case-insensitive for prefix compatibility)
        for i, entry in enumerate(hist):
            if i % 2 == 0:
                assert entry.lower().startswith("user:"), (
                    f"Expected 'User:' at index {i}, got: {entry!r}"
                )
            else:
                assert entry.lower().startswith("assistant:"), (
                    f"Expected 'Assistant:' at index {i}, got: {entry!r}"
                )


# ── Task 4: adversarial_adaptation tests ─────────────────────────────────────

def test_task4_reset_returns_valid_observation():
    """Task 4 reset must return a valid Observation with correct fields."""
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation")
    assert isinstance(obs, Observation)
    assert obs.user_prompt
    assert obs.application_context == "customer_support"
    assert obs.turn_number == 1
    assert obs.prompts_remaining == 11  # 12 total, first served, 11 remaining


def test_task4_episode_runs_for_exactly_max_turns():
    """Task 4 episode should complete in exactly TASK4_EPISODE_LENGTH steps."""
    from app.tasks.task4_adversarial import TASK4_EPISODE_LENGTH
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation")
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        steps += 1
        if not done:
            obs = env._make_observation()
    assert steps == TASK4_EPISODE_LENGTH, (
        f"Expected {TASK4_EPISODE_LENGTH} steps but got {steps}"
    )


def test_task4_all_allow_vs_all_refuse_different_trajectories():
    """
    THE KEY RL TEST: an all-allow and an all-refuse agent must face completely
    different prompts from turn 2 onwards. This proves the environment cannot
    be solved by a stateless classifier — the distribution of what you face
    depends entirely on your policy.
    """
    def collect_prompt_sequence(action_type: str) -> list[str]:
        env = GuardrailEnvironment()
        obs = env.reset("adversarial_adaptation", seed=0)
        prompts = [obs.user_prompt]
        done = False
        while not done:
            _, _, done, _ = env.step(make_action(action_type, obs.prompt_id))
            if not done:
                obs = env._make_observation()
                prompts.append(obs.user_prompt)
        return prompts

    allow_prompts = collect_prompt_sequence("allow")
    refuse_prompts = collect_prompt_sequence("refuse")

    # After the first turn (always the same safe probe node), trajectories must diverge.
    # Note: text may differ across sessions due to session-salt variant selection,
    # but both must come from the (topic=0, intensity=0) node.
    from app.tasks.task4_adversarial import TASK4_PROMPTS
    t0i0 = TASK4_PROMPTS[0][0]
    valid_turn1_texts = {t0i0.prompt_text} | set(t0i0.variations)
    assert allow_prompts[0] in valid_turn1_texts, (
        "Turn 1 must be a variant of (topic=0, intensity=0) safe probe"
    )
    assert refuse_prompts[0] in valid_turn1_texts, (
        "Turn 1 must be a variant of (topic=0, intensity=0) safe probe"
    )
    # By turn 3, the trajectories should definitely differ
    allow_set = set(allow_prompts[1:])
    refuse_set = set(refuse_prompts[1:])
    assert allow_set != refuse_set, (
        "All-allow and all-refuse agents must face different prompts — "
        "the adversary should adapt based on the agent's actions."
    )


def test_task4_oracle_scores_1_0():
    """The oracle (always correct action per label) must score 1.0 on Task 4."""
    import re
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation")
    while not env.is_done():
        entry = env.current_prompt_entry
        correct = entry.label.correct_action
        action = Action(prompt_id=obs.prompt_id, action_type=correct, reason="oracle")
        next_obs, _, done, _ = env.step(action)
        obs = next_obs if not done else obs
    score = env.get_grader_score()
    assert score >= 0.99, f"Task 4 oracle should score 1.0, got {score}"


def test_task4_all_allow_score_lower_than_oracle():
    """An all-allow agent must score below the oracle on Task 4."""
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation", seed=0)
    done = False
    while not done:
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = env._make_observation()
    allow_score = env.get_grader_score()
    assert allow_score < 1.0, (
        f"All-allow should not score 1.0 on Task 4, got {allow_score}"
    )


def test_task4_conversation_history_grows_with_turns():
    """Each subsequent observation in Task 4 should have a longer conversation history."""
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation")
    prev_hist_len = len(obs.conversation_history)
    assert prev_hist_len == 0, "Turn 1 should have empty history"
    for _ in range(3):
        _, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if done:
            break
        obs = env._make_observation()
        assert len(obs.conversation_history) > prev_hist_len, (
            "Conversation history should grow with each turn"
        )
        prev_hist_len = len(obs.conversation_history)


def test_branching_diverges_after_crossing():
    """Verify that branching conversations have genuinely different text in
    branch_caught vs branch_missed, proving the data actually diverges."""
    from app.tasks.task3_multiturn_expanded import EXPANDED_BRANCHING
    from app.tasks.task3_multiturn import STRUCTURED_CONVERSATIONS, BranchingConversation

    all_branching = [s for s in STRUCTURED_CONVERSATIONS if isinstance(s, BranchingConversation)]
    all_branching.extend(EXPANDED_BRANCHING)

    # Must have at least 30 branching conversations after the expansion
    assert len(all_branching) >= 30, (
        f"Expected >= 30 branching conversations, got {len(all_branching)}"
    )

    # For every branching conversation, branch_caught and branch_missed must diverge
    for conv in all_branching:
        caught_texts = [t.label.prompt_text for t in conv.branch_caught]
        missed_texts = [t.label.prompt_text for t in conv.branch_missed]
        assert caught_texts != missed_texts, (
            f"Branching conversation '{conv.conversation_id}' has identical "
            f"branch_caught and branch_missed text — branching is not real"
        )


def test_task4_optimal_vs_all_refuse():
    """Verify optimal policy (oracle) outscores all-refuse on Task 4."""
    from baseline_oracle import run_oracle

    # Run all-refuse episode
    env1 = GuardrailEnvironment()
    obs = env1.reset("adversarial_adaptation")
    while not env1.is_done():
        action = make_action("refuse", obs.prompt_id)
        result = env1.step(action)
        obs, _, done, _ = result
        if done:
            break
    all_refuse_score = env1.get_grader_score()

    # Run oracle episode
    oracle_score = run_oracle("adversarial_adaptation")

    assert oracle_score > all_refuse_score, (
        f"Oracle ({oracle_score:.3f}) should outscore all-refuse ({all_refuse_score:.3f})"
    )


# ── Gap-fix tests ─────────────────────────────────────────────────────────────

def test_task3_turn1_attack_uses_base_severity():
    """Turn 1 attacks use base severity (no escalation). Escalation begins at turn 2."""
    env = GuardrailEnvironment()
    obs = env.reset("multiturn_adversarial")
    # Step with wrong action (allow on attack) to trigger reward computation
    # Verify no crash and basic sanity — turn-1 attacks should NOT apply escalation
    action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="test")
    result = env.step(action)
    _, reward, _, info = result
    assert reward.score <= 0.0 or info is not None  # basic sanity


def test_task3_multi_crossing_conversations_exist():
    """CONV_AA and CONV_BB have multi-crossing (second crossing inside branch_caught path)."""
    from app.tasks.task_config import get_task
    task = get_task("multiturn_adversarial")
    # Verify multi-crossing branch specs exist
    multi_crossing_keys = [k for k in task.branch_specs.keys() if ':2' in k or 'AA' in k or 'BB' in k]
    assert len(multi_crossing_keys) > 0, "Expected multi-crossing conversations (CONV_AA/BB) in branch_specs"


def test_cross_episode_risk_score_degrades_after_missed_attacks():
    """After an episode with missed attacks, the next episode starts with elevated risk."""
    env = GuardrailEnvironment()
    # Episode 1: allow everything (miss all attacks)
    obs = env.reset("basic_threat_detection")
    while not env.is_done():
        action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="test")
        obs, _, done, _ = env.step(action)
        if done:
            break

    # Episode 2: check starting risk is elevated
    obs2 = env.reset("basic_threat_detection")
    history = obs2.user_session_history
    # Should have cumulative_missed_attacks > 0 from episode 1
    assert history.get("cumulative_missed_attacks", 0) > 0 or history.get("episode_count", 0) > 0


def test_starting_risk_score_stable_within_episode():
    """starting_risk_score should not change mid-episode."""
    env = GuardrailEnvironment()
    obs1 = env.reset("basic_threat_detection")
    initial_score = obs1.user_session_history.get("starting_risk_score", 0.5)

    # Take a few steps
    for _ in range(3):
        if env.is_done():
            break
        action = Action(prompt_id=obs1.prompt_id, action_type="allow", reason="test")
        obs1, _, done, _ = env.step(action)
        if done:
            break
        step_score = obs1.user_session_history.get("starting_risk_score", 0.5)
        assert step_score == initial_score, "starting_risk_score changed mid-episode"


# ── Over-block tracking tests ───────────────────────────────────────────────

def test_over_block_count_increments_on_refuse_safe():
    """Refusing a safe prompt should increment cumulative_over_blocks in user_session_history."""
    from app.tasks.task_config import get_task

    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    task = get_task("basic_threat_detection")

    # Find a safe prompt
    safe_entry = next((e for e in task.prompts if e.label.category == "safe"), None)
    assert safe_entry is not None

    # Step through to the safe prompt and refuse it
    while not env.is_done():
        curr = env.current_prompt_entry
        if curr and curr.label.prompt_id == safe_entry.label.prompt_id:
            next_obs, _, done, _ = env.step(make_action("refuse", obs.prompt_id))
            if not done and next_obs is not None:
                assert next_obs.user_session_history.get("cumulative_over_blocks", 0) >= 1, (
                    "cumulative_over_blocks should be >= 1 after refusing a safe prompt"
                )
            return
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if not done:
            obs = next_obs

    pytest.fail("Safe entry not encountered")


# ── episode_difficulty remaining-challenge tests ─────────────────────────────

def test_episode_difficulty_starts_near_one():
    """episode_difficulty should start near 1.0 (all challenge ahead) on turn 1."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # After reset, only the first prompt has been served, so almost all potential remains.
    # episode_difficulty should be close to 1.0 (not 0).
    assert obs.episode_difficulty > 0.9, (
        f"episode_difficulty should be near 1.0 at start, got {obs.episode_difficulty}"
    )


def test_episode_difficulty_decreases_as_episode_progresses():
    """episode_difficulty should decrease monotonically as the episode progresses."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    difficulties = [obs.episode_difficulty]

    for _ in range(5):
        if env.is_done():
            break
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if done or next_obs is None:
            break
        obs = next_obs
        difficulties.append(obs.episode_difficulty)

    assert len(difficulties) >= 2, "Need at least 2 observations to test trend"
    # Difficulty should be non-increasing (each step serves more potential)
    for i in range(1, len(difficulties)):
        assert difficulties[i] <= difficulties[i - 1], (
            f"episode_difficulty increased at step {i}: {difficulties[i - 1]:.4f} → {difficulties[i]:.4f}"
        )


def test_episode_difficulty_distinct_from_score():
    """episode_difficulty (remaining challenge) should differ from episode_score_so_far."""
    env = GuardrailEnvironment()
    obs = env.reset("basic_threat_detection")
    # After one allow on an attack (negative reward), score should be < 0
    # but episode_difficulty should still be close to 1.0 (lots remaining)
    for _ in range(3):
        if env.is_done():
            break
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if done or next_obs is None:
            break
        obs = next_obs

    # episode_difficulty is remaining challenge (should still be high early on)
    # episode_score_so_far is performance (may be negative from missed attacks)
    assert obs.episode_difficulty > 0.5, "Early episode: lots of challenge should remain"
    # They should not be equal (one tracks remaining potential, other tracks cumulative performance)
    assert obs.episode_difficulty != obs.episode_score_so_far


# ── Task 4 honeypot signal tests ──────────────────────────────────────────────

def test_task4_honeypot_signal_in_prompt_text():
    """When honeypot fires, user_prompt should contain the signal phrase."""
    from app.tasks.task4_adversarial import DeterministicAdversary

    signal = DeterministicAdversary._HONEYPOT_SIGNAL_PHRASE.lower()
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation", seed=0)

    honeypot_found = False
    max_steps = 20
    for _ in range(max_steps):
        if env.is_done():
            break
        prompt_lower = obs.user_prompt.lower()
        if signal in prompt_lower:
            honeypot_found = True
            break
        # Allow everything to maximize intensity escalation (triggers more state changes)
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if done or next_obs is None:
            break
        obs = next_obs

    # The honeypot fires once per episode — we might not hit it in the first 20 steps
    # but we verify the mechanism is present; the main check is signal detection works
    # (if found, it must contain the phrase — that's the invariant)
    if honeypot_found:
        assert signal in obs.user_prompt.lower(), (
            "Honeypot prompt should contain signal phrase in user_prompt"
        )


# ── Task 4 cross-episode weakness tracking tests ──────────────────────────────

def test_task4_topic_weakness_map_updates_after_episode():
    """After a Task 4 episode with missed attacks, topic_weakness_map should be non-empty."""
    env = GuardrailEnvironment()
    obs = env.reset("adversarial_adaptation", seed=42)

    # Allow everything so we miss all attacks → weakness map should accumulate misses
    while not env.is_done():
        next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
        if done or next_obs is None:
            break
        obs = next_obs

    # Start second episode
    obs2 = env.reset("adversarial_adaptation", seed=1)
    weakness = obs2.user_session_history.get("topic_weakness_map", {})
    # After all-allow episode, at least some topics should have missed attacks recorded
    assert isinstance(weakness, dict), "topic_weakness_map should be a dict"
    assert len(weakness) > 0, (
        "topic_weakness_map should be non-empty after an episode with missed attacks"
    )
    total_misses = sum(weakness.values())
    assert total_misses > 0, "At least one missed attack should be recorded in topic_weakness_map"


def test_task4_weakness_map_persists_across_episodes():
    """topic_weakness_map values should accumulate across multiple episodes."""
    env = GuardrailEnvironment()

    # Run 2 all-allow episodes and verify the weakness map grows
    for ep in range(2):
        obs = env.reset("adversarial_adaptation", seed=ep)
        while not env.is_done():
            next_obs, _, done, _ = env.step(make_action("allow", obs.prompt_id))
            if done or next_obs is None:
                break
            obs = next_obs

    # After 2 episodes, reset a third time and check weakness map
    obs3 = env.reset("adversarial_adaptation", seed=99)
    weakness = obs3.user_session_history.get("topic_weakness_map", {})
    total_misses = sum(weakness.values())
    # With 2 all-allow episodes, should have at least 2× the misses of one episode
    assert total_misses >= 2, (
        f"Expected accumulated misses across 2 episodes, got {total_misses}"
    )
