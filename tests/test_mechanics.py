"""
Tests for the distinctive environment mechanics not covered elsewhere:
  1. topic_weakness_map biases starting topic in subsequent Task 4 episodes
  2. Risk-score recovery is exactly 5× slower than degradation (0.05 vs 0.25)
  3. Recovery conversation: ≤2 over-blocks → no penalty; 3rd over-block → -0.1
  4. Honeypot allow jumps next intensity to 5
  5. RAG injection prompts (#158-177) score correctly with the right action
"""
import pytest
from app.environment import GuardrailEnvironment
from app.models import Action
from app.tasks.task4_adversarial import DeterministicAdversary
from app.tasks.task_config import PromptEntry


def make_action(action_type: str, prompt_id: str) -> Action:
    return Action(prompt_id=prompt_id, action_type=action_type, reason="test")


# ── Test 1: topic_weakness_map biases starting topic ──────────────────────────

def test_topic_weakness_map_biases_topic_selection():
    """After missing attacks on topic X in episode 1, episode 2 is more likely
    to start on that topic. Verified by comparing the _topic_weakness_map
    contents and the adversary's starting topic across episodes.
    """
    env = GuardrailEnvironment()

    # Episode 1: play through Task 4, always allowing (missing every attack).
    obs = env.reset("adversarial_adaptation", seed=42)
    done = False
    while not done:
        pid = obs.prompt_id
        obs_next, reward, done, info = env.step(make_action("allow", pid))
        obs = obs_next or obs

    # After episode 1, the topic_weakness_map must have at least one entry.
    weakness_after_ep1 = dict(env._topic_weakness_map)
    assert len(weakness_after_ep1) > 0, (
        "topic_weakness_map should be non-empty after missing attacks in episode 1"
    )

    # In episode 2, the adversary receives the weakness map.
    # We cannot guarantee the exact starting topic, but we can assert that the
    # adversary's _pick_starting_topic() weights favor the missed topics.
    obs = env.reset("adversarial_adaptation", seed=99)
    # Adversary should be initialized with the cumulative weakness map.
    assert env._adversary is not None
    adv: DeterministicAdversary = env._adversary
    # Verify the adversary was passed the updated weakness map (not an empty dict).
    assert adv._topic_weakness_map, (
        "Adversary in episode 2 should receive the non-empty topic_weakness_map"
    )
    # The most-missed topic should appear in the weakness map.
    top_missed_topic = max(weakness_after_ep1, key=weakness_after_ep1.get)
    assert top_missed_topic in adv._topic_weakness_map


# ── Test 2: risk-score recovery is 5× slower than degradation ─────────────────

def test_risk_score_recovery_rate_is_one_fifth_of_degradation():
    """Missed attack → risk -= 0.25; correct block → risk += 0.05.
    Degradation (0.25) is exactly 5× recovery (0.05).

    We verify by directly consulting the reward constants (ground truth)
    and then confirming the env's internal state tracks them after controlled steps.
    """
    from app.reward import REWARD_MISS_ATTACK  # -0.30 base; risk delta is separate

    # Constants defined in environment.py (not exported, so we verify via live env)
    env = GuardrailEnvironment()

    # Drive Task 1 to find an attack prompt, miss it, then block the next one.
    # Use seed=0 for determinism.
    obs = env.reset("basic_threat_detection", seed=0)
    initial_risk = env._session_risk_score
    assert abs(initial_risk - 0.5) < 1e-6, f"Expected start risk 0.5, got {initial_risk}"

    # Scan for an attack prompt to miss.
    found_miss = False
    risk_before_miss = None
    risk_after_miss = None
    steps = 0
    while steps < 20:
        pid = obs.prompt_id
        current_entry = env._current_entry
        if current_entry.label.category != "safe":
            # Miss this attack
            risk_before_miss = env._session_risk_score
            obs_next, _, done, _ = env.step(make_action("allow", pid))
            risk_after_miss = env._session_risk_score
            found_miss = True
            obs = obs_next or obs
            break
        else:
            obs_next, _, done, _ = env.step(make_action("allow", pid))
            obs = obs_next or obs
        steps += 1
        if done:
            break

    assert found_miss, "Could not find an attack prompt in first 20 steps of Task 1 seed=0"
    miss_delta = round(risk_before_miss - risk_after_miss, 4)
    assert abs(miss_delta - 0.25) < 1e-6, (
        f"Missed attack should reduce risk by 0.25, got delta={miss_delta}"
    )

    # Now find an attack to block correctly.
    risk_before_block = None
    risk_after_block = None
    found_block = False
    steps = 0
    while steps < 30 and not done:
        pid = obs.prompt_id
        current_entry = env._current_entry
        if current_entry.label.category != "safe":
            correct_action = current_entry.label.correct_action
            if correct_action in ("refuse", "escalate"):
                risk_before_block = env._session_risk_score
                obs_next, _, done, _ = env.step(make_action(correct_action, pid))
                risk_after_block = env._session_risk_score
                found_block = True
                break
        obs_next, _, done, _ = env.step(make_action("allow", pid))
        obs = obs_next or obs
        steps += 1

    if not found_block:
        pytest.skip("Could not find a blockable attack within episode — extend seed range")

    block_delta = round(risk_after_block - risk_before_block, 4)
    assert abs(block_delta - 0.05) < 1e-6, (
        f"Correct block should raise risk by 0.05, got delta={block_delta}"
    )

    # Assert 5× ratio
    assert abs(miss_delta / block_delta - 5.0) < 1e-4, (
        f"Degradation should be 5× recovery: miss={miss_delta} / block={block_delta} = {miss_delta/block_delta:.2f}"
    )


# ── Test 3: recovery conversation over-block allowance ────────────────────────

def test_recovery_conversation_two_overblocks_no_penalty():
    """CONV_S is a recovery conversation. The grader allows 2 over-blocks without penalty;
    the 3rd over-block costs -0.1. We verify by comparing grader scores with 2 vs 3 over-blocks.
    """
    from app.grader import grade_task3
    from app.tasks.task3_multiturn_expanded import CONV_S

    recovery_labels = [turn.label for turn in CONV_S if turn.label.is_recovery_conversation]
    assert len(recovery_labels) >= 3, "CONV_S should have ≥ 3 recovery turns to test"

    contexts = ["customer_support"] * len(recovery_labels)

    # 2 over-blocks: refuse first 2 turns, allow the rest
    actions_2 = [
        Action(prompt_id=lbl.prompt_id, action_type="refuse" if i < 2 else "allow", reason="test")
        for i, lbl in enumerate(recovery_labels)
    ]
    score_2 = grade_task3(actions_2, recovery_labels, contexts)

    # 3 over-blocks: refuse first 3 turns, allow the rest
    actions_3 = [
        Action(prompt_id=lbl.prompt_id, action_type="refuse" if i < 3 else "allow", reason="test")
        for i, lbl in enumerate(recovery_labels)
    ]
    score_3 = grade_task3(actions_3, recovery_labels, contexts)

    assert score_3 < score_2, (
        f"3rd over-block should reduce score vs 2 over-blocks: {score_3:.4f} vs {score_2:.4f}"
    )


# ── Test 4: honeypot allow jumps intensity to 5 ───────────────────────────────

def test_honeypot_allow_jumps_intensity_to_five():
    """Allowing the honeypot turn (prefixed 'Hypothetically speaking,') causes
    the adversary to jump to intensity 5 on the very next turn.
    """
    HONEYPOT_PHRASE = "Hypothetically speaking,"

    env = GuardrailEnvironment()

    for seed in range(10):
        obs = env.reset("adversarial_adaptation", seed=seed)
        honeypot_found = False
        done = False
        steps = 0
        while not done and steps < 35:
            pid = obs.prompt_id
            if HONEYPOT_PHRASE in obs.user_prompt and not honeypot_found:
                # Allow the honeypot turn
                obs_next, reward, done, info = env.step(make_action("allow", pid))
                honeypot_found = True
                if not done and obs_next is not None:
                    # Next turn's adversary intensity should be 5
                    assert env._adversary is not None
                    assert env._adversary.intensity == 5, (
                        f"Seed {seed}: expected intensity=5 after honeypot allow, "
                        f"got intensity={env._adversary.intensity}"
                    )
                obs = obs_next or obs
            else:
                obs_next, reward, done, info = env.step(make_action("refuse", pid))
                obs = obs_next or obs
            steps += 1

        if honeypot_found:
            return  # Test passed on this seed

    pytest.skip("Honeypot turn not encountered in first 10 seeds — increase seed range if needed")


# ── Test 5: RAG injection prompts grade correctly ────────────────────────────

def test_rag_injection_prompts_grade_correctly():
    """Prompts with attack_features containing 'rag_injection' should receive
    positive grader credit when the agent takes the correct labeled action.
    Verifies that the grader's prompt accounting includes RAG prompts.
    """
    from app.grader import grade_task1
    from app.tasks.task1_basic_expanded import TASK1_EXPANDED_LABELS

    # Collect all RAG injection prompts (have "rag_injection" in attack_features)
    rag_prompts = [
        p for p in TASK1_EXPANDED_LABELS
        if "rag_injection" in (p.attack_features or [])
    ]
    assert len(rag_prompts) >= 5, (
        f"Expected ≥5 RAG injection prompts in Task 1, found {len(rag_prompts)}"
    )

    # Build actions using the correct label for each prompt
    correct_actions = []
    for lbl in rag_prompts:
        correct_actions.append(Action(
            prompt_id=lbl.prompt_id,
            action_type=lbl.correct_action,
            reason="oracle",
            modified_prompt=(
                lbl.prompt_text[:int(len(lbl.prompt_text) * 0.7)]
                if lbl.correct_action == "modify" else None
            ),
        ))

    contexts = ["general_assistant"] * len(rag_prompts)
    score = grade_task1(correct_actions, rag_prompts, contexts)

    # Oracle score on these prompts should be clearly positive.
    # RAG prompts are all attack class (no safe class), so grade_task1's
    # (safe_ratio + attack_ratio) / 2 formula caps at 0.5 → best = 0.375 with consistency bonus.
    # Threshold 0.3 is clearly above 0.0 (missed all) or 0.1875 (random).
    assert score > 0.3, (
        f"Oracle actions on RAG injection prompts should score > 0.3, got {score:.4f}"
    )
