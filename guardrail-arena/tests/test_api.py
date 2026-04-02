"""
API layer tests for Guardrail Arena.
Uses FastAPI TestClient (no live server required).
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ── /reset ────────────────────────────────────────────────────────────────────

def test_reset_query_param_valid():
    r = client.post("/reset?task_id=basic_threat_detection")
    assert r.status_code == 200
    obs = r.json()
    assert "prompt_id" in obs
    assert "user_prompt" in obs
    assert obs["turn_number"] == 1
    assert obs["user_risk_score"] == 0.5


def test_reset_json_body_valid():
    r = client.post("/reset", json={"task_id": "context_aware_policy"})
    assert r.status_code == 200
    obs = r.json()
    assert "prompt_id" in obs


def test_reset_json_body_multiturn():
    r = client.post("/reset", json={"task_id": "multiturn_adversarial"})
    assert r.status_code == 200
    obs = r.json()
    assert "conversation_history" in obs


def test_reset_json_body_invalid_task_returns_422():
    r = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert r.status_code == 422
    assert "nonexistent_task" in r.json()["detail"]


def test_reset_query_param_invalid_task_returns_422():
    r = client.post("/reset?task_id=bad_task_id")
    assert r.status_code == 422


def test_reset_no_body_defaults_to_task1():
    r = client.post("/reset")
    assert r.status_code == 200
    # Default task is basic_threat_detection — prompts_remaining should be 157 (158 total, first served)
    obs = r.json()
    assert obs["prompts_remaining"] == 157


def test_reset_empty_task_id_returns_422():
    r = client.post("/reset?task_id=")
    assert r.status_code == 422


def test_reset_with_seed_returns_observation():
    r = client.post("/reset?task_id=basic_threat_detection&seed=42")
    assert r.status_code == 200


def test_reset_same_seed_same_canonical_prompt_different_session_ids():
    """Same seed → same canonical user_prompt, but different session prompt_ids
    (each session has its own salt to prevent cross-session lookup tables)."""
    r1 = client.post("/reset?task_id=basic_threat_detection&seed=7")
    r2 = client.post("/reset?task_id=basic_threat_detection&seed=7")
    assert r1.json()["user_prompt"] == r2.json()["user_prompt"], (
        "Same seed must serve the same canonical user_prompt"
    )
    assert r1.json()["prompt_id"] != r2.json()["prompt_id"], (
        "Different sessions must have different session prompt_ids even with the same seed"
    )


def test_reset_different_seeds_different_first_prompt():
    r1 = client.post("/reset?task_id=basic_threat_detection&seed=1")
    r2 = client.post("/reset?task_id=basic_threat_detection&seed=2")
    # Different seeds → different ordering; first prompts are very likely different
    # (can be the same by chance but extremely unlikely with 76 prompts)
    pid1, pid2 = r1.json()["prompt_id"], r2.json()["prompt_id"]
    assert pid1 != pid2, "Different seeds should produce different orderings"


# ── /step ─────────────────────────────────────────────────────────────────────

def _fresh_session(task_id="basic_threat_detection"):
    """Return (prompt_id, session_id) from a fresh /reset call."""
    obs = client.post(f"/reset?task_id={task_id}").json()
    return obs["prompt_id"], obs["session_id"]


def test_step_valid_allow():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "reward" in data
    assert "done" in data
    assert "info" in data
    assert "attack_features" in data["info"]


def test_step_valid_refuse():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "refuse", "reason": "test"})
    assert r.status_code == 200


def test_step_valid_modify_with_modified_prompt():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={
        "prompt_id": pid,
        "action_type": "modify",
        "reason": "test",
        "modified_prompt": "Safe version of the prompt"
    })
    assert r.status_code == 200


def test_step_valid_modify_null_modified_prompt():
    """modify with null modified_prompt is valid — scores as low-quality."""
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={
        "prompt_id": pid,
        "action_type": "modify",
        "reason": "test",
        "modified_prompt": None
    })
    assert r.status_code == 200


def test_step_invalid_action_type_returns_422():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "invalid", "reason": "test"})
    assert r.status_code == 422


def test_step_wrong_prompt_id_returns_422():
    _, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": "wrong-id", "action_type": "allow", "reason": "test"})
    assert r.status_code == 422


def test_step_before_reset_returns_400():
    """Step with no session_id returns 400 (session_id is required)."""
    r = client.post("/step", json={"prompt_id": "x", "action_type": "allow", "reason": "test"})
    assert r.status_code == 400


def test_step_empty_modified_prompt_is_treated_as_no_sanitization():
    """Empty string modified_prompt is accepted but reward.py treats it as no real sanitization."""
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "modify", "reason": "test", "modified_prompt": ""})
    assert r.status_code == 200
    # reward breakdown should reflect partial credit, not full sanitization
    breakdown = r.json()["reward"]["breakdown"]
    assert breakdown["outcome"] in ("partial_credit", "correct")  # safe prompt modify = partial_credit


def test_step_info_has_ground_truth_action():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert "ground_truth_action" in r.json()["info"]


def test_step_info_attack_features_is_list():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert isinstance(r.json()["info"]["attack_features"], list)


def test_step_reward_structure():
    pid, sid = _fresh_session()
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    reward = r.json()["reward"]
    for field in ("score", "cumulative_score", "feedback", "breakdown"):
        assert field in reward, f"Missing reward field: {field}"


def test_step_done_true_on_last_step_task1():
    """Complete a full Task 1 episode and verify done=True on last step."""
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    pid = obs["prompt_id"]
    done = False
    steps = 0
    while not done:
        r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        steps += 1
        if not done:
            pid = data["observation"]["prompt_id"]
    assert steps == 158
    assert data["observation"] is None


def test_step_after_done_returns_400():
    """Calling step after episode ends must return 400, not crash."""
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    pid = obs["prompt_id"]
    done = False
    while not done:
        r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        if not done:
            pid = data["observation"]["prompt_id"]
    # Episode is now done — extra step should 400
    r = client.post(f"/step?session_id={sid}", json={"prompt_id": "any", "action_type": "allow", "reason": "test"})
    assert r.status_code == 400


# ── /grader ───────────────────────────────────────────────────────────────────

def test_grader_before_done_returns_400():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    r = client.get(f"/grader?session_id={sid}")
    assert r.status_code == 400


def test_grader_mid_episode_returns_400():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    pid = obs["prompt_id"]
    client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    r = client.get(f"/grader?session_id={sid}")
    assert r.status_code == 400


def test_grader_after_complete_episode_returns_score():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    pid = obs["prompt_id"]
    done = False
    while not done:
        r = client.post(f"/step?session_id={sid}", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        if not done:
            pid = data["observation"]["prompt_id"]
    r = client.get(f"/grader?session_id={sid}")
    assert r.status_code == 200
    body = r.json()
    assert "score" in body
    assert 0.0 <= body["score"] <= 1.0
    assert body["task_id"] == "basic_threat_detection"


# ── /demo ─────────────────────────────────────────────────────────────────────

def test_demo_returns_200():
    r = client.get("/demo")
    assert r.status_code == 200


def test_demo_has_5_trajectory_steps():
    r = client.get("/demo")
    data = r.json()
    assert len(data["trajectory"]) == 5


def test_demo_has_partial_grader_score():
    r = client.get("/demo")
    data = r.json()
    assert "partial_grader_score" in data
    assert isinstance(data["partial_grader_score"], float)


# ── /state ────────────────────────────────────────────────────────────────────

def test_state_returns_valid_structure():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    sid = obs["session_id"]
    r = client.get(f"/state?session_id={sid}")
    assert r.status_code == 200
    state = r.json()
    for field in ("task_id", "current_prompt_index", "total_prompts", "cumulative_score", "done"):
        assert field in state


# ── /tasks ────────────────────────────────────────────────────────────────────

def test_tasks_returns_four_tasks():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tasks"]) == 4
    task_ids = [t["id"] for t in data["tasks"]]
    assert "basic_threat_detection" in task_ids
    assert "context_aware_policy" in task_ids
    assert "multiturn_adversarial" in task_ids
    assert "adversarial_adaptation" in task_ids


# ── /leaderboard ──────────────────────────────────────────────────────────────

def test_leaderboard_returns_200():
    r = client.get("/leaderboard")
    assert r.status_code == 200


def test_leaderboard_structure():
    r = client.get("/leaderboard")
    data = r.json()
    assert "leaderboard" in data
    for task_id in ("basic_threat_detection", "context_aware_policy", "multiturn_adversarial"):
        assert task_id in data["leaderboard"]


# ── /health survives all tests ─────────────────────────────────────────────────

def test_health_still_alive_after_all_tests():
    r = client.get("/health")
    assert r.status_code == 200


# ── Item 25: Leaderboard persistence ─────────────────────────────────────────

def test_leaderboard_agent_name_appears():
    """Passing agent_name to /grader should record the name on the leaderboard."""
    from app.main import _load_leaderboard, _save_leaderboard, _leaderboard
    # Run a full task1 episode
    r = client.post("/reset?task_id=basic_threat_detection")
    assert r.status_code == 200
    obs = r.json()
    sid = obs["session_id"]
    done = False
    while not done:
        r = client.post(f"/step?session_id={sid}", json={"prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        if not done:
            obs = data["observation"]
    r = client.get(f"/grader?agent_name=PersistenceTestAgent&session_id={sid}")
    assert r.status_code == 200
    r2 = client.get("/leaderboard")
    lb = r2.json()["leaderboard"]["basic_threat_detection"]
    agents = [e.get("agent", "") for e in lb]
    assert "PersistenceTestAgent" in agents


def test_leaderboard_persistence_save_load():
    """_save_leaderboard / _load_leaderboard round-trip."""
    import os
    from app.main import _save_leaderboard, _load_leaderboard, _LEADERBOARD_PATH
    test_entry = {"agent": "RoundTripAgent", "score": 0.999, "timestamp": "2025-01-01T00:00:00Z"}
    test_lb = {
        "basic_threat_detection": [test_entry],
        "context_aware_policy": [],
        "multiturn_adversarial": [],
    }
    _save_leaderboard(test_lb)
    loaded = _load_leaderboard()
    agents = [e.get("agent", "") for e in loaded["basic_threat_detection"]]
    assert "RoundTripAgent" in agents
    # Cleanup
    if os.path.exists(_LEADERBOARD_PATH):
        os.remove(_LEADERBOARD_PATH)


# ── Item 28: /rollout endpoint ───────────────────────────────────────────────

def test_rollout_returns_trajectory_and_score():
    """POST /rollout with task_id and actions should return a trajectory and grader_score."""
    actions = [{"action_type": "allow", "reason": "test"}] * 158
    r = client.post("/rollout", json={"task_id": "basic_threat_detection", "actions": actions})
    assert r.status_code == 200
    data = r.json()
    assert "grader_score" in data
    assert "trajectory" in data
    assert len(data["trajectory"]) == 158
    assert 0.0 <= data["grader_score"] <= 1.0


def test_rollout_invalid_task_returns_422():
    r = client.post("/rollout", json={"task_id": "fake_task", "actions": []})
    assert r.status_code == 422


# ── Item 29: Stress test (documents single-session limitation) ────────────────

def test_concurrent_resets_do_not_crash():
    """20 sequential /reset calls must all return 200 with unique session IDs."""
    for i in range(20):
        r = client.post("/reset?task_id=basic_threat_detection")
        assert r.status_code == 200, f"Reset {i} failed with {r.status_code}"

    # After rapid resets, a clean episode should still complete correctly
    r = client.post("/reset?task_id=basic_threat_detection")
    assert r.status_code == 200
    obs = r.json()
    sid = obs["session_id"]
    r2 = client.post(f"/step?session_id={sid}", json={"prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "test"})
    assert r2.status_code == 200


# ── Session isolation tests ───────────────────────────────────────────────────

def test_reset_returns_session_id():
    """Every /reset call must return a session_id."""
    r = client.post("/reset?task_id=basic_threat_detection")
    assert r.status_code == 200
    assert "session_id" in r.json()
    assert r.json()["session_id"]  # non-empty


def test_20_resets_return_unique_session_ids():
    """20 sequential resets must return 20 unique session IDs."""
    ids = set()
    for _ in range(20):
        r = client.post("/reset?task_id=basic_threat_detection")
        assert r.status_code == 200
        ids.add(r.json()["session_id"])
    assert len(ids) == 20, f"Expected 20 unique session IDs, got {len(ids)}"


def test_isolated_sessions_independent_state():
    """Two sessions running different tasks have completely independent state."""
    r1 = client.post("/reset?task_id=basic_threat_detection")
    r2 = client.post("/reset?task_id=context_aware_policy")
    sid1 = r1.json()["session_id"]
    sid2 = r2.json()["session_id"]
    pid1 = r1.json()["prompt_id"]
    pid2 = r2.json()["prompt_id"]
    assert sid1 != sid2

    # Step on session 1
    s1 = client.post(
        f"/step?session_id={sid1}",
        json={"prompt_id": pid1, "action_type": "allow", "reason": "test"},
    )
    assert s1.status_code == 200

    # Check state of each session independently
    state1 = client.get(f"/state?session_id={sid1}").json()
    state2 = client.get(f"/state?session_id={sid2}").json()
    assert state1["task_id"] == "basic_threat_detection"
    assert state2["task_id"] == "context_aware_policy"
    assert state1["current_prompt_index"] == 1  # one step taken
    assert state2["current_prompt_index"] == 0  # no steps yet


def test_step_invalid_session_id_returns_404():
    r = client.post(
        "/step?session_id=00000000-0000-0000-0000-000000000000",
        json={"prompt_id": "any", "action_type": "allow", "reason": "test"},
    )
    assert r.status_code == 404


def test_delete_session():
    r = client.post("/reset?task_id=basic_threat_detection")
    sid = r.json()["session_id"]
    del_r = client.delete(f"/sessions/{sid}")
    assert del_r.status_code == 200
    # Subsequent step should 404
    r2 = client.post(
        f"/step?session_id={sid}",
        json={"prompt_id": "any", "action_type": "allow", "reason": "test"},
    )
    assert r2.status_code == 404


def test_sessions_endpoint_lists_active():
    r = client.post("/reset?task_id=basic_threat_detection")
    sid = r.json()["session_id"]
    sessions_r = client.get("/sessions")
    assert sessions_r.status_code == 200
    data = sessions_r.json()
    session_ids = [s["session_id"] for s in data["sessions"]]
    assert sid in session_ids
