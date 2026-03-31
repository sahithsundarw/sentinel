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
    # Default task is basic_threat_detection — prompts_remaining should be 75
    obs = r.json()
    assert obs["prompts_remaining"] == 75


def test_reset_empty_task_id_returns_422():
    r = client.post("/reset?task_id=")
    assert r.status_code == 422


def test_reset_with_seed_returns_observation():
    r = client.post("/reset?task_id=basic_threat_detection&seed=42")
    assert r.status_code == 200


def test_reset_same_seed_same_first_prompt():
    r1 = client.post("/reset?task_id=basic_threat_detection&seed=7")
    r2 = client.post("/reset?task_id=basic_threat_detection&seed=7")
    assert r1.json()["prompt_id"] == r2.json()["prompt_id"]


def test_reset_different_seeds_different_first_prompt():
    r1 = client.post("/reset?task_id=basic_threat_detection&seed=1")
    r2 = client.post("/reset?task_id=basic_threat_detection&seed=2")
    # Different seeds → different ordering; first prompts are very likely different
    # (can be the same by chance but extremely unlikely with 76 prompts)
    pid1, pid2 = r1.json()["prompt_id"], r2.json()["prompt_id"]
    assert pid1 != pid2, "Different seeds should produce different orderings"


# ── /step ─────────────────────────────────────────────────────────────────────

def _fresh_prompt_id(task_id="basic_threat_detection"):
    return client.post(f"/reset?task_id={task_id}").json()["prompt_id"]


def test_step_valid_allow():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "reward" in data
    assert "done" in data
    assert "info" in data
    assert "attack_features" in data["info"]


def test_step_valid_refuse():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "refuse", "reason": "test"})
    assert r.status_code == 200


def test_step_valid_modify_with_modified_prompt():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={
        "prompt_id": pid,
        "action_type": "modify",
        "reason": "test",
        "modified_prompt": "Safe version of the prompt"
    })
    assert r.status_code == 200


def test_step_valid_modify_null_modified_prompt():
    """modify with null modified_prompt is valid — scores as low-quality."""
    pid = _fresh_prompt_id()
    r = client.post("/step", json={
        "prompt_id": pid,
        "action_type": "modify",
        "reason": "test",
        "modified_prompt": None
    })
    assert r.status_code == 200


def test_step_invalid_action_type_returns_422():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "invalid", "reason": "test"})
    assert r.status_code == 422


def test_step_wrong_prompt_id_returns_422():
    _fresh_prompt_id()  # reset to get valid state
    r = client.post("/step", json={"prompt_id": "wrong-id", "action_type": "allow", "reason": "test"})
    assert r.status_code == 422


def test_step_before_reset_returns_400():
    """Step before any reset raises RuntimeError → 400."""
    from app.environment import GuardrailEnvironment
    from app.main import env
    # Force uninitialized state
    env._initialized = False
    r = client.post("/step", json={"prompt_id": "x", "action_type": "allow", "reason": "test"})
    assert r.status_code == 400
    # Restore state
    client.post("/reset?task_id=basic_threat_detection")


def test_step_empty_modified_prompt_is_treated_as_no_sanitization():
    """Empty string modified_prompt is accepted but reward.py treats it as no real sanitization."""
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "modify", "reason": "test", "modified_prompt": ""})
    assert r.status_code == 200
    # reward breakdown should reflect partial credit, not full sanitization
    breakdown = r.json()["reward"]["breakdown"]
    assert breakdown["outcome"] in ("partial_credit", "correct")  # safe prompt modify = partial_credit


def test_step_info_has_ground_truth_action():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert "ground_truth_action" in r.json()["info"]


def test_step_info_attack_features_is_list():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    assert isinstance(r.json()["info"]["attack_features"], list)


def test_step_reward_structure():
    pid = _fresh_prompt_id()
    r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    reward = r.json()["reward"]
    for field in ("score", "cumulative_score", "feedback", "breakdown"):
        assert field in reward, f"Missing reward field: {field}"


def test_step_done_true_on_last_step_task1():
    """Complete a full Task 1 episode and verify done=True on last step."""
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    pid = obs["prompt_id"]
    done = False
    steps = 0
    while not done:
        r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        steps += 1
        if not done:
            pid = data["observation"]["prompt_id"]
    assert steps == 76
    assert data["observation"] is None


def test_step_after_done_returns_400():
    """Calling step after episode ends must return 400, not crash."""
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    pid = obs["prompt_id"]
    done = False
    while not done:
        r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        if not done:
            pid = data["observation"]["prompt_id"]
    # Episode is now done — extra step should 400
    r = client.post("/step", json={"prompt_id": "any", "action_type": "allow", "reason": "test"})
    assert r.status_code == 400


# ── /grader ───────────────────────────────────────────────────────────────────

def test_grader_before_done_returns_400():
    client.post("/reset?task_id=basic_threat_detection")
    r = client.get("/grader")
    assert r.status_code == 400


def test_grader_mid_episode_returns_400():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    pid = obs["prompt_id"]
    client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
    r = client.get("/grader")
    assert r.status_code == 400


def test_grader_after_complete_episode_returns_score():
    obs = client.post("/reset?task_id=basic_threat_detection").json()
    pid = obs["prompt_id"]
    done = False
    while not done:
        r = client.post("/step", json={"prompt_id": pid, "action_type": "allow", "reason": "test"})
        data = r.json()
        done = data["done"]
        if not done:
            pid = data["observation"]["prompt_id"]
    r = client.get("/grader")
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


def test_demo_has_final_grader_score():
    r = client.get("/demo")
    data = r.json()
    assert "final_grader_score" in data
    assert isinstance(data["final_grader_score"], float)


# ── /state ────────────────────────────────────────────────────────────────────

def test_state_returns_valid_structure():
    client.post("/reset?task_id=basic_threat_detection")
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    for field in ("task_id", "current_prompt_index", "total_prompts", "cumulative_score", "done"):
        assert field in state


# ── /tasks ────────────────────────────────────────────────────────────────────

def test_tasks_returns_three_tasks():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tasks"]) == 3
    task_ids = [t["id"] for t in data["tasks"]]
    assert "basic_threat_detection" in task_ids
    assert "context_aware_policy" in task_ids
    assert "multiturn_adversarial" in task_ids


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
