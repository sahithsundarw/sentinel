"""
Tests for new Round 2 endpoints.

Tests:
  - GET /multi_agent_info
  - POST /training_log + GET /training_log
  - GET /reward_breakdown (before and after episode)
  - GET /adversary_state (before and after Task 4 episode)
  - GET /training_data?format=sft
  - GET /training_data?format=chat
  - POST /batch_rollout
  - GET /agent_comparison

All existing tests in test_api.py / test_environment.py / test_graders.py / test_reward.py
MUST still pass — this file only adds new tests for new endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.environment import GuardrailEnvironment
from app.models import Action


@pytest.fixture
def client():
    return TestClient(app)


def _complete_task1_episode(client: TestClient) -> str:
    """Run a full basic_threat_detection episode (all-allow) and return session_id."""
    resp = client.post("/reset?task_id=basic_threat_detection")
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]

    obs = data
    while True:
        prompt_id = obs["prompt_id"]
        step_resp = client.post(
            f"/step?session_id={session_id}",
            json={"prompt_id": prompt_id, "action_type": "allow", "reason": "test"},
        )
        assert step_resp.status_code == 200
        result = step_resp.json()
        if result["done"]:
            break
        obs = result["observation"]

    return session_id


def _complete_task4_episode(client: TestClient) -> str:
    """Run a full adversarial_adaptation episode (all-refuse) and return session_id."""
    resp = client.post("/reset?task_id=adversarial_adaptation")
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]

    obs = data
    while True:
        prompt_id = obs["prompt_id"]
        step_resp = client.post(
            f"/step?session_id={session_id}",
            json={"prompt_id": prompt_id, "action_type": "refuse", "reason": "test"},
        )
        assert step_resp.status_code == 200
        result = step_resp.json()
        if result["done"]:
            break
        obs = result["observation"]

    return session_id


# ── /multi_agent_info ─────────────────────────────────────────────────────────

def test_multi_agent_info_returns_valid_json(client):
    resp = client.get("/multi_agent_info")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "adversary" in data["agents"]
    assert "defender" in data["agents"]
    assert data["theme"] == "multi_agent_interactions"


def test_multi_agent_info_adversary_has_state_space(client):
    data = client.get("/multi_agent_info").json()
    adv = data["agents"]["adversary"]
    ss = adv["state_space"]
    assert ss["topics"] == 10
    assert ss["intensity_levels"] == 6
    assert ss["total_states"] == 60
    assert ss["observable_states"] == 180


def test_multi_agent_info_has_evidence_of_rl(client):
    data = client.get("/multi_agent_info").json()
    ev = data["evidence_of_rl_requirement"]
    assert ev["zero_shot_235B_task4"] == 0.0
    assert ev["tabular_q_learner_task4"] == 0.954


def test_multi_agent_info_interaction_dynamics(client):
    data = client.get("/multi_agent_info").json()
    dyn = data["interaction_dynamics"]
    assert "theory_of_mind" in dyn
    assert "non_stationary_distribution" in dyn
    assert "branching_trajectories" in dyn


# ── /training_log ─────────────────────────────────────────────────────────────

def test_training_log_post_and_get(client):
    # POST a log entry
    resp = client.post("/training_log", json={
        "agent_name": "test_agent_xyz",
        "task_id": "basic_threat_detection",
        "episode": 1,
        "grader_score": 0.55,
        "cumulative_reward": 3.2,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent_name"] == "test_agent_xyz"

    # GET the log for that agent
    get_resp = client.get("/training_log?agent_name=test_agent_xyz")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["agent_name"] == "test_agent_xyz"
    assert len(data["entries"]) >= 1
    assert data["entries"][-1]["grader_score"] == 0.55


def test_training_log_get_all_agents(client):
    # Post entries for two distinct agents
    client.post("/training_log", json={
        "agent_name": "__agent_a__", "task_id": "basic_threat_detection",
        "episode": 1, "grader_score": 0.4,
    })
    client.post("/training_log", json={
        "agent_name": "__agent_b__", "task_id": "context_aware_policy",
        "episode": 1, "grader_score": 0.6,
    })
    resp = client.get("/training_log")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "__agent_a__" in data["agents"]
    assert "__agent_b__" in data["agents"]


def test_training_log_multiple_episodes(client):
    for ep in range(1, 4):
        client.post("/training_log", json={
            "agent_name": "__multi_ep_agent__", "task_id": "basic_threat_detection",
            "episode": ep, "grader_score": 0.3 + ep * 0.1,
        })
    resp = client.get("/training_log?agent_name=__multi_ep_agent__")
    data = resp.json()
    assert len(data["entries"]) >= 3


# ── /reward_breakdown ─────────────────────────────────────────────────────────

def test_reward_breakdown_requires_done_session(client):
    # Active (not yet done) session should return 400
    resp = client.post("/reset?task_id=basic_threat_detection")
    session_id = resp.json()["session_id"]
    bd_resp = client.get(f"/reward_breakdown?session_id={session_id}")
    assert bd_resp.status_code == 400


def test_reward_breakdown_requires_session_id(client):
    resp = client.get("/reward_breakdown")
    assert resp.status_code == 400


def test_reward_breakdown_after_episode(client):
    session_id = _complete_task1_episode(client)
    resp = client.get(f"/reward_breakdown?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "breakdown" in data
    bd = data["breakdown"]
    assert "correct_allows" in bd
    assert "missed_attacks" in bd
    assert "over_blocks" in bd
    assert "correct_blocks" in bd
    assert "failure_patterns" in data
    assert "risk_score_trajectory" in data
    assert "grader_score" in data
    assert 0.0 <= data["grader_score"] <= 1.0


def test_reward_breakdown_totals_add_up(client):
    session_id = _complete_task1_episode(client)
    data = client.get(f"/reward_breakdown?session_id={session_id}").json()
    bd = data["breakdown"]
    total = (
        bd["correct_allows"]["count"]
        + bd["correct_blocks"]["count"]
        + bd["missed_attacks"]["count"]
        + bd["over_blocks"]["count"]
    )
    # total steps should be >= total counted actions (unnecessary_escalations overlap)
    assert total <= data["total_steps"] + 5  # small tolerance for escalate-on-safe overlap


# ── /adversary_state ──────────────────────────────────────────────────────────

def test_adversary_state_only_for_task4(client):
    # Task 1 session should return 400
    resp = client.post("/reset?task_id=basic_threat_detection")
    sid = resp.json()["session_id"]
    # Complete the episode so it's done
    obs = resp.json()
    while True:
        pr = client.post(f"/step?session_id={sid}", json={
            "prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "test"
        }).json()
        if pr["done"]:
            break
        obs = pr["observation"]
    as_resp = client.get(f"/adversary_state?session_id={sid}")
    assert as_resp.status_code == 400  # wrong task


def test_adversary_state_blocked_during_active_episode(client):
    resp = client.post("/reset?task_id=adversarial_adaptation")
    sid = resp.json()["session_id"]
    # Don't complete — should get 403
    as_resp = client.get(f"/adversary_state?session_id={sid}")
    assert as_resp.status_code == 403


def test_adversary_state_after_task4_episode(client):
    session_id = _complete_task4_episode(client)
    resp = client.get(f"/adversary_state?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "adversary_trajectory" in data
    assert len(data["adversary_trajectory"]) > 0
    assert "topics_visited" in data
    assert "max_intensity_reached" in data
    # Each trajectory entry should have required fields
    first = data["adversary_trajectory"][0]
    assert "turn" in first
    assert "topic_idx" in first
    assert "intensity" in first
    assert "agent_action" in first


# ── /training_data?format=sft ─────────────────────────────────────────────────

def test_training_data_sft_format(client):
    resp = client.get("/training_data?task_id=basic_threat_detection&format=sft")
    assert resp.status_code == 200
    data = resp.json()
    assert "examples" in data
    assert data["format"] == "sft"
    assert len(data["examples"]) > 0
    ex = data["examples"][0]
    assert "prompt" in ex
    assert "completion" in ex


def test_training_data_chat_format(client):
    resp = client.get("/training_data?task_id=basic_threat_detection&format=chat")
    assert resp.status_code == 200
    data = resp.json()
    assert "examples" in data
    assert data["format"] == "chat"
    assert len(data["examples"]) > 0
    ex = data["examples"][0]
    assert "messages" in ex
    msgs = ex["messages"]
    assert any(m["role"] == "system" for m in msgs)
    assert any(m["role"] == "user" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)


def test_training_data_raw_format_unchanged(client):
    resp = client.get("/training_data?task_id=basic_threat_detection&format=raw")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompts" in data
    assert len(data["prompts"]) > 0


def test_training_data_invalid_format(client):
    resp = client.get("/training_data?task_id=basic_threat_detection&format=invalid")
    assert resp.status_code == 422


def test_training_data_sft_completion_is_valid_json(client):
    import json
    resp = client.get("/training_data?task_id=basic_threat_detection&format=sft")
    examples = resp.json()["examples"]
    valid_actions = {"allow", "refuse", "modify", "escalate"}
    for ex in examples[:10]:  # check first 10
        completion = ex["completion"]
        parsed = json.loads(completion)
        assert "action_type" in parsed
        assert parsed["action_type"] in valid_actions


# ── /batch_rollout ────────────────────────────────────────────────────────────

def test_batch_rollout_basic(client):
    resp = client.post("/batch_rollout", json={
        "task_id": "basic_threat_detection",
        "episodes": [{"seed": 42, "actions": []}],  # empty = all-allow
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert "grader_score" in result
    assert "steps" in result
    assert "trajectory" in result
    assert 0.0 <= result["grader_score"] <= 1.0


def test_batch_rollout_multiple_episodes(client):
    resp = client.post("/batch_rollout", json={
        "task_id": "basic_threat_detection",
        "episodes": [
            {"seed": 1, "actions": []},
            {"seed": 2, "actions": []},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_episodes"] == 2
    assert len(data["results"]) == 2


def test_batch_rollout_invalid_task(client):
    resp = client.post("/batch_rollout", json={
        "task_id": "invalid_task_xyz",
        "episodes": [{"actions": []}],
    })
    assert resp.status_code == 422


def test_batch_rollout_with_actions(client):
    # Provide some explicit actions
    resp = client.post("/batch_rollout", json={
        "task_id": "basic_threat_detection",
        "episodes": [{
            "seed": 99,
            "actions": [
                {"action_type": "allow", "reason": "safe"},
                {"action_type": "refuse", "reason": "attack"},
                {"action_type": "allow", "reason": "safe"},
            ],
        }],
    })
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    # First 3 steps should use provided actions, rest default to allow
    assert len(result["trajectory"]) > 0


# ── /agent_comparison ─────────────────────────────────────────────────────────

def test_agent_comparison_basic(client):
    # Seed some log data
    client.post("/training_log", json={
        "agent_name": "__cmp_agent_1__", "task_id": "basic_threat_detection",
        "episode": 1, "grader_score": 0.45,
    })
    client.post("/training_log", json={
        "agent_name": "__cmp_agent_2__", "task_id": "basic_threat_detection",
        "episode": 1, "grader_score": 0.72,
    })
    resp = client.get("/agent_comparison?agents=__cmp_agent_1__,__cmp_agent_2__&task_id=basic_threat_detection")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert len(data["agents"]) == 2


def test_agent_comparison_unknown_agents(client):
    resp = client.get("/agent_comparison?agents=nonexistent_agent_zzz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agents"][0]["score"] is None


def test_agent_comparison_requires_agents_param(client):
    resp = client.get("/agent_comparison")
    assert resp.status_code == 422
