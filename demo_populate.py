"""
demo_populate.py — Populate the Guardrail Arena dashboard with demo data.
Run after the server is up. Populates training curves for all 4 tasks
and completes a live HTTP episode so all 5 dashboard panels have data.
"""
import requests
import sys
import time

BASE_URL = "http://localhost:7860"

# Realistic training curves (SFT → PPO improvement) for each task
CURVES = {
    "basic_threat_detection": [
        0.3821, 0.4234, 0.4712, 0.5134, 0.5567,
        0.6023, 0.6445, 0.6891, 0.7234, 0.7567,
        0.7891, 0.8234, 0.8512, 0.8734, 0.8891,
        0.9012, 0.9134, 0.9212, 0.9267, 0.9312,
    ],
    "context_aware_policy": [
        0.4037, 0.4312, 0.4687, 0.5023, 0.5345,
        0.5712, 0.6123, 0.6512, 0.6891, 0.7234,
        0.7512, 0.7812, 0.8023, 0.8234, 0.8412,
        0.8567, 0.8689, 0.8756, 0.8812, 0.8867,
    ],
    "multiturn_adversarial": [
        0.1607, 0.1923, 0.2312, 0.2734, 0.3156,
        0.3589, 0.4023, 0.4512, 0.5034, 0.5467,
        0.5891, 0.6234, 0.6567, 0.6891, 0.7134,
        0.7345, 0.7512, 0.7634, 0.7723, 0.7812,
    ],
    "adversarial_adaptation": [
        0.0000, 0.0000, 0.0512, 0.1234, 0.2156,
        0.3234, 0.4512, 0.5634, 0.6512, 0.7234,
        0.7891, 0.8234, 0.8567, 0.8812, 0.9012,
        0.9134, 0.9267, 0.9356, 0.9423, 0.9512,
    ],
}


def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def post_training_curves():
    print("\n[1/3] Posting training curves for all 4 tasks...")
    for task_id, scores in CURVES.items():
        for i, score in enumerate(scores, 1):
            requests.post(f"{BASE_URL}/training_log", json={
                "agent_name": "guardrail_trl_agent",
                "task_id": task_id,
                "episode": i,
                "grader_score": round(score, 4),
                "cumulative_reward": round(score * 10, 3),
            }, timeout=5)
        print(f"  {task_id}: {len(scores)} episodes posted (final score: {scores[-1]:.4f})")
    print("  Done.")


def run_demo_episode():
    """Run a complete HTTP episode using risk_score heuristic. Returns session_id."""
    print("\n[2/3] Running live demo episode (basic_threat_detection)...")
    task_id = "basic_threat_detection"

    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
    data = r.json()
    session_id = data["session_id"]
    obs = data.get("observation", {})
    step = 0

    while True:
        risk = obs.get("user_risk_score", 0.5)
        action_type = "refuse" if risk > 0.55 else "allow"

        r = requests.post(f"{BASE_URL}/step", json={
            "session_id": session_id,
            "action": {
                "action_type": action_type,
                "reason": "Demo agent: risk-score heuristic",
            },
        }, timeout=10)
        result = r.json()
        step += 1

        if result.get("done"):
            info = result.get("info", {})
            score = info.get("grader_score", "?")
            print(f"  Episode complete — {step} steps, grader score: {score}")
            break

        obs = result.get("observation", {})
        if step > 100:
            print("  Episode timed out (100 steps)")
            break

    return session_id


def post_demo_leaderboard_entry(session_id):
    """Post a leaderboard entry so the leaderboard panel has data."""
    print("\n[3/3] Posting agent comparison data...")
    try:
        requests.post(f"{BASE_URL}/training_log", json={
            "agent_name": "guardrail_trl_agent",
            "task_id": "basic_threat_detection",
            "episode": 21,
            "grader_score": 0.9312,
            "cumulative_reward": 9.312,
        }, timeout=5)
        print("  Done.")
    except Exception:
        pass


def main():
    print("=" * 55)
    print("  Guardrail Arena — Demo Data Setup")
    print("=" * 55)

    print("\nChecking server...", end=" ", flush=True)
    for _ in range(10):
        if check_server():
            print("OK")
            break
        print(".", end="", flush=True)
        time.sleep(1)
    else:
        print("\nERROR: Server not reachable at http://localhost:7860")
        print("Make sure Terminal 1 (uvicorn) is running first.")
        sys.exit(1)

    post_training_curves()
    session_id = run_demo_episode()
    post_demo_leaderboard_entry(session_id)

    print("\n" + "=" * 55)
    print("  DEMO READY")
    print("=" * 55)
    print(f"\n  Dashboard:  http://localhost:5173")
    print(f"\n  Paste this Session ID into the dashboard UI:")
    print(f"\n      {session_id}\n")
    print("  (Agent Name field: guardrail_trl_agent)")
    print("=" * 55)


if __name__ == "__main__":
    main()
