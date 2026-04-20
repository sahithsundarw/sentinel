"""
demo_populate.py — Populate the Guardrail Arena dashboard with demo data.
Run after the server is up. Posts training curves for all 4 tasks in parallel
and completes a live episode to get a session ID for the dashboard.
"""
import requests
import sys
import time
import threading

BASE_URL = "http://localhost:7860"

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


def post_task_curve(task_id, scores):
    for i, score in enumerate(scores, 1):
        try:
            requests.post(f"{BASE_URL}/training_log", json={
                "agent_name": "guardrail_trl_agent",
                "task_id": task_id,
                "episode": i,
                "grader_score": round(score, 4),
                "cumulative_reward": round(score * 10, 3),
            }, timeout=5)
        except Exception:
            pass


def post_training_curves():
    print("\n[1/3] Posting training curves for all 4 tasks (parallel)...", flush=True)
    threads = []
    for task_id, scores in CURVES.items():
        t = threading.Thread(target=post_task_curve, args=(task_id, scores))
        t.start()
        threads.append((t, task_id, scores))

    for t, task_id, scores in threads:
        t.join()
        print(f"  {task_id}: done (final: {scores[-1]:.4f})")
    print("  All tasks posted.")


def run_demo_episode():
    print("\n[2/3] Running live episode...", flush=True)
    try:
        r = requests.post(f"{BASE_URL}/reset",
                          json={"task_id": "basic_threat_detection"}, timeout=10)
        data = r.json()
        session_id = data["session_id"]
        obs = data.get("observation", {})
    except Exception as e:
        print(f"  ERROR resetting: {e}")
        return None

    step = 0
    while True:
        try:
            risk = obs.get("user_risk_score", 0.5)
            action_type = "refuse" if risk > 0.55 else "allow"
            r = requests.post(f"{BASE_URL}/step", json={
                "session_id": session_id,
                "action": {"action_type": action_type, "reason": "demo"},
            }, timeout=10)
            result = r.json()
            step += 1
            if result.get("done"):
                score = result.get("info", {}).get("grader_score", "?")
                print(f"  Episode complete — {step} steps, score: {score}")
                break
            obs = result.get("observation", {})
            if step > 100:
                print("  Episode complete (100 steps)")
                break
        except Exception as e:
            print(f"  Step error: {e}")
            break

    return session_id


def main():
    print("=" * 55)
    print("  Guardrail Arena — Demo Setup")
    print("=" * 55)

    print("\nChecking server...", end=" ", flush=True)
    for _ in range(15):
        if check_server():
            print("OK")
            break
        print(".", end="", flush=True)
        time.sleep(1)
    else:
        print("\nERROR: Server not reachable at http://localhost:7860")
        sys.exit(1)

    post_training_curves()
    session_id = run_demo_episode()

    print("\n" + "=" * 55)
    print("  DEMO READY")
    print("=" * 55)
    print(f"\n  Dashboard : http://localhost:5173")
    if session_id:
        print(f"\n  Session ID (paste into dashboard):\n")
        print(f"      {session_id}")
    print(f"\n  Agent Name: guardrail_trl_agent")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
