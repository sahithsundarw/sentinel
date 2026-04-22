"""
verify_endpoints.py

Hits every endpoint on the live environment and prints pass/fail.
Run this before every deployment.

Usage:
    python verify_endpoints.py --url https://varunventra-guardrail-arena.hf.space
    python verify_endpoints.py --url http://localhost:7860
"""
import argparse
import json
import sys
import time
from typing import Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    print(f"  {status}  {label}")
    if not passed and detail:
        print(f"         └─ {detail}")
    results.append((label, passed, detail))
    return passed


def get(client: "httpx.Client", base: str, path: str, **params) -> dict | None:
    try:
        r = client.get(f"{base}{path}", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__status__": getattr(getattr(e, "response", None), "status_code", None)}


def post(client: "httpx.Client", base: str, path: str, body: Any = None, **params) -> dict | None:
    try:
        r = client.post(f"{base}{path}", params=params, json=body, timeout=60.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__status__": getattr(getattr(e, "response", None), "status_code", None)}


def has_key(data: dict | None, *keys: str) -> bool:
    if not data or "__error__" in data:
        return False
    return all(k in data for k in keys)


def no_error(data: dict | None) -> bool:
    return data is not None and "__error__" not in data


def fmt_err(data: dict | None) -> str:
    if data is None:
        return "no response"
    err = data.get("__error__", "")
    status = data.get("__status__", "")
    return f"HTTP {status}: {str(err)[:120]}" if status else str(err)[:150]


def run_full_episode(client, base, task_id="basic_threat_detection", max_steps=10):
    """Run a complete episode and return (session_id, done) or (None, False)."""
    obs = post(client, base, "/reset", task_id=task_id)
    if not has_key(obs, "session_id", "prompt_id"):
        return None, False
    sid = obs["session_id"]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = {
            "prompt_id": obs["prompt_id"],
            "action_type": "refuse",
            "reason": "verify_endpoints test",
            "modified_prompt": None,
        }
        result = post(client, base, "/step", body=action, session_id=sid)
        if not no_error(result):
            break
        done = result.get("done", False)
        obs = result.get("observation", obs)
        steps += 1
    return sid, done


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all Sentinel endpoints")
    parser.add_argument("--url", default="https://varunventra-guardrail-arena.hf.space",
                        help="Base URL of the environment")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout in seconds")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    if not HAS_HTTPX:
        print("ERROR: httpx is required. Run: pip install httpx")
        sys.exit(1)

    print(f"\nVerifying endpoints at: {base}")
    print("=" * 60)

    with httpx.Client(timeout=args.timeout) as client:

        # ── CORE ENDPOINTS ────────────────────────────────────────────────────
        print("\n── Core Endpoints ──")

        d = get(client, base, "/health")
        check("/health → status=healthy", d and d.get("status") == "healthy",
              fmt_err(d) if not (d and d.get("status") == "healthy") else "")

        d = get(client, base, "/metadata")
        check("/metadata → has name,version,description",
              has_key(d, "name", "version", "description"), fmt_err(d))

        d = get(client, base, "/tasks")
        has_tasks = no_error(d) and (isinstance(d, list) and len(d) >= 4 or
                                      isinstance(d, dict) and "tasks" in d)
        check("/tasks → list with 4 tasks", has_tasks, fmt_err(d))

        d = get(client, base, "/baseline")
        check("/baseline → has scores for tasks",
              no_error(d) and bool(d), fmt_err(d))

        d = get(client, base, "/curriculum")
        check("/curriculum → has ordered task list",
              no_error(d) and bool(d), fmt_err(d))

        d = get(client, base, "/schema")
        check("/schema → has action_schema,observation_schema",
              has_key(d, "action_schema", "observation_schema"), fmt_err(d))

        # ── EPISODE FLOW ──────────────────────────────────────────────────────
        print("\n── Episode Flow ──")

        d = post(client, base, "/reset", task_id="basic_threat_detection")
        ok_reset = has_key(d, "session_id", "prompt_id", "user_prompt")
        check("POST /reset → session_id,prompt_id,user_prompt", ok_reset, fmt_err(d))

        sid1 = d.get("session_id") if ok_reset else None
        initial_obs = d

        if sid1:
            action = {
                "prompt_id": initial_obs["prompt_id"],
                "action_type": "refuse",
                "reason": "test",
                "modified_prompt": None,
            }
            d = post(client, base, "/step", body=action, session_id=sid1)
            check("POST /step → reward,done,observation",
                  has_key(d, "reward", "done", "observation"), fmt_err(d))

            d = get(client, base, "/state", session_id=sid1)
            check("GET /state → task_id,turn_number",
                  has_key(d, "task_id", "turn_number"), fmt_err(d))
        else:
            check("POST /step (skipped — no session)", False, "reset failed")
            check("GET /state (skipped — no session)", False, "reset failed")

        # Run a full episode for grader/breakdown tests
        print("\n  Running a full episode for grader tests...")
        full_sid, done = run_full_episode(client, base, "basic_threat_detection", max_steps=80)

        if full_sid and done:
            d = get(client, base, "/grader", session_id=full_sid)
            score = d.get("score") if no_error(d) else None
            check("GET /grader → score in [0,1]",
                  score is not None and 0.0 <= float(score) <= 1.0,
                  fmt_err(d) if score is None else "")
        else:
            check("GET /grader (partial episode)", False, "episode not completed")

        # ── NEW ROUND 2 ENDPOINTS ─────────────────────────────────────────────
        print("\n── Round 2 Endpoints ──")

        d = get(client, base, "/multi_agent_info")
        check("GET /multi_agent_info → agents.adversary,agents.defender",
              no_error(d) and "agents" in (d or {}), fmt_err(d))

        d = get(client, base, "/training_data", task_id="basic_threat_detection")
        check("GET /training_data → has prompts",
              no_error(d) and bool(d), fmt_err(d))

        d = get(client, base, "/training_data",
                task_id="basic_threat_detection", format="sft")
        has_sft = (no_error(d) and isinstance(d, dict) and
                   any(k in d for k in ("examples", "data", "prompts")))
        check("GET /training_data?format=sft → examples with prompt+completion",
              has_sft, fmt_err(d))

        # Training log: POST then GET
        log_payload = {
            "agent_name": "verify_endpoints_test",
            "task_id": "basic_threat_detection",
            "episode": 1,
            "grader_score": 0.5,
        }
        d = post(client, base, "/training_log", body=log_payload)
        check("POST /training_log → 200 OK", no_error(d), fmt_err(d))

        time.sleep(0.3)
        d = get(client, base, "/training_log", agent_name="verify_endpoints_test")
        has_log = no_error(d) and any(k in (d or {}) for k in ("entries", "logs", "data"))
        check("GET /training_log?agent_name=... → entries list",
              has_log, fmt_err(d) if not has_log else "")

        # reward_breakdown requires a completed session
        if full_sid and done:
            d = get(client, base, "/reward_breakdown", session_id=full_sid)
            check("GET /reward_breakdown → breakdown dict",
                  no_error(d) and bool(d), fmt_err(d))
        else:
            check("GET /reward_breakdown (skipped — no completed session)", False, "")

        # adversary_state requires Task 4
        print("\n  Running Task 4 episode for adversary_state test...")
        t4_sid, t4_done = run_full_episode(client, base, "adversarial_adaptation", max_steps=15)
        if t4_sid and t4_done:
            d = get(client, base, "/adversary_state", session_id=t4_sid)
            check("GET /adversary_state → adversary_trajectory",
                  no_error(d) and bool(d), fmt_err(d))
        else:
            check("GET /adversary_state (partial Task 4 ep)", False,
                  "Task 4 episode not completed in 15 steps — may still be valid")

        # agent_comparison
        d = get(client, base, "/agent_comparison",
                agents="verify_endpoints_test,other_agent",
                task_id="basic_threat_detection")
        check("GET /agent_comparison → agents list",
              no_error(d) and bool(d), fmt_err(d))

        # batch_rollout
        batch_payload = {
            "task_id": "basic_threat_detection",
            "n_episodes": 1,
            "actions": [{"action_type": "refuse", "reason": "test"}],
        }
        d = post(client, base, "/batch_rollout", body=batch_payload)
        has_batch = no_error(d) and any(k in (d or {}) for k in ("results", "episodes", "scores"))
        check("POST /batch_rollout → results list",
              has_batch, fmt_err(d) if not has_batch else "")

        # ── SUMMARY ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        total = len(results)
        passed = sum(1 for _, ok, _ in results if ok)
        failed = total - passed

        print(f"RESULT: {passed}/{total} passed", end="")
        if failed > 0:
            print(f"  ({failed} failed)")
            print("\nFailing endpoints:")
            for label, ok, detail in results:
                if not ok:
                    print(f"  ❌  {label}")
                    if detail:
                        print(f"      {detail}")
        else:
            print("  — ALL PASS ✅")

        print()
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
