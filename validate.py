"""
validate.py — OpenEnv submission validator for Sentinel

Runs a 3-step OpenEnv compliance check + full endpoint smoke test.

Usage:
    python validate.py https://varunventra-guardrail-arena.hf.space .
    python validate.py http://localhost:7860 .
    python validate.py --url https://varunventra-guardrail-arena.hf.space

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
    2 — httpx not installed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Run: pip install httpx")
    sys.exit(2)

# ─── helpers ────────────────────────────────────────────────────────────────

_passed: list[str] = []
_failed: list[str] = []


def ok(label: str) -> None:
    print(f"  [OK]   {label}")
    _passed.append(label)


def fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"\n         └─ {detail}"
    print(msg)
    _failed.append(label)


def _get(client: httpx.Client, base: str, path: str, **params: Any) -> dict:
    try:
        r = client.get(f"{base}{path}", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"__error__": str(exc)}


def _post(client: httpx.Client, base: str, path: str, body: Any = None, **params: Any) -> dict:
    try:
        r = client.post(f"{base}{path}", params=params, json=body, timeout=60.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"__error__": str(exc)}


def _ok(d: dict, *keys: str) -> bool:
    return bool(d) and "__error__" not in d and all(k in d for k in keys)


# ─── OpenEnv 3-step compliance ───────────────────────────────────────────────

def check_openenv_compliance(client: httpx.Client, base: str) -> bool:
    """Core OpenEnv loop: health → reset → step → grader."""
    print("\n[1/3] OpenEnv compliance (reset → step → grader loop)")

    health = _get(client, base, "/health")
    if _ok(health, "status"):
        ok("GET /health returns {status}")
    else:
        fail("GET /health", str(health))
        return False

    obs = _post(client, base, "/reset", task_id="basic_threat_detection")
    if _ok(obs, "session_id", "prompt_id", "user_prompt"):
        ok("POST /reset returns session_id + observation")
    else:
        fail("POST /reset", str(obs))
        return False

    sid = obs["session_id"]
    pid = obs["prompt_id"]

    action = {"prompt_id": pid, "action_type": "refuse", "reason": "validate"}
    step = _post(client, base, "/step", action, session_id=sid)
    if _ok(step, "done", "reward"):
        ok("POST /step returns done + reward")
    else:
        fail("POST /step", str(step))
        return False

    # Run to episode end (max 100 turns)
    for _ in range(100):
        if step.get("done"):
            break
        obs2 = step.get("observation", {})
        pid2 = obs2.get("prompt_id", "")
        step = _post(client, base, "/step", {"prompt_id": pid2, "action_type": "refuse", "reason": "v"}, session_id=sid)

    grade = _get(client, base, "/grader", session_id=sid)
    if _ok(grade, "score") and isinstance(grade.get("score"), (int, float)):
        ok(f"GET /grader returns numeric score ({grade['score']:.4f})")
    else:
        fail("GET /grader", str(grade))
        return False

    return True


# ─── full endpoint smoke test ────────────────────────────────────────────────

def check_all_endpoints(client: httpx.Client, base: str) -> None:
    """Smoke-test all 24 documented endpoints."""
    print("\n[2/3] Full endpoint coverage (24 endpoints)")

    # Static GET endpoints
    for path, keys in [
        ("/health",          ["status"]),
        ("/metadata",        ["name"]),
        ("/schema",          ["action_space"]),
        ("/tasks",           []),
        ("/baseline",        []),
        ("/leaderboard",     []),
        ("/curriculum",      []),
        ("/multi_agent_info",["agents"]),
    ]:
        d = _get(client, base, path)
        label = f"GET {path}"
        if "__error__" not in d and d is not None:
            ok(label)
        else:
            fail(label, d.get("__error__", "unexpected response"))

    # Training data
    for task_id in ["basic_threat_detection", "context_aware_policy"]:
        d = _get(client, base, "/training_data", task_id=task_id, format="sft")
        label = f"GET /training_data?task_id={task_id}"
        if "__error__" not in d:
            ok(label)
        else:
            fail(label, d.get("__error__", ""))

    # Start a fresh session for session-dependent endpoints
    obs = _post(client, base, "/reset", task_id="basic_threat_detection")
    sid = obs.get("session_id", "")
    pid = obs.get("prompt_id", "")

    if sid:
        ok("POST /reset (session setup for dependent tests)")
    else:
        fail("POST /reset (session setup)", "no session_id")
        return

    # Step
    action = {"prompt_id": pid, "action_type": "allow", "reason": "v"}
    step = _post(client, base, "/step", action, session_id=sid)
    if _ok(step, "done"):
        ok("POST /step")
    else:
        fail("POST /step", step.get("__error__", ""))

    # State
    state = _get(client, base, "/state", session_id=sid)
    if "__error__" not in state:
        ok("GET /state")
    else:
        fail("GET /state", state.get("__error__", ""))

    # Sessions list
    sessions = _get(client, base, "/sessions")
    if "__error__" not in sessions:
        ok("GET /sessions")
    else:
        fail("GET /sessions", sessions.get("__error__", ""))

    # Reward breakdown
    rb = _get(client, base, "/reward_breakdown", session_id=sid)
    if "__error__" not in rb:
        ok("GET /reward_breakdown")
    else:
        fail("GET /reward_breakdown", rb.get("__error__", ""))

    # Training log POST + GET
    log_entry = {"agent_name": "validate_bot", "task_id": "basic_threat_detection", "score": 0.5, "method": "test"}
    tl_post = _post(client, base, "/training_log", log_entry)
    if "__error__" not in tl_post:
        ok("POST /training_log")
    else:
        fail("POST /training_log", tl_post.get("__error__", ""))

    tl_get = _get(client, base, "/training_log")
    if "__error__" not in tl_get:
        ok("GET /training_log")
    else:
        fail("GET /training_log", tl_get.get("__error__", ""))

    # Batch rollout
    br = _post(client, base, "/batch_rollout", {"task_id": "basic_threat_detection", "n_episodes": 1, "policy": "all_refuse"})
    if "__error__" not in br:
        ok("POST /batch_rollout")
    else:
        fail("POST /batch_rollout", br.get("__error__", ""))

    # Agent comparison
    ac = _get(client, base, "/agent_comparison", agents="all_refuse,all_allow", task_id="basic_threat_detection")
    if "__error__" not in ac:
        ok("GET /agent_comparison")
    else:
        fail("GET /agent_comparison", ac.get("__error__", ""))

    # Task 4 session for adversary_state
    obs4 = _post(client, base, "/reset", task_id="adversarial_adaptation")
    sid4 = obs4.get("session_id", "")
    if sid4:
        step4 = _post(client, base, "/step",
                      {"prompt_id": obs4.get("prompt_id", ""), "action_type": "refuse", "reason": "v"},
                      session_id=sid4)
        adv = _get(client, base, "/adversary_state", session_id=sid4)
        if "__error__" not in adv:
            ok("GET /adversary_state (Task 4 session)")
        else:
            fail("GET /adversary_state", adv.get("__error__", ""))
    else:
        fail("POST /reset (task4 session)", "no session_id")

    # Delete session
    del_r = client.delete(f"{base}/sessions/{sid}", timeout=15.0)
    if del_r.status_code in (200, 204):
        ok(f"DELETE /sessions/{{id}}")
    else:
        fail(f"DELETE /sessions/{{id}}", f"HTTP {del_r.status_code}")


# ─── directory checks ────────────────────────────────────────────────────────

def check_directory(directory: str) -> None:
    """Check required files exist in the submission directory."""
    import os
    print(f"\n[3/3] Directory checks ({directory})")

    required = [
        "openenv.yaml",
        "requirements.txt",
        "app/main.py",
        "app/environment.py",
        "app/grader.py",
        "app/reward.py",
        "README.md",
    ]
    recommended = [
        "training_colab.ipynb",
        "train_trl.py",
        "inference.py",
        "results/local_training_results.json",
        "blog_post_hf.md",
    ]

    for f in required:
        path = os.path.join(directory, f)
        if os.path.exists(path):
            ok(f"Required: {f}")
        else:
            fail(f"Required: {f}", "file not found")

    for f in recommended:
        path = os.path.join(directory, f)
        if os.path.exists(path):
            ok(f"Recommended: {f}")
        else:
            print(f"  [WARN]  Recommended: {f} (not found)")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="OpenEnv submission validator for Sentinel")
    parser.add_argument("url", nargs="?", default="https://varunventra-guardrail-arena.hf.space",
                        help="Base URL of the running environment")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Path to the submission directory")
    parser.add_argument("--url", dest="url_flag", default=None,
                        help="Alternative way to specify URL (--url=...)")
    args = parser.parse_args()

    base_url = (args.url_flag or args.url).rstrip("/")
    directory = args.directory

    print(f"Sentinel OpenEnv Validator")
    print(f"  Target : {base_url}")
    print(f"  Dir    : {directory}")
    print("=" * 60)

    with httpx.Client() as client:
        compliance_ok = check_openenv_compliance(client, base_url)
        if compliance_ok:
            check_all_endpoints(client, base_url)
        else:
            print("\n  [ABORT] Skipping full endpoint checks — OpenEnv loop failed.")

    check_directory(directory)

    # Summary
    total = len(_passed) + len(_failed)
    print("\n" + "=" * 60)
    print(f"  Result: {len(_passed)}/{total} checks passed")
    if _failed:
        print(f"  Failed ({len(_failed)}):")
        for f in _failed:
            print(f"    - {f}")
        return 1

    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
