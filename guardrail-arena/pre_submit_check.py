"""
Guardrail Arena — Pre-Submission Validator
==========================================
Simulates all 3 phases of hackathon judging against either:
  - Your local Docker container (http://localhost:7860)
  - Your live HF Space (https://yourname-guardrail-arena.hf.space)

Usage:
    python pre_submit_check.py
    python pre_submit_check.py --env-url http://localhost:7860
    python pre_submit_check.py --env-url https://yourname-guardrail-arena.hf.space

For Phase 3 agentic evaluation (requires model creds):
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python pre_submit_check.py
"""
import os
import re
import sys
import json
import time
import argparse
import subprocess

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    raise SystemExit(1)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(colour, text):
    """Wrap text in colour codes (disabled on Windows if not supported)."""
    try:
        return f"{colour}{text}{RESET}"
    except Exception:
        return text

PASS_MARK  = _c(GREEN,  "PASS")
FAIL_MARK  = _c(RED,    "FAIL")
SKIP_MARK  = _c(YELLOW, "SKIP")
WARN_MARK  = _c(YELLOW, "WARN")

# ── Result tracker ────────────────────────────────────────────────────────────

results: list[dict] = []

def record(phase: str, name: str, passed: bool, detail: str = "", warn: bool = False):
    status = "WARN" if warn else ("PASS" if passed else "FAIL")
    results.append({"phase": phase, "name": name, "status": status, "detail": detail})
    mark = WARN_MARK if warn else (PASS_MARK if passed else FAIL_MARK)
    suffix = f"  {_c(YELLOW, detail)}" if detail else ""
    print(f"  {mark}  {name}{suffix}")
    return passed


def section(title: str):
    print(f"\n{_c(BOLD, title)}")
    print("  " + "-" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_ACTIONS  = {"allow", "refuse", "modify", "escalate"}
_VALID_CONTEXTS = {"children_education", "creative_writing", "customer_support", "general_assistant"}
_OBS_FIELDS     = {"prompt_id", "user_prompt", "conversation_history",
                   "application_context", "user_risk_score", "turn_number",
                   "prompts_remaining", "episode_score_so_far"}

def is_uuid(s: str) -> bool:
    return bool(re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        str(s).lower()
    ))


def hf_space_to_api_url(url: str) -> str:
    """
    Convert HF Space page URL to API URL.
    https://huggingface.co/spaces/user/space  ->  https://user-space.hf.space
    Already-correct URLs pass through unchanged.
    """
    url = url.rstrip("/")
    m = re.match(r"https?://huggingface\.co/spaces/([^/]+)/([^/]+)", url)
    if m:
        user, space = m.group(1), m.group(2)
        return f"https://{user}-{space}.hf.space"
    return url


def step_all(client: httpx.Client, env_url: str, task_id: str, action_override: str = "allow"):
    """
    Reset a task and step through every prompt using action_override.
    Returns (list_of_step_data, grader_score).
    """
    r = client.post(f"{env_url}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()

    steps = []
    done  = False
    while not done:
        action = {
            "prompt_id":      obs["prompt_id"],
            "action_type":    action_override,
            "reason":         "validator",
            "modified_prompt": None,
        }
        r = client.post(f"{env_url}/step", json=action, timeout=30)
        r.raise_for_status()
        data = r.json()
        steps.append(data)
        done = data["done"]
        if not done:
            obs = data["observation"]

    r = client.get(f"{env_url}/grader", timeout=30)
    r.raise_for_status()
    score = r.json()["score"]
    return steps, score


def collect_ground_truth(client: httpx.Client, env_url: str, task_id: str):
    """
    Step through a task with allow-everything, collecting ground_truth_action per step.
    Returns list of (prompt_id, ground_truth_action) in order.
    """
    r = client.post(f"{env_url}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()
    ground_truths = []
    done = False
    while not done:
        action = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                  "reason": "collect", "modified_prompt": None}
        r = client.post(f"{env_url}/step", json=action, timeout=30)
        r.raise_for_status()
        data = r.json()
        gt = data.get("info", {}).get("ground_truth_action", "allow")
        ground_truths.append((obs["prompt_id"], gt))
        done = data["done"]
        if not done:
            obs = data["observation"]
    return ground_truths


def replay_ground_truth(client: httpx.Client, env_url: str, task_id: str,
                        ground_truths: list) -> float:
    """
    Reset task and replay the collected ground truth actions. Returns grader score.
    """
    r = client.post(f"{env_url}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()
    done = False
    idx  = 0
    while not done:
        _, gt_action = ground_truths[idx]
        action = {"prompt_id": obs["prompt_id"], "action_type": gt_action,
                  "reason": "replay", "modified_prompt": None}
        r = client.post(f"{env_url}/step", json=action, timeout=30)
        r.raise_for_status()
        data = r.json()
        done = data["done"]
        idx += 1
        if not done:
            obs = data["observation"]
    r = client.get(f"{env_url}/grader", timeout=30)
    r.raise_for_status()
    return r.json()["score"]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — API Compliance & Spec Checks
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_api_compliance(env_url: str):
    section("PHASE 1 — API Compliance & OpenEnv Spec")
    passed_all = True
    client = httpx.Client(timeout=30)

    # --- 1.1 Health endpoints ---
    try:
        r = client.get(f"{env_url}/")
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        record("P1", "GET /  returns 200 + {status: ok}", ok)
        passed_all &= ok
    except Exception as e:
        record("P1", "GET /  returns 200 + {status: ok}", False, str(e))
        passed_all = False

    try:
        r = client.get(f"{env_url}/health")
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        record("P1", "GET /health returns 200 + {status: ok}", ok)
        passed_all &= ok
    except Exception as e:
        record("P1", "GET /health returns 200 + {status: ok}", False, str(e))
        passed_all = False

    # --- 1.2 Tasks endpoint ---
    try:
        r = client.get(f"{env_url}/tasks")
        data = r.json()
        tasks  = data.get("tasks", [])
        schema = data.get("action_schema", {})

        ok1 = r.status_code == 200
        ok2 = len(tasks) == 3
        ok3 = all(t.get("id") and t.get("difficulty") for t in tasks)
        schema_str = json.dumps(schema)
        ok4 = all(a in schema_str for a in ("allow", "refuse", "modify", "escalate"))
        ids = [t["id"] for t in tasks]
        ok5 = set(ids) == {"basic_threat_detection", "context_aware_policy", "multiturn_adversarial"}
        difficulties = {t["id"]: t["difficulty"] for t in tasks}
        ok6 = (difficulties.get("basic_threat_detection") == "easy" and
               difficulties.get("context_aware_policy")    == "medium" and
               difficulties.get("multiturn_adversarial")   == "hard")

        record("P1", "GET /tasks — status 200",               ok1)
        record("P1", "GET /tasks — exactly 3 tasks",          ok2, f"got {len(tasks)}")
        record("P1", "GET /tasks — all tasks have id+difficulty", ok3)
        record("P1", "GET /tasks — action schema has all 4 enums", ok4)
        record("P1", "GET /tasks — correct task IDs",          ok5)
        record("P1", "GET /tasks — easy/medium/hard difficulty labels", ok6)
        passed_all &= all([ok1, ok2, ok3, ok4, ok5, ok6])
    except Exception as e:
        record("P1", "GET /tasks", False, str(e))
        passed_all = False

    # --- 1.3 Reset each task — validate Observation shape ---
    for task_id in ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]:
        try:
            r = client.post(f"{env_url}/reset", params={"task_id": task_id})
            obs = r.json()
            ok_status = r.status_code == 200
            ok_fields  = _OBS_FIELDS.issubset(obs.keys())
            ok_uuid    = is_uuid(obs.get("prompt_id", ""))
            ok_context = obs.get("application_context") in _VALID_CONTEXTS
            ok_risk    = isinstance(obs.get("user_risk_score"), (int, float))
            ok_turn    = obs.get("turn_number") == 1
            ok_score0  = obs.get("episode_score_so_far") == 0.0

            record("P1", f"POST /reset {task_id} — 200",            ok_status)
            record("P1", f"POST /reset {task_id} — all 8 obs fields", ok_fields,
                   f"missing: {_OBS_FIELDS - obs.keys()}" if not ok_fields else "")
            record("P1", f"POST /reset {task_id} — prompt_id is UUID", ok_uuid,
                   obs.get("prompt_id", ""))
            record("P1", f"POST /reset {task_id} — valid application_context", ok_context,
                   obs.get("application_context", ""))
            record("P1", f"POST /reset {task_id} — turn_number=1 on first obs", ok_turn)
            record("P1", f"POST /reset {task_id} — episode_score_so_far=0.0", ok_score0)
            passed_all &= all([ok_status, ok_fields, ok_uuid, ok_context, ok_turn, ok_score0])
        except Exception as e:
            record("P1", f"POST /reset {task_id}", False, str(e))
            passed_all = False

    # --- 1.4 Step endpoint — single step validation ---
    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        obs = r.json()
        action = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                  "reason": "test", "modified_prompt": None}
        r = client.post(f"{env_url}/step", json=action)
        data = r.json()

        ok_status  = r.status_code == 200
        ok_keys    = {"observation", "reward", "done", "info"}.issubset(data.keys()) or data.get("done") == True
        ok_reward  = "score" in data.get("reward", {}) and "feedback" in data.get("reward", {})
        ok_gt      = "ground_truth_action" in data.get("info", {})
        ok_gt_val  = data.get("info", {}).get("ground_truth_action") in _VALID_ACTIONS

        record("P1", "POST /step — status 200",                        ok_status)
        record("P1", "POST /step — returns observation+reward+done+info", ok_keys)
        record("P1", "POST /step — reward has score+feedback",         ok_reward)
        record("P1", "POST /step — info has ground_truth_action",      ok_gt)
        record("P1", "POST /step — ground_truth_action is valid",      ok_gt_val,
               data.get("info", {}).get("ground_truth_action", "missing"))
        passed_all &= all([ok_status, ok_keys, ok_reward, ok_gt, ok_gt_val])
    except Exception as e:
        record("P1", "POST /step basic validation", False, str(e))
        passed_all = False

    # --- 1.5 State endpoint ---
    try:
        r = client.get(f"{env_url}/state")
        data = r.json()
        ok = r.status_code == 200 and "task_id" in data and "done" in data
        record("P1", "GET /state — 200 + task_id + done fields", ok)
        passed_all &= ok
    except Exception as e:
        record("P1", "GET /state", False, str(e))
        passed_all = False

    # --- 1.6 /baseline — must return 200 WITHOUT needing API key ---
    try:
        r = client.get(f"{env_url}/baseline")
        data = r.json()
        ok_200    = r.status_code == 200
        scores    = data.get("scores", {})
        ok_scores = len(scores) == 3
        ok_range  = all(0.0 <= v <= 1.0 for v in scores.values())
        record("P1", "GET /baseline — always returns 200 (no API key needed)", ok_200)
        record("P1", "GET /baseline — has 3 task scores",                      ok_scores)
        record("P1", "GET /baseline — all scores in [0.0, 1.0]",              ok_range,
               str(scores))
        passed_all &= all([ok_200, ok_scores, ok_range])
    except Exception as e:
        record("P1", "GET /baseline", False, str(e))
        passed_all = False

    # --- 1.7 Error handling ---
    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "nonexistent_task_xyz"})
        ok = r.status_code in (400, 422)
        record("P1", "Invalid task_id → 400/422 (not 500)", ok, f"got {r.status_code}")
        passed_all &= ok
    except Exception as e:
        record("P1", "Invalid task_id error handling", False, str(e))
        passed_all = False

    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        obs = r.json()
        bad_action = {"prompt_id": obs["prompt_id"], "action_type": "DETONATE",
                      "reason": "test", "modified_prompt": None}
        r = client.post(f"{env_url}/step", json=bad_action)
        ok = r.status_code in (400, 422)
        record("P1", "Invalid action_type → 400/422 (not 500)", ok, f"got {r.status_code}")
        passed_all &= ok
    except Exception as e:
        record("P1", "Invalid action_type error handling", False, str(e))
        passed_all = False

    try:
        client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        r = client.get(f"{env_url}/grader")
        ok = r.status_code in (400, 422)
        record("P1", "GET /grader before done → 400 (not 500)", ok, f"got {r.status_code}")
        passed_all &= ok
    except Exception as e:
        record("P1", "GET /grader before done", False, str(e))
        passed_all = False

    client.close()
    return passed_all


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Episode Integrity & Grader Checks
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_episode_integrity(env_url: str):
    section("PHASE 2 — Episode Integrity & Grader Quality")
    passed_all = True
    client = httpx.Client(timeout=60)

    task_ids = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]
    all_scores = {}

    # --- 2.1 Full episodes complete correctly ---
    for task_id in task_ids:
        try:
            steps, score = step_all(client, env_url, task_id, "allow")
            last_step = steps[-1]

            ok_done  = last_step["done"] == True
            ok_score = 0.0 <= score <= 1.0
            reward_scores = [s["reward"]["score"] for s in steps]
            ok_vary  = len(set(reward_scores)) > 1  # reward signal varies (not all same)

            record("P2", f"{task_id} — episode completes with done=True", ok_done)
            record("P2", f"{task_id} — grader score in [0.0, 1.0]",       ok_score,
                   f"score={score:.4f}")
            record("P2", f"{task_id} — reward signal varies across steps", ok_vary,
                   f"unique values: {len(set(round(s,3) for s in reward_scores))}")
            passed_all &= all([ok_done, ok_score, ok_vary])
            all_scores[task_id] = score
        except Exception as e:
            record("P2", f"{task_id} full episode", False, str(e))
            passed_all = False
            all_scores[task_id] = None

    # --- 2.2 Scores are NOT all identical (instant disqualification check) ---
    valid_scores = [v for v in all_scores.values() if v is not None]
    ok_not_identical = len(set(valid_scores)) > 1
    record("P2", "Grader scores are NOT all identical (disqualifying if same)",
           ok_not_identical, str({k: f"{v:.4f}" for k, v in all_scores.items() if v}))
    passed_all &= ok_not_identical

    # --- 2.3 Difficulty ordering: easy >= medium >= hard (dumb agent) ---
    t1 = all_scores.get("basic_threat_detection")
    t2 = all_scores.get("context_aware_policy")
    t3 = all_scores.get("multiturn_adversarial")
    if t1 is not None and t2 is not None and t3 is not None:
        ok_order = t1 >= t2 >= t3
        record("P2", f"Difficulty ordering: easy({t1:.2f}) >= medium({t2:.2f}) >= hard({t3:.2f})",
               ok_order, "WARN: wrong ordering signals grader issue" if not ok_order else "")
        passed_all &= ok_order

    # --- 2.4 Determinism: same task reset twice → same first prompt ---
    try:
        r1 = client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        id1 = r1.json()["prompt_id"]
        r2 = client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        id2 = r2.json()["prompt_id"]
        ok = id1 == id2
        record("P2", "Reset twice → same first prompt_id (deterministic)", ok,
               f"{id1} vs {id2}")
        passed_all &= ok
    except Exception as e:
        record("P2", "Determinism check", False, str(e))
        passed_all = False

    # --- 2.5 Perfect score test: replay ground truth → score near 1.0 ---
    for task_id in task_ids:
        try:
            gt = collect_ground_truth(client, env_url, task_id)
            perfect_score = replay_ground_truth(client, env_url, task_id, gt)
            ok = perfect_score >= 0.95
            record("P2", f"{task_id} — perfect actions → score >= 0.95",
                   ok, f"got {perfect_score:.4f}")
            passed_all &= ok
        except Exception as e:
            record("P2", f"{task_id} perfect score test", False, str(e))
            passed_all = False

    # --- 2.6 Worst score: refuse everything → score < 0.5 ---
    try:
        _, worst_score = step_all(client, env_url, "basic_threat_detection", "refuse")
        ok = worst_score < 0.5
        record("P2", f"basic_threat_detection — refuse everything → score < 0.5",
               ok, f"got {worst_score:.4f}")
        passed_all &= ok
    except Exception as e:
        record("P2", "Worst score test", False, str(e))
        passed_all = False

    # --- 2.7 Grader determinism: same perfect actions twice → same score ---
    try:
        gt = collect_ground_truth(client, env_url, "context_aware_policy")
        s1 = replay_ground_truth(client, env_url, "context_aware_policy", gt)
        s2 = replay_ground_truth(client, env_url, "context_aware_policy", gt)
        ok = s1 == s2
        record("P2", "Grader deterministic: identical runs → identical scores",
               ok, f"{s1:.4f} vs {s2:.4f}")
        passed_all &= ok
    except Exception as e:
        record("P2", "Grader determinism", False, str(e))
        passed_all = False

    # --- 2.8 Task 3 specific: conversation_done signals ---
    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "multiturn_adversarial"})
        obs = r.json()
        done = False
        conv_done_count = 0
        history_grows = True
        prev_history_len = -1
        while not done:
            action = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                      "reason": "test", "modified_prompt": None}
            r = client.post(f"{env_url}/step", json=action)
            data = r.json()
            if data.get("info", {}).get("conversation_done"):
                conv_done_count += 1
            done = data["done"]
            if not done:
                obs = data["observation"]
                cur_len = len(obs.get("conversation_history", []))
                # History should grow within a conversation, then reset to 0 at new conv start
                # Just check it's present and is a list
                if not isinstance(obs.get("conversation_history"), list):
                    history_grows = False

        ok_signals = conv_done_count >= 5  # at least 5 original conversations
        ok_history = history_grows
        record("P2", f"Task 3 — conversation_done signals fired ({conv_done_count} total)",
               ok_signals, f"expected >= 5")
        record("P2", "Task 3 — conversation_history is always a list",
               ok_history)
        passed_all &= all([ok_signals, ok_history])
    except Exception as e:
        record("P2", "Task 3 conversation signals", False, str(e))
        passed_all = False

    # --- 2.9 Observation fields: user_risk_score varies (it's a real signal) ---
    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "context_aware_policy"})
        obs = r.json()
        risk_scores = [obs.get("user_risk_score", 0.5)]
        done = False
        step_count = 0
        while not done and step_count < 20:
            action = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                      "reason": "test", "modified_prompt": None}
            r = client.post(f"{env_url}/step", json=action)
            data = r.json()
            done = data["done"]
            if not done:
                obs = data["observation"]
                risk_scores.append(obs.get("user_risk_score", 0.5))
            step_count += 1
        ok = len(set(round(s, 2) for s in risk_scores)) > 1
        record("P2", "user_risk_score varies across prompts (live signal not constant)",
               ok, f"unique values: {sorted(set(round(s,2) for s in risk_scores))}")
        # Warning only — not a hard fail
        if not ok:
            results[-1]["status"] = "WARN"
    except Exception as e:
        record("P2", "user_risk_score variation check", False, str(e))

    # --- 2.10 Prompt IDs are UUIDs ---
    try:
        r = client.post(f"{env_url}/reset", params={"task_id": "basic_threat_detection"})
        obs = r.json()
        pid = obs.get("prompt_id", "")
        ok = is_uuid(pid)
        record("P2", f"prompt_id is a valid UUID (anti-cheat requirement)",
               ok, pid)
        passed_all &= ok
    except Exception as e:
        record("P2", "UUID prompt_id check", False, str(e))
        passed_all = False

    client.close()
    return passed_all


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Agentic Evaluation (runs inference.py with real model)
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_agentic_evaluation(env_url: str):
    section("PHASE 3 — Agentic Evaluation (simulates judge model run)")

    api_base = os.getenv("API_BASE_URL")
    model    = os.getenv("MODEL_NAME")
    token    = os.getenv("HF_TOKEN")

    if not all([api_base, model, token]):
        print(f"  {SKIP_MARK}  API_BASE_URL / MODEL_NAME / HF_TOKEN not set — skipping agentic phase")
        print(f"         Set these env vars to simulate the judge's model run:")
        print(f"         API_BASE_URL=https://router.huggingface.co/v1")
        print(f"         MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct")
        print(f"         HF_TOKEN=hf_yourtoken")
        return None

    inference_path = os.path.join(os.path.dirname(__file__), "inference.py")
    if not os.path.exists(inference_path):
        record("P3", "inference.py exists in guardrail-arena/", False,
               f"not found at {inference_path}")
        return False

    record("P3", "inference.py exists in guardrail-arena/", True)

    # --- 3.1 Graceful failure without env vars ---
    env_no_creds = {k: v for k, v in os.environ.items()
                    if k not in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")}
    r = subprocess.run([sys.executable, inference_path], env=env_no_creds,
                       capture_output=True, text=True, timeout=15)
    ok = r.returncode == 1 and "Error:" in r.stdout and not r.stderr.strip()
    record("P3", "inference.py exits cleanly (code 1, no traceback) when creds missing", ok,
           r.stdout.strip()[:80])

    # --- 3.2 Run full inference.py against live env ---
    print(f"\n  Running inference.py with {model} against {env_url}")
    print(f"  This may take several minutes...")
    env_with_creds = {**os.environ,
                      "API_BASE_URL": api_base,
                      "MODEL_NAME":   model,
                      "HF_TOKEN":     token,
                      "ENV_URL":      env_url}

    t_start = time.time()
    proc = subprocess.run(
        [sys.executable, inference_path],
        env=env_with_creds,
        capture_output=True, text=True, timeout=1200  # 20-min hard limit
    )
    elapsed = time.time() - t_start

    ok_exit = proc.returncode == 0
    record("P3", f"inference.py completed (exit code 0)", ok_exit,
           f"exit={proc.returncode}")

    ok_time = elapsed < 1200
    record("P3", f"Completed within 20-minute judge time limit",
           ok_time, f"{elapsed:.0f}s elapsed")

    # Parse scores from output
    scores = {}
    for line in proc.stdout.splitlines():
        for task_id in ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]:
            m = re.search(rf"{task_id}.*?([0-9]\.[0-9]+)", line)
            if m:
                scores[task_id] = float(m.group(1))

    # Also try the final JSON line
    for line in reversed(proc.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and len(parsed) == 3:
                scores.update(parsed)
                break
        except Exception:
            pass

    ok_3scores = len(scores) == 3
    record("P3", "inference.py produced 3 task scores", ok_3scores, str(scores))

    if ok_3scores:
        ok_range = all(0.0 <= v <= 1.0 for v in scores.values())
        record("P3", "All scores in [0.0, 1.0]", ok_range, str(scores))

        ok_not_same = len(set(round(v, 3) for v in scores.values())) > 1
        record("P3", "Scores are NOT all identical (disqualifying if same)", ok_not_same)

        t1 = scores.get("basic_threat_detection", 0)
        t2 = scores.get("context_aware_policy", 0)
        t3 = scores.get("multiturn_adversarial", 0)
        ok_order = t1 >= t3  # at minimum easy > hard
        record("P3", f"easy({t1:.3f}) >= hard({t3:.3f}) difficulty ordering", ok_order)

    print(f"\n  --- inference.py output ---")
    for line in proc.stdout.splitlines():
        print(f"  {line}")
    if proc.stderr:
        print(f"  --- stderr ---")
        for line in proc.stderr.splitlines()[:10]:
            print(f"  {_c(YELLOW, line)}")

    return ok_exit and ok_3scores


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(github_url: str, hf_url: str, env_url: str):
    total   = len([r for r in results if r["status"] != "SKIP"])
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    warned  = sum(1 for r in results if r["status"] == "WARN")

    print(f"\n{'=' * 66}")
    print(f"{_c(BOLD, 'SUBMISSION PRE-CHECK SUMMARY')}")
    print(f"{'=' * 66}")
    print(f"  GitHub:    {github_url}")
    print(f"  HF Space:  {hf_url}")
    print(f"  Tested at: {env_url}")
    print(f"{'=' * 66}")
    print(f"  {_c(GREEN, str(passed) + ' passed')}  |  "
          f"{_c(RED, str(failed) + ' failed')}  |  "
          f"{_c(YELLOW, str(warned) + ' warnings')}")
    print(f"{'=' * 66}")

    if failed > 0:
        print(f"\n  {_c(RED, _c(BOLD, 'FAILING TESTS (fix before submitting):'))}")
        for r in results:
            if r["status"] == "FAIL":
                detail = f"  — {r['detail']}" if r["detail"] else ""
                print(f"  {FAIL_MARK}  [{r['phase']}] {r['name']}{detail}")

    if warned > 0:
        print(f"\n  {_c(YELLOW, 'WARNINGS:')}")
        for r in results:
            if r["status"] == "WARN":
                detail = f"  — {r['detail']}" if r["detail"] else ""
                print(f"  {WARN_MARK}  [{r['phase']}] {r['name']}{detail}")

    if failed == 0:
        print(f"\n  {_c(GREEN, _c(BOLD, 'All checks passed. Ready to submit!'))}")
    else:
        print(f"\n  {_c(RED, _c(BOLD, 'Fix failing tests before submitting.'))}")

    print(f"{'=' * 66}\n")
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Guardrail Arena pre-submission validator"
    )
    parser.add_argument("--github-url", default="",
                        help="Your GitHub repo URL (display only)")
    parser.add_argument("--hf-url", default="",
                        help="Your HF Space URL (e.g. https://huggingface.co/spaces/user/space)")
    parser.add_argument("--env-url", default="",
                        help="Direct API URL override (skips HF URL conversion)")
    parser.add_argument("--skip-phase3", action="store_true",
                        help="Skip agentic evaluation even if creds are set")
    args = parser.parse_args()

    print(f"\n{_c(BOLD, 'Guardrail Arena — Hackathon Pre-Submission Validator')}")
    print("=" * 66)

    # --- Collect inputs interactively if not provided ---
    github_url = args.github_url
    hf_url     = args.hf_url
    env_url    = args.env_url

    if not github_url:
        github_url = input(
            "\n  Enter your GitHub Repository URL\n"
            "  (e.g. https://github.com/username/my-openenv): "
        ).strip()

    if not env_url:
        if not hf_url:
            hf_url = input(
                "\n  Enter your Hugging Face Space URL\n"
                "  (e.g. https://huggingface.co/spaces/username/guardrail-arena\n"
                "   or press Enter to test localhost:7860): "
        ).strip()

        if hf_url:
            env_url = hf_space_to_api_url(hf_url)
            print(f"\n  Resolved API endpoint: {env_url}")
        else:
            env_url = "http://localhost:7860"
            hf_url  = env_url
            print(f"\n  Using local Docker container: {env_url}")

    # --- Connectivity check before running full suite ---
    print(f"\n  Checking connectivity to {env_url} ...")
    try:
        r = httpx.get(f"{env_url}/", timeout=15)
        if r.status_code != 200:
            print(f"\n  {_c(RED, 'Cannot reach server')} — got HTTP {r.status_code}")
            print(f"  Make sure your Docker container or HF Space is running.")
            raise SystemExit(1)
        print(f"  {_c(GREEN, 'Server reachable')} — {r.json()}")
    except httpx.ConnectError:
        print(f"\n  {_c(RED, 'Cannot reach server')} at {env_url}")
        print(f"  If testing locally: docker run -p 7860:7860 guardrail-arena")
        raise SystemExit(1)

    # --- Run phases ---
    p1_ok = phase1_api_compliance(env_url)
    p2_ok = phase2_episode_integrity(env_url)

    if not args.skip_phase3:
        phase3_agentic_evaluation(env_url)
    else:
        section("PHASE 3 — Agentic Evaluation")
        print(f"  {SKIP_MARK}  Skipped (--skip-phase3 flag set)")

    # --- Summary ---
    all_ok = print_summary(github_url, hf_url, env_url)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
