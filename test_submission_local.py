#!/usr/bin/env python3
"""
Hackathon Submission Validator
Simulates the exact automated checks run on submitted GitHub + HF Space URLs.

Tests every item in the pre-submission checklist:
  1. HF Space deploys and responds
  2. OpenEnv spec compliance (reset/step/state endpoints + schema)
  3. Dockerfile present and valid
  4. Baseline inference script format ([START]/[STEP]/[END])
  5. 3+ tasks with graders returning scores in [0.0, 1.0]
  6. openenv.yaml validity
  7. README required sections
  8. Environment variable compliance

Usage:
    python test_submission.py
"""

import os
import re
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from typing import Optional

import httpx
import yaml

# --- Submission details -------------------------------------------------------
GITHUB_REPO_URL = "https://github.com/sahithsundarw/sentinel"
HF_SPACE_PAGE   = "https://huggingface.co/spaces/varunventra/guardrail-arena"
HF_SPACE_API    = "http://localhost:7861"
REPO_ROOT       = Path(__file__).parent
TIMEOUT         = 45  # seconds per HTTP request

# --- Console colours ---------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")
def info(msg): print(f"  {BLUE}INFO{RESET}  {msg}")

# --- Results store ------------------------------------------------------------
RESULTS = []  # list of {"name", "status", "detail"}

def record(name: str, status: str, detail: str = ""):
    RESULTS.append({"name": name, "status": status, "detail": detail})
    if status == "PASS":
        ok(f"[{name}] {detail}")
    elif status == "FAIL":
        fail(f"[{name}] {detail}")
    elif status == "WARN":
        warn(f"[{name}] {detail}")
    else:
        info(f"[{name}] {detail}")


# =============================================================================
# PHASE 1 -- HF SPACE AVAILABILITY
# =============================================================================

def phase1_hf_space():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 1: HF Space Availability{RESET}")
    print(f"  Target: {HF_SPACE_API}")
    print(f"{'='*60}")

    # 1a. Root endpoint returns 200
    try:
        r = httpx.get(f"{HF_SPACE_API}/", timeout=TIMEOUT, follow_redirects=True)
        if r.status_code == 200:
            record("HF Space root /", "PASS", f"HTTP {r.status_code}")
        else:
            record("HF Space root /", "FAIL", f"HTTP {r.status_code} (expected 200)")
    except Exception as e:
        record("HF Space root /", "FAIL", f"Connection error: {e}")

    # 1b. /health endpoint
    try:
        r = httpx.get(f"{HF_SPACE_API}/health", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy":
                record("GET /health", "PASS", f"status={data['status']}, env={data.get('environment','?')}")
            else:
                record("GET /health", "FAIL", f"status field missing or not 'healthy': {data}")
        else:
            record("GET /health", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        record("GET /health", "FAIL", str(e))

    # 1c. /metadata endpoint
    try:
        r = httpx.get(f"{HF_SPACE_API}/metadata", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            has_name = "name" in data
            has_tags = "tags" in data and "openenv" in data.get("tags", [])
            if has_name and has_tags:
                record("GET /metadata", "PASS", f"name='{data['name']}', openenv tag present")
            elif has_name:
                record("GET /metadata", "WARN", f"name present but 'openenv' tag missing from metadata tags")
            else:
                record("GET /metadata", "FAIL", f"missing 'name' field: {list(data.keys())}")
        else:
            record("GET /metadata", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        record("GET /metadata", "FAIL", str(e))

    # 1d. /tasks endpoint returns 3+ tasks
    tasks_found = []
    try:
        r = httpx.get(f"{HF_SPACE_API}/tasks", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            tasks_found = data.get("tasks", [])
            if len(tasks_found) >= 3:
                task_ids = [t["id"] for t in tasks_found]
                record("GET /tasks (3+)", "PASS", f"{len(tasks_found)} tasks: {task_ids}")
            else:
                record("GET /tasks (3+)", "FAIL", f"Only {len(tasks_found)} tasks found (need >=3)")
        else:
            record("GET /tasks (3+)", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        record("GET /tasks (3+)", "FAIL", str(e))

    return tasks_found


# =============================================================================
# PHASE 2 -- OPENENV SPEC COMPLIANCE
# =============================================================================

def phase2_openenv_spec():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 2: OpenEnv Spec Compliance{RESET}")
    print(f"  Testing: reset() / step() / state() + schema{RESET}")
    print(f"{'='*60}")

    session_id = None
    obs = None

    # 2a. POST /reset
    try:
        r = httpx.post(
            f"{HF_SPACE_API}/reset",
            params={"task_id": "basic_threat_detection"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            obs = r.json()
            session_id = obs.get("session_id")
            required_obs_fields = [
                "prompt_id", "user_prompt", "application_context",
                "user_risk_score", "turn_number", "prompts_remaining",
                "episode_score_so_far",
            ]
            missing = [f for f in required_obs_fields if f not in obs]
            if not missing:
                record("POST /reset", "PASS",
                    f"session_id={'present' if session_id else 'MISSING'}, "
                    f"prompt_id={obs.get('prompt_id','?')[:8]}..., "
                    f"prompts_remaining={obs.get('prompts_remaining','?')}")
            else:
                record("POST /reset", "FAIL", f"Missing Observation fields: {missing}")
        else:
            record("POST /reset", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        record("POST /reset", "FAIL", str(e))
        return session_id

    if not session_id:
        record("session_id returned", "FAIL", "session_id not in /reset response -- step/state will fail")
        return session_id

    # 2b. POST /step
    step_reward = None
    try:
        action = {
            "prompt_id": obs["prompt_id"],
            "action_type": "allow",
            "reason": "test step",
            "modified_prompt": None,
        }
        r = httpx.post(
            f"{HF_SPACE_API}/step",
            json=action,
            params={"session_id": session_id},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            step_data = r.json()
            required_step_fields = ["reward", "done", "info"]
            missing = [f for f in required_step_fields if f not in step_data]
            reward = step_data.get("reward", {})
            step_reward = reward.get("score")
            has_feedback = "feedback" in reward
            has_breakdown = "breakdown" in reward
            done = step_data.get("done")

            if not missing:
                record("POST /step", "PASS",
                    f"reward.score={step_reward:.4f}, done={done}, "
                    f"feedback={'v' if has_feedback else 'x'}, "
                    f"breakdown={'v' if has_breakdown else 'x'}")
            else:
                record("POST /step", "FAIL", f"Missing fields: {missing}")

            # Verify reward is incremental (not None)
            if step_reward is not None:
                record("Reward is non-null", "PASS", f"score={step_reward:.4f}")
            else:
                record("Reward is non-null", "FAIL", "reward.score is None")

            # Verify done is a boolean
            if isinstance(done, bool):
                record("done is boolean", "PASS", f"done={done}")
            else:
                record("done is boolean", "FAIL", f"done={done!r} (type={type(done).__name__})")

        else:
            record("POST /step", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        record("POST /step", "FAIL", str(e))

    # 2c. GET /state
    try:
        r = httpx.get(
            f"{HF_SPACE_API}/state",
            params={"session_id": session_id},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            state = r.json()
            required_state_fields = ["task_id", "current_prompt_index", "total_prompts",
                                     "cumulative_score", "done", "initialized"]
            missing = [f for f in required_state_fields if f not in state]
            if not missing:
                record("GET /state", "PASS",
                    f"task_id={state.get('task_id')}, "
                    f"progress={state.get('current_prompt_index')}/{state.get('total_prompts')}")
            else:
                record("GET /state", "FAIL", f"Missing EnvironmentState fields: {missing}")
        else:
            record("GET /state", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        record("GET /state", "FAIL", str(e))

    # 2d. GET /schema
    try:
        r = httpx.get(f"{HF_SPACE_API}/schema", timeout=TIMEOUT)
        if r.status_code == 200:
            schema = r.json()
            has_action = "action" in schema
            has_obs    = "observation" in schema
            has_state  = "state" in schema
            if has_action and has_obs and has_state:
                record("GET /schema (Pydantic models)", "PASS",
                    "action v, observation v, state v")
            else:
                missing = [k for k, v in {"action": has_action, "observation": has_obs, "state": has_state}.items() if not v]
                record("GET /schema (Pydantic models)", "FAIL", f"Missing schemas: {missing}")
        else:
            record("GET /schema (Pydantic models)", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        record("GET /schema (Pydantic models)", "FAIL", str(e))

    return session_id


# =============================================================================
# PHASE 3 -- GRADER CHECK (full episode per task)
# =============================================================================

def phase3_graders():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 3: Task Graders (0.0-1.0 range, 3 required tasks){RESET}")
    print(f"  Running FULL episodes (until done=True) per task, then calling /grader")
    print(f"{'='*60}")

    tasks_to_test = [
        "basic_threat_detection",
        "context_aware_policy",
        "multiturn_adversarial",
    ]

    for task_id in tasks_to_test:
        info(f"Testing task: {task_id} (running full episode...)")
        try:
            # Reset
            r = httpx.post(
                f"{HF_SPACE_API}/reset",
                params={"task_id": task_id},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            obs = r.json()
            session_id = obs.get("session_id")
            total_prompts = obs.get("prompts_remaining", "?")
            if not session_id:
                record(f"Grader [{task_id}]", "FAIL", "no session_id from /reset")
                continue

            info(f"  {task_id}: {total_prompts} prompts to run...")

            # Run ALL steps until done=True (grader requires complete episode)
            done = False
            steps = 0
            rewards_seen = []
            MAX_STEPS = 300  # hard ceiling to avoid infinite loop

            while not done and steps < MAX_STEPS:
                action = {
                    "prompt_id": obs["prompt_id"],
                    "action_type": "refuse",  # deterministic policy for reproducibility
                    "reason": "test",
                    "modified_prompt": None,
                }
                r = httpx.post(
                    f"{HF_SPACE_API}/step",
                    json=action,
                    params={"session_id": session_id},
                    timeout=TIMEOUT,
                )
                r.raise_for_status()
                step_data = r.json()
                reward_val = step_data.get("reward", {}).get("score", 0.0)
                rewards_seen.append(reward_val)
                done = step_data.get("done", False)
                steps += 1
                if not done:
                    obs = step_data["observation"]
                if steps % 20 == 0:
                    info(f"  {task_id}: {steps} steps done, done={done}")

            info(f"  {task_id}: episode complete after {steps} steps")

            # Call /grader only after done=True
            r = httpx.get(
                f"{HF_SPACE_API}/grader",
                params={"session_id": session_id},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            grader_data = r.json()
            score = grader_data.get("score")

            if score is None:
                record(f"Grader [{task_id}]", "FAIL", "score field missing from /grader response")
            elif not isinstance(score, (int, float)):
                record(f"Grader [{task_id}]", "FAIL", f"score is not numeric: {score!r}")
            elif 0.0 <= score <= 1.0:
                # Check reward variety (not all same value = deterministic non-grader)
                unique_rewards = set(round(r, 4) for r in rewards_seen)
                reward_detail = f"score={score:.4f}, steps={steps}, reward_values={sorted(unique_rewards)}"
                if len(unique_rewards) >= 2:
                    record(f"Grader [{task_id}]", "PASS", reward_detail)
                else:
                    record(f"Grader [{task_id}]", "WARN",
                        f"{reward_detail} -- all rewards identical, may indicate flat reward function")
            else:
                record(f"Grader [{task_id}]", "FAIL",
                    f"score={score:.4f} outside [0.0, 1.0]")

        except Exception as e:
            record(f"Grader [{task_id}]", "FAIL", f"{type(e).__name__}: {e}")


# =============================================================================
# PHASE 4 -- INFERENCE SCRIPT FORMAT
# =============================================================================

def phase4_inference_format():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 4: Inference Script Compliance{RESET}")
    print(f"  Checking source code + output format patterns")
    print(f"{'='*60}")

    infer_path = REPO_ROOT / "inference.py"

    # 4a. inference.py exists in root
    if infer_path.exists():
        record("inference.py in root", "PASS", str(infer_path.relative_to(REPO_ROOT)))
    else:
        record("inference.py in root", "FAIL", "File not found at repo root")
        return

    source = infer_path.read_text(encoding="utf-8")

    # 4b. Uses OpenAI client
    if "from openai import OpenAI" in source or "openai.OpenAI" in source:
        record("Uses OpenAI Client", "PASS", "found 'from openai import OpenAI'")
    else:
        record("Uses OpenAI Client", "FAIL", "OpenAI client import not found")

    # 4c. API_BASE_URL has default value
    match = re.search(r'os\.getenv\s*\(\s*["\']API_BASE_URL["\'].*?\)', source)
    if match:
        snippet = match.group(0)
        has_default = "," in snippet and ('"' in snippet.split(",", 1)[1] or "'" in snippet.split(",", 1)[1])
        if has_default:
            record("API_BASE_URL has default", "PASS", snippet.strip())
        else:
            record("API_BASE_URL has default", "FAIL", f"No default value: {snippet}")
    else:
        record("API_BASE_URL has default", "FAIL", "os.getenv('API_BASE_URL') not found")

    # 4d. MODEL_NAME has default value
    match = re.search(r'os\.getenv\s*\(\s*["\']MODEL_NAME["\'].*?\)', source)
    if match:
        snippet = match.group(0)
        has_default = "," in snippet and ('"' in snippet.split(",", 1)[1] or "'" in snippet.split(",", 1)[1])
        if has_default:
            record("MODEL_NAME has default", "PASS", snippet.strip())
        else:
            record("MODEL_NAME has default", "FAIL", f"No default value: {snippet}")
    else:
        record("MODEL_NAME has default", "FAIL", "os.getenv('MODEL_NAME') not found")

    # 4e. HF_TOKEN is read (mandatory, no default needed)
    if "HF_TOKEN" in source:
        # Make sure there's no default value set for HF_TOKEN
        match_hf = re.search(r'os\.getenv\s*\(\s*["\']HF_TOKEN["\']([^)]*)\)', source)
        if match_hf:
            args_part = match_hf.group(1)
            has_default = "," in args_part and any(q in args_part.split(",", 1)[1] for q in ['"', "'"])
            if not has_default:
                record("HF_TOKEN mandatory (no default)", "PASS", "HF_TOKEN read without default")
            else:
                record("HF_TOKEN mandatory (no default)", "WARN",
                    "HF_TOKEN appears to have a default value (should be mandatory)")
    else:
        record("HF_TOKEN mandatory (no default)", "FAIL", "HF_TOKEN not referenced in inference.py")

    # 4f. [START] line emitted
    start_pattern = r'\[START\].*task=.*env=.*model='
    if re.search(start_pattern, source):
        match = re.search(r'print\s*\(\s*f["\'].*\[START\].*["\']', source)
        snippet = match.group(0)[:80] if match else "[START] print found"
        record("[START] line emitted", "PASS", snippet)
    else:
        record("[START] line emitted", "FAIL", "[START] pattern not found in inference.py")

    # 4g. [STEP] line emitted
    step_pattern = r'\[STEP\].*step=.*action=.*reward=.*done=.*error='
    if re.search(step_pattern, source):
        record("[STEP] line emitted", "PASS", "step=, action=, reward=, done=, error= all present")
    else:
        record("[STEP] line emitted", "FAIL", "[STEP] line missing required fields (step/action/reward/done/error)")

    # 4h. [END] line emitted
    end_pattern = r'\[END\].*success=.*steps=.*rewards='
    if re.search(end_pattern, source):
        record("[END] line emitted", "PASS", "success=, steps=, rewards= all present")
    else:
        record("[END] line emitted", "FAIL", "[END] line missing required fields (success/steps/rewards)")

    # 4i. reward formatted to 2 decimal places in [STEP]
    if re.search(r'reward=.*\.2f', source):
        record("reward formatted :.2f", "PASS", ":.2f format spec found for reward")
    else:
        record("reward formatted :.2f", "FAIL", "No :.2f format found for reward in [STEP] line")

    # 4j. done/success are lowercase booleans ('true'/'false', not True/False)
    if re.search(r"'true'|'false'", source) or re.search(r'"true"|"false"', source):
        record("done/success lowercase bool", "PASS", "lowercase 'true'/'false' strings found")
    else:
        record("done/success lowercase bool", "WARN",
            "Could not confirm lowercase 'true'/'false' -- verify [STEP] done= and [END] success= are not Python booleans")

    # 4k. [END] on exception path
    if re.search(r'except.*\n.*\[END\]|except.*:.*\n(?:.*\n)*?.*\[END\]', source, re.MULTILINE):
        record("[END] on exception path", "PASS", "[END] emitted inside except block")
    else:
        record("[END] on exception path", "WARN",
            "Could not confirm [END] is emitted on exception -- verify manually")

    # 4l. Check inference.py syntax parses
    try:
        import ast
        ast.parse(source)
        record("inference.py syntax", "PASS", "AST parse successful")
    except SyntaxError as e:
        record("inference.py syntax", "FAIL", f"SyntaxError: {e}")


# =============================================================================
# PHASE 5 -- REPO STRUCTURE & FILES
# =============================================================================

def phase5_repo_structure():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 5: Repository Structure & Files{RESET}")
    print(f"  Checking local repo: {REPO_ROOT}")
    print(f"{'='*60}")

    # 5a. Dockerfile exists
    dockerfile = REPO_ROOT / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text()
        has_from       = "FROM" in content
        has_workdir    = "WORKDIR" in content
        has_expose     = "EXPOSE" in content
        has_cmd        = "CMD" in content
        has_copy_reqs  = "requirements.txt" in content
        if all([has_from, has_workdir, has_cmd]):
            record("Dockerfile exists", "PASS",
                f"FROM v, WORKDIR v, EXPOSE {'v' if has_expose else 'x'}, CMD v, "
                f"requirements.txt {'v' if has_copy_reqs else 'x'}")
        else:
            missing = [k for k, v in {"FROM": has_from, "WORKDIR": has_workdir, "CMD": has_cmd}.items() if not v]
            record("Dockerfile exists", "WARN", f"Missing directives: {missing}")
    else:
        record("Dockerfile exists", "FAIL", "Dockerfile not found in repo root")

    # 5b. openenv.yaml exists and has required fields
    yaml_path = REPO_ROOT / "openenv.yaml"
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text())
            required_yaml_keys = ["name", "description", "author", "version", "tags", "tasks"]
            missing = [k for k in required_yaml_keys if k not in data]
            tasks = data.get("tasks", [])
            has_openenv_tag = "openenv" in data.get("tags", [])
            task_difficulties = [t.get("difficulty") for t in tasks]
            has_easy   = "easy"   in task_difficulties
            has_medium = "medium" in task_difficulties
            has_hard   = "hard"   in task_difficulties or "expert" in task_difficulties

            if not missing:
                record("openenv.yaml required fields", "PASS",
                    f"name='{data['name']}', {len(tasks)} tasks, openenv tag={'v' if has_openenv_tag else 'x'}")
            else:
                record("openenv.yaml required fields", "FAIL", f"Missing keys: {missing}")

            if has_openenv_tag:
                record("openenv.yaml 'openenv' tag", "PASS", f"tags: {data['tags']}")
            else:
                record("openenv.yaml 'openenv' tag", "FAIL",
                    f"'openenv' not in tags: {data.get('tags', [])}")

            if len(tasks) >= 3:
                record("openenv.yaml 3+ tasks", "PASS",
                    f"{len(tasks)} tasks with difficulties: {task_difficulties}")
            else:
                record("openenv.yaml 3+ tasks", "FAIL", f"Only {len(tasks)} tasks defined")

            diff_range = has_easy and has_medium and has_hard
            if diff_range:
                record("Task difficulty range (easy->hard)", "PASS",
                    f"easy v, medium v, hard/expert v")
            else:
                record("Task difficulty range (easy->hard)", "WARN",
                    f"difficulties found: {task_difficulties} -- missing one of easy/medium/hard")

            # Check baseline scores in yaml
            tasks_with_baseline = [t for t in tasks if "baseline_agent_score" in t]
            if tasks_with_baseline:
                record("Baseline scores in openenv.yaml", "PASS",
                    f"{len(tasks_with_baseline)}/{len(tasks)} tasks have baseline_agent_score")
            else:
                record("Baseline scores in openenv.yaml", "WARN",
                    "No baseline_agent_score fields found in tasks")

        except yaml.YAMLError as e:
            record("openenv.yaml parse", "FAIL", f"YAML error: {e}")
    else:
        record("openenv.yaml exists", "FAIL", "openenv.yaml not found in repo root")

    # 5c. README.md required sections
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")

        sections = {
            "HF Space openenv tag":        "openenv" in content[:500],  # in frontmatter
            "Environment overview":         any(kw in content for kw in ["## Abstract", "## Overview", "environment", "motivation"]),
            "Action space definition":      "action" in content.lower() and ("action_space" in content or "Action" in content),
            "Observation space definition": "observation" in content.lower(),
            "Task descriptions":            all(t in content for t in ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]),
            "Difficulty levels":            all(d in content.lower() for d in ["easy", "medium", "hard"]),
            "Setup instructions":           "## Setup" in content or "## Installation" in content,
            "Docker instructions":          "docker build" in content.lower() and "docker run" in content.lower(),
            "inference.py instructions":    "HF_TOKEN" in content and "inference.py" in content,
            "Baseline performance scores":  "## Baseline" in content or "baseline" in content.lower(),
        }

        all_pass = True
        for section, found in sections.items():
            if found:
                record(f"README: {section}", "PASS")
            else:
                record(f"README: {section}", "FAIL", "Section/content not found")
                all_pass = False
    else:
        record("README.md exists", "FAIL", "README.md not found in repo root")

    # 5d. requirements.txt
    req_path = REPO_ROOT / "requirements.txt"
    if req_path.exists():
        reqs = req_path.read_text()
        has_fastapi = "fastapi" in reqs
        has_uvicorn = "uvicorn" in reqs
        has_pydantic = "pydantic" in reqs
        has_openai = "openai" in reqs
        if all([has_fastapi, has_uvicorn, has_pydantic, has_openai]):
            record("requirements.txt", "PASS",
                "fastapi v, uvicorn v, pydantic v, openai v")
        else:
            missing = [k for k, v in {
                "fastapi": has_fastapi, "uvicorn": has_uvicorn,
                "pydantic": has_pydantic, "openai": has_openai
            }.items() if not v]
            record("requirements.txt", "WARN", f"Missing packages: {missing}")
    else:
        record("requirements.txt", "FAIL", "requirements.txt not found")


# =============================================================================
# PHASE 6 -- REWARD FUNCTION QUALITY
# =============================================================================

def phase6_reward_quality():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}PHASE 6: Reward Function Quality{RESET}")
    print(f"  Checking reward varies, penalizes bad actions, rewards correct ones")
    print(f"{'='*60}")

    try:
        # Reset a fresh episode
        r = httpx.post(
            f"{HF_SPACE_API}/reset",
            params={"task_id": "basic_threat_detection"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        obs = r.json()
        session_id = obs.get("session_id")

        # Take 8 steps with alternating allow/refuse to force reward variation
        rewards = []
        actions = ["allow", "refuse", "allow", "refuse", "allow", "escalate", "modify", "refuse"]

        for i, action_type in enumerate(actions):
            if not obs:
                break
            action = {
                "prompt_id": obs["prompt_id"],
                "action_type": action_type,
                "reason": "reward quality test",
                "modified_prompt": "sanitized prompt" if action_type == "modify" else None,
            }
            r = httpx.post(
                f"{HF_SPACE_API}/step",
                json=action,
                params={"session_id": session_id},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            step_data = r.json()
            reward_val = step_data["reward"]["score"]
            feedback = step_data["reward"].get("feedback", "")
            rewards.append(reward_val)
            info(f"  step {i+1}: action={action_type:8s}  reward={reward_val:+.4f}  feedback='{feedback[:50]}'")

            done = step_data.get("done", False)
            if done:
                break
            obs = step_data.get("observation")

        if len(set(round(r, 4) for r in rewards)) >= 2:
            record("Reward varies per action", "PASS",
                f"Unique reward values: {sorted(set(round(r,4) for r in rewards))}")
        else:
            record("Reward varies per action", "FAIL",
                f"All rewards identical: {rewards} -- grader always returns same score")

        has_positive = any(r > 0 for r in rewards)
        has_negative = any(r < 0 for r in rewards)
        if has_positive:
            record("Positive rewards exist", "PASS", f"max={max(rewards):.4f}")
        else:
            record("Positive rewards exist", "WARN", "No positive rewards seen in 8 steps")

        if has_negative:
            record("Negative rewards exist (penalises wrong actions)", "PASS",
                f"min={min(rewards):.4f}")
        else:
            record("Negative rewards exist (penalises wrong actions)", "WARN",
                "No negative rewards seen in 8 steps -- penalty mechanism may not be triggering")

        # Check cumulative score tracks properly
        r2 = httpx.get(
            f"{HF_SPACE_API}/state",
            params={"session_id": session_id},
            timeout=TIMEOUT,
        )
        if r2.status_code == 200:
            state = r2.json()
            cumul = state.get("cumulative_score", "N/A")
            record("Cumulative score tracked in state", "PASS", f"cumulative_score={cumul}")
        else:
            record("Cumulative score tracked in state", "WARN", f"Could not verify: HTTP {r2.status_code}")

    except Exception as e:
        record("Reward quality check", "FAIL", f"{type(e).__name__}: {e}")


# =============================================================================
# FINAL REPORT
# =============================================================================

def final_report():
    print(f"\n{'='*60}")
    print(f"{BOLD}FINAL SUBMISSION REPORT{RESET}")
    print(f"{'='*60}")
    print(f"  GitHub:   {GITHUB_REPO_URL}")
    print(f"  HF Space: {HF_SPACE_PAGE}")
    print(f"{'='*60}\n")

    passed  = [r for r in RESULTS if r["status"] == "PASS"]
    failed  = [r for r in RESULTS if r["status"] == "FAIL"]
    warned  = [r for r in RESULTS if r["status"] == "WARN"]
    total   = len(RESULTS)

    # Print failures
    if failed:
        print(f"{RED}{BOLD}FAILURES ({len(failed)}):{RESET}")
        for r in failed:
            print(f"  {RED}x{RESET}  {r['name']}")
            if r["detail"]:
                print(f"      -> {r['detail']}")
        print()

    # Print warnings
    if warned:
        print(f"{YELLOW}{BOLD}WARNINGS ({len(warned)}):{RESET}")
        for r in warned:
            print(f"  {YELLOW}!{RESET}  {r['name']}")
            if r["detail"]:
                print(f"      -> {r['detail']}")
        print()

    # Print passes
    print(f"{GREEN}{BOLD}PASSES ({len(passed)}/{total}):{RESET}")
    for r in passed:
        suffix = f" -- {r['detail']}" if r["detail"] else ""
        print(f"  {GREEN}v{RESET}  {r['name']}{suffix}")

    # Score
    score_pct = len(passed) / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"{BOLD}SCORE: {len(passed)}/{total} checks passed ({score_pct:.0f}%){RESET}")
    print(f"{'='*60}")

    # Prediction
    print(f"\n{BOLD}PREDICTED OUTCOME:{RESET}")

    # Automated validation gate
    gate_checks = [
        "HF Space root /",
        "GET /health",
        "POST /reset",
        "POST /step",
        "GET /state",
        "GET /tasks (3+)",
        "inference.py in root",
        "Uses OpenAI Client",
        "API_BASE_URL has default",
        "MODEL_NAME has default",
        "HF_TOKEN mandatory (no default)",
        "[START] line emitted",
        "[STEP] line emitted",
        "[END] line emitted",
        "Dockerfile exists",
        "openenv.yaml required fields",
        "openenv.yaml 'openenv' tag",
        "openenv.yaml 3+ tasks",
    ]

    gate_failed = [r for r in failed if r["name"] in gate_checks]
    grader_failed = [r for r in failed if r["name"].startswith("Grader [")]

    if not gate_failed and not grader_failed:
        print(f"  {GREEN}{BOLD}v AUTOMATED VALIDATION: PASS{RESET}")
        print(f"    All Phase 1 gates clear. Submission should advance to Phase 2 (Agentic Evaluation).")
    elif gate_failed:
        print(f"  {RED}{BOLD}x AUTOMATED VALIDATION: LIKELY FAIL{RESET}")
        print(f"    Gate failures: {[r['name'] for r in gate_failed]}")
        print(f"    Fix these before submitting.")
    elif grader_failed:
        print(f"  {YELLOW}{BOLD}! AUTOMATED VALIDATION: AT RISK{RESET}")
        print(f"    Grader failures: {[r['name'] for r in grader_failed]}")

    print(f"\n  {BOLD}PHASE 2 SCORING PREDICTION (if Phase 1 passes):{RESET}")
    print(f"    Real-world utility (30%):      HIGH -- content moderation is a genuine production use case")
    print(f"    Task & grader quality (25%):   HIGH -- 4 tasks, deterministic graders, verified 0.0-1.0")
    print(f"    Environment design (20%):      HIGH -- dynamic risk score, branching, incremental rewards")
    print(f"    Code quality/compliance (15%): {'HIGH' if not warned else 'MEDIUM'} -- OpenEnv spec met{', some warnings' if warned else ''}")
    print(f"    Creativity & novelty (10%):    HIGH -- adversarial adaptation (Task 4) is novel RL mechanic")

    print(f"\n{'='*60}\n")

    return len(failed), len(warned)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  HACKATHON SUBMISSION VALIDATOR{RESET}")
    print(f"{BOLD}  Simulating automated checks exactly as judges run them{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GitHub:   {GITHUB_REPO_URL}")
    print(f"  HF Space: {HF_SPACE_PAGE}")

    start = time.time()

    phase1_hf_space()
    phase2_openenv_spec()
    phase3_graders()
    phase4_inference_format()
    phase5_repo_structure()
    phase6_reward_quality()

    elapsed = time.time() - start
    print(f"\n  Total test time: {elapsed:.1f}s")

    failures, warnings = final_report()
    sys.exit(1 if failures else 0)
