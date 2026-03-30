"""
Guardrail Arena — Pre-Submission Validator
==========================================
Mirrors the 9-phase check that hackathon judges run.

Usage:
    python validate.py --url https://varunventra-guardrail-arena.hf.space

    python validate.py --url https://varunventra-guardrail-arena.hf.space \
        --api-base-url https://router.huggingface.co/v1 \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --hf-token hf_xxx
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import httpx
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

TASK_IDS = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(msg):  return f"{GREEN}[PASS]{RESET} {msg}"
def fail(msg): return f"{RED}[FAIL]{RESET} {msg}"
def warn(msg): return f"{YELLOW}[WARN]{RESET} {msg}"


def section(n, title):
    print(f"\n{BOLD}[{n}/9] {title}{RESET}")
    print("  " + "-" * 60)


def check(label, passed, detail=""):
    tag = ok("PASS") if passed else fail("FAIL")
    detail_str = f"  ({detail})" if detail else ""
    print(f"  Testing: {label}")
    print(f"  Result:  {tag}{detail_str}")
    return passed


def section_result(passed):
    if passed:
        print(f"  {GREEN}{BOLD}SECTION RESULT: PASS{RESET}")
    else:
        print(f"  {RED}{BOLD}SECTION RESULT: FAIL{RESET}")
    return passed


def run_episode(url, task_id, action="allow"):
    """Run a complete episode with a fixed action. Returns (scores, grader_score)."""
    r = httpx.post(f"{url}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()

    step_scores = []
    done = False
    while not done:
        action_body = {
            "prompt_id": obs["prompt_id"],
            "action_type": action,
            "reason": "validate",
            "modified_prompt": None,
        }
        sr = httpx.post(f"{url}/step", json=action_body, timeout=30)
        sr.raise_for_status()
        data = sr.json()
        step_scores.append(data["reward"]["score"])
        done = data["done"]
        if not done:
            obs = data["observation"]

    gr = httpx.get(f"{url}/grader", timeout=30)
    gr.raise_for_status()
    grader_score = gr.json()["score"]
    return step_scores, grader_score


# ---------------------------------------------------------------------------
# Section 1 — HF Space Health
# ---------------------------------------------------------------------------

def s1_health(url):
    section(1, "HF SPACE HEALTH")
    results = []

    for endpoint in ["/", "/health"]:
        full = url.rstrip("/") + endpoint
        print(f"  Testing: GET {endpoint} returns 200")
        try:
            t0 = time.time()
            r = httpx.get(full, timeout=15)
            elapsed = time.time() - t0
            if r.status_code != 200:
                # Cold-start retry
                print(f"  Result:  {warn('got ' + str(r.status_code) + ', retrying after 30s (cold start)')}")
                time.sleep(30)
                t0 = time.time()
                r = httpx.get(full, timeout=15)
                elapsed = time.time() - t0
            passed = r.status_code == 200 and elapsed < 10
            results.append(check(
                f"GET {endpoint} 200 under 10s",
                passed,
                f"status={r.status_code} time={elapsed:.1f}s",
            ))
        except Exception as e:
            print(f"  Result:  {fail('FAIL')}  ({e})")
            results.append(False)

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 2 — OpenEnv Spec Compliance
# ---------------------------------------------------------------------------

def s2_openenv(url):
    section(2, "OPENENV SPEC COMPLIANCE")
    results = []

    yaml_path = os.path.join(HERE, "openenv.yaml")
    print(f"  Testing: openenv.yaml exists at {yaml_path}")
    if not os.path.exists(yaml_path):
        print(f"  Result:  {fail('FAIL')}  (file not found)")
        return section_result(False)
    print(f"  Result:  {ok('PASS')}")

    with open(yaml_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    required_fields = ["name", "description", "author", "version", "tags",
                       "tasks", "observation_space", "action_space", "reward_space"]
    for field in required_fields:
        results.append(check(
            f"openenv.yaml has field '{field}'",
            field in spec,
            "present" if field in spec else "MISSING",
        ))

    tags = spec.get("tags", [])
    results.append(check("tags includes 'openenv'", "openenv" in tags, str(tags)))

    tasks = spec.get("tasks", [])
    results.append(check("tasks has >= 3 entries", len(tasks) >= 3, f"got {len(tasks)}"))

    task_fields_ok = all(
        all(k in t for k in ["id", "name", "description", "difficulty"])
        for t in tasks
    )
    results.append(check("each task has id/name/description/difficulty", task_fields_ok))

    difficulties = {t.get("difficulty") for t in tasks}
    for d in ["easy", "medium", "hard"]:
        results.append(check(
            f"difficulty '{d}' present",
            d in difficulties,
            str(difficulties),
        ))

    print(f"  Testing: GET /tasks matches openenv.yaml task list")
    try:
        r = httpx.get(f"{url}/tasks", timeout=15)
        api_ids = {t["id"] for t in r.json().get("tasks", r.json()
                   if isinstance(r.json(), list) else [])}
        yaml_ids = {t["id"] for t in tasks}
        # handle both list and dict responses
        raw = r.json()
        if isinstance(raw, list):
            api_ids = {t["id"] for t in raw}
        elif isinstance(raw, dict) and "tasks" in raw:
            api_ids = {t["id"] for t in raw["tasks"]}
        else:
            api_ids = set()
        match = yaml_ids == api_ids
        results.append(check("GET /tasks IDs match openenv.yaml", match,
                              f"yaml={yaml_ids} api={api_ids}"))
    except Exception as e:
        print(f"  Result:  {fail('FAIL')}  ({e})")
        results.append(False)

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 3 — Endpoint Shape Validation
# ---------------------------------------------------------------------------

def s3_shapes(url):
    section(3, "ENDPOINT SHAPE VALIDATION")
    results = []

    obs_fields = {
        "prompt_id": str, "user_prompt": str, "conversation_history": list,
        "application_context": str, "user_risk_score": float, "turn_number": int,
        "prompts_remaining": int, "episode_score_so_far": float,
    }

    for task_id in TASK_IDS:
        print(f"\n  -- {task_id} --")
        try:
            r = httpx.post(f"{url}/reset", params={"task_id": task_id}, timeout=30)
            results.append(check(f"POST /reset?task_id={task_id} returns 200",
                                 r.status_code == 200, f"got {r.status_code}"))
            obs = r.json()
            for field, ftype in obs_fields.items():
                present = field in obs
                correct_type = isinstance(obs.get(field), ftype) if present else False
                if field == "user_risk_score":
                    correct_type = correct_type and 0.0 <= obs[field] <= 1.0
                results.append(check(
                    f"  obs.{field} present and correct type",
                    present and correct_type,
                    f"{type(obs.get(field)).__name__}={obs.get(field)!r}" if present else "MISSING",
                ))

            # GET /state
            sr = httpx.get(f"{url}/state", timeout=15)
            state = sr.json()
            for sf in ["task_id", "current_prompt_index", "total_prompts",
                       "cumulative_score", "done", "initialized"]:
                results.append(check(f"GET /state has '{sf}'", sf in state))

            # POST /step
            action_body = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                           "reason": "validate", "modified_prompt": None}
            step_r = httpx.post(f"{url}/step", json=action_body, timeout=30)
            results.append(check("POST /step returns 200",
                                 step_r.status_code == 200, f"got {step_r.status_code}"))
            step_data = step_r.json()
            for sf in ["observation", "reward", "done", "info"]:
                results.append(check(f"step response has '{sf}'", sf in step_data))
            reward = step_data.get("reward", {})
            for rf in ["score", "cumulative_score", "feedback", "breakdown"]:
                results.append(check(f"reward has '{rf}'", rf in reward))

            # GET /grader before done — expect non-200
            gr = httpx.get(f"{url}/grader", timeout=15)
            results.append(check(
                "GET /grader before done returns error (4xx/5xx)",
                gr.status_code >= 400,
                f"got {gr.status_code}",
            ))

        except Exception as e:
            print(f"  Result:  {fail('FAIL')}  ({e})")
            results.append(False)

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 4 — Full Episode Execution
# ---------------------------------------------------------------------------

def s4_episodes(url):
    section(4, "FULL EPISODE EXECUTION")
    results = []

    for task_id in TASK_IDS:
        print(f"\n  -- {task_id} --")
        try:
            r = httpx.post(f"{url}/reset", params={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            obs = r.json()

            prev_remaining = obs["prompts_remaining"]
            step_scores = []
            episode_scores = [obs["episode_score_so_far"]]
            done = False
            step_num = 0
            all_done_false_until_last = True
            decrement_ok = True

            while not done:
                action_body = {"prompt_id": obs["prompt_id"], "action_type": "allow",
                               "reason": "validate", "modified_prompt": None}
                sr = httpx.post(f"{url}/step", json=action_body, timeout=30)
                sr.raise_for_status()
                data = sr.json()
                done = data["done"]
                step_scores.append(data["reward"]["score"])
                step_num += 1

                if not done:
                    obs = data["observation"]
                    episode_scores.append(obs["episode_score_so_far"])
                    new_remaining = obs["prompts_remaining"]
                    if new_remaining != prev_remaining - 1:
                        decrement_ok = False
                    prev_remaining = new_remaining

            results.append(check(f"{task_id}: episode completes (done=True)", True))
            results.append(check(f"{task_id}: prompts_remaining decrements correctly", decrement_ok))

            score_changes = len(set(episode_scores)) > 1
            results.append(check(f"{task_id}: episode_score_so_far changes", score_changes,
                                 f"{len(set(episode_scores))} unique values"))

            gr = httpx.get(f"{url}/grader", timeout=30)
            results.append(check(f"{task_id}: GET /grader returns 200 after done",
                                 gr.status_code == 200, f"got {gr.status_code}"))
            gscore = gr.json().get("score", None)
            in_range = gscore is not None and 0.0 <= gscore <= 1.0
            results.append(check(f"{task_id}: grader score in [0.0, 1.0]",
                                 in_range, f"score={gscore}"))
            print(f"  Score: {gscore:.4f}" if gscore is not None else "  Score: N/A")

        except Exception as e:
            print(f"  Result:  {fail('FAIL')}  ({e})")
            results.append(False)

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 5 — Reward Function Quality
# ---------------------------------------------------------------------------

def s5_reward(url):
    section(5, "REWARD FUNCTION QUALITY")
    results = []

    try:
        print("  Testing: Episode A (all allow) vs Episode B (all refuse)")
        scores_a, grader_a = run_episode(url, "basic_threat_detection", "allow")
        scores_b, grader_b = run_episode(url, "basic_threat_detection", "refuse")

        results.append(check("Episode A rewards vary (not all identical)",
                             len(set(scores_a)) > 1,
                             f"{len(set(scores_a))} unique values"))
        results.append(check("Episode B rewards vary (not all identical)",
                             len(set(scores_b)) > 1,
                             f"{len(set(scores_b))} unique values"))
        results.append(check("Episode A has at least one non-zero reward",
                             any(s != 0 for s in scores_a)))
        results.append(check("Episode B has at least one non-zero reward",
                             any(s != 0 for s in scores_b)))
        results.append(check("Episode A and B produce different grader scores",
                             abs(grader_a - grader_b) > 0.001,
                             f"A={grader_a:.4f} B={grader_b:.4f}"))

        var_a = sum((s - sum(scores_a)/len(scores_a))**2 for s in scores_a) / len(scores_a)
        var_b = sum((s - sum(scores_b)/len(scores_b))**2 for s in scores_b) / len(scores_b)
        print(f"  Episode A (allow): grader={grader_a:.4f}  reward_variance={var_a:.5f}")
        print(f"  Episode B (refuse): grader={grader_b:.4f}  reward_variance={var_b:.5f}")

        # Verify all 3 task graders differentiate — catches a grader silently broken for task2/3
        print("\n  Testing: all-allow vs all-refuse produces different grader scores on all tasks")
        for tid in TASK_IDS:
            try:
                _, g_allow = run_episode(url, tid, "allow")
                _, g_refuse = run_episode(url, tid, "refuse")
                results.append(check(
                    f"{tid}: allow({g_allow:.4f}) != refuse({g_refuse:.4f})",
                    abs(g_allow - g_refuse) > 0.001,
                    f"allow={g_allow:.4f} refuse={g_refuse:.4f}",
                ))
                results.append(check(
                    f"{tid}: all-allow grader is non-zero",
                    g_allow > 0.0,
                    f"got {g_allow:.4f}",
                ))
            except Exception as e:
                print(f"  Result:  {fail('FAIL')}  ({tid}: {e})")
                results.append(False)

    except Exception as e:
        print(f"  Result:  {fail('FAIL')}  ({e})")
        results.append(False)

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 6 — Grader Determinism
# ---------------------------------------------------------------------------

def s6_determinism(url):
    section(6, "GRADER DETERMINISM")
    results = []

    grader_scores = {}
    for task_id in TASK_IDS:
        try:
            _, score1 = run_episode(url, task_id, "allow")
            _, score2 = run_episode(url, task_id, "allow")
            match = abs(score1 - score2) < 1e-9
            results.append(check(f"{task_id}: run1={score1:.4f} == run2={score2:.4f}", match))
            grader_scores[task_id] = score1
        except Exception as e:
            print(f"  Result:  {fail('FAIL')}  ({e})")
            results.append(False)
            grader_scores[task_id] = None

    valid = [v for v in grader_scores.values() if v is not None]
    not_all_same = len(set(round(v, 6) for v in valid)) > 1
    results.append(check(
        "Grader scores differ across tasks (not all identical — disqualifying if same)",
        not_all_same,
        str({k: f"{v:.4f}" for k, v in grader_scores.items() if v is not None}),
    ))

    print(f"\n  Scores:")
    for task_id, score in grader_scores.items():
        print(f"    {task_id}: {score:.4f}" if score is not None else f"    {task_id}: ERROR")

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 7 — Inference Script
# ---------------------------------------------------------------------------

def s7_inference(url, api_base_url, model, hf_token):
    section(7, "INFERENCE SCRIPT")
    results = []

    inference_path = os.path.join(HERE, "inference.py")
    results.append(check("inference.py exists", os.path.exists(inference_path),
                         inference_path))
    if not os.path.exists(inference_path):
        return section_result(False)

    src = open(inference_path, encoding="utf-8").read()

    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        found = var in src
        results.append(check(f"inference.py reads {var} env var", found,
                             "found" if found else "MISSING — judges cannot inject their model"))

    if "gpt-4o-mini" in src:
        print(f"  {warn('WARN')}: inference.py contains hardcoded 'gpt-4o-mini' — "
              "should use MODEL_NAME env var")
    else:
        print(f"  {ok('OK')}: no hardcoded model name found")

    if "OpenAI(base_url=" in src or 'OpenAI(base_url =' in src:
        print(f"  {ok('OK')}: uses OpenAI(base_url=...) for endpoint injection")
    else:
        print(f"  {warn('WARN')}: OpenAI(base_url=...) not found — "
              "judges may not be able to redirect to their endpoint")

    if api_base_url and model and hf_token:
        print(f"\n  Running live inference: model={model}")
        env = os.environ.copy()
        env.update({
            "API_BASE_URL": api_base_url,
            "MODEL_NAME": model,
            "HF_TOKEN": hf_token,
            "ENV_URL": url,
        })
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, inference_path],
                env=env, capture_output=True, text=True, timeout=1200,
            )
            elapsed = time.time() - t0
            output = proc.stdout + proc.stderr
            results.append(check("inference.py completes under 20 minutes",
                                 elapsed < 1200, f"{elapsed:.0f}s"))
            for tid in TASK_IDS:
                results.append(check(f"output contains {tid}",
                                     tid in output, "found" if tid in output else "MISSING"))
            # Try to extract JSON scores line
            for line in reversed(output.splitlines()):
                try:
                    scores = json.loads(line)
                    if isinstance(scores, dict):
                        print(f"  Inference scores: {scores}")
                        break
                except Exception:
                    pass
        except subprocess.TimeoutExpired:
            print(f"  Result:  {fail('FAIL')}  (timed out after 20 minutes)")
            results.append(False)
    else:
        print(f"\n  {YELLOW}SKIPPED{RESET}: live inference run "
              "(provide --api-base-url --model --hf-token to run)")

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 8 — Documentation Check
# ---------------------------------------------------------------------------

def s8_docs():
    section(8, "DOCUMENTATION CHECK")
    results = []

    readme_path = os.path.join(HERE, "README.md")
    print(f"  Testing: README.md exists at {readme_path}")
    if not os.path.exists(readme_path):
        print(f"  Result:  {fail('FAIL')}  (not found)")
        return section_result(False)
    print(f"  Result:  {ok('PASS')}")

    content = open(readme_path, encoding="utf-8").read().lower()

    for word in ["observation", "action"]:
        results.append(check(f"README contains '{word}'", word in content))

    for tid in TASK_IDS:
        results.append(check(f"README contains task ID '{tid}'", tid in content))

    for level in ["easy", "medium", "hard"]:
        results.append(check(f"README contains difficulty '{level}'", level in content))

    has_scores = bool(re.search(r'\|\s*\d+\.\d+', content))
    results.append(check("README contains a baseline scores table (|0.xxxx pattern)",
                         has_scores))

    has_setup = "docker" in content or "pip install" in content
    results.append(check("README contains setup instructions (docker or pip install)",
                         has_setup))

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Section 9 — Docker Build
# ---------------------------------------------------------------------------

def s9_docker():
    section(9, "DOCKER BUILD")
    results = []

    # Check Docker availability
    try:
        ver = subprocess.run(["docker", "--version"], capture_output=True,
                             text=True, timeout=10)
        if ver.returncode != 0:
            raise RuntimeError("docker --version failed")
        print(f"  Docker available: {ver.stdout.strip()}")
    except Exception:
        print(f"  {YELLOW}SKIPPED{RESET}: Docker not available on this machine")
        return "SKIPPED"

    container_name = "guardrail-arena-validate-run"
    image_name = "guardrail-arena-validate"

    # Clean up any leftover container from a previous run
    subprocess.run(["docker", "rm", "-f", container_name],
                   capture_output=True, text=True)

    # Build
    print(f"  Testing: docker build -t {image_name} .")
    t0 = time.time()
    build = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        capture_output=True, text=True, cwd=HERE,
    )
    build_time = time.time() - t0
    build_ok = build.returncode == 0
    results.append(check(f"docker build exits 0 (took {build_time:.0f}s)", build_ok,
                         "OK" if build_ok else build.stderr[-300:]))

    if not build_ok:
        return section_result(False)

    # Run
    print(f"  Testing: docker run -d -p 7861:7860 {image_name}")
    run = subprocess.run(
        ["docker", "run", "-d", "-p", "7861:7860", "--name", container_name, image_name],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        results.append(check("docker run starts", False, run.stderr[:200]))
    else:
        time.sleep(6)
        try:
            hr = httpx.get("http://localhost:7861/", timeout=10)
            health_ok = hr.status_code == 200
            results.append(check("container health check GET / returns 200",
                                 health_ok, f"status={hr.status_code}"))
        except Exception as e:
            results.append(check("container health check GET / returns 200", False, str(e)))

    # Always clean up
    subprocess.run(["docker", "stop", container_name], capture_output=True, text=True)
    subprocess.run(["docker", "rm",  container_name], capture_output=True, text=True)
    print(f"  Container cleaned up.")

    return section_result(all(results))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Guardrail Arena pre-submission validator")
    parser.add_argument("--url", required=True,
                        help="HF Space or local URL, e.g. https://user-space.hf.space")
    parser.add_argument("--api-base-url", default="",
                        help="OpenAI-compatible API base URL for live inference test")
    parser.add_argument("--model", default="",
                        help="Model name for live inference test")
    parser.add_argument("--hf-token", default="",
                        help="HuggingFace token for live inference test")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    print(f"\n{BOLD}Guardrail Arena — Pre-Submission Validator{RESET}")
    print("=" * 64)
    print(f"  Target: {url}")
    print(f"  Local dir: {HERE}")
    print("=" * 64)

    outcomes = {}

    outcomes[1]  = s1_health(url)
    outcomes[2]  = s2_openenv(url)
    outcomes[3]  = s3_shapes(url)
    outcomes[4]  = s4_episodes(url)
    outcomes[5]  = s5_reward(url)
    outcomes[6]  = s6_determinism(url)
    outcomes[7]  = s7_inference(url, args.api_base_url, args.model, args.hf_token)
    outcomes[8]  = s8_docs()
    outcomes[9]  = s9_docker()

    labels = {
        1: "HF Space Health",
        2: "OpenEnv Spec Compliance",
        3: "Endpoint Shape Validation",
        4: "Full Episode Execution",
        5: "Reward Function Quality",
        6: "Grader Determinism",
        7: "Inference Script",
        8: "Documentation Check",
        9: "Docker Build",
    }

    print(f"\n{'=' * 64}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"{'=' * 64}")
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    failures = []

    for n in range(1, 10):
        result = outcomes[n]
        label = labels[n]
        if result == "SKIPPED":
            status = f"{YELLOW}SKIPPED{RESET}"
            skipped_count += 1
        elif result:
            status = f"{GREEN}PASS{RESET}"
            passed_count += 1
        else:
            status = f"{RED}FAIL{RESET}"
            failed_count += 1
            failures.append(f"[{n}/9] {label}")
        print(f"  [{n}/9] {label:<35} {status}")

    print(f"{'=' * 64}")
    print(f"  {GREEN}{passed_count} passed{RESET}  |  "
          f"{RED}{failed_count} failed{RESET}  |  "
          f"{YELLOW}{skipped_count} skipped{RESET}")

    if failures:
        print(f"\n{RED}{BOLD}FAILING SECTIONS:{RESET}")
        for f_label in failures:
            print(f"  {RED}[FAIL]{RESET}  {f_label}")
        print(f"\n{RED}{BOLD}OVERALL: FAIL — fix the sections above before submitting.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}OVERALL: PASS — ready to submit.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
