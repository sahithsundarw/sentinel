#!/usr/bin/env python3
"""
Guardrail Arena — Submission Validator
=======================================
Mirrors the official OpenEnv validate-submission.sh check:

  Step 1/3  Ping HF Space  (POST /reset returns 200)
  Step 2/3  Docker build
  Step 3/3  openenv validate

Usage:
    python validate.py <ping_url> [repo_dir]

    python validate.py https://varunventra-guardrail-arena.hf.space
    python validate.py https://varunventra-guardrail-arena.hf.space ./guardrail-arena

Arguments:
    ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
    repo_dir   Path to your repo directory containing Dockerfile (default: current dir)
"""
import json
import os
import subprocess
import sys
import time

import httpx

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

if sys.stdout.isatty():
    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BOLD   = "\033[1m"
    NC     = "\033[0m"
else:
    RED = GREEN = YELLOW = BOLD = NC = ""

PASS_COUNT = 0


def log(msg):
    t = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{t}] {msg}")


def passed(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    log(f"{GREEN}PASSED{NC} -- {msg}")


def failed(msg):
    log(f"{RED}FAILED{NC} -- {msg}")


def hint(msg):
    print(f"  {YELLOW}Hint:{NC} {msg}")


def stop_at(step):
    print()
    print(f"{RED}{BOLD}Validation stopped at {step}.{NC} Fix the above before continuing.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ping_url> [repo_dir]")
        print()
        print("  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)")
        print("  repo_dir   Path to your repo (default: current directory)")
        sys.exit(1)

    ping_url = sys.argv[1].rstrip("/")
    repo_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else ".")

    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{BOLD}  OpenEnv Submission Validator{NC}")
    print(f"{BOLD}========================================{NC}")
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {ping_url}")
    print()

    # ------------------------------------------------------------------
    # Step 1/3 — Ping HF Space
    # ------------------------------------------------------------------
    log(f"{BOLD}Step 1/3: Pinging HF Space{NC} ({ping_url}/reset) ...")

    http_code = 0
    try:
        r = httpx.post(f"{ping_url}/reset", content=b"{}", timeout=30,
                       headers={"Content-Type": "application/json"})
        http_code = r.status_code
    except Exception as exc:
        failed(f"HF Space not reachable — {exc}")
        hint("Check your network connection and that the Space is running.")
        hint(f"Try: curl -s -o /dev/null -w '%{{http_code}}' -X POST {ping_url}/reset")
        stop_at("Step 1")

    if http_code == 200:
        passed("HF Space is live and responds to /reset")
    else:
        failed(f"HF Space /reset returned HTTP {http_code} (expected 200)")
        hint("Make sure your Space is running and the URL is correct.")
        hint(f"Try opening {ping_url} in your browser first.")
        stop_at("Step 1")

    # ------------------------------------------------------------------
    # Step 2/3 — Docker build
    # ------------------------------------------------------------------
    log(f"{BOLD}Step 2/3: Running docker build{NC} ...")

    # Find docker
    docker_cmd = None
    for candidate in ["docker", "docker.exe"]:
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True, timeout=10)
            if r.returncode == 0:
                docker_cmd = [candidate]
                break
        except Exception:
            pass

    if docker_cmd is None:
        failed("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        stop_at("Step 2")

    # Find Dockerfile
    if os.path.isfile(os.path.join(repo_dir, "Dockerfile")):
        docker_context = repo_dir
    elif os.path.isfile(os.path.join(repo_dir, "server", "Dockerfile")):
        docker_context = os.path.join(repo_dir, "server")
    else:
        failed("No Dockerfile found in repo root or server/ directory")
        stop_at("Step 2")

    log(f"  Found Dockerfile in {docker_context}")

    try:
        cmd = docker_cmd + ["build", docker_context]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        build_ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        failed("Docker build timed out after 600s")
        stop_at("Step 2")

    if build_ok:
        passed("Docker build succeeded")
    else:
        failed("Docker build failed")
        tail = result.stdout[-1000:] + result.stderr[-500:]
        print(tail)
        stop_at("Step 2")

    # ------------------------------------------------------------------
    # Step 3/3 — openenv validate
    # ------------------------------------------------------------------
    log(f"{BOLD}Step 3/3: Running openenv validate{NC} ...")

    # Try bare `openenv` first, fall back to `python -m openenv.cli`
    openenv_cmd = None
    for candidate in ["openenv", "openenv.exe"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True, timeout=10)
            openenv_cmd = [candidate]
            break
        except Exception:
            pass
    if openenv_cmd is None:
        try:
            subprocess.run([sys.executable, "-m", "openenv.cli", "--help"],
                           capture_output=True, check=True, timeout=10)
            openenv_cmd = [sys.executable, "-m", "openenv.cli"]
        except Exception:
            pass

    if openenv_cmd is None:
        failed("openenv command not found")
        hint("Install it: pip install openenv-core")
        stop_at("Step 3")

    try:
        result = subprocess.run(
            openenv_cmd + ["validate"],
            capture_output=True, text=True, cwd=repo_dir, timeout=120,
        )
        validate_ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        failed("openenv validate timed out after 120s")
        stop_at("Step 3")

    output = (result.stdout + result.stderr).strip()
    if validate_ok:
        passed("openenv validate passed")
        if output:
            log(f"  {output}")
    else:
        failed("openenv validate failed")
        print(output)
        stop_at("Step 3")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{GREEN}{BOLD}  All 3/3 checks passed!{NC}")
    print(f"{GREEN}{BOLD}  Your submission is ready to submit.{NC}")
    print(f"{BOLD}========================================{NC}")
    print()


if __name__ == "__main__":
    main()
