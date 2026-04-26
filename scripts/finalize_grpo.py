"""
Run after GRPO training finishes on the HF Training Space.
Reads live /training_log data, updates README and charts with GRPO results,
then pushes the updated codebase to GitHub AND to the HF API Space.

The HF API Space restart is safe after GRPO training finishes because:
- training_log.json is persisted to disk on every POST
- app/main.py now loads training_log.json on startup
- So the /training_log endpoint is preserved across restarts

Usage:
    python scripts/finalize_grpo.py [--push-hf]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_URL = "https://varunventra-guardrail-arena.hf.space"
README_MD = ROOT / "README.md"
CHARTS_PY = ROOT / "generate_charts.py"


def fetch_training_log():
    r = requests.get(f"{ENV_URL}/training_log", timeout=30)
    r.raise_for_status()
    return r.json()


def summarize_grpo(log_data: dict) -> dict:
    """Extract pre/post scores and learning curve from training_log."""
    summary = {}
    all_entries = log_data.get("all_entries", {})
    for agent_name, entries in all_entries.items():
        if not entries:
            continue
        by_task: dict[str, list] = {}
        for e in entries:
            tid = e.get("task_id", "unknown")
            by_task.setdefault(tid, []).append(e)
        for task_id, task_entries in by_task.items():
            task_entries.sort(key=lambda x: x.get("episode", 0))
            pre  = task_entries[0]["grader_score"]
            post = task_entries[-1]["grader_score"]
            n    = len(task_entries)
            scores = [e["grader_score"] for e in task_entries]
            summary[task_id] = {
                "agent": agent_name,
                "episodes": n,
                "pre_score": round(pre, 4),
                "post_score": round(post, 4),
                "delta": round(post - pre, 4),
                "scores": [round(s, 4) for s in scores],
            }
            print(f"  {task_id}: {n} episodes, {pre:.4f} -> {post:.4f} (delta {post-pre:+.4f})")
    return summary


def update_readme(summary: dict):
    text = README_MD.read_text(encoding="utf-8")

    # Find the Llama GRPO row and update it
    # Current: | Llama-3.1-8B | REINFORCE (20 ep, LoRA) | 0.0929 | — | — | — |
    t3 = summary.get("multiturn_adversarial", {})
    t2 = summary.get("context_aware_policy", {})
    t1 = summary.get("basic_threat_detection", {})

    t1_score = f"**{t1['post_score']:.4f}**" if t1 else "—"
    t2_score = f"**{t2['post_score']:.4f}**" if t2 else "—"
    t3_score = f"**{t3['post_score']:.4f}**" if t3 else "—"

    old_row = r"\| Llama-3\.1-8B \| REINFORCE \(20 ep.*?\|"
    new_row = (
        f"| Llama-3.1-8B | GRPO (20 ep × 3 tasks, LoRA, L40S) "
        f"| {t1_score} | {t2_score} | {t3_score} | — |"
    )
    updated = re.sub(old_row, new_row, text)
    if updated == text:
        print("WARNING: Could not find Llama GRPO row in README. Manual update needed.")
    else:
        README_MD.write_text(updated, encoding="utf-8")
        print("README.md updated with GRPO scores.")


def regen_charts():
    print("\nRegenerating charts...")
    result = subprocess.run([sys.executable, str(CHARTS_PY)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Chart generation FAILED:", result.stderr[-1000:])
    else:
        print("Charts regenerated OK.")


def git_commit_push_github(summary: dict):
    subprocess.run(["git", "add",
                    "README.md",
                    "generate_charts.py",
                    "results/"], check=True, cwd=ROOT)
    status = subprocess.run(["git", "diff", "--cached", "--stat"],
                            capture_output=True, text=True, cwd=ROOT)
    if not status.stdout.strip():
        print("Nothing to commit to GitHub.")
        return
    print(status.stdout)
    tasks_done = list(summary.keys())
    msg = (
        f"results: GRPO Llama-3.1-8B training complete ({', '.join(tasks_done)})\n\n"
        + "\n".join(
            f"  {tid}: {v['pre_score']:.4f} -> {v['post_score']:.4f} ({v['episodes']} ep)"
            for tid, v in summary.items()
        )
        + "\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], check=True, cwd=ROOT)
    subprocess.run(["git", "push", "origin", "main"], check=True, cwd=ROOT)
    print("Pushed to GitHub.")


def push_to_hf():
    print("\nPushing to HF Space (this will restart the Space)...")
    print("NOTE: training_log.json is persisted on disk, so /training_log survives restart.")
    result = subprocess.run(["git", "push", "hf", "main"],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print("HF push failed:", result.stderr)
    else:
        print("Pushed to HF Space. Space will restart in ~30 seconds.")
        print("After restart, /results will include all training files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-hf", action="store_true",
                        help="Also push to HF Space (restarts the Space)")
    args = parser.parse_args()

    print("=" * 60)
    print("Finalizing GRPO training results")
    print("=" * 60)

    log_data = fetch_training_log()
    agents = log_data.get("agents", [])
    print(f"Agents logged: {agents}")

    if not agents:
        print("No GRPO data in training_log. Is training still running?")
        sys.exit(1)

    summary = summarize_grpo(log_data)
    if not summary:
        print("No valid GRPO episodes found.")
        sys.exit(1)

    update_readme(summary)
    regen_charts()
    git_commit_push_github(summary)

    if args.push_hf:
        push_to_hf()
    else:
        print("\nSkipped HF push. Run with --push-hf when GRPO training is fully complete.")

    print("\nDone. Judges can see GRPO training data at:")
    print("  https://varunventra-guardrail-arena.hf.space/training_log")
    print("  https://github.com/sahithsundarw/sentinel/tree/main/results")


if __name__ == "__main__":
    main()
