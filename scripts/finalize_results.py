"""
Run after Q-learner training finishes.
Reads results/qlearner_task4_eval.json, updates generate_charts.py KNOWN_SCORES,
regenerates all charts, commits results to git, and pushes to GitHub.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

EVAL_PATH = ROOT / "results" / "qlearner_task4_eval.json"
CHARTS_PY  = ROOT / "generate_charts.py"
README_MD  = ROOT / "README.md"


def load_eval():
    if not EVAL_PATH.exists():
        print(f"ERROR: {EVAL_PATH} not found. Q-learner training may not have finished.")
        sys.exit(1)
    with open(EVAL_PATH) as f:
        return json.load(f)


def update_readme(eval_data):
    mean  = eval_data["mean"]
    std   = eval_data["std"]
    seeds = eval_data["seeds"]
    scores = eval_data["scores"]
    print(f"Q-Learner Task 4: mean={mean:.4f} ± std={std:.4f} over seeds {seeds}")
    print(f"Per-seed: {scores}")

    text = README_MD.read_text(encoding="utf-8")

    # Update the Q-Learner row in the results table
    old_row = r"\| \*\*Tabular Q-Learner\*\* \|.*?\| \*\*0\.9540\*\* \|"
    new_row = (
        f"| **Tabular Q-Learner** | **RL (keyword features)** "
        f"| ~0.46 | 0.507 | **0.487** | **{mean:.4f} ± {std:.4f}** |"
    )
    updated = re.sub(old_row, new_row, text)
    if updated == text:
        print("WARNING: Could not find Q-Learner row in README to update. Skipping.")
    else:
        README_MD.write_text(updated, encoding="utf-8")
        print("README.md updated with multi-seed eval results.")


def update_charts(eval_data):
    mean = eval_data["mean"]
    text = CHARTS_PY.read_text(encoding="utf-8")

    # Update Q-Learner task4 score in KNOWN_SCORES if it changed
    old = '"Q-Learner (trained)":{"task1": 0.4600, "task2": 0.5070, "task3": 0.4870, "task4": 0.9540},'
    new = f'"Q-Learner (trained)":{{\"task1\": 0.4600, "task2": 0.5070, "task3": 0.4870, "task4": {mean:.4f}}},'
    updated = text.replace(old, new)
    if updated != text:
        CHARTS_PY.write_text(updated, encoding="utf-8")
        print(f"generate_charts.py updated: Q-Learner task4 → {mean:.4f}")
    else:
        print("generate_charts.py: KNOWN_SCORES already matches or pattern not found.")


def regen_charts():
    print("\nRegenerating charts...")
    python = sys.executable
    result = subprocess.run([python, str(CHARTS_PY)], capture_output=True, text=True)
    if result.returncode != 0:
        print("Chart generation failed:")
        print(result.stderr[-2000:])
    else:
        print("Charts regenerated successfully.")
        if result.stdout:
            print(result.stdout[-500:])


def git_commit_push():
    files = [
        "results/qlearner_task4.json",
        "results/qlearner_task4_eval.json",
        "results/qlearner_task4_run.log",
        "results/hero_learning_curve.png",
        "results/multi_model_comparison.png",
        "results/heatmap.png",
        "results/action_distribution.png",
        "results/before_after_table.png",
        "results/sft_loss_curve.png",
        "results/training_comparison.png",
        "results/full_training_curve.png",
        "generate_charts.py",
        "README.md",
    ]
    existing = [f for f in files if Path(f).exists()]
    print(f"\nStaging {len(existing)} files...")
    subprocess.run(["git", "add"] + existing, check=True)

    status = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("Nothing to commit.")
        return

    print(status.stdout)
    eval_data = load_eval()
    msg = (
        f"results: Q-learner Task 4 multi-seed eval {eval_data['mean']:.4f} ± {eval_data['std']:.4f}\n\n"
        f"seeds={eval_data['seeds']}, per-seed={eval_data['scores']}\n"
        f"explore={eval_data['explore_episodes']}, exploit={eval_data['exploit_episodes']}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    print("Committed. Pushing to GitHub...")
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Pushed to GitHub.")


def main():
    print("=" * 60)
    print("Finalizing Q-Learner results")
    print("=" * 60)

    eval_data = load_eval()
    update_readme(eval_data)
    update_charts(eval_data)
    regen_charts()
    git_commit_push()

    print("\nDone. Judges can see all results at:")
    print("  https://varunventra-guardrail-arena.hf.space/results")
    print("  https://varunventra-guardrail-arena.hf.space/training_log")
    print("  https://github.com/sahithsundarw/sentinel/tree/main/results")


if __name__ == "__main__":
    main()
