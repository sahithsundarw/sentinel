"""
Run after GRPO training completes to update all artifacts.
Usage: python scripts/integrate_grpo_results.py
"""
import json
import os
from pathlib import Path


def find_grpo_results():
    candidates = [
        "results/grpo_scores.json",
        "results/llama_grpo_scores.json",
        "grpo_results.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None


def main():
    data, path = find_grpo_results()
    if data is None:
        print("No GRPO results found. Training may still be running.")
        print("Expected one of: results/grpo_scores.json, results/llama_grpo_scores.json, grpo_results.json")
        return

    print(f"Found GRPO results at {path}:")
    print(json.dumps(data, indent=2))

    if os.path.exists("generate_charts.py"):
        os.system("python generate_charts.py")
        print("Charts regenerated with GRPO data.")
    else:
        print("generate_charts.py not found — skipping chart regeneration")

    print("\nNext steps:")
    print("1. Review results above")
    print("2. Update README.md results table with GRPO row")
    print("3. Update training_results.html with GRPO row")
    print("4. git add -A && git commit -m 'feat: GRPO training results'")
    print("5. Redeploy to HF Space")


if __name__ == "__main__":
    main()
