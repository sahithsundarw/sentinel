"""
update_readme_results.py

Updates README.md with real training results and adds the Colab badge.

Usage:
    python update_readme_results.py
    python update_readme_results.py --results-file results/local_training_results.json
"""
import argparse
import json
import os
import re
import sys

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb)"
)

RESULTS_FILES = [
    "results/local_training_results.json",
    "results/notebook_training_results.json",
]


def load_results(path: str | None = None) -> dict | None:
    candidates = [path] if path else RESULTS_FILES
    for f in candidates:
        if f and os.path.exists(f):
            try:
                with open(f) as fh:
                    return json.load(fh)
            except Exception:
                pass
    return None


def add_colab_badge(content: str) -> tuple[str, bool]:
    """Add Colab badge after the first H1 heading if not already present."""
    if "colab-badge.svg" in content:
        return content, False

    # Find first H1
    lines = content.split("\n")
    new_lines = []
    inserted = False
    for i, line in enumerate(lines):
        new_lines.append(line)
        if not inserted and line.startswith("# "):
            new_lines.append("")
            new_lines.append(COLAB_BADGE)
            new_lines.append("")
            inserted = True
    return "\n".join(new_lines), inserted


def update_training_table(content: str, results: dict) -> tuple[str, bool]:
    """Update the Llama-3.1-8B row in the Training Results table."""
    task_id = results.get("task_id", "basic_threat_detection")
    final = results.get("final_score")
    if final is None:
        return content, False

    task_idx = {
        "basic_threat_detection": 0,
        "context_aware_policy": 1,
        "multiturn_adversarial": 2,
        "adversarial_adaptation": 3,
    }.get(task_id, 0)

    # Find the Llama-3.1-8B row in the markdown table
    # Pattern: | Llama-3.1-8B | score | score | score | score |
    pattern = re.compile(
        r'(\|\s*(?:Llama-3\.1-8B|Llama 3\.1 8B)[^|]*\|)([^|]*\|){4}',
        re.IGNORECASE
    )

    def replace_row(m: re.Match) -> str:
        full = m.group(0)
        # Split into cells
        cells = full.split("|")
        # cells[0] = '', cells[1] = model name, cells[2..5] = task scores, cells[6] = ''
        if len(cells) >= 6:
            # Replace the cell at task_idx+2 with the new score
            cells[task_idx + 2] = f" **{final:.4f}** (trained) "
        return "|".join(cells)

    new_content, count = re.subn(pattern, replace_row, content, count=1)
    return new_content, count > 0


def print_diff(original: str, updated: str) -> None:
    orig_lines = original.split("\n")
    upd_lines = updated.split("\n")
    changes = 0
    for i, (o, u) in enumerate(zip(orig_lines, upd_lines)):
        if o != u:
            print(f"  Line {i+1}:")
            print(f"    - {o[:100]}")
            print(f"    + {u[:100]}")
            changes += 1
            if changes >= 10:
                print("  ... (more changes not shown)")
                break
    # Check for added lines
    if len(upd_lines) > len(orig_lines):
        for j in range(len(orig_lines), min(len(orig_lines) + 5, len(upd_lines))):
            print(f"  Line {j+1} (added): {upd_lines[j][:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update README.md with real training results")
    parser.add_argument("--results-file", help="Path to training results JSON")
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    args = parser.parse_args()

    if not os.path.exists(args.readme):
        print(f"ERROR: {args.readme} not found")
        sys.exit(1)

    with open(args.readme) as f:
        original = f.read()

    content = original
    changes: list[str] = []

    # 1. Add Colab badge
    content, badge_added = add_colab_badge(content)
    if badge_added:
        changes.append("Added Colab badge after H1 heading")
    else:
        print("  Colab badge already present.")

    # 2. Update training results table
    results = load_results(args.results_file)
    if results:
        print(f"  Loaded training results: {results.get('task_id')} / method={results.get('method')}")
        print(f"  Zero-shot: {results.get('zero_shot_score'):.4f}  Final: {results.get('final_score'):.4f}")
        content, table_updated = update_training_table(content, results)
        if table_updated:
            changes.append(f"Updated Llama-3.1-8B Training Results row (final={results['final_score']:.4f})")
        else:
            print("  WARNING: Could not find Llama-3.1-8B row in Training Results table.")
    else:
        print("  No training results JSON found — skipping table update.")
        print(f"  Run 'python run_training_local.py --mock' first to generate results.")

    if content != original:
        print("\nChanges:")
        for c in changes:
            print(f"  + {c}")
        print()
        print_diff(original, content)
        with open(args.readme, "w") as f:
            f.write(content)
        print(f"\nUpdated: {args.readme}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
