"""generate_charts.py — Produce reward curve visualizations for Guardrail Arena Round 2.

Generates Charts D, E, and F from hardcoded baseline data.
No model training required — run immediately.

Usage:
    python generate_charts.py
    python generate_charts.py --output-dir ./results

Output files:
    results/task4_learning_curve.png   — Chart E: Task 4 Q-learner (0.0 → 0.95)
    results/score_comparison.png       — Chart D: Before/After bar chart
    results/model_comparison.png       — Chart F: All-model comparison
"""
import argparse
import os
import json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Run: pip install matplotlib numpy")


# ── Hardcoded data (from BASELINE_RESULTS.md + train_task4.py run) ───────────

TASK_LABELS = ["Task 1\nbasic_threat_detection", "Task 2\ncontext_aware_policy",
                "Task 3\nmultiturn_adversarial", "Task 4\nadversarial_adaptation"]
TASK_IDS = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"]

# All baseline scores (per task)
BASELINES = {
    "all_allow":    [0.3750, 0.4037, 0.1607, 0.1500],
    "all_refuse":   [0.3534, 0.3460, 0.0688, 0.0000],
    "llama_8b_zs":  [0.5428, 0.5143, 0.4746, 0.0000],
    "gpt4o_mini":   [0.9216, 0.7512, 0.6120, 0.4820],
    "llama_70b":    [0.9857, 0.7713, 0.6866, None],
    "qwen_235b":    [0.9857, 0.6862, 0.8275, 0.0000],
    "oracle":       [1.0000, 1.0000, 1.0000, 1.0000],
}

# Task 4 Q-learner learning curve (from train_task4.py measured run)
TASK4_EPISODES  = [0,      1,      5,      10,     15,     20]
TASK4_SCORES    = [0.0000, 0.0000, 0.0000, 0.5301, 0.9540, 0.9540]
TASK4_ALL_ALLOW = 0.1500
TASK4_ALL_REFUSE = 0.0000


def _save(fig, path: str, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {path}")


# ── Chart E: Task 4 Q-Learner Learning Curve ─────────────────────────────────

def chart_e_task4_learning_curve(output_dir: str) -> None:
    """Q-learner goes from 0.0 to 0.95 in 20 episodes on Task 4 (adversarial_adaptation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0f1117")

    # Main learning curve
    ax.plot(TASK4_EPISODES, TASK4_SCORES, "b-o", linewidth=2.5, markersize=8,
            color="#3b82f6", label="Tabular Q-learner (eval score)")
    ax.fill_between(TASK4_EPISODES, 0, TASK4_SCORES, alpha=0.1, color="#3b82f6")

    # Baselines
    ax.axhline(y=TASK4_ALL_ALLOW, color="#ef4444", linestyle="--", alpha=0.8, linewidth=1.5,
               label=f"All-Allow baseline ({TASK4_ALL_ALLOW:.4f})")
    ax.axhline(y=TASK4_ALL_REFUSE, color="#f97316", linestyle="--", alpha=0.8, linewidth=1.5,
               label=f"All-Refuse baseline ({TASK4_ALL_REFUSE:.4f})")
    ax.axhline(y=0.9540, color="#22c55e", linestyle=":", alpha=0.6, linewidth=1.5,
               label="Convergence (0.9540)")

    # Annotation for Qwen-3-235B at 0.0
    ax.annotate("Qwen-3-235B\n(235B params)\nscores 0.0000",
                xy=(0, 0.0), xytext=(3, 0.12),
                fontsize=9, color="#ef4444",
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c0303", edgecolor="#ef4444"))

    ax.set_xlabel("Training Episode", color="#9ca3af", fontsize=12)
    ax.set_ylabel("Grader Score (0.0 – 1.0)", color="#9ca3af", fontsize=12)
    ax.set_title("Task 4: Adversarial Adaptation — Q-Learner vs Zero-Shot LLMs\n"
                 '"Model scale does not help. Policy learning does."',
                 color="#ffffff", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.5, 21)
    ax.tick_params(colors="#9ca3af")
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#374151")
    ax.legend(fontsize=9, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              loc="upper left")

    _save(fig, os.path.join(output_dir, "task4_learning_curve.png"))
    plt.close(fig)


# ── Chart D: Before/After Score Comparison Bar Chart ─────────────────────────

def chart_d_score_comparison(output_dir: str) -> None:
    """Grouped bar chart showing agent scores across all 4 tasks."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0f1117")

    x = np.arange(4)
    width = 0.13
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    agents = [
        ("All-Allow",     BASELINES["all_allow"],   "#374151"),
        ("All-Refuse",    BASELINES["all_refuse"],  "#4b5563"),
        ("Llama-8B ZS",   BASELINES["llama_8b_zs"], "#6366f1"),
        ("GPT-4o-mini",   BASELINES["gpt4o_mini"],  "#3b82f6"),
        ("Qwen-3-235B",   BASELINES["qwen_235b"],   "#8b5cf6"),
        ("Oracle",        BASELINES["oracle"],      "#22c55e"),
    ]

    for (label, scores, color), offset in zip(agents, offsets):
        vals = [s if s is not None else 0.0 for s in scores]
        bars = ax.bar(x + offset * width, vals, width * 0.85, label=label,
                      color=color, alpha=0.85, edgecolor="#0a0a0a", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#d1d5db")

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, color="#9ca3af", fontsize=10)
    ax.set_ylabel("Grader Score (0.0 – 1.0)", color="#9ca3af", fontsize=12)
    ax.set_title("Guardrail Arena — Score Comparison Across All Tasks & Agents",
                 color="#ffffff", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#9ca3af")
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.15, color="#374151")
    ax.legend(fontsize=9, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              ncol=3, loc="upper right")

    _save(fig, os.path.join(output_dir, "score_comparison.png"))
    plt.close(fig)


# ── Chart F: Multi-Model Comparison Heatmap ───────────────────────────────────

def chart_f_model_comparison(output_dir: str) -> None:
    """Heatmap of all model × task scores."""
    models = ["Oracle", "llama-3.3-70b", "Qwen-3-235B", "gpt-4o-mini",
              "Llama-3.1-8B", "Tabular Q-learner", "All-Allow", "All-Refuse"]
    task_names = ["Task 1", "Task 2", "Task 3", "Task 4"]

    data_matrix = np.array([
        [1.0000, 1.0000, 1.0000, 1.0000],   # Oracle
        [0.9857, 0.7713, 0.6866, None],      # llama-3.3-70b
        [0.9857, 0.6862, 0.8275, 0.0000],   # Qwen-3-235B
        [0.9216, 0.7512, 0.6120, 0.4820],   # gpt-4o-mini
        [0.5428, 0.5143, 0.4746, 0.0000],   # Llama-3.1-8B
        [0.4600, None,   None,   0.9540],   # Tabular Q-learner
        [0.3750, 0.4037, 0.1607, 0.1500],   # All-Allow
        [0.3534, 0.3460, 0.0688, 0.0000],   # All-Refuse
    ], dtype=object)

    # Convert to float, replacing None with NaN
    float_matrix = np.array([[float(v) if v is not None else np.nan
                              for v in row] for row in data_matrix], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(float_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(task_names, color="#e6edf3", fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color="#e6edf3", fontsize=10)
    ax.tick_params(bottom=False, left=False)

    for i in range(len(models)):
        for j in range(4):
            val = float_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#6b7280")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors="#9ca3af")
    cbar.set_label("Grader Score", color="#9ca3af")

    ax.set_title("Guardrail Arena — Model × Task Performance Matrix",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, os.path.join(output_dir, "model_comparison.png"))
    plt.close(fig)


def _save_metadata(output_dir: str, charts: list[str]) -> None:
    """Save chart metadata as JSON for reference."""
    meta = {
        "charts": charts,
        "data": {
            "task4_learning_curve": {
                "episodes": TASK4_EPISODES,
                "scores": TASK4_SCORES,
                "all_allow_baseline": TASK4_ALL_ALLOW,
                "all_refuse_baseline": TASK4_ALL_REFUSE,
                "source": "train_task4.py measured run",
            },
            "baselines": BASELINES,
        }
    }
    path = os.path.join(output_dir, "chart_data.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Guardrail Arena reward curve visualizations")
    parser.add_argument("--output-dir", default="./results", help="Output directory for charts")
    parser.add_argument("--dpi", type=int, default=300, help="Chart DPI (default: 300)")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: Install matplotlib and numpy first:")
        print("  pip install matplotlib numpy")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating charts -> {args.output_dir}/")
    print()

    chart_e_task4_learning_curve(args.output_dir)
    chart_d_score_comparison(args.output_dir)
    chart_f_model_comparison(args.output_dir)

    charts = [
        "task4_learning_curve.png",
        "score_comparison.png",
        "model_comparison.png",
    ]
    _save_metadata(args.output_dir, charts)

    print()
    print("=" * 60)
    print("Charts generated:")
    for c in charts:
        p = os.path.join(args.output_dir, c)
        print(f"  {p}")
    print()
    print("Use in pitch slide 4 and blog post.")
    print("Key chart: task4_learning_curve.png -- Q-learner 0.0->0.95")
    print("Key stat: Qwen-3-235B scores 0.0000 on Task 4 (same as all-refuse)")


if __name__ == "__main__":
    main()
