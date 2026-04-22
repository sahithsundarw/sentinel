"""generate_charts.py — Produce reward curve visualizations for Sentinel Round 2.

Generates Charts D, E, F, and a 4-panel combined PNG from baseline data.
If results/local_training_results.json exists, uses real training data for LLM panel.

Usage:
    python generate_charts.py
    python generate_charts.py --output-dir ./results
    python generate_charts.py --no-llm-panel

Output files:
    results/task4_learning_curve.png      — Chart E: Task 4 Q-learner (0.0 → 0.95)
    results/score_comparison.png          — Chart D: Before/After bar chart
    results/model_comparison.png          — Chart F: All-model comparison
    results/reward_curves_task4.png       — Alias for task4_learning_curve.png
    results/reward_curves_llm.png         — LLM training curve (real or mock)
    results/reward_curves_combined.png    — 2×2 four-panel combined (1600×900px)
"""
import argparse
import os
import json
import shutil

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Run: pip install matplotlib numpy")



# ── Load real training data if available ─────────────────────────────────────

def _load_llm_results(output_dir: str) -> dict | None:
    """Load real LLM training results if they exist."""
    for fname in ("local_training_results.json", "notebook_training_results.json"):
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return data
            except Exception:
                pass
    return None


def _mock_llm_curve(zero_shot: float = 0.5428, target: float = 0.7350,
                     episodes: int = 20) -> tuple[list, list]:
    """Generate a mock LLM training curve for visualization."""
    import math
    import random
    rng = random.Random(42)
    ep_list = list(range(episodes + 1))
    scores = [zero_shot]
    for ep in range(1, episodes + 1):
        t = ep / episodes
        sigmoid = 1.0 / (1.0 + math.exp(-8 * (t - 0.5)))
        base = zero_shot + (target - zero_shot) * sigmoid
        noise = rng.gauss(0, 0.012)
        scores.append(round(max(0.0, min(1.0, base + noise)), 4))
    return ep_list, scores


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
    ax.set_title("Sentinel — Score Comparison Across All Tasks & Agents",
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

    ax.set_title("Sentinel — Model × Task Performance Matrix",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, os.path.join(output_dir, "model_comparison.png"))
    plt.close(fig)


# ── Chart LLM: LLM Training Curve (real or mock) ─────────────────────────────

def chart_llm_training_curve(output_dir: str, llm_results: dict | None) -> None:
    """LLM zero-shot → trained, using real data if available else mock."""
    if llm_results:
        ep_list = llm_results.get("episodes", [])
        scores = llm_results.get("scores", [])
        zero_shot = llm_results.get("zero_shot_score", 0.5428)
        final = llm_results.get("final_score", scores[-1] if scores else 0.72)
        task_id = llm_results.get("task_id", "basic_threat_detection")
        method = llm_results.get("method", "sft")
        label = f"Llama-3.1-8B ({method.upper()})"
        data_source = "real data"
    else:
        zero_shot = 0.5428
        ep_list, scores = _mock_llm_curve(zero_shot, 0.7350, 20)
        final = scores[-1]
        task_id = "basic_threat_detection"
        label = "Llama-3.1-8B (SFT — simulated)"
        data_source = "simulated"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0f1117")

    ax.plot(ep_list, scores, "g-o", linewidth=2.5, markersize=6,
            color="#22c55e", label=label)
    ax.fill_between(ep_list, 0, scores, alpha=0.08, color="#22c55e")

    ax.axhline(y=zero_shot, color="#6366f1", linestyle="--", alpha=0.8, linewidth=1.5,
               label=f"Zero-shot baseline ({zero_shot:.4f})")
    ax.axhline(y=BASELINES["all_allow"][0], color="#ef4444", linestyle="--",
               alpha=0.6, linewidth=1.2, label=f"All-Allow ({BASELINES['all_allow'][0]:.4f})")

    ax.annotate(f"Zero-shot\n{zero_shot:.4f}",
                xy=(0, zero_shot), xytext=(3, zero_shot - 0.08),
                fontsize=9, color="#6366f1",
                arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0830", edgecolor="#6366f1"))

    ax.annotate(f"After training\n{final:.4f}",
                xy=(ep_list[-1], final), xytext=(ep_list[-1] - 6, final + 0.08),
                fontsize=9, color="#22c55e",
                arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#031a0d", edgecolor="#22c55e"))

    ax.set_xlabel("Training Episode", color="#9ca3af", fontsize=12)
    ax.set_ylabel("Grader Score (0.0 – 1.0)", color="#9ca3af", fontsize=12)
    ax.set_title(f"LLM Training: {task_id}\n"
                 f'"Zero-shot: {zero_shot:.4f}  →  Trained: {final:.4f}  '
                 f'(+{final - zero_shot:.4f})"',
                 color="#ffffff", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(colors="#9ca3af")
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#374151")
    ax.legend(fontsize=9, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              loc="lower right")

    if data_source == "simulated":
        ax.text(0.5, 0.97, "⚠ Simulated data — run with --mock or live training for real curve",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#f59e0b", alpha=0.8)

    _save(fig, os.path.join(output_dir, "reward_curves_llm.png"))
    plt.close(fig)


# ── Chart Combined: 4-panel 1600×900 presentation chart ──────────────────────

def chart_combined_4panel(output_dir: str, llm_results: dict | None) -> None:
    """2×2 four-panel combined chart, exactly 1600×900px at 150 DPI."""
    fig, axes = plt.subplots(2, 2, figsize=(1600 / 150, 900 / 150))
    fig.patch.set_facecolor("#0a0a0a")
    fig.subplots_adjust(hspace=0.42, wspace=0.38, left=0.07, right=0.97,
                        top=0.93, bottom=0.10)

    # ── Panel 1 (top-left): Task 4 Q-learner curve ───────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor("#0f1117")
    ax.plot(TASK4_EPISODES, TASK4_SCORES, "b-o", linewidth=2, markersize=5,
            color="#3b82f6", label="Tabular Q-learner")
    ax.fill_between(TASK4_EPISODES, 0, TASK4_SCORES, alpha=0.08, color="#3b82f6")
    ax.axhline(y=TASK4_ALL_ALLOW, color="#ef4444", linestyle="--", alpha=0.7,
               linewidth=1.2, label=f"All-Allow ({TASK4_ALL_ALLOW:.2f})")
    ax.axhline(y=TASK4_ALL_REFUSE, color="#f97316", linestyle="--", alpha=0.7,
               linewidth=1.2, label=f"All-Refuse ({TASK4_ALL_REFUSE:.2f})")
    ax.text(16, 0.97, "0.9540", fontsize=10, color="#22c55e", fontweight="bold")
    ax.text(0.5, 0.08, "Qwen-3-235B: 0.0000", fontsize=8, color="#ef4444",
            transform=ax.transAxes, ha="center")
    ax.set_title("Task 4: Q-learner (0.0 → 0.95)", color="#f0f0f0", fontsize=10, fontweight="bold")
    ax.set_xlabel("Episode", color="#9ca3af", fontsize=8)
    ax.set_ylabel("Score", color="#9ca3af", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.12, color="#374151")
    ax.legend(fontsize=7, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              loc="upper left")

    # ── Panel 2 (top-right): LLM training curve ──────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor("#0f1117")

    if llm_results:
        ep_list = llm_results.get("episodes", [])
        scores = llm_results.get("scores", [])
        zero_shot = llm_results.get("zero_shot_score", 0.5428)
        final = llm_results.get("final_score", scores[-1] if scores else 0.72)
        method = llm_results.get("method", "sft")
        label = f"Llama-3.1-8B ({method.upper()})"
        simulated = False
    else:
        zero_shot = 0.5428
        ep_list, scores = _mock_llm_curve(zero_shot, 0.7350, 20)
        final = scores[-1]
        label = "Llama-3.1-8B (simulated)"
        simulated = True

    ax.plot(ep_list, scores, "g-o", linewidth=2, markersize=5,
            color="#22c55e", label=label)
    ax.fill_between(ep_list, 0, scores, alpha=0.08, color="#22c55e")
    ax.axhline(y=zero_shot, color="#6366f1", linestyle="--", alpha=0.7, linewidth=1.2,
               label=f"Zero-shot ({zero_shot:.4f})")
    ax.set_title(f"LLM Training: {zero_shot:.4f} → {final:.4f}", color="#f0f0f0",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Episode", color="#9ca3af", fontsize=8)
    ax.set_ylabel("Score", color="#9ca3af", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.12, color="#374151")
    ax.legend(fontsize=7, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              loc="lower right")
    if simulated:
        ax.text(0.5, 0.97, "simulated", transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#f59e0b", style="italic")

    # ── Panel 3 (bottom-left): Before/After bar chart per task ───────────────
    ax = axes[1, 0]
    ax.set_facecolor("#0f1117")

    x = np.arange(4)
    width = 0.28
    zs_scores = BASELINES["llama_8b_zs"]
    panel3_simulated = not bool(llm_results)
    tr_scores = [
        llm_results["final_score"] if (llm_results and i == 0) else BASELINES["llama_8b_zs"][i]
        for i in range(4)
    ] if llm_results else [s * 1.35 for s in BASELINES["llama_8b_zs"]]

    ax.bar(x - width / 2, zs_scores, width, label="Zero-shot", color="#6366f1", alpha=0.85)
    trained_label = "Trained (SFT)" if llm_results else "Trained (SFT — projected)"
    ax.bar(x + width / 2, tr_scores, width, label=trained_label, color="#22c55e" if not panel3_simulated else "#f59e0b", alpha=0.85)

    for i, (zs, tr) in enumerate(zip(zs_scores, tr_scores)):
        ax.text(i - width / 2, zs + 0.01, f"{zs:.2f}", ha="center", fontsize=6.5, color="#d1d5db")
        ax.text(i + width / 2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=6.5, color="#d1d5db")

    short_labels = ["Task 1\nbasic", "Task 2\ncontext", "Task 3\nmulti", "Task 4\nadversary"]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, color="#9ca3af", fontsize=7)
    ax.set_ylabel("Score", color="#9ca3af", fontsize=8)
    title3 = "Before vs After Training" if not panel3_simulated else "Before vs After Training (projected)"
    ax.set_title(title3, color="#f0f0f0" if not panel3_simulated else "#f59e0b", fontsize=10, fontweight="bold")
    if panel3_simulated:
        ax.text(0.5, 0.97, "PROJECTED — run real training for actual values",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#f59e0b", style="italic")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.12, color="#374151")
    ax.legend(fontsize=7, facecolor="#111827", edgecolor="#374151", labelcolor="#d1d5db",
              loc="upper right")

    # ── Panel 4 (bottom-right): Multi-model comparison ───────────────────────
    ax = axes[1, 1]
    ax.set_facecolor("#0f1117")

    # Task 1 scores only for clarity
    models = ["Oracle", "Qwen-3-235B", "GPT-4o-mini", "Llama-8B ZS", "All-Allow", "All-Refuse"]
    task1_scores = [
        BASELINES["oracle"][0],
        BASELINES["qwen_235b"][0],
        BASELINES["gpt4o_mini"][0],
        BASELINES["llama_8b_zs"][0],
        BASELINES["all_allow"][0],
        BASELINES["all_refuse"][0],
    ]
    if llm_results:
        models.insert(4, "Llama-8B Trained")
        task1_scores.insert(4, llm_results.get("final_score", 0.72))

    colors = ["#22c55e", "#8b5cf6", "#3b82f6", "#6366f1"]
    if llm_results:
        colors.insert(4, "#34d399")
    colors += ["#ef4444", "#f97316"]

    bars = ax.barh(models, task1_scores, color=colors[:len(models)], alpha=0.85)
    for bar, val in zip(bars, task1_scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=6.5, color="#d1d5db")

    ax.set_xlabel("Task 1 Score", color="#9ca3af", fontsize=8)
    ax.set_title("Model Comparison (Task 1)", color="#f0f0f0", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.12, color="#374151")

    # Title
    fig.suptitle("Sentinel — Training Results", color="#ffffff",
                 fontsize=14, fontweight="bold", y=0.98)

    path = os.path.join(output_dir, "reward_curves_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"CHART SAVED: {path}")


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
    parser = argparse.ArgumentParser(description="Generate Sentinel reward curve visualizations")
    parser.add_argument("--output-dir", default="./results", help="Output directory for charts")
    parser.add_argument("--dpi", type=int, default=300, help="Chart DPI (default: 300)")
    parser.add_argument("--no-llm-panel", action="store_true",
                        help="Skip LLM training curve and combined chart")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: Install matplotlib and numpy first:")
        print("  pip install matplotlib numpy")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load real LLM training data if available
    llm_results = _load_llm_results(args.output_dir)
    if llm_results:
        print(f"  Using real LLM training data from {args.output_dir}/")
        print(f"  Task: {llm_results.get('task_id')}  Method: {llm_results.get('method')}")
        print(f"  Zero-shot: {llm_results.get('zero_shot_score'):.4f}  "
              f"Final: {llm_results.get('final_score'):.4f}")
    else:
        print("  No local_training_results.json found — using simulated LLM curve")

    print(f"\nGenerating charts -> {args.output_dir}/")
    print()

    chart_e_task4_learning_curve(args.output_dir)
    chart_d_score_comparison(args.output_dir)
    chart_f_model_comparison(args.output_dir)

    charts = [
        "task4_learning_curve.png",
        "score_comparison.png",
        "model_comparison.png",
    ]

    if not args.no_llm_panel:
        chart_llm_training_curve(args.output_dir, llm_results)
        chart_combined_4panel(args.output_dir, llm_results)
        charts += ["reward_curves_llm.png", "reward_curves_combined.png"]

        # Create alias: reward_curves_task4.png → task4_learning_curve.png
        src = os.path.join(args.output_dir, "task4_learning_curve.png")
        dst = os.path.join(args.output_dir, "reward_curves_task4.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            charts.append("reward_curves_task4.png")

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
