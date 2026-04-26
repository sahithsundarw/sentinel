"""
Apply comprehensive error-handling fixes to training_colab.ipynb.
Run with: python scripts/fix_notebook.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "training_colab.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

def get_src(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src

def set_src(cell, text):
    cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in text.splitlines()]
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

def find_cell(cell_id=None, contains=None):
    for c in nb["cells"]:
        if cell_id and c.get("id") == cell_id:
            return c
        if contains and contains in get_src(c):
            return c
    return None

changes = []

# ── Fix 1: Cell 4 _post/_get — handle HTTP 410 (Gone / session expired) ──────
c4 = find_cell("cell-4")
if c4:
    src = get_src(c4)
    old = "                r.raise_for_status()\n                return r.json()\n            except (httpx.ConnectError, httpx.TimeoutException) as e:"
    new = "                if r.status_code == 410:\n                    raise RuntimeError(f'Session expired (410). Call env.reset() to start a new episode.')\n                r.raise_for_status()\n                return r.json()\n            except (httpx.ConnectError, httpx.TimeoutException) as e:"
    if old in src and new not in src:
        # Apply to both _post and _get (appears twice)
        src = src.replace(old, new)
        set_src(c4, src)
        changes.append("Cell 4: added HTTP 410 handling to _post/_get")
    else:
        changes.append("Cell 4: 410 handling already present or pattern not matched")

# ── Fix 2: Cell 4 health_check — add retry loop ───────────────────────────────
if c4:
    src = get_src(c4)
    old_hc = """    def health_check(self):
        try:
            r = self.client.get(f"{self.base_url}/health", timeout=30)
            return r.status_code == 200
        except Exception:
            return False"""
    new_hc = """    def health_check(self, retries=6, delay=10):
        for attempt in range(retries):
            try:
                r = self.client.get(f"{self.base_url}/health", timeout=30)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            if attempt < retries - 1:
                print(f"  Health check attempt {attempt+1}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
        return False"""
    if old_hc in src and new_hc not in src:
        src = src.replace(old_hc, new_hc)
        set_src(c4, src)
        changes.append("Cell 4: health_check now retries 6x with 10s delay")
    else:
        changes.append("Cell 4: health_check retry already present or pattern not matched")

# ── Fix 3: Cell 4 post_training_log — surface warnings ───────────────────────
if c4:
    src = get_src(c4)
    old_ptl = "        except Exception:\n            pass"
    new_ptl = "        except Exception as _e:\n            print(f'  Warning: training_log POST failed: {_e}')"
    if old_ptl in src and new_ptl not in src:
        src = src.replace(old_ptl, new_ptl, 1)  # only replace the first occurrence (in post_training_log)
        set_src(c4, src)
        changes.append("Cell 4: post_training_log now prints warnings on failure")
    else:
        changes.append("Cell 4: post_training_log warning already present or pattern not matched")

# ── Fix 4: Cell 5 — add `import torch` at top ────────────────────────────────
c5 = find_cell("cell-5")
if c5:
    src = get_src(c5)
    if "import torch" not in src:
        # Add after the cell header comment block
        src = src.replace(
            "SYSTEM_PROMPT = ",
            "import torch\n\nSYSTEM_PROMPT = ",
            1
        )
        set_src(c5, src)
        changes.append("Cell 5: added `import torch`")
    else:
        changes.append("Cell 5: torch already imported")

# ── Fix 5: Cell 5 evaluate_greedy — guard against None result + grader error ─
if c5:
    src = get_src(c5)
    old_grader = """        # Try grader; if episode not done fall back to normalised cumulative reward
        try:
            score = env.grader()
        except Exception:
            cum = 0.0
            if result and "info" in result:
                cum = result["info"].get("cumulative_reward", result.get("cumulative_score", 0.0))
            score = min(max(float(cum), 0.0), 1.0)"""
    if old_grader not in src:
        # Pattern already different or already fixed — check simpler form
        changes.append("Cell 5: grader try/except already present")
    else:
        changes.append("Cell 5: grader error handling already present")

# ── Fix 6: Cell 6 — wrap get_training_data in try/except ─────────────────────
c6 = find_cell("cell-6")
if c6:
    src = get_src(c6)
    old_gtd = """        examples = env.get_training_data(fmt="chat")
        all_examples.extend(examples)
        print(f"  {task_id:35s}  {len(examples)} raw examples")"""
    new_gtd = """        try:
            examples = env.get_training_data(fmt="chat")
        except Exception as _e:
            print(f"  Warning: could not fetch training data for {task_id}: {_e}")
            examples = []
        all_examples.extend(examples)
        print(f"  {task_id:35s}  {len(examples)} raw examples")"""
    if old_gtd in src and new_gtd not in src:
        src = src.replace(old_gtd, new_gtd)
        set_src(c6, src)
        changes.append("Cell 6: get_training_data wrapped in try/except")
    else:
        changes.append("Cell 6: get_training_data guard already present or pattern not matched")

# ── Fix 7: Cell 6 — guard min() on empty by_action + early exit ──────────────
if c6:
    src = get_src(c6)
    old_min = "    min_count = min(len(v) for v in by_action.values())"
    new_min = """    if not by_action:
        print("  ERROR: No training examples collected. Check environment connectivity.")
        task_metrics = {}
        metrics = {"sft_pre": 0.0, "sft_post": 0.0, "task_metrics": {}}
    else:
        pass  # continue below
    min_count = min((len(v) for v in by_action.values()), default=0)
    if min_count == 0:
        print("  WARNING: One or more action classes has 0 examples; skipping balance step.")"""
    if old_min in src and new_min not in src:
        src = src.replace(old_min, new_min)
        set_src(c6, src)
        changes.append("Cell 6: min() guarded against empty by_action")
    else:
        changes.append("Cell 6: min() guard already present or pattern not matched")

# ── Fix 8: Q-learner cell — wrap env.grader() in _t4_episode ─────────────────
cq = find_cell(contains="_t4_episode")
if cq:
    src = get_src(cq)
    old_g = "    return env.grader()"
    new_g = """    try:
        return env.grader()
    except Exception as _e:
        print(f"  Warning: grader call failed: {_e}")
        return 0.0"""
    if old_g in src and new_g not in src:
        src = src.replace(old_g, new_g, 1)
        set_src(cq, src)
        changes.append("Q-learner: _t4_episode env.grader() wrapped in try/except")
    else:
        changes.append("Q-learner: _t4_episode grader guard already present or not matched")

# ── Fix 9: Q-learner cell — wrap env.grader() in _t4_eval_multi ──────────────
if cq:
    src = get_src(cq)
    old_mg = "        scores.append(env.grader())"
    new_mg = """        try:
            scores.append(env.grader())
        except Exception as _e:
            print(f"  Warning: grader failed for seed {seed}: {_e}")
            scores.append(0.0)"""
    if old_mg in src and new_mg not in src:
        src = src.replace(old_mg, new_mg)
        set_src(cq, src)
        changes.append("Q-learner: _t4_eval_multi grader wrapped in try/except per seed")
    else:
        changes.append("Q-learner: _t4_eval_multi grader guard already present or not matched")

# ── Fix 10: Q-learner cell — safe default for task4_result before MOCK_MODE ──
if cq:
    src = get_src(cq)
    safe_default = 'task4_result = {"zero_shot": 0.0, "post_rl": 0.0, "peak": 0.0, "method": "tabular_q_learning"}\n'
    # Insert safe default before the MOCK_MODE check for task4_result
    mock_check = 'if MOCK_MODE:\n    task4_result = {"zero_shot": 0.0, "post_rl": 0.7493'
    if mock_check in src and safe_default not in src:
        src = src.replace(mock_check, safe_default + mock_check)
        set_src(cq, src)
        changes.append("Q-learner: added safe task4_result default before MOCK_MODE branch")
    else:
        changes.append("Q-learner: task4_result safe default already present or not matched")

# ── Fix 11: Cell 9 — use CHECKPOINT_DIR for results path ─────────────────────
c9 = find_cell("8d0da462")
if c9:
    src = get_src(c9)
    old_path = 'with open("results/notebook_training_results.json", "w") as f:'
    new_path = 'results_dir = os.path.join(CHECKPOINT_DIR, "results")\nos.makedirs(results_dir, exist_ok=True)\nwith open(os.path.join(results_dir, "notebook_training_results.json"), "w") as f:'
    if old_path in src and new_path not in src:
        src = src.replace('os.makedirs("results", exist_ok=True)\n', '')
        src = src.replace(old_path, new_path)
        # Fix the print statement too
        src = src.replace(
            'print("Results saved: results/notebook_training_results.json")',
            'print(f"Results saved: {results_dir}/notebook_training_results.json")'
        )
        set_src(c9, src)
        changes.append("Cell 9: results path now uses CHECKPOINT_DIR")
    else:
        changes.append("Cell 9: CHECKPOINT_DIR path already present or not matched")

# ── Fix 12: Add wandb.finish() at the very end of Cell 9 ─────────────────────
if c9:
    src = get_src(c9)
    wandb_finish = """
# Finish W&B run if active
try:
    import wandb as _wandb
    if _wandb.run is not None:
        _wandb.finish()
        print("W&B run finished.")
except Exception:
    pass"""
    if "wandb.finish()" not in src and "_wandb.finish()" not in src:
        src = src + wandb_finish
        set_src(c9, src)
        changes.append("Cell 9: added wandb.finish() at end")
    else:
        changes.append("Cell 9: wandb.finish() already present")

# ── Save ──────────────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"Applied {len(changes)} changes to {NB.name}:")
for ch in changes:
    print(f"  - {ch}")
print("Done.")
