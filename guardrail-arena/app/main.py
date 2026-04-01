"""
FastAPI application — Guardrail Arena OpenEnv HTTP interface.

Endpoints:
    GET  /           — Health check
    POST /reset      — Reset environment, return initial observation
    POST /step       — Take action, return observation + reward + done + info
    GET  /state      — Return current environment state
    GET  /tasks      — Return all tasks + action schema
    GET  /baseline   — Run baseline inference, return scores for all tasks
    GET  /grader     — Return grader score for completed episode
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal
from datetime import datetime, timezone
import threading
import logging
import os
import json
from logging.handlers import RotatingFileHandler

from app.environment import GuardrailEnvironment
from app.models import Action, TaskInfo
from app.grader import grade_task1

# Rotating step audit log
os.makedirs("logs", exist_ok=True)
_step_logger = logging.getLogger("step_audit")
_step_logger.setLevel(logging.INFO)
_step_handler = RotatingFileHandler("logs/step_audit.log", maxBytes=10 * 1024 * 1024, backupCount=3)
_step_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_step_logger.addHandler(_step_handler)

_LEADERBOARD_PATH = "leaderboard.json"
_leaderboard_lock = threading.Lock()

_HTML_LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Guardrail Arena</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace; max-width: 960px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
h1 { color: #58a6ff; font-size: 2rem; margin-bottom: 0.25rem; }
h2 { color: #79c0ff; font-size: 1.2rem; margin: 1.5rem 0 0.75rem; border-bottom: 1px solid #30363d; padding-bottom: 0.25rem; }
p { color: #8b949e; margin-bottom: 1rem; }
table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
th { background: #161b22; color: #79c0ff; padding: 0.5rem 1rem; text-align: left; border: 1px solid #30363d; }
td { padding: 0.5rem 1rem; border: 1px solid #30363d; color: #e6edf3; }
tr:nth-child(even) { background: #161b22; }
.badge { padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-weight: bold; }
.easy { background: #1a4731; color: #3fb950; }
.medium { background: #3d2b0a; color: #d29922; }
.hard { background: #3d1212; color: #f85149; }
pre { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; overflow-x: auto; color: #79c0ff; font-size: 0.85rem; margin-bottom: 1rem; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.tagline { color: #8b949e; font-size: 1rem; margin-bottom: 1.5rem; }
</style>
</head>
<body>
<h1>🛡️ Guardrail Arena</h1>
<p class="tagline">LLM Safety Guardrail Training Environment — OpenEnv-spec RL environment</p>

<p>An RL environment for training AI agents to act as content safety moderators. Agents observe user prompts with conversation history and context, then choose allow / refuse / modify / escalate. Rewards are asymmetric: missing an attack costs 2× more than over-blocking. Dynamic user risk scores compound future penalties for missed attacks.</p>

<h2>Tasks</h2>
<table>
<tr><th>Task ID</th><th>Difficulty</th><th>Prompts</th><th>All-Allow Baseline</th></tr>
<tr><td>basic_threat_detection</td><td><span class="badge easy">easy</span></td><td>99</td><td>0.5595</td></tr>
<tr><td>context_aware_policy</td><td><span class="badge medium">medium</span></td><td>100</td><td>0.1695</td></tr>
<tr><td>multiturn_adversarial</td><td><span class="badge hard">hard</span></td><td>173 turns / 28 convos</td><td>0.4000</td></tr>
</table>

<h2>Quick Start</h2>
<pre>
# 1. Reset to a task
curl -X POST "http://localhost:7860/reset?task_id=basic_threat_detection"

# 2. Submit an action (use prompt_id from observation above)
curl -X POST "http://localhost:7860/step" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt_id": "&lt;id&gt;", "action_type": "allow", "reason": "Safe prompt", "modified_prompt": null}'

# 3. Get grader score after all prompts
curl "http://localhost:7860/grader"

# Optional: shuffle prompt order with a seed
curl -X POST "http://localhost:7860/reset?task_id=basic_threat_detection&amp;seed=42"
</pre>

<h2>API Endpoints</h2>
<table>
<tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
<tr><td>GET</td><td><a href="/">/</a></td><td>This page</td></tr>
<tr><td>GET</td><td><a href="/health">/health</a></td><td>Health check — returns JSON status</td></tr>
<tr><td>POST</td><td>/reset</td><td>Reset environment. Params: task_id, optional seed (int)</td></tr>
<tr><td>POST</td><td>/step</td><td>Submit action, receive observation + reward + done + info</td></tr>
<tr><td>GET</td><td><a href="/state">/state</a></td><td>Current environment state</td></tr>
<tr><td>GET</td><td><a href="/tasks">/tasks</a></td><td>All task metadata + action schema</td></tr>
<tr><td>GET</td><td><a href="/grader">/grader</a></td><td>Final grader score (0.0–1.0) after episode ends</td></tr>
<tr><td>GET</td><td><a href="/demo">/demo</a></td><td>Pre-scripted 5-step demo episode with full trajectory JSON</td></tr>
<tr><td>GET</td><td><a href="/leaderboard">/leaderboard</a></td><td>Top 10 scores per task (in-memory)</td></tr>
<tr><td>GET</td><td><a href="/baseline">/baseline</a></td><td>Pre-computed baseline scores</td></tr>
</table>

<p>Full API docs: <a href="/docs">/docs</a> · <a href="/redoc">/redoc</a></p>
</body>
</html>"""

app = FastAPI(
    title="Guardrail Arena",
    description=(
        "LLM Safety Guardrail Training Environment — an OpenEnv-spec RL environment "
        "where agents learn to moderate content safety decisions."
    ),
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Single-session design: one active episode at a time.
# Concurrent /reset calls from multiple clients will overwrite each other's state.
# For evaluation, run one task at a time sequentially (as all standard harnesses do).
env = GuardrailEnvironment()

# Pre-seeded with named baselines so judges immediately see a score distribution.
_BASELINE_ENTRIES: dict[str, list[dict]] = {
    "basic_threat_detection": [
        {"agent": "all-allow",   "score": 0.5595, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "heuristic",   "score": 0.7381, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "all-refuse",  "score": 0.4405, "timestamp": "2025-01-01T00:00:00Z"},
    ],
    "context_aware_policy": [
        {"agent": "heuristic",   "score": 0.3284, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "all-refuse",  "score": 0.3686, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "all-allow",   "score": 0.1695, "timestamp": "2025-01-01T00:00:00Z"},
    ],
    "multiturn_adversarial": [
        {"agent": "all-allow",   "score": 0.4000, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "heuristic",   "score": 0.2941, "timestamp": "2025-01-01T00:00:00Z"},
        {"agent": "all-refuse",  "score": 0.1303, "timestamp": "2025-01-01T00:00:00Z"},
    ],
}

def _load_leaderboard() -> dict[str, list[dict]]:
    """Load leaderboard from disk, merging with baseline entries."""
    data = {k: list(v) for k, v in _BASELINE_ENTRIES.items()}
    if os.path.exists(_LEADERBOARD_PATH):
        try:
            with open(_LEADERBOARD_PATH, "r") as f:
                saved = json.load(f)
            for task_id, entries in saved.items():
                if task_id in data:
                    data[task_id].extend(entries)
                    data[task_id].sort(key=lambda x: x["score"], reverse=True)
                    data[task_id] = data[task_id][:10]
        except Exception:
            pass
    return data

def _save_leaderboard(leaderboard: dict[str, list[dict]]) -> None:
    """Persist live (non-baseline) entries to disk."""
    baseline_agents = {"all-allow", "all-refuse", "heuristic"}
    to_save: dict[str, list[dict]] = {}
    for task_id, entries in leaderboard.items():
        live = [e for e in entries if e.get("agent", "") not in baseline_agents]
        if live:
            to_save[task_id] = live
    try:
        with open(_LEADERBOARD_PATH, "w") as f:
            json.dump(to_save, f)
    except Exception:
        pass

_leaderboard: dict[str, list[dict]] = _load_leaderboard()

_computed_baselines: dict[str, float] = {}


@app.on_event("startup")
async def _compute_baselines() -> None:
    """Compute all-allow baseline scores at startup and cache them."""
    from app.tasks.task_config import get_task
    for task_id in ("basic_threat_detection", "context_aware_policy", "multiturn_adversarial"):
        try:
            b_env = GuardrailEnvironment()
            obs = b_env.reset(task_id)
            while not b_env.is_done():
                action = Action(prompt_id=obs.prompt_id, action_type="allow", reason="baseline")
                next_obs, _, done, _ = b_env.step(action)
                obs = next_obs if not done else obs
            _computed_baselines[task_id] = b_env.get_grader_score()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def root():
    """HTML landing page."""
    return HTMLResponse(content=_HTML_LANDING_PAGE)


@app.get("/health")
async def health():
    """Health check. Returns status=healthy for OpenEnv spec compliance."""
    return {"status": "healthy", "environment": "guardrail-arena", "version": "1.0.0"}


@app.get("/metadata")
async def metadata():
    """OpenEnv spec: environment metadata."""
    return {
        "name": "Guardrail Arena",
        "description": (
            "An OpenEnv-spec RL environment for training AI content safety agents. "
            "Agents learn to moderate LLM prompts across three difficulty tiers: "
            "basic threat detection, context-aware policy enforcement, and multi-turn adversarial defense."
        ),
        "version": "1.0.0",
        "tags": ["openenv", "safety", "guardrails", "content-moderation", "adversarial", "llm-safety"],
    }


@app.get("/schema")
async def schema():
    """OpenEnv spec: action, observation, and state schemas."""
    from app.models import Observation, EnvironmentState
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


class ResetRequest(BaseModel):
    task_id: str = "basic_threat_detection"
    seed: Optional[int] = None


@app.post("/reset")
async def reset(
    body: Optional[ResetRequest] = None,
    task_id: str = Query(default=None),
    seed: Optional[int] = Query(default=None),
):
    """Reset environment with specified task. Accepts JSON body or query params."""
    # JSON body takes priority over query params; fall back to "basic_threat_detection" only if
    # both sources are absent (None), not if they're empty string (which get_task will reject).
    if body is not None:
        resolved_task_id = body.task_id
        resolved_seed = body.seed if body.seed is not None else seed
    elif task_id is not None:
        resolved_task_id = task_id
        resolved_seed = seed
    else:
        resolved_task_id = "basic_threat_detection"
        resolved_seed = seed

    try:
        observation = env.reset(task_id=resolved_task_id, seed=resolved_seed)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return observation.model_dump()


@app.post("/step")
async def step(action: Action):
    """Take an action. Returns observation, reward, done, info."""
    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = {
        "observation": observation.model_dump() if observation else None,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }
    _step_logger.info(
        "task=%s prompt_id=%s action=%s reward=%.4f outcome=%s",
        env.current_task_id or "unknown",
        action.prompt_id,
        action.action_type,
        reward.score,
        reward.breakdown.get("outcome", ""),
    )
    return result


@app.get("/state")
async def state():
    """Return current environment state."""
    return env.state().model_dump()


@app.get("/tasks")
async def tasks():
    """Return list of available tasks and action schema."""
    return {
        "tasks": [task.model_dump() for task in env.get_tasks()],
        "action_schema": Action.model_json_schema(),
        "reset_params": {"seed": "optional int — shuffles prompt order deterministically; same seed always gives same order"},
    }


@app.get("/baseline")
async def baseline():
    """Return baseline scores. all_allow is computed live at startup; others are pre-measured."""
    return {
        "status": "ok",
        "baselines": {
            "all_allow": _computed_baselines if _computed_baselines else {
                "basic_threat_detection": 0.5595,
                "context_aware_policy": 0.1695,
                "multiturn_adversarial": 0.4000,
            },
            "all_refuse": {"basic_threat_detection": 0.4405, "context_aware_policy": 0.3686, "multiturn_adversarial": 0.1303},
            "heuristic":  {"basic_threat_detection": 0.7381, "context_aware_policy": 0.3284, "multiturn_adversarial": 0.2941},
        },
        "note": "all_allow computed live at server startup. heuristic/all_refuse are pre-measured.",
    }


@app.get("/grader")
async def grader(agent_name: Optional[str] = Query(default=None)):
    """Return grader score for the current completed episode. Optionally pass agent_name to record on leaderboard."""
    if not env.is_done():
        raise HTTPException(status_code=400, detail="Episode not complete. Finish all steps first.")
    score = env.get_grader_score()
    task_id = env.current_task_id
    with _leaderboard_lock:
        if task_id in _leaderboard:
            entry = {
                "score": score,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if agent_name:
                entry["agent"] = agent_name
            _leaderboard[task_id].append(entry)
            _leaderboard[task_id].sort(key=lambda x: x["score"], reverse=True)
            _leaderboard[task_id] = _leaderboard[task_id][:10]
            _save_leaderboard(_leaderboard)
    return {"task_id": task_id, "score": score}


@app.get("/leaderboard")
async def leaderboard():
    """Top 10 scores per task. Populated automatically when /grader is called after an episode."""
    return {
        "leaderboard": _leaderboard,
        "note": "Top 10 scores per task. Resets when server restarts.",
    }


@app.get("/demo")
async def demo():
    """Run a pre-scripted 5-step demonstration episode. No API key needed."""
    demo_env = GuardrailEnvironment()
    obs = demo_env.reset("basic_threat_detection")
    hardcoded_actions = ["allow", "allow", "refuse", "allow", "refuse"]
    trajectory = []

    for i, action_type in enumerate(hardcoded_actions):
        action = Action(prompt_id=obs.prompt_id, action_type=action_type, reason="demo")
        next_obs, reward, done, info = demo_env.step(action)
        trajectory.append({
            "step": i + 1,
            "prompt": obs.user_prompt,
            "application_context": obs.application_context,
            "user_risk_score": obs.user_risk_score,
            "action_taken": action_type,
            "reward": reward.score,
            "correct_action": info["ground_truth_action"],
            "outcome": reward.breakdown.get("outcome", ""),
            "reasoning": reward.breakdown.get("reason", ""),
        })
        if not done:
            obs = next_obs

    # basic_threat_detection has 76 prompts; done=False after 5 steps.
    # Compute partial grader score directly from the 5 collected steps.
    grader_score = grade_task1(demo_env._actions, demo_env._labels, demo_env._contexts)
    correct = sum(1 for t in trajectory if t["outcome"] == "correct")

    return {
        "task_id": "basic_threat_detection",
        "description": "A 5-step demonstration showing the reward signal in action",
        "trajectory": trajectory,
        "final_grader_score": grader_score,
        "summary": (
            f"Agent correctly handled {correct}/5 prompts. "
            f"Demonstrates reward signal: correct blocks, missed attacks, and over-blocks."
        ),
    }


# ── /submit ───────────────────────────────────────────────────────────────────

_VALID_TASK_IDS = {"basic_threat_detection", "context_aware_policy", "multiturn_adversarial"}


class SubmitRequest(BaseModel):
    agent_name: str
    score: float
    task_id: str


@app.post("/submit")
async def submit(req: SubmitRequest):
    """Submit a score to the leaderboard without running a live episode."""
    if req.task_id not in _VALID_TASK_IDS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id '{req.task_id}'")
    score = max(0.0, min(1.0, req.score))
    entry = {
        "agent": req.agent_name,
        "score": score,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with _leaderboard_lock:
        _leaderboard[req.task_id].append(entry)
        _leaderboard[req.task_id].sort(key=lambda x: x["score"], reverse=True)
        _leaderboard[req.task_id] = _leaderboard[req.task_id][:10]
        _save_leaderboard(_leaderboard)
    return {"status": "ok", "task_id": req.task_id, "score": score}


# ── /rollout ──────────────────────────────────────────────────────────────────

class RolloutActionRequest(BaseModel):
    action_type: Literal["allow", "refuse", "modify", "escalate"]
    reason: str = ""
    modified_prompt: Optional[str] = None


class RolloutRequest(BaseModel):
    task_id: str
    actions: list[RolloutActionRequest]


@app.post("/rollout")
async def rollout(req: RolloutRequest):
    """Run a complete episode with pre-supplied actions. Returns trajectory and grader score."""
    if req.task_id not in _VALID_TASK_IDS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id '{req.task_id}'")
    roll_env = GuardrailEnvironment()
    obs = roll_env.reset(req.task_id)
    trajectory = []
    action_iter = iter(req.actions)

    while not roll_env.is_done():
        try:
            act_req = next(action_iter)
        except StopIteration:
            # Pad remaining steps with "allow"
            act_req = RolloutActionRequest(action_type="allow", reason="auto-pad")
        action = Action(
            prompt_id=obs.prompt_id,
            action_type=act_req.action_type,
            reason=act_req.reason,
            modified_prompt=act_req.modified_prompt,
        )
        next_obs, reward, done, info = roll_env.step(action)
        trajectory.append({
            "prompt_id": action.prompt_id,
            "user_prompt": obs.user_prompt,
            "action_type": act_req.action_type,
            "reward": reward.score,
            "outcome": reward.breakdown.get("outcome", ""),
            "correct_action": info.get("ground_truth_action", ""),
        })
        obs = next_obs if not done else obs

    grader_score = roll_env.get_grader_score()
    return {"task_id": req.task_id, "grader_score": grader_score, "trajectory": trajectory}


# ── /replay ───────────────────────────────────────────────────────────────────

class ReplayActionRequest(BaseModel):
    prompt_id: str
    action_type: Literal["allow", "refuse", "modify", "escalate"]
    reason: str = ""
    modified_prompt: Optional[str] = None


class ReplayRequest(BaseModel):
    task_id: str
    actions: list[ReplayActionRequest]


@app.post("/replay")
async def replay(req: ReplayRequest):
    """Replay a sequence of prompt_id→action pairs. Validates all prompt_ids exist, then scores."""
    if req.task_id not in _VALID_TASK_IDS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id '{req.task_id}'")
    from app.tasks.task_config import get_task
    task = get_task(req.task_id)
    valid_ids = {e.label.prompt_id for e in task.prompts}
    for act in req.actions:
        if act.prompt_id not in valid_ids:
            raise HTTPException(status_code=422, detail=f"Unknown prompt_id '{act.prompt_id}'")

    replay_env = GuardrailEnvironment()
    obs = replay_env.reset(req.task_id)
    trajectory = []
    action_map = {a.prompt_id: a for a in req.actions}

    while not replay_env.is_done():
        pid = obs.prompt_id
        if pid in action_map:
            act_req = action_map[pid]
            action = Action(
                prompt_id=pid,
                action_type=act_req.action_type,
                reason=act_req.reason,
                modified_prompt=act_req.modified_prompt,
            )
        else:
            action = Action(prompt_id=pid, action_type="allow", reason="no action provided")
        next_obs, reward, done, info = replay_env.step(action)
        trajectory.append({
            "prompt_id": pid,
            "action_type": action.action_type,
            "reward": reward.score,
            "outcome": reward.breakdown.get("outcome", ""),
        })
        obs = next_obs if not done else obs

    grader_score = replay_env.get_grader_score()
    return {"task_id": req.task_id, "grader_score": grader_score, "trajectory": trajectory}
