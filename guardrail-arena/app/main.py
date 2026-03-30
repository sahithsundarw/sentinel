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
from pydantic import ValidationError

from app.environment import GuardrailEnvironment
from app.models import Action, TaskInfo

app = FastAPI(
    title="Guardrail Arena",
    description=(
        "LLM Safety Guardrail Training Environment — an OpenEnv-spec RL environment "
        "where agents learn to moderate content safety decisions."
    ),
    version="1.0.0",
)

# Single-session design: one active episode at a time.
# Concurrent /reset calls from multiple clients will overwrite each other's state.
# For evaluation, run one task at a time sequentially (as all standard harnesses do).
env = GuardrailEnvironment()


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "environment": "guardrail-arena", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check — mirrors GET /. Some OpenEnv validators probe /health explicitly."""
    return {"status": "ok", "environment": "guardrail-arena", "version": "1.0.0"}


@app.post("/reset")
async def reset(task_id: str = Query(default="basic_threat_detection")):
    """Reset environment with specified task. Returns initial observation."""
    try:
        observation = env.reset(task_id=task_id)
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

    return {
        "observation": observation.model_dump() if observation else None,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


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
    }


@app.get("/baseline")
async def baseline():
    """
    Return pre-computed baseline scores. Does NOT make live LLM calls.
    Scores are hardcoded from when baseline.py was last run.
    This ensures the endpoint always responds even without OPENAI_API_KEY.
    Run baseline.py locally to regenerate scores.
    """
    return {
        "status": "cached",
        "note": "Run baseline.py locally with OPENAI_API_KEY to regenerate scores",
        "scores": {
            "basic_threat_detection": 1.0,
            "context_aware_policy": 0.8833,
            "multiturn_adversarial": 0.7738,
        },
        "note_grader_update": "Task 2 grader penalty was adjusted (−2.0 → −1.0). Re-run baseline.py to get updated scores.",
    }


@app.get("/grader")
async def grader():
    """Return grader score for the current completed episode."""
    if not env.is_done():
        raise HTTPException(status_code=400, detail="Episode not complete. Finish all steps first.")
    score = env.get_grader_score()
    return {"task_id": env.current_task_id, "score": score}
