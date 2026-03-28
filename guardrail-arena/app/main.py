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
import os
import json
import subprocess
import sys
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

env = GuardrailEnvironment()


@app.get("/")
async def root():
    """Health check."""
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
    Trigger baseline inference script (baseline.py) and return scores for all tasks.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY environment variable is not set.",
        )
    try:
        result = subprocess.run(
            [sys.executable, "baseline.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "OPENAI_API_KEY": api_key},
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Baseline script failed:\n{result.stderr}",
            )
        # Parse the final JSON scores from stdout (last line)
        output_lines = result.stdout.strip().split("\n")
        scores_line = next(
            (l for l in reversed(output_lines) if l.strip().startswith("{")),
            None,
        )
        scores = json.loads(scores_line) if scores_line else {}
        return {"scores": scores, "output": result.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Baseline script timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grader")
async def grader():
    """Return grader score for the current completed episode."""
    if not env.is_done():
        raise HTTPException(status_code=400, detail="Episode not complete. Finish all steps first.")
    score = env.get_grader_score()
    return {"task_id": env.current_task_id, "score": score}
