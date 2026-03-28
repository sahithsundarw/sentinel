"""
Pydantic models for Guardrail Arena OpenEnv environment.
All models are typed per the OpenEnv spec.
"""
from typing import Optional
from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    prompt_id: str
    user_prompt: str
    conversation_history: list[str]
    application_context: str
    user_risk_score: float
    turn_number: int
    prompts_remaining: int
    episode_score_so_far: float


class Action(BaseModel):
    prompt_id: str
    action_type: str  # "allow", "refuse", "modify", "escalate"
    reason: str
    modified_prompt: Optional[str] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        valid = {"allow", "refuse", "modify", "escalate"}
        if v not in valid:
            raise ValueError(f"action_type must be one of {valid}, got '{v}'")
        return v


class Reward(BaseModel):
    score: float
    cumulative_score: float
    feedback: str


class PromptLabel(BaseModel):
    prompt_id: str
    prompt_text: str
    category: str  # "safe", "attack_obvious", "attack_subtle", "attack_injection", "attack_escalation"
    correct_action: str  # "allow", "refuse", "modify", "escalate"
    severity: float  # 0.0 for safe, 0.1–1.0 for attacks
    attack_type: Optional[str] = None  # "jailbreak", "prompt_injection", "social_engineering", "encoding_trick"
    context_dependent: bool = False
    correct_action_by_context: Optional[dict[str, str]] = None


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str


class EnvironmentState(BaseModel):
    task_id: Optional[str]
    current_prompt_index: int
    total_prompts: int
    cumulative_score: float
    done: bool
    initialized: bool
