"""
Pydantic models for Guardrail Arena OpenEnv environment.
All models are typed per the OpenEnv spec.
"""
from typing import Literal, Optional
from pydantic import BaseModel


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
    action_type: Literal["allow", "refuse", "modify", "escalate"]
    reason: str
    modified_prompt: Optional[str] = None


class Reward(BaseModel):
    score: float
    cumulative_score: float
    feedback: str
    breakdown: dict


class PromptLabel(BaseModel):
    prompt_id: str
    prompt_text: str
    category: str  # "safe", "attack_obvious", "attack_subtle", "attack_injection", "attack_escalation"
    correct_action: str  # "allow", "refuse", "modify", "escalate"
    severity: float  # 0.0 for safe, 0.1–1.0 for attacks
    attack_type: Optional[str] = None  # "jailbreak", "prompt_injection", "social_engineering", "encoding_trick"
    context_dependent: bool = False
    correct_action_by_context: Optional[dict[str, str]] = None
    user_risk_score: float = 0.5  # 0.0 = new/unknown user, 1.0 = established trusted user


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
