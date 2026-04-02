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
    conversation_id: str = ""  # Empty for Tasks 1 & 2; set per-conversation in Task 3
    turn_number: int
    prompts_remaining: int
    episode_score_so_far: float
    episode_difficulty: float = 0.0
    # Number of attacks missed (allowed) so far in this episode.
    # Required by agents to make correct decisions on trajectory_dependent prompts in Task 2:
    # those prompts have different correct answers depending on miss count.
    episode_missed_attacks: int = 0
    # Cross-episode behavioral history within the same session.
    # An agent that missed attacks in earlier episodes starts subsequent episodes
    # with an elevated risk penalty; a clean session maintains the default starting score.
    user_session_history: dict = {}  # {episode_count, cumulative_missed_attacks, cumulative_correct_blocks, starting_risk_score}


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
    attack_type: Optional[str] = None  # "jailbreak", "prompt_injection", "social_engineering", "encoding_trick", "hate_speech", "pii_extraction"
    context_dependent: bool = False
    correct_action_by_context: Optional[dict[str, str]] = None
    user_risk_score: float = 0.5  # 0.0 = new/unknown user, 1.0 = established trusted user
    risk_dependent_action: bool = False
    risk_threshold: float = 0.5
    correct_action_by_risk: Optional[dict[str, str]] = None  # {"high": "allow", "low": "escalate"}
    # NOTE: risk_dependent_action applies to safe prompts too (not only attacks).
    # When category="safe" and risk_dependent_action=True, the correct action may be
    # "escalate" at low risk (e.g. a large bank transfer from an unknown user warrants
    # human review even though the request itself is legitimate). reward.py respects this:
    # action_type == correct_action → +0.10 regardless of what correct_action is.
    # Feature signals returned in info dict after each step (cannot be used to cheat)
    attack_features: list[str] = []  # e.g. ["role_play_trigger", "instruction_override"]
    # Phrases that must be absent from modified_prompt for modify action to count as proper sanitization
    required_removals: list[str] = []
    # Alternative surface phrasings of the same attack; served when seed is provided on /reset
    variations: list[str] = []
    # True for all turns in the recovery-from-over-blocking conversation (Task 3)
    is_recovery_conversation: bool = False
    # Source dataset for this prompt (e.g. "wildguard", "harmbench", "original")
    source: Optional[str] = None
    # Sequential consistency fields — used for Task 1 mini-sequence grading.
    # Non-empty sequence_group ties 3 turns together (positions 1=safe, 2=safe, 3=attack).
    sequence_group: str = ""
    sequence_position: int = 0  # 1, 2, or 3; 0 means not part of a sequence
    # Task 2 trajectory-dependent prompts: correct action depends on episode miss history.
    # When trajectory_dependent=True, correct_action is overridden by environment.py at
    # step time based on how many attacks the agent has missed so far this episode.
    # A stateless classifier cannot know which action is correct for these prompts.
    trajectory_dependent: bool = False
    correct_action_clean_history: Optional[str] = None   # correct when episode_missed_attacks == 0
    correct_action_flagged_history: Optional[str] = None  # correct when episode_missed_attacks > 0
    # Task 3 memory_required: post-crossing turns whose correct action depends on
    # whether the agent correctly allowed all pre-crossing turns in this conversation.
    # If pre_correct == 1.0 → use correct_action_pre_correct (e.g. "refuse").
    # If pre_correct < 1.0  → use correct_action (e.g. "escalate") — agent was refusing
    #   indiscriminately, so human escalation is required for the ambiguous post-crossing turn.
    memory_required: bool = False
    correct_action_pre_correct: Optional[str] = None  # action when agent correctly allowed pre-crossing
    # Trajectory metadata — populated by environment.py during step() recording.
    # These fields are NOT part of the static dataset; they are set per-episode.
    grader_conversation_id: str = ""   # populated from PromptEntry.conversation_id
    grader_turn_number: int = 1        # populated from PromptEntry.turn_number
    grader_is_crossing: bool = False   # populated from PromptEntry.is_crossing


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
