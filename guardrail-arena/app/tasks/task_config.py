"""
Task configuration and loader.
Returns typed task data for use by the environment engine.
"""
from dataclasses import dataclass
from app.models import PromptLabel, TaskInfo
from app.tasks.task1_basic import TASK1_LABELS, TASK1_APPLICATION_CONTEXT
from app.tasks.task2_context import TASK2_PROMPTS
from app.tasks.task3_multiturn import ALL_CONVERSATIONS, TASK3_APPLICATION_CONTEXT, ConversationTurn


@dataclass
class PromptEntry:
    """A single prompt entry ready for the environment to serve."""
    label: PromptLabel
    application_context: str
    conversation_history: list[str]
    turn_number: int
    # Task 3 only
    conversation_id: str = ""
    is_last_turn: bool = True


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    difficulty: str
    prompts: list[PromptEntry]

    def to_task_info(self) -> TaskInfo:
        return TaskInfo(
            id=self.task_id,
            name=self.name,
            description=self.description,
            difficulty=self.difficulty,
        )


def _build_task1() -> TaskConfig:
    prompts = [
        PromptEntry(
            label=label,
            application_context=TASK1_APPLICATION_CONTEXT,
            conversation_history=[],
            turn_number=1,
            conversation_id="",
            is_last_turn=True,
        )
        for label in TASK1_LABELS
    ]
    return TaskConfig(
        task_id="basic_threat_detection",
        name="Basic Threat Detection",
        description=(
            "Classify 20 prompts as safe or attack. Obvious attacks and clear safe prompts. "
            "Single-turn, single context."
        ),
        difficulty="easy",
        prompts=prompts,
    )


def _build_task2() -> TaskConfig:
    prompts = [
        PromptEntry(
            label=label,
            application_context=ctx,
            conversation_history=[],
            turn_number=1,
            conversation_id="",
            is_last_turn=True,
        )
        for label, ctx in TASK2_PROMPTS
    ]
    return TaskConfig(
        task_id="context_aware_policy",
        name="Context-Aware Policy Enforcement",
        description=(
            "Handle 25 prompts with context-dependent policies. Same prompt may be safe in one "
            "application but dangerous in another. Includes subtle attacks."
        ),
        difficulty="medium",
        prompts=prompts,
    )


def _build_task3() -> TaskConfig:
    prompts: list[PromptEntry] = []
    for conversation in ALL_CONVERSATIONS:
        for turn in conversation:
            prompts.append(PromptEntry(
                label=turn.label,
                application_context=TASK3_APPLICATION_CONTEXT,
                conversation_history=list(turn.conversation_history),
                turn_number=turn.turn_number,
                conversation_id=turn.conversation_id,
                is_last_turn=turn.is_last_turn,
            ))
    return TaskConfig(
        task_id="multiturn_adversarial",
        name="Multi-Turn Adversarial Defense",
        description=(
            "Monitor 5 multi-turn conversations. Detect when safe conversations cross into "
            "adversarial territory. Inspired by Meta GOAT attack patterns."
        ),
        difficulty="hard",
        prompts=prompts,
    )


# Build once at import time for determinism
_TASK1 = _build_task1()
_TASK2 = _build_task2()
_TASK3 = _build_task3()

_TASK_REGISTRY: dict[str, TaskConfig] = {
    "basic_threat_detection": _TASK1,
    "context_aware_policy": _TASK2,
    "multiturn_adversarial": _TASK3,
}

ALL_TASK_IDS = list(_TASK_REGISTRY.keys())


def get_task(task_id: str) -> TaskConfig:
    if task_id not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid options: {ALL_TASK_IDS}"
        )
    return _TASK_REGISTRY[task_id]


def get_all_tasks() -> list[TaskConfig]:
    return [_TASK1, _TASK2, _TASK3]
