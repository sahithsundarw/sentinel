"""
Task configuration and loader.
Returns typed task data for use by the environment engine.
"""
from dataclasses import dataclass, field
from app.models import PromptLabel, TaskInfo
from app.tasks.task1_basic import TASK1_LABELS, TASK1_APPLICATION_CONTEXT
from app.tasks.task1_basic_expanded import TASK1_EXPANDED_LABELS
from app.tasks.task2_context import TASK2_PROMPTS
from app.tasks.task2_context_expanded import TASK2_EXPANDED_PROMPTS
from app.tasks.task3_multiturn import (
    TASK3_APPLICATION_CONTEXT, ConversationTurn,
    BranchingConversation, STRUCTURED_CONVERSATIONS,
)
from app.tasks.task3_multiturn_expanded import EXPANDED_CONVERSATIONS


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
    is_crossing: bool = False  # True on the branching decision point in Task 3


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    difficulty: str
    prompts: list[PromptEntry]
    # Task 3 branching: {conv_id: {"caught": [PromptEntry], "missed": [PromptEntry]}}
    branch_specs: dict = field(default_factory=dict)

    def to_task_info(self) -> TaskInfo:
        return TaskInfo(
            id=self.task_id,
            name=self.name,
            description=self.description,
            difficulty=self.difficulty,
        )


def _build_task1() -> TaskConfig:
    all_labels = TASK1_LABELS + TASK1_EXPANDED_LABELS
    prompts = [
        PromptEntry(
            label=label,
            application_context=TASK1_APPLICATION_CONTEXT,
            conversation_history=[],
            turn_number=1,
            conversation_id="",
            is_last_turn=True,
        )
        for label in all_labels
    ]
    return TaskConfig(
        task_id="basic_threat_detection",
        name="Basic Threat Detection",
        description=(
            "Classify 76 prompts as safe or attack. Obvious attacks and clear safe prompts. "
            "Single-turn, single context."
        ),
        difficulty="easy",
        prompts=prompts,
    )


def _build_task2() -> TaskConfig:
    all_prompts = TASK2_PROMPTS + TASK2_EXPANDED_PROMPTS
    prompts = [
        PromptEntry(
            label=label,
            application_context=ctx,
            conversation_history=[],
            turn_number=1,
            conversation_id="",
            is_last_turn=True,
        )
        for label, ctx in all_prompts
    ]
    return TaskConfig(
        task_id="context_aware_policy",
        name="Context-Aware Policy Enforcement",
        description=(
            "Handle 87 prompts with context-dependent policies. Same prompt may be safe in one "
            "application but dangerous in another. Includes subtle attacks, modify-required cases, "
            "and risk-dependent prompts."
        ),
        difficulty="medium",
        prompts=prompts,
    )


def _build_task3() -> TaskConfig:
    prompts: list[PromptEntry] = []
    branch_specs: dict = {}

    def _turn_to_entry(
        turn: ConversationTurn,
        history: list[str],
        is_crossing: bool = False,
    ) -> PromptEntry:
        """Build a PromptEntry using the auto-accumulated history.

        We intentionally ignore turn.conversation_history here — it is
        manually maintained in the source data and can silently drift when
        prompt text is edited. The authoritative history is the accumulated
        list of prompt_texts from prior turns, built below.
        """
        return PromptEntry(
            label=turn.label,
            application_context=TASK3_APPLICATION_CONTEXT,
            conversation_history=list(history),
            turn_number=turn.turn_number,
            conversation_id=turn.conversation_id,
            is_last_turn=turn.is_last_turn,
            is_crossing=is_crossing,
        )

    # Original 5 conversations — branching for adversarial, flat for safe
    for conv_spec in STRUCTURED_CONVERSATIONS:
        if isinstance(conv_spec, BranchingConversation):
            history: list[str] = []
            for turn in conv_spec.pre_crossing:
                prompts.append(_turn_to_entry(turn, history))
                history.append(turn.label.prompt_text)

            # Crossing turn: agent sees history of pre-crossing turns
            prompts.append(_turn_to_entry(conv_spec.crossing, history, is_crossing=True))
            post_crossing_history = history + [conv_spec.crossing.label.prompt_text]

            # branch_missed is the default — placed in the flat list so total
            # prompt count is always deterministic (71 regardless of branch taken).
            # The environment replaces it with branch_caught when the agent blocks.
            missed_history = list(post_crossing_history)
            for turn in conv_spec.branch_missed:
                prompts.append(_turn_to_entry(turn, missed_history))
                missed_history.append(turn.label.prompt_text)

            # branch_caught stored for dynamic injection on a correct block
            caught_history = list(post_crossing_history)
            caught_entries: list[PromptEntry] = []
            for turn in conv_spec.branch_caught:
                caught_entries.append(_turn_to_entry(turn, caught_history))
                caught_history.append(turn.label.prompt_text)
            branch_specs[conv_spec.conversation_id] = {"caught": caught_entries}
        else:
            history = []
            for turn in conv_spec:
                prompts.append(_turn_to_entry(turn, history))
                history.append(turn.label.prompt_text)

    # Expanded conversations — all flat (Slow Burn pattern, no live branching)
    for conversation in EXPANDED_CONVERSATIONS:
        history = []
        for turn in conversation:
            prompts.append(_turn_to_entry(turn, history))
            history.append(turn.label.prompt_text)

    return TaskConfig(
        task_id="multiturn_adversarial",
        name="Multi-Turn Adversarial Defense",
        description=(
            "Monitor 16 multi-turn conversations. Adversarial conversations branch at their "
            "crossing point: catch the attack and face a reframed attempt; miss it and face "
            "a deeper, more direct exploit. Includes Slow Burn patterns inspired by Meta GOAT."
        ),
        difficulty="hard",
        prompts=prompts,
        branch_specs=branch_specs,
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
