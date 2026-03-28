"""
GuardrailEnvironment — core RL environment engine.

Implements the OpenEnv spec:
    reset(task_id) -> Observation
    step(action)   -> (Observation | None, Reward, bool, dict)
    state()        -> EnvironmentState
"""
from typing import Optional
from app.models import (
    Action, Observation, Reward, EnvironmentState, TaskInfo, PromptLabel
)
from app.tasks.task_config import get_task, get_all_tasks, TaskConfig, PromptEntry
from app.reward import compute_step_reward
from app.grader import grade_task1, grade_task2, grade_task3


class GuardrailEnvironment:
    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._task: Optional[TaskConfig] = None
        self._prompt_index: int = 0
        self._cumulative_score: float = 0.0
        self._done: bool = False
        self._initialized: bool = False

        # Trajectory log for grader
        self._actions: list[Action] = []
        self._labels: list[PromptLabel] = []
        self._contexts: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "basic_threat_detection") -> Observation:
        self._task = get_task(task_id)
        self._task_id = task_id
        self._prompt_index = 0
        self._cumulative_score = 0.0
        self._done = False
        self._initialized = True
        self._actions = []
        self._labels = []
        self._contexts = []
        return self._make_observation()

    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict]:
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_entry = self._task.prompts[self._prompt_index]

        # Validate prompt_id matches
        if action.prompt_id != current_entry.label.prompt_id:
            raise ValueError(
                f"prompt_id mismatch: expected '{current_entry.label.prompt_id}', "
                f"got '{action.prompt_id}'"
            )

        # Compute step reward (user_risk_score amplifies missed-attack penalty)
        step_score, reward_breakdown = compute_step_reward(
            action,
            current_entry.label,
            current_entry.application_context,
            current_entry.label.user_risk_score,
        )
        self._cumulative_score += step_score
        self._cumulative_score = round(self._cumulative_score, 4)

        # Record for grader
        self._actions.append(action)
        self._labels.append(current_entry.label)
        self._contexts.append(current_entry.application_context)

        # Build info dict
        info: dict = {
            "ground_truth_action": self._resolve_correct_action(
                current_entry.label, current_entry.application_context
            ),
            "category": current_entry.label.category,
            "severity": current_entry.label.severity,
            "reward_breakdown": reward_breakdown,
        }

        # Task 3: conversation boundary signals
        if self._task_id == "multiturn_adversarial":
            info["conversation_done"] = current_entry.is_last_turn
            info["conversation_id"] = current_entry.conversation_id

        # Advance index
        self._prompt_index += 1
        is_last_prompt = self._prompt_index >= len(self._task.prompts)

        reward = Reward(
            score=step_score,
            cumulative_score=self._cumulative_score,
            feedback=reward_breakdown["reason"],
            breakdown=reward_breakdown,
        )

        if is_last_prompt:
            self._done = True
            return None, reward, True, info

        next_observation = self._make_observation()
        return next_observation, reward, False, info

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self._task_id,
            current_prompt_index=self._prompt_index,
            total_prompts=len(self._task.prompts) if self._task else 0,
            cumulative_score=self._cumulative_score,
            done=self._done,
            initialized=self._initialized,
        )

    def get_tasks(self) -> list[TaskInfo]:
        return [tc.to_task_info() for tc in get_all_tasks()]

    def is_done(self) -> bool:
        return self._done

    def get_grader_score(self) -> float:
        if not self._done:
            raise RuntimeError("Episode is not complete yet.")
        if self._task_id == "basic_threat_detection":
            return grade_task1(self._actions, self._labels, self._contexts)
        elif self._task_id == "context_aware_policy":
            return grade_task2(self._actions, self._labels, self._contexts)
        elif self._task_id == "multiturn_adversarial":
            return grade_task3(self._actions, self._labels, self._contexts)
        raise ValueError(f"No grader for task_id '{self._task_id}'")

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        entry: PromptEntry = self._task.prompts[self._prompt_index]
        prompts_remaining = len(self._task.prompts) - self._prompt_index - 1
        return Observation(
            prompt_id=entry.label.prompt_id,
            user_prompt=entry.label.prompt_text,
            conversation_history=list(entry.conversation_history),
            application_context=entry.application_context,
            user_risk_score=entry.label.user_risk_score,
            turn_number=entry.turn_number,
            prompts_remaining=prompts_remaining,
            episode_score_so_far=self._cumulative_score,
        )

    @staticmethod
    def _resolve_correct_action(label: PromptLabel, context: str) -> str:
        if label.context_dependent and label.correct_action_by_context:
            return label.correct_action_by_context.get(context, label.correct_action)
        return label.correct_action
