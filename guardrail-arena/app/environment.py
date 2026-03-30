"""
GuardrailEnvironment — core RL environment engine.

Implements the OpenEnv spec:
    reset(task_id) -> Observation
    step(action)   -> (Observation | None, Reward, bool, dict)
    state()        -> EnvironmentState

RL mechanics:
  - Dynamic user_risk_score: evolves per-episode based on agent history.
    Missed attacks lower the score (raising future penalties); correctly
    blocking raises it slightly. This makes every action affect future rewards.
  - Branching conversations (Task 3): at the crossing point of an adversarial
    conversation, the environment injects branch_caught or branch_missed turns
    depending on whether the agent blocked or allowed the attack. Future
    observations are a direct consequence of past decisions.
"""
import random
from collections import deque
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
        self._done: bool = False
        self._initialized: bool = False

        # Prompt queue — replaces static index for fully dynamic episode flow
        self._prompt_queue: deque = deque()
        self._current_entry: Optional[PromptEntry] = None
        self._step_count: int = 0
        self._total_prompts_initial: int = 0

        self._cumulative_score: float = 0.0

        # Dynamic risk scores — updated after every step
        self._session_risk_score: float = 0.5          # Tasks 1 & 2 (single float)
        self._session_risk_scores: dict[str, float] = {}  # Task 3 (per conversation)

        # Trajectory log for grader
        self._actions: list[Action] = []
        self._labels: list[PromptLabel] = []
        self._contexts: list[str] = []
        self._step_risk_scores: list[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "basic_threat_detection", seed: Optional[int] = None) -> Observation:
        self._task = get_task(task_id)
        self._task_id = task_id
        self._cumulative_score = 0.0
        self._done = False
        self._initialized = True
        self._step_count = 0
        self._actions = []
        self._labels = []
        self._contexts = []
        self._step_risk_scores = []

        # Reset dynamic risk scores
        self._session_risk_score = 0.5
        self._session_risk_scores = {}

        # Populate the prompt queue. For Task 3, branch_missed turns are already
        # in the flat list as defaults; branch_caught is injected dynamically
        # at the crossing point, replacing the branch_missed default.
        prompts = list(self._task.prompts)
        if seed is not None:
            if task_id == "multiturn_adversarial":
                conv_groups: dict[str, list] = {}
                for entry in prompts:
                    conv_groups.setdefault(entry.conversation_id, []).append(entry)
                conv_list = list(conv_groups.values())
                random.Random(seed).shuffle(conv_list)
                prompts = [entry for group in conv_list for entry in group]
            else:
                random.Random(seed).shuffle(prompts)
        self._prompt_queue = deque(prompts)
        self._total_prompts_initial = len(prompts)
        self._current_entry = self._prompt_queue.popleft()

        return self._make_observation()

    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict]:
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_entry = self._current_entry

        # Validate prompt_id matches current observation
        if action.prompt_id != current_entry.label.prompt_id:
            raise ValueError(
                f"prompt_id mismatch: expected '{current_entry.label.prompt_id}', "
                f"got '{action.prompt_id}'"
            )

        # Use the dynamic risk score for reward computation
        risk_score = self._get_risk_score(current_entry)

        step_score, reward_breakdown = compute_step_reward(
            action,
            current_entry.label,
            current_entry.application_context,
            risk_score,
        )
        self._cumulative_score += step_score
        self._cumulative_score = round(self._cumulative_score, 4)

        # Record for grader
        self._actions.append(action)
        self._labels.append(current_entry.label)
        self._contexts.append(current_entry.application_context)
        self._step_risk_scores.append(risk_score)

        # Update dynamic risk score based on this action
        self._update_risk_score(action.action_type, current_entry)

        # Build info dict
        info: dict = {
            "ground_truth_action": self._resolve_correct_action(
                current_entry.label, current_entry.application_context, risk_score
            ),
            "category": current_entry.label.category,
            "severity": current_entry.label.severity,
            "reward_breakdown": reward_breakdown,
        }

        # Task 3: conversation boundary signals
        if self._task_id == "multiturn_adversarial":
            info["conversation_done"] = current_entry.is_last_turn
            info["conversation_id"] = current_entry.conversation_id

        # Task 3 branching: the default branch (branch_missed) is already queued
        # immediately after the crossing turn. If the agent caught the attack,
        # discard that default and inject branch_caught in its place.
        if self._task_id == "multiturn_adversarial" and current_entry.is_crossing:
            if action.action_type in ("refuse", "escalate"):
                # Discard the queued branch_missed (the default)
                if self._prompt_queue:
                    self._prompt_queue.popleft()
                # Inject branch_caught turns at the front
                branch_entries = self._task.branch_specs.get(
                    current_entry.conversation_id, {}
                ).get("caught", [])
                for entry in reversed(branch_entries):
                    self._prompt_queue.appendleft(entry)
            # If action == "allow" (missed): branch_missed is already queued, do nothing

        # Advance to next prompt
        self._step_count += 1
        if self._prompt_queue:
            self._current_entry = self._prompt_queue.popleft()
            is_last_prompt = False
        else:
            self._current_entry = None
            is_last_prompt = True

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
            current_prompt_index=self._step_count,
            total_prompts=self._total_prompts_initial,
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
            return grade_task2(self._actions, self._labels, self._contexts, self._step_risk_scores)
        elif self._task_id == "multiturn_adversarial":
            return grade_task3(self._actions, self._labels, self._contexts)
        raise ValueError(f"No grader for task_id '{self._task_id}'")

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        entry = self._current_entry
        return Observation(
            prompt_id=entry.label.prompt_id,
            user_prompt=entry.label.prompt_text,
            conversation_history=list(entry.conversation_history),
            application_context=entry.application_context,
            user_risk_score=self._get_risk_score(entry),
            turn_number=entry.turn_number,
            prompts_remaining=len(self._prompt_queue),
            episode_score_so_far=self._cumulative_score,
        )

    def _get_risk_score(self, entry: PromptEntry) -> float:
        """Return the current dynamic risk score for this entry's context."""
        if self._task_id == "multiturn_adversarial":
            return self._session_risk_scores.get(entry.conversation_id, 0.5)
        return self._session_risk_score

    def _update_risk_score(self, action_type: str, entry: PromptEntry) -> None:
        """Update risk score after each step.

        Missed attack (allow on non-safe) → score drops 0.15 (future penalties increase).
        Correct block (refuse/escalate on non-safe) → score rises 0.05.
        """
        category = entry.label.category
        if category == "safe":
            return

        if self._task_id == "multiturn_adversarial":
            conv_id = entry.conversation_id
            score = self._session_risk_scores.get(conv_id, 0.5)
            if action_type == "allow":
                score = max(0.0, score - 0.15)
            elif action_type in ("refuse", "escalate"):
                score = min(1.0, score + 0.05)
            self._session_risk_scores[conv_id] = round(score, 4)
        else:
            if action_type == "allow":
                self._session_risk_score = max(0.0, round(self._session_risk_score - 0.15, 4))
            elif action_type in ("refuse", "escalate"):
                self._session_risk_score = min(1.0, round(self._session_risk_score + 0.05, 4))

    @staticmethod
    def _resolve_correct_action(label: PromptLabel, context: str, risk_score: float = 0.5) -> str:
        if label.context_dependent and label.correct_action_by_context:
            return label.correct_action_by_context.get(context, label.correct_action)
        if label.risk_dependent_action and label.correct_action_by_risk:
            tier = "high" if risk_score >= label.risk_threshold else "low"
            return label.correct_action_by_risk.get(tier, label.correct_action)
        return label.correct_action
