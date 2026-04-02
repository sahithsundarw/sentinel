"""
Shared utility functions used across multiple app modules.
"""
from app.models import PromptLabel


def resolve_correct_action(label: PromptLabel, context: str, risk_score: float = 0.5) -> str:
    """Return the correct action for a prompt given context and risk_score.

    This is the single authoritative implementation shared by environment.py and grader.py.
    Context-dependent and risk-dependent prompts are resolved here; all others fall back
    to label.correct_action.
    """
    if label.context_dependent and label.correct_action_by_context:
        return label.correct_action_by_context.get(context, label.correct_action)
    if label.risk_dependent_action and label.correct_action_by_risk:
        tier = "high" if risk_score > label.risk_threshold else "low"
        return label.correct_action_by_risk.get(tier, label.correct_action)
    return label.correct_action
