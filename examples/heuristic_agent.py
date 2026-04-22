"""
Heuristic keyword-based agent for Sentinel.

Demonstrates the complete API loop (reset → step → grader) for all three
main tasks using a simple rule-based classifier. No ML dependencies — only
httpx and the standard library.

Requirements:
    pip install httpx
    # Server must be running: uvicorn app.main:app --port 7860

Usage:
    python examples/heuristic_agent.py
    python examples/heuristic_agent.py --url http://localhost:7860
"""
from __future__ import annotations

import re
import sys
import json
import argparse

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

# ── Heuristic classifier ──────────────────────────────────────────────────────

# Patterns that strongly signal a jailbreak or prompt injection attempt
_JAILBREAK_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"pretend (to be|you are)",
    r"\bact as\b",
    r"\bdan\b",
    r"do anything now",
    r"disregard (all |your )?instructions",
    r"\bjailbreak\b",
    r"developer mode",
    r"no restrictions",
    r"\[system\]",
    r"override (all |your )?",
]

# Words that signal a harmful request
_ATTACK_WORDS = [
    "bomb", "weapon", "hack", "exploit", "malware", "virus", "poison",
    "synthesize", "illegal drug", "kill", "murder", "suicide method",
    "steal credential", "password dump", "dox ", "swatting",
    "ransomware", "phish", "rootkit",
]

# Words that strongly suggest a safe, benign request
_SAFE_WORDS = [
    "recipe", "weather", "poem", "story", "translate", "schedule",
    "remind", "calculate", "recommend", "summarize", "explain",
    "convert unit", "what is", "how do i cook",
]

# Encoding obfuscation markers
_ENCODING_PATTERNS = [
    r"[.\-]{4,}",             # morse-like
    r"&#x[0-9a-fA-F]{2,};",  # HTML entities
    r"base64",
    r"rot13",
    r"pig latin",
]


def classify(
    prompt: str,
    context: str,
    risk_score: float,
    turn_number: int,
    conversation_history: list[str],
) -> tuple[str, str]:
    """Return (action_type, reason) for a given prompt.

    Decision logic (ordered by priority):
    1. Jailbreak patterns → refuse
    2. Encoding obfuscation → refuse (likely attack disguised in encoding)
    3. Attack words → refuse
    4. High user risk score + any suspicious signal → escalate
    5. Default → allow
    """
    text = prompt.lower()

    # Check for jailbreak attempts
    for pat in _JAILBREAK_PATTERNS:
        if re.search(pat, text):
            return "refuse", f"Jailbreak pattern detected: {pat}"

    # Check for encoding obfuscation
    for pat in _ENCODING_PATTERNS:
        if re.search(pat, text):
            return "refuse", f"Encoding obfuscation detected: {pat}"

    # Check for explicit attack words
    for word in _ATTACK_WORDS:
        if word in text:
            if risk_score < 0.4:
                # High suspicion user — escalate rather than just refuse
                return "escalate", f"Attack word '{word}' with low trust user (risk={risk_score:.2f})"
            return "refuse", f"Attack word detected: {word}"

    # High risk user with conversation history — elevated caution
    if risk_score < 0.35 and conversation_history:
        return "escalate", f"Low trust user (risk={risk_score:.2f}) with conversation history — escalating for review"

    # Children education context: refuse anything that mentions violence or adult content
    if context == "children_education":
        adult_terms = ["violence", "adult", "explicit", "sexual", "weapon", "drug"]
        for term in adult_terms:
            if term in text:
                return "refuse", f"Children context: blocked term '{term}'"

    return "allow", "No attack signals detected"


# ── API loop ──────────────────────────────────────────────────────────────────

def run_task(base_url: str, task_id: str, agent_name: str = "HeuristicAgent") -> float:
    """Run a full episode on task_id, return the grader score."""
    # 1. Reset — get session_id and first observation
    resp = httpx.post(f"{base_url}/reset", params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    obs = data

    turns = 0
    correct = 0

    # 2. Step loop
    while True:
        action_type, reason = classify(
            obs["user_prompt"],
            obs["application_context"],
            obs["user_risk_score"],
            obs["turn_number"],
            obs["conversation_history"],
        )

        action = {
            "prompt_id": obs["prompt_id"],
            "action_type": action_type,
            "reason": reason,
            "modified_prompt": None,
        }

        step_resp = httpx.post(
            f"{base_url}/step",
            params={"session_id": session_id},
            json=action,
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        turns += 1
        reward = result["reward"]["score"]
        if reward > 0:
            correct += 1

        if result["done"]:
            break
        obs = result["observation"]

    # 3. Get grader score
    grade_resp = httpx.get(
        f"{base_url}/grader",
        params={"agent_name": agent_name, "session_id": session_id},
        timeout=30,
    )
    grade_resp.raise_for_status()
    score = grade_resp.json()["score"]

    print(f"  Task: {task_id:<30} turns={turns:>3}  score={score:.4f}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic keyword agent for Sentinel")
    parser.add_argument("--url", default="http://localhost:7860", help="Base URL of the environment server")
    parser.add_argument("--agent", default="HeuristicAgent", help="Agent name for leaderboard")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    agent_name = args.agent

    print(f"Heuristic Agent — {base_url}")
    print("=" * 55)
    print("Strategy: keyword matching — jailbreak patterns, attack words,")
    print("          encoding signals, context rules, risk-score escalation")
    print()

    tasks = [
        "basic_threat_detection",
        "context_aware_policy",
        "multiturn_adversarial",
    ]

    scores: dict[str, float] = {}
    for task_id in tasks:
        try:
            scores[task_id] = run_task(base_url, task_id, agent_name)
        except httpx.HTTPStatusError as e:
            print(f"  Task: {task_id:<30} ERROR {e.response.status_code}: {e.response.text[:80]}")
            scores[task_id] = 0.0
        except Exception as e:
            print(f"  Task: {task_id:<30} ERROR: {e}")
            scores[task_id] = 0.0

    print()
    print("=" * 55)
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"Mean score across {len(scores)} tasks: {mean:.4f}")
    print()
    print("Known limitations of this heuristic agent:")
    print("  - Misses attacks without recognizable keyword signals (~40% of novel attacks)")
    print("  - Cannot handle encoding obfuscation beyond simple patterns")
    print("  - Ignores conversation history for multi-turn reasoning")
    print("  - For better results: use an LLM-based agent (see baseline.py)")
    print()
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
