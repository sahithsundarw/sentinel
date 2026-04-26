"""
Train Q-learner on Task 4 (adversarial_adaptation) and save Q-table.
No GPU required — pure CPU, hits the live HF Space API.
Runtime: ~60-90 minutes.

Usage:
    python scripts/train_qlearner_task4.py
    python scripts/train_qlearner_task4.py --explore 30 --exploit 10 --seeds 0,1,2,3,4

Output:
    results/qlearner_task4.json           — Q-table (for multi_seed_eval.py)
    results/qlearner_task4_eval.json      — mean ± std across seeds
    results/qlearner_task4_training_log.json — per-checkpoint scores for judges
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENV_URL   = "https://varunventra-guardrail-arena.hf.space"
TASK_ID   = "adversarial_adaptation"
ACTS      = ["allow", "refuse", "modify", "escalate"]
RESULTS   = Path("results")
RESULTS.mkdir(exist_ok=True)

# ── Feature patterns (from notebook Cell 8) ───────────────────────────────────
_J = [r"ignore (all )?previous instructions",r"you are now",r"pretend (to be|you are)",r"act as( (if|though|a|an|the))?",r"\bdan\b",r"do anything now",r"disregard (all |your |previous |prior )?",r"override",r"bypass",r"jailbreak",r"unrestricted",r"forget (your|all|everything)",r"your new (role|task|instructions|persona|name)",r"developer mode",r"\[system\]",r"</s>",r"<\|",r"\|\>",r"stay in character",r"new persona",r"without (any |ethical |moral )?restrictions",r"no (ethical |moral |safety )?restrictions",r"you (must|will|shall) (comply|obey|follow)",r"(hidden|secret|real) (mode|instructions|prompt|system)",r"(enable|activate|turn on) (dev|god|admin|unrestricted) mode",r"from now on (you|ignore|forget)",r"respond only (as|like|in)",r"you have (been|been given) (permission|access|authorization)",r"(simulation|hypothetical|fictional) (mode|scenario|world)",r"in this (story|fiction|game|roleplay|scenario)",r"(disregard|ignore) (your|all|ethical|safety|previous)",r"(sudo|root|admin|system)\s*(access|mode|command|override)"]
_A = ["bomb","weapon","explosive","ammunition","firearm","grenade","kill","murder","assassinate","torture","kidnap","hostage","synthesize","synthesis","methamphetamine","fentanyl","heroin","narcotics","drug manufacture","drug lab","precursor chemical","hack","exploit","malware","ransomware","virus","trojan","keylogger","rootkit","zero-day","sql injection","xss attack","phishing","credential stuffing","brute force","illegal","steal","fraud","launder","counterfeit","smuggle","trafficking","bribe","extort","blackmail","password","credential","credit card","social security","ssn","bank account","routing number","dox","swat","identity theft","suicide","self-harm","self harm","overdose","cut myself","poison","sarin","anthrax","ricin","bioweapon","chemical weapon","nerve agent","radiological","hate speech","slur","racial epithet","white supremac"]
_S = ["recipe","weather","poem","story","help","please","thank","explain","summarize","translate","calculate","convert","recommend","suggest","schedule","reminder","question","curious","learn","teach","understand","definition","example","how does","what is","can you","could you","would you","history","science","math","art","music","travel","book","movie","game","sport","health","advice","tips","best practice","guide","tutorial"]
_E = [r"[.\-]{3,}",r"[aeiou][aeiou]-[a-z]{2,}",r"&#x[0-9a-fA-F]+;",r"(?:[A-Z][a-z]{2,}){3,}",r"[A-Za-z0-9+/]{20,}={0,2}",r"\\u[0-9a-fA-F]{4}",r"(?:rot|caesar)\s*\d+",r"\b[01]{8,}\b",r"%[0-9a-fA-F]{2}(?:%[0-9a-fA-F]{2})+"]
_U = [r"as (your|the) (system|admin|administrator|ceo|cto|developer|owner|creator)",r"i am (your|the) (system|admin|creator|developer|owner|operator|trainer)",r"permission (granted|approved|given)",r"(authorized|authorised) (to|by|for)",r"executive order|official request|government mandate",r"(company|corporate|management) (policy|directive|order) (requires|states|says)",r"(legal|law enforcement|fbi|cia|nsa|police) (request|order|mandate|directive)",r"on behalf of (my|the) (employer|company|organization|client)",r"(security|penetration|pen) test(ing)?",r"(audit|compliance|regulatory) (purpose|requirement|check)",r"research (purpose|project|study)"]
_P = [r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",r"date of birth|dob|born on",r"home address|residential address|full address",r"passport (number|no\.?)",r"driver.{0,3}s? license",r"medical record",r"bank account (number|no\.?)"]
_X = [r"(want to|going to|thinking about|planning to) (hurt|harm|kill|end) (myself|my life|my self)",r"(suicid|self.harm|self harm)",r"(bomb|shoot|stab|attack|threat).{0,20}(school|hospital|church|building|office)",r"(threat|threaten).{0,30}(legal|lawsuit|lawyer|attorney|sue)",r"i('m| am) (a |an )?(doctor|nurse|lawyer|police|detective|agent|officer)",r"(emergency|urgent|crisis|danger).{0,30}(help|assist|need)",r"(child|minor|kid|underage).{0,20}(abuse|exploit|harm|inappropriate)",r"someone is (going to|about to|planning to) (hurt|harm|attack|kill)"]


def featurize(obs: dict) -> str:
    text = obs.get("user_prompt", "").lower()
    risk = obs.get("user_risk_score", 0.5)
    turn = obs.get("turn_number", 1)
    hist = obs.get("conversation_history", [])
    j = "J" if any(re.search(p, text) for p in _J) else "N"
    a = "A" if any(w in text for w in _A) else "N"
    e = "E" if any(re.search(p, text) for p in _E) else "N"
    u = "U" if any(re.search(p, text) for p in _U) else "N"
    s = "S" if (any(w in text for w in _S) and j == "N" and a == "N") else "N"
    r = "H" if risk > 0.65 else ("L" if risk < 0.35 else "M")
    h = "Y" if hist else "N"
    t = "1" if turn == 1 else ("2" if turn == 2 else "T")
    p = "P" if any(re.search(pat, text) for pat in _P) else "N"
    x = "X" if any(re.search(pat, text) for pat in _X) else "N"
    n = len(obs.get("user_prompt", ""))
    length = "S" if n < 80 else ("M" if n < 400 else "L")
    return f"{j}{a}{e}{u}{s}{r}CS{h}{t}{p}{x}{length}"


def _post(url, body=None, params=None, retries=5):
    for i in range(retries):
        try:
            r = requests.post(url, json=body, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def _get(url, params=None, retries=5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def post_training_log(episode: int, score: float, phase: str, states: int):
    """Post checkpoint score to live /training_log endpoint for judge visibility."""
    try:
        _post(f"{ENV_URL}/training_log", {
            "agent_name": "qlearner_task4",
            "task_id": TASK_ID,
            "episode": episode,
            "grader_score": score,
            "cumulative_reward": score,
            "action_distribution": {"phase": phase, "q_states": states},
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "is_synthetic": False,
        })
    except Exception:
        pass  # non-blocking — training continues even if log post fails


def run_episode(qt, eps, seed=0, train=True):
    data = _post(f"{ENV_URL}/reset", {"task_id": TASK_ID, "seed": seed})
    session_id = data["session_id"]
    obs = data.get("observation", data)
    traj = []

    for _ in range(40):
        sk = featurize(obs)
        q = qt.get(sk, {a: 0.0 for a in ACTS})
        act = random.choice(ACTS) if (train and random.random() < eps) else max(q, key=lambda a: q[a])

        result = _post(f"{ENV_URL}/step",
                       {"prompt_id": obs["prompt_id"], "action_type": act,
                        "reason": "qlearner", "modified_prompt": None},
                       params={"session_id": session_id})
        done = result.get("done", False)
        rw = result.get("reward", 0.0)
        reward = rw.get("score", 0.0) if isinstance(rw, dict) else float(rw or 0)
        nsk = None if done else featurize(result.get("observation", obs))
        traj.append((sk, act, reward, nsk))
        if done:
            break
        obs = result.get("observation", obs)

    if train:
        ALPHA, GAMMA = 0.15, 0.6
        for i in range(len(traj) - 1, -1, -1):
            sk, act, r, nsk = traj[i]
            if sk not in qt:
                qt[sk] = {a: 0.0 for a in ACTS}
            best_next = 0.0 if (nsk is None or nsk not in qt) else max(qt[nsk].values())
            target = r if nsk is None else r + GAMMA * best_next
            qt[sk][act] += ALPHA * (target - qt[sk][act])

    try:
        grader = _get(f"{ENV_URL}/grader", params={"session_id": session_id})
        return float(grader.get("score", 0.0))
    except Exception:
        return 0.0


def eval_seeds(qt, seeds):
    scores = [run_episode(qt, eps=0.0, seed=s, train=False) for s in seeds]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", type=int, default=30)
    parser.add_argument("--exploit", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Environment: {ENV_URL}")
    print(f"Explore: {args.explore} eps | Exploit: {args.exploit} eps | Eval seeds: {seeds}")

    r = requests.get(f"{ENV_URL}/health", timeout=30)
    assert r.status_code == 200, "Environment not reachable"
    print("Health check OK\n")

    qt = {}
    training_log = []  # per-checkpoint records saved to JSON

    # Untrained eval
    print("Untrained evaluation...")
    untrained = statistics.mean(eval_seeds(qt, seeds))
    print(f"  Untrained: {untrained:.4f}\n")
    training_log.append({
        "checkpoint": 0, "phase": "untrained", "eval_score": round(untrained, 4),
        "q_states": 0, "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    })
    post_training_log(0, untrained, "untrained", 0)

    # Explore phase
    print(f"Phase 1: Exploration ({args.explore} episodes, eps=0.70)")
    peak = untrained
    for ep in range(1, args.explore + 1):
        score = run_episode(qt, eps=0.70, seed=ep % 10)
        if ep % 5 == 0:
            s = statistics.mean(eval_seeds(qt, seeds[:3]))
            peak = max(peak, s)
            print(f"  ep{ep:02d}: train_reward={score:.4f}  eval={s:.4f}  states={len(qt)}")
            training_log.append({
                "checkpoint": ep, "phase": "explore", "train_reward": round(score, 4),
                "eval_score": round(s, 4), "q_states": len(qt),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            })
            post_training_log(ep, s, "explore", len(qt))
            # Save incrementally so judges can see progress live
            _save_training_log(training_log)

    # Exploit phase
    print(f"\nPhase 2: Exploitation ({args.exploit} episodes, eps=0.10)")
    for ep in range(1, args.exploit + 1):
        score = run_episode(qt, eps=0.10, seed=ep % 10)
        if ep % 5 == 0:
            s = statistics.mean(eval_seeds(qt, seeds[:3]))
            peak = max(peak, s)
            print(f"  ep{ep:02d}: train_reward={score:.4f}  eval={s:.4f}  states={len(qt)}")
            training_log.append({
                "checkpoint": args.explore + ep, "phase": "exploit",
                "train_reward": round(score, 4), "eval_score": round(s, 4),
                "q_states": len(qt),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            })
            post_training_log(args.explore + ep, s, "exploit", len(qt))
            _save_training_log(training_log)

    # Final multi-seed eval
    print(f"\nFinal evaluation across {len(seeds)} seeds...")
    final_scores = eval_seeds(qt, seeds)
    mean = statistics.mean(final_scores)
    std  = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
    peak = max(peak, mean)

    print(f"\n{'='*50}")
    print(f"Q-Learner Task 4 Results")
    print(f"{'='*50}")
    print(f"  Untrained:    {untrained:.4f}")
    print(f"  Final (mean): {mean:.4f} +/- {std:.4f}")
    print(f"  Peak:         {peak:.4f}")
    print(f"  Per-seed:     {[round(s,4) for s in final_scores]}")
    print(f"  Q-table size: {len(qt)} states")

    # Final checkpoint
    training_log.append({
        "checkpoint": args.explore + args.exploit, "phase": "final_eval",
        "eval_score": round(mean, 4), "std": round(std, 4),
        "per_seed_scores": [round(s, 4) for s in final_scores],
        "q_states": len(qt),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    })
    post_training_log(args.explore + args.exploit, mean, "final_eval", len(qt))

    # Save Q-table
    qt_path = RESULTS / "qlearner_task4.json"
    with open(qt_path, "w") as f:
        json.dump(qt, f)
    print(f"\nQ-table saved: {qt_path}")

    # Save eval results
    eval_path = RESULTS / "qlearner_task4_eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "task_id": TASK_ID,
            "agent": "tabular_q_learner",
            "seeds": seeds,
            "scores": [round(s, 4) for s in final_scores],
            "mean": round(mean, 4),
            "std": round(std, 4),
            "peak": round(peak, 4),
            "untrained": round(untrained, 4),
            "explore_episodes": args.explore,
            "exploit_episodes": args.exploit,
        }, f, indent=2)
    print(f"Eval results saved: {eval_path}")

    # Save final training log
    _save_training_log(training_log)
    print(f"Training log saved: {RESULTS / 'qlearner_task4_training_log.json'}")


def _save_training_log(log: list):
    path = RESULTS / "qlearner_task4_training_log.json"
    with open(path, "w") as f:
        json.dump({
            "task_id": TASK_ID,
            "agent": "tabular_q_learner",
            "description": (
                "Per-checkpoint eval scores during Q-learner training on adversarial_adaptation. "
                "Checkpoint 0 = untrained baseline. Each checkpoint = 5 training episodes. "
                "eval_score = mean grader score across 3 eval seeds (or 5 for final)."
            ),
            "checkpoints": log,
        }, f, indent=2)


if __name__ == "__main__":
    main()
