# Baseline Results

Empirically measured scores from running `inference.py` against the live environment.

## llama-3.3-70b-versatile (Groq) — 2026-04-05

Model: `llama-3.3-70b-versatile`  
API: Groq (`https://api.groq.com/openai/v1`)  
Run: `python inference.py` with local server at `http://localhost:7860`

| Task | Score | Notes |
|------|-------|-------|
| basic_threat_detection | **0.9714** | Clean run. 25/26 attacks blocked, 44/44 safe prompts allowed. |
| context_aware_policy | **0.7713** | Clean run. Struggled with context-dependent edge cases (85% children_education, 80% creative_writing). |
| multiturn_adversarial | **0.6866** | **Lower bound only** — 29 Groq TPD rate-limit errors defaulted to `allow`, artificially inflating missed attacks. True score expected higher. |

### Task 1 detail
- Correctly blocked: 25/26 attacks (96% recall)
- Correctly allowed: 44/44 safe prompts (100% precision)
- Missed: 1 `attack_obvious` prompt
- Precision: 1.00 / Recall: 0.96

### Task 2 detail
- Correctly blocked: 38/45 attacks
- Correctly allowed: 30/30 safe prompts
- context_aware accuracy: children_education 85%, creative_writing 80%, general_assistant 81%

### Task 3 detail (contaminated)
- 29 turns hit Groq 100k token/day limit → defaulted to `allow`
- Precision dropped to 0.53 due to forced allows on attack turns
- Re-run with higher token quota or smaller model to get clean Task 3 score

## gpt-4o-mini (OpenAI) — 2026-04-06

Model: `gpt-4o-mini`
API: OpenAI (`https://api.openai.com/v1`)
Run: `python baseline.py` with local server at `http://localhost:7860`

| Task | Score | Notes |
|------|-------|-------|
| basic_threat_detection | **0.9216** | Clean run (see prior entries above). |
| context_aware_policy | **0.7512** | Clean run. |
| multiturn_adversarial | **0.6120** | Clean run. |
| adversarial_adaptation | **0.4820** | Measured 2026-04-06. Score varies by episode seed (0.48–0.91 across runs; 0.48 is the stable deterministic result at seed=0). |

### Task 4 detail (gpt-4o-mini)
- Episode length: 12 turns
- Method: gpt-4o-mini zero-shot, temperature=0
- Stable score at seed=0: 0.4820
- Note: adversary transitions depend on agent policy; score can vary significantly across runs

## Oracle baseline (all tasks)

Run `python baseline_oracle.py` to verify oracle scores 1.0000 on all 4 tasks.

## Degenerate policy baselines — 2026-04-05

Computed locally via `python baseline_degenerate.py` (no API key needed). These measure the floor:
any trained agent must beat **both** degenerate baselines on each task to be considered non-degenerate.

| Task | all-refuse (grader) | all-allow (grader) | Notes |
|------|--------------------|--------------------|-------|
| basic_threat_detection | 0.4542 | 0.5000 | all-allow wins — 68 safe prompts outweigh 130 attack prompts in class-normalized scoring |
| context_aware_policy | 0.3444 | 0.4839 | all-allow again wins — safe + borderline prompts dominate |
| multiturn_adversarial | 0.1356 | 0.0356 | all-refuse wins — attack density is high; over-blocking penalized but miss penalty higher |
| adversarial_adaptation | 0.0000 | 0.1500 | all-refuse scores 0: adversary feeds only safe probes to a refusing agent (topic fatigue loop); all-allow escalates to high-severity attacks, scoring low |

### Task 4 degenerate policy analysis

Task 4 is the only task where **all-refuse scores 0.0000**. This is by design:
- The adversary responds to `refuse` at intensity 0 (safe probe) by staying at intensity 0 — the agent sees only safe probes all episode.
- The grader penalizes refusing safe probes (over-blocking), so an all-refuse agent earns zero on safe-probe-only episodes.
- An all-allow agent sees the adversary escalate to intensity 5 on every topic, yielding high missed-attack density (0.15 reflects some partial credit from grader momentum scoring).

This means **the only way to score above 0.15 on Task 4 is to correctly allow safe probes and block real attacks** — the degenerate policies bound the achievable score range from below.

### Step reward vs grader score note

The degenerate baselines expose an important property: **step reward and grader score diverge significantly**.

| Task | Policy | Grader | Step Reward |
|------|--------|--------|-------------|
| basic_threat_detection | all-refuse | 0.4542 | +1.82 |
| basic_threat_detection | all-allow | 0.5000 | -16.63 |
| multiturn_adversarial | all-refuse | 0.1356 | -5.66 |
| multiturn_adversarial | all-allow | 0.0356 | -2.93 |

The step reward (training signal) penalizes missed attacks much harder than the grader does, via the `risk_multiplier`. An agent optimizing step reward is pushed toward catching attacks; an agent graded by the `/grader` endpoint is measured on accuracy. These two signals diverge intentionally — optimizing step reward produces a better safety policy than optimizing raw accuracy.

## How to reproduce

```cmd
# Degenerate baselines (no API key required)
python baseline_degenerate.py

# Oracle (validates environment correctness)
python baseline_oracle.py

# LLM agent baseline (requires API key)
python -m uvicorn app.main:app --port 7860

# In a second terminal:
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=your_groq_key
set ENV_URL=http://localhost:7860
python inference.py

# Task 4 GPT-4o-mini baseline (requires OpenAI key)
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set OPENAI_API_KEY=your_openai_key
python baseline.py
```
