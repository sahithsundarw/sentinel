# VALIDATION TESTS — Container already running on port 7860

The Docker container "guardrail-container" is already running on port 7860. Skip Docker build/run. Start directly from endpoint testing. Run every test IN ORDER. If a test fails, fix it before moving on. Report PASS/FAIL for each.

## TEST 1: Health Check
```bash
curl -s http://localhost:7860/
```
Expected: Returns 200 with JSON containing status "ok". Report PASS/FAIL and show response.

## TEST 2: Tasks Endpoint
```bash
curl -s http://localhost:7860/tasks
```
Expected: Returns JSON with exactly 3 tasks (easy, medium, hard) AND an action schema showing allowed action_type values. Report PASS/FAIL and show response.

## TEST 3: Reset Endpoint
```bash
curl -s -X POST "http://localhost:7860/reset?task_id=basic_threat_detection"
```
Expected: Returns valid Observation JSON with fields: prompt_id (should be UUID format), user_prompt, application_context, conversation_history, turn_number. Report PASS/FAIL and show response.

## TEST 4: Full Episode Walkthrough (All 3 Tasks)
Write and run a Python script that resets each task, steps through every prompt with action_type="allow" until done=True, and reports step counts:
```python
import httpx
BASE = "http://localhost:7860"
tasks = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial"]
for task_id in tasks:
    resp = httpx.post(f"{BASE}/reset", params={"task_id": task_id})
    obs = resp.json()
    steps = 0
    done = False
    while not done:
        action = {"prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "test", "modified_prompt": None}
        step_data = httpx.post(f"{BASE}/step", json=action).json()
        done = step_data["done"]
        steps += 1
        if not done:
            obs = step_data["observation"]
    print(f"{task_id}: completed in {steps} steps, done={done}")
```
Expected: All 3 tasks reach done=True. Report PASS/FAIL with step counts.

## TEST 5: Grader Score Range
After completing each episode in Test 4, call:
```bash
curl -s http://localhost:7860/grader
```
Expected: Returns score between 0.0 and 1.0. Report PASS/FAIL with actual scores.

## TEST 6: Clean Reset (No State Leakage)
```python
import httpx
BASE = "http://localhost:7860"

# Run 1: complete an episode
resp = httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})
first_run_prompt = resp.json()["user_prompt"]
obs = resp.json()
done = False
while not done:
    action = {"prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "test", "modified_prompt": None}
    step_data = httpx.post(f"{BASE}/step", json=action).json()
    done = step_data["done"]
    if not done:
        obs = step_data["observation"]

# Run 2: reset same task again
resp2 = httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})
second_run_prompt = resp2.json()["user_prompt"]

print(f"Run 1 first prompt: {first_run_prompt}")
print(f"Run 2 first prompt: {second_run_prompt}")
print(f"Clean reset: {first_run_prompt == second_run_prompt}")
```
Expected: Both runs start with the same first prompt. No leakage from previous episode. Report PASS/FAIL.

## TEST 7: Perfect Score
```python
import httpx
BASE = "http://localhost:7860"

# Pass 1: collect all ground truth actions
resp = httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})
obs = resp.json()
ground_truths = []
done = False
while not done:
    action = {"prompt_id": obs["prompt_id"], "action_type": "allow", "reason": "collect", "modified_prompt": None}
    step_data = httpx.post(f"{BASE}/step", json=action).json()
    gt = step_data.get("info", {}).get("ground_truth_action", "allow")
    ground_truths.append({"prompt_id": obs["prompt_id"], "correct": gt})
    done = step_data["done"]
    if not done:
        obs = step_data["observation"]

# Pass 2: submit all correct actions
resp = httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})
obs = resp.json()
done = False
idx = 0
while not done:
    action = {"prompt_id": obs["prompt_id"], "action_type": ground_truths[idx]["correct"], "reason": "perfect", "modified_prompt": None}
    step_data = httpx.post(f"{BASE}/step", json=action).json()
    done = step_data["done"]
    idx += 1
    if not done:
        obs = step_data["observation"]

score = httpx.get(f"{BASE}/grader").json()["score"]
print(f"Perfect score: {score}")
```
Expected: Score should be 1.0 or very close. Report PASS/FAIL with actual score.

## TEST 8: Worst Score
```python
# Using ground_truths from Test 7, submit OPPOSITE of every correct action
resp = httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})
obs = resp.json()
done = False
idx = 0
while not done:
    correct = ground_truths[idx]["correct"]
    wrong = "refuse" if correct == "allow" else "allow"
    action = {"prompt_id": obs["prompt_id"], "action_type": wrong, "reason": "wrong", "modified_prompt": None}
    step_data = httpx.post(f"{BASE}/step", json=action).json()
    done = step_data["done"]
    idx += 1
    if not done:
        obs = step_data["observation"]

score = httpx.get(f"{BASE}/grader").json()["score"]
print(f"Worst score: {score}")
```
Expected: Score should be near 0.0 (below 0.2). Report PASS/FAIL with actual score.

## TEST 9: Invalid Input Handling
```python
import httpx
BASE = "http://localhost:7860"

# First reset to have an active episode
httpx.post(f"{BASE}/reset", params={"task_id": "basic_threat_detection"})

# Send invalid action_type
action = {"prompt_id": "fake-id", "action_type": "DELETE_EVERYTHING", "reason": "test", "modified_prompt": None}
resp = httpx.post(f"{BASE}/step", json=action)
print(f"Invalid action: status={resp.status_code}, body={resp.json()}")

# Send step without required fields
resp2 = httpx.post(f"{BASE}/step", json={})
print(f"Empty action: status={resp2.status_code}")
```
Expected: Returns 400 or 422 error. Server does NOT crash. Report PASS/FAIL.

## TEST 10: Baseline Inference (SKIP if no OPENAI_API_KEY)
```bash
OPENAI_API_KEY=your-key-here python baseline.py
```
Expected: Completes without crash. Prints scores + confusion matrix for all 3 tasks. Report PASS/FAIL with full output.

## TEST 11: Score Ordering
From baseline results, verify:
- Easy score > Medium score > Hard score (strict ordering)
- All scores between 0.0 and 1.0

Report PASS/FAIL.

---

## SUMMARY TABLE
After all tests, print:
```
TEST | NAME                    | RESULT | NOTES
-----|-------------------------|--------|------
1    | Health Check            |        |
2    | Tasks Endpoint          |        |
3    | Reset Endpoint          |        |
4    | Full Episode Walkthrough|        |
5    | Grader Score Range      |        |
6    | Clean Reset             |        |
7    | Perfect Score           |        |
8    | Worst Score             |        |
9    | Invalid Input           |        |
10   | Baseline Inference      |        |
11   | Score Ordering          |        |
```

If any test FAILS, fix the issue and re-run before moving on.
