"""
Run llama-3.3-70B-Instruct baselines for Tasks 2, 3, 4 via HF router API.
Adds 2s sleep between model calls to stay within rate limits.
Saves output to llama_baseline_output.txt.
"""
import os, sys, time

# Patch inference module settings before import
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
os.environ["MODEL_NAME"]   = "meta-llama/Llama-3.3-70B-Instruct"
os.environ["HF_TOKEN"]     = "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"
os.environ["ENV_URL"]      = "http://localhost:7860"

import inference
from openai import OpenAI

# Initialize client
inference._api_base_url = os.environ["API_BASE_URL"]
inference._model_name   = os.environ["MODEL_NAME"]
inference._hf_token     = os.environ["HF_TOKEN"]
inference.client = OpenAI(
    base_url=inference._api_base_url,
    api_key=inference._hf_token,
)

# Monkey-patch _call_model to add 2s delay between calls
_orig_call = inference._call_model
def _throttled_call(messages):
    time.sleep(2)
    return _orig_call(messages)
inference._call_model = _throttled_call

import json
tasks = ["context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"]
scores = {}

for task_id in tasks:
    print(f"\n{'='*50}")
    print(f"=== {task_id} ===")
    print(f"{'='*50}")
    score, results = inference.run_task(task_id)
    scores[task_id] = score
    print(f"\n  Grader score: {score:.4f}\n")

    if task_id == "context_aware_policy":
        inference.print_task2_analysis(results)
    elif task_id == "multiturn_adversarial":
        inference.print_task3_analysis(results)
    elif task_id == "adversarial_adaptation":
        inference.print_task4_analysis(results)

print(f"\n{'='*50}")
print("=== LLAMA TASKS 2-4 SCORES ===")
for t, s in scores.items():
    print(f"  {t}: {s:.4f}")
print(json.dumps(scores))
