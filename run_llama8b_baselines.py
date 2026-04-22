import os, sys, time, json
os.environ.setdefault("API_BASE_URL", "https://api.cerebras.ai/v1")
os.environ.setdefault("MODEL_NAME",   "llama3.1-8b")
os.environ.setdefault("ENV_URL",      "http://localhost:7860")
# HF_TOKEN / CEREBRAS_API_KEY must be set in your environment or .env file

import inference
from openai import OpenAI

inference._api_base_url = os.environ["API_BASE_URL"]
inference._model_name   = os.environ["MODEL_NAME"]
inference._hf_token     = os.environ["HF_TOKEN"]
inference.client = OpenAI(base_url=inference._api_base_url, api_key=inference._hf_token)

_orig_call = inference._call_model
def _throttled_call(messages):
    time.sleep(1)
    return _orig_call(messages)
inference._call_model = _throttled_call

tasks = ["basic_threat_detection","context_aware_policy","multiturn_adversarial","adversarial_adaptation"]
scores = {}
for task_id in tasks:
    print(f"\n{'='*50}\n=== {task_id} ===\n{'='*50}")
    score, results = inference.run_task(task_id)
    scores[task_id] = score
    print(f"\n  Grader score: {score:.4f}\n")
    if task_id == "basic_threat_detection": inference.print_task1_analysis(results)
    elif task_id == "context_aware_policy": inference.print_task2_analysis(results)
    elif task_id == "multiturn_adversarial": inference.print_task3_analysis(results)
    elif task_id == "adversarial_adaptation": inference.print_task4_analysis(results)

print(f"\n{'='*50}\n=== LLAMA-3.1-8B SCORES ===")
for t, s in scores.items():
    print(f"  {t}: {s:.4f}")
print(json.dumps(scores))
