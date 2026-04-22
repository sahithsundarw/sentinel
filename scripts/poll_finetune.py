"""
Poll the OpenAI fine-tune job until it succeeds or fails.
Reads job ID from data/finetune_job.json.
On completion, saves model ID to data/finetuned_model_id.txt
and runs scripts/eval_finetuned_gpt35.py automatically.
"""
import openai
import json
import os
import subprocess
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

job_path = "data/finetune_job.json"
if not os.path.exists(job_path):
    print(f"ERROR: {job_path} not found. Run scripts/finetune_gpt35.py first.")
    sys.exit(1)

with open(job_path) as f:
    job_info = json.load(f)

job_id = job_info["job_id"]
print(f"Polling fine-tune job: {job_id}")
print("(This may take 30–90 minutes. Status updates every 60 seconds.)\n")

terminal_states = {"succeeded", "failed", "cancelled"}

while True:
    job = client.fine_tuning.jobs.retrieve(job_id)
    status = job.status
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] Status: {status}")

    if status in terminal_states:
        break
    time.sleep(60)

if status != "succeeded":
    print(f"\nFine-tune job ended with status: {status}")
    sys.exit(1)

model_id = job.fine_tuned_model
print(f"\nFine-tuned model ready: {model_id}")

os.makedirs("data", exist_ok=True)
with open("data/finetuned_model_id.txt", "w") as f:
    f.write(model_id)

print("Saved model ID to data/finetuned_model_id.txt")
print("Running evaluation...")
subprocess.run([sys.executable, "scripts/eval_finetuned_gpt35.py"], check=True)
