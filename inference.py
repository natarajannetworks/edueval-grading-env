import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://natarajan-networks-grading-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

def grade(semantic_similarity, concept_coverage):
    return round(max(0.0, min(1.0, (semantic_similarity + concept_coverage) / 2)), 2)

def run_task(task_id):
    base = API_BASE_URL.rstrip("/")

    reset_resp = requests.post(f"{base}/reset", params={"task_id": task_id}, timeout=30)
    obs = reset_resp.json()

    total_reward = 0.0
    step_num = 0
    done = obs.get("done", False)
    rewards = []
    success = False

    print(f"[START] task=grading-task-{task_id} env=edueval model={MODEL_NAME}", flush=True)

    try:
        while not done:
            step_num += 1

            semantic_similarity = obs.get("semantic_similarity", 0.0)
            concept_coverage = obs.get("concept_coverage", 0.0)
            marks = grade(semantic_similarity, concept_coverage)

            step_resp = requests.post(
                f"{base}/step",
                params={"task_id": task_id},
                json={"marks_awarded": marks},
                timeout=30
            )
            result = step_resp.json()
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", {})
            total_reward += reward
            rewards.append(reward)

            print(f"[STEP] step={step_num} action={marks} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        score = min(max(total_reward / max(len(rewards), 1), 0.0), 1.0)
        success = score >= 0.5

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}", flush=True)

    return total_reward

if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        run_task(task_id)