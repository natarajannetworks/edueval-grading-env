import os
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://natarajan-networks-grading-env.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "dummy",
    base_url=API_BASE_URL if API_BASE_URL and "hf.space" not in API_BASE_URL else "https://api.openai.com/v1"
)

def grade_with_llm(question_text, student_answer, answer_key, semantic_similarity, concept_coverage):
    prompt = f"""You are an expert answer grader. Grade the student's answer between 0.0 and 1.0.

Question: {question_text}
Answer Key: {answer_key}
Student Answer: {student_answer}
Semantic Similarity Score: {semantic_similarity}
Concept Coverage Score: {concept_coverage}

Return only a single float number between 0.0 and 1.0. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        mark = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, mark))
    except Exception:
        return round(max(0.0, min(1.0, (semantic_similarity + concept_coverage) / 2)), 2)

def run_task(task_id):
    base = API_BASE_URL.rstrip("/")

    reset_resp = requests.post(f"{base}/reset", params={"task_id": task_id})
    obs = reset_resp.json()

    total_reward = 0.0
    step_num = 0
    done = False

    print(f"[START] task_id={task_id}")

    while not done:
        step_num += 1
        marks = grade_with_llm(
            obs["question_text"],
            obs["student_answer"],
            obs["answer_key"],
            obs["semantic_similarity"],
            obs["concept_coverage"]
        )

        step_resp = requests.post(
            f"{base}/step",
            params={"task_id": task_id},
            json={"marks_awarded": marks}
        )
        result = step_resp.json()
        reward = result["reward"]
        done = result["done"]
        obs = result["observation"]
        total_reward += reward

        print(f"[STEP] task_id={task_id} step={step_num} marks_awarded={marks} reward={reward} done={done}")

    print(f"[END] task_id={task_id} total_reward={round(total_reward, 2)} steps={step_num}")
    return total_reward

if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        run_task(task_id)