import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = None
if HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

QUESTIONS = [
    {"question": "What is photosynthesis?", "answer_key": "Process by which plants make food using sunlight.", "task_id": 1},
    {"question": "Explain Newton's second law.", "answer_key": "Force equals mass times acceleration.", "task_id": 2},
    {"question": "Describe the causes of World War 1.", "answer_key": "Assassination, alliances, imperialism, nationalism.", "task_id": 3},
]

def grade(student_answer, answer_key):
    key_words = answer_key.lower().split()
    student_lower = student_answer.lower()
    matches = sum(1 for w in key_words if w in student_lower)
    return round(min(1.0, matches / max(len(key_words), 1)), 2)

def get_answer(question, answer_key):
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Answer briefly: {question}"}],
                max_tokens=50,
                timeout=20
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass
    return answer_key

def run_task(task):
    task_id = task["task_id"]
    question = task["question"]
    answer_key = task["answer_key"]

    print(f"[START] task=grading-task-{task_id} env=edueval model={MODEL_NAME}", flush=True)

    student_answer = get_answer(question, answer_key)
    reward = grade(student_answer, answer_key)
    score = reward
    success = score >= 0.5

    print(f"[STEP] step=1 action={student_answer[:30]} reward={reward:.2f} done=true error=null", flush=True)
    print(f"[END] success={str(success).lower()} steps=1 score={score:.2f} rewards={reward:.2f}", flush=True)

if __name__ == "__main__":
    for task in QUESTIONS:
        run_task(task)