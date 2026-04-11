import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASKS = [
    {
        "task_name": "grading-easy",
        "question": "What is photosynthesis?",
        "answer_key": "Process by which plants make food using sunlight water and carbon dioxide."
    },
    {
        "task_name": "grading-medium",
        "question": "Explain Newton's second law of motion.",
        "answer_key": "Force equals mass times acceleration. F equals ma."
    },
    {
        "task_name": "grading-hard",
        "question": "Describe the main causes of World War 1.",
        "answer_key": "Assassination of Archduke Franz Ferdinand militarism alliances imperialism and nationalism."
    },
]

def grade(student_answer, answer_key):
    key_words = [w for w in answer_key.lower().split() if len(w) > 3]
    student_lower = student_answer.lower()
    matches = sum(1 for w in key_words if w in student_lower)
    return round(min(1.0, matches / max(len(key_words), 1)), 2)

def get_answer(question):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Answer this question briefly in 1-2 sentences: {question}"}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"error: {e}"

def run_task(task):
    task_name = task["task_name"]
    question = task["question"]
    answer_key = task["answer_key"]
    rewards = []
    success = False

    print(f"[START] task={task_name} env=edueval model={MODEL_NAME}", flush=True)

    try:
        student_answer = get_answer(question)
        reward = grade(student_answer, answer_key)
        rewards.append(reward)
        success = reward >= 0.5
        print(f"[STEP] step=1 action={student_answer[:50].replace(chr(10), ' ')} reward={reward:.2f} done=true error=null", flush=True)
    except Exception as e:
        rewards.append(0.0)
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={str(e)}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps=1 rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)