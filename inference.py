import os
import requests
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

ENV_BASE_URL = os.getenv("ENV_BASE_URL", os.getenv("SPACE_URL", "https://natarajan-networks-grading-env.hf.space"))

def llm_grade(question, student_answer, answer_key, semantic_similarity, concept_coverage):
    prompt = f"""You are an expert educational grader. Grade the following student answer.

Question: {question}
Reference Answer: {answer_key}
Student Answer: {student_answer}
Semantic Similarity Score: {semantic_similarity:.2f}
Concept Coverage Score: {concept_coverage:.2f}

Instructions:
- Award marks between 0.1 and 0.9 only
- 0.9 = excellent answer covering all key concepts
- 0.5 = partial answer with some correct concepts
- 0.1 = completely wrong or irrelevant answer
- Consider the semantic similarity and concept coverage scores as hints
- Be fair and consistent

Respond with ONLY a single decimal number between 0.1 and 0.9. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        mark = float(text)
        return round(max(0.1, min(0.9, mark)), 2)
    except Exception:
        return round(max(0.1, min(0.9, (semantic_similarity + concept_coverage) / 2 + 0.1)), 2)

def run_task(task_id):
    base = ENV_BASE_URL.rstrip("/")
    rewards = []
    step_num = 0
    success = False
    score = 0.5

    task_names = {1: "grading-easy", 2: "grading-medium", 3: "grading-hard"}
    task_name = task_names.get(task_id, f"grading-task-{task_id}")

    print(f"[START] task={task_name} env=edueval model={MODEL_NAME}", flush=True)

    try:
        reset_resp = requests.post(
            f"{base}/reset",
            params={"task_id": task_id},
            timeout=30
        )
        obs = reset_resp.json()
        done = obs.get("done", False)

        while not done:
            step_num += 1

            question = obs.get("question_text", "")
            student_answer = obs.get("student_answer", "")
            answer_key = obs.get("answer_key", "")
            semantic_similarity = obs.get("semantic_similarity", 0.0)
            concept_coverage = obs.get("concept_coverage", 0.0)

            marks = llm_grade(
                question,
                student_answer,
                answer_key,
                semantic_similarity,
                concept_coverage
            )

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
            rewards.append(reward)

            print(f"[STEP] step={step_num} action={marks} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        score = round(sum(rewards) / max(len(rewards), 1), 2)
        score = max(0.05, min(0.95, score))
        success = score >= 0.5

    except Exception as e:
        if not rewards:
            rewards.append(0.05)
        print(f"[STEP] step={step_num+1} action=error reward=0.05 done=true error={str(e)}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.05"
        score_clamped = round(max(0.01, min(0.99, score)), 2)
        print(f"[END] success={str(success).lower()} steps={step_num} score={score_clamped} rewards={rewards_str}", flush=True)

    return score

if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        run_task(task_id)