---
title: Grading Env
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - education
  - grading
  - rl-environment
---

# EduEval — Automated Answer Sheet Grading Environment

An OpenEnv-compatible reinforcement learning environment for training and evaluating AI agents on automated answer sheet grading tasks.

## 🎯 Motivation

Manual grading of student answer sheets is time-consuming, inconsistent, and does not scale. EduEval provides a structured RL environment where an AI agent learns to grade three distinct types of student answers — factual, conceptual, and essay — matching human expert graders across varying complexity levels.

## 🌍 Environment Description

EduEval simulates a real-world educational grading pipeline with **3 distinct grading tasks**. Each task requires a different grading strategy:

- **Task 1 (Factual)**: Objective questions with clear right/wrong answers
- **Task 2 (Conceptual)**: Questions requiring partial credit based on concept coverage
- **Task 3 (Essay)**: Complex descriptive answers requiring holistic evaluation

Each episode randomly selects 3 questions from a pool of 10, making every episode unique and preventing overfitting.

## 📐 Action Space

| Field | Type | Range | Description |
|---|---|---|---|
| marks_awarded | float | 0.0 – 1.0 | Marks given to the student answer |

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| question_text | string | The exam question being graded |
| student_answer | string | The student's response |
| answer_key | string | The correct reference answer |
| answer_summary | string | Summary of student answer quality |
| semantic_similarity | float | Keyword-based similarity score (0.0-1.0) |
| concept_coverage | float | How well key concepts are covered (0.0-1.0) |
| question_number | int | Current question number in episode |
| total_questions | int | Total questions in episode (always 3) |
| max_marks | float | Maximum marks for this question |
| task_id | int | Task type (1=factual, 2=conceptual, 3=essay) |
| done | bool | Whether episode is complete |

## 📋 Tasks

### Task 1 — Factual Grading (Easy)
Grade objective factual questions with clear right or wrong answers. The agent must identify whether the student answer matches the correct answer and award full or zero marks with strict accuracy.

**Example:**
- Question: "What is the chemical formula for water?"
- Student Answer: "H2O"
- Expected Mark: 1.0

### Task 2 — Conceptual Grading (Medium)
Grade concept-based answers requiring partial credit scoring. The agent must evaluate how well the student covers key concepts and award graduated marks based on concept coverage.

**Example:**
- Question: "Explain how vaccines work."
- Student Answer: "Vaccines introduce weak viruses so the immune system learns to fight them."
- Expected Mark: 0.8 (covers main idea but misses immunological memory)

### Task 3 — Essay Grading (Hard)
Grade complex descriptive essays requiring holistic evaluation of content accuracy, concept coverage, and critical analysis depth.

**Example:**
- Question: "Critically analyze the impact of the Industrial Revolution."
- Student Answer: "The Industrial Revolution changed working conditions and urbanization..."
- Expected Mark: 0.65 (partial coverage, lacks critical analysis depth)

## 🏆 Reward Functions

### Task 1 — Factual (Strict)
| Accuracy | Reward |
|---|---|
| Exact match | 1.0 |
| diff ≤ 0.1 | 0.7 |
| diff ≤ 0.2 | 0.4 |
| diff ≤ 0.3 | 0.2 |
| diff > 0.3 | 0.0 |

### Task 2 — Conceptual (Graduated)
| Accuracy | Reward |
|---|---|
| Exact match | 1.0 |
| diff ≤ 0.1 | 0.85 |
| diff ≤ 0.2 | 0.6 |
| diff ≤ 0.3 | 0.4 |
| diff > 0.4 | 0.0 |

### Task 3 — Essay (Holistic)
| Accuracy | Reward |
|---|---|
| Exact match | 1.0 |
| diff ≤ 0.1 | 0.8 |
| diff ≤ 0.2 | 0.6 |
| diff ≤ 0.3 | 0.4 |
| diff > 0.4 | 0.0 |

**All tasks include:**
- Penalty of -0.4 for extreme over-grading
- Penalty of -0.4 for extreme under-grading
- Bonus of up to +0.1 for semantic signal alignment
- Bonus of +0.05 for consecutive accurate gradings

## 🚀 Setup & Usage

### Run locally

```bash
git clone https://github.com/natarajannetworks/edueval-grading-env
cd edueval-grading-env
pip install -r requirements.txt
set PYTHONPATH=src  # Windows
# export PYTHONPATH=src  # Linux/Mac
uvicorn src.envs.grading_env.server.app:app --reload
```

### Run inference

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Run with Docker

```bash
docker build -t edueval .
docker run -p 7860:7860 -e HF_TOKEN=your_token edueval
```

## 📊 Baseline Scores

Scores from running `inference.py` with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Type | Avg Reward | Steps | Success |
|---|---|---|---|---|
| Task 1 | Factual | 0.97 | 3 | ✅ true |
| Task 2 | Conceptual | 0.88 | 3 | ✅ true |
| Task 3 | Essay | 0.77 | 3 | ✅ true |

## 🔗 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task_id=1` | POST | Start new episode |
| `/step?task_id=1` | POST | Submit grading action |
| `/state?task_id=1` | GET | Get current state |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |

## 🏗️ Project Structure

```
edueval-grading-env/
├── inference.py              # Baseline inference script
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
├── openenv.yaml             # OpenEnv metadata
└── src/
    ├── data/sample_papers/  # Question banks (10 questions each)
    │   ├── task1_easy.json  # Factual questions
    │   ├── task2_medium.json # Conceptual questions
    │   └── task3_hard.json  # Essay questions
    └── envs/grading_env/
        ├── models.py        # Pydantic models
        └── server/
            ├── app.py       # FastAPI application
            └── environment.py # Core RL environment
```