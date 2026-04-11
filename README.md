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

Manual grading of student answer sheets is time-consuming, inconsistent, and does not scale. EduEval provides a structured RL environment where an AI agent learns to grade student answers accurately — matching human expert graders — across varying difficulty levels. This is a real-world task that educational institutions face daily.

## 🌍 Environment Description

EduEval simulates a real-world educational grading pipeline. At each step, the agent receives a student's answer along with the correct answer key, semantic similarity score, and concept coverage score. The agent must award marks between 0.0 and 1.0. The environment rewards accurate grading and penalizes over-grading or under-grading. Each episode randomly selects 3 questions from a pool of 10, making every episode unique.

## 📐 Action Space

| Field | Type | Range | Description |
|---|---|---|---|
| marks_awarded | float | 0.0 – 1.0 | Marks given to the student answer |

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| question_text | string | The exam question |
| student_answer | string | The student's response |
| answer_key | string | The correct reference answer |
| answer_summary | string | Summary of the student answer |
| semantic_similarity | float | Keyword-based similarity score (0.0-1.0) |
| concept_coverage | float | How well key concepts are covered (0.0-1.0) |
| question_number | int | Current question index |
| total_questions | int | Total questions in episode (always 3) |
| max_marks | float | Maximum marks for this question |
| task_id | int | Task difficulty level (1=easy, 2=medium, 3=hard) |
| done | bool | Whether episode is complete |

## 📋 Tasks

| Task | Difficulty | Description | Questions Pool |
|---|---|---|---|
| Task 1 | Easy | Basic factual questions with simple one-line answers | 10 questions |
| Task 2 | Medium | Multi-concept questions requiring partial credit scoring | 10 questions |
| Task 3 | Hard | Complex descriptive answers requiring deep semantic evaluation | 10 questions |

Each episode randomly samples 3 questions from the pool of 10, ensuring variety and preventing overfitting.

## 🏆 Reward Function

The reward function provides dense signals throughout the episode:

| Accuracy | Reward |
|---|---|
| Exact match (diff = 0.0) | 1.0 |
| Very close (diff ≤ 0.05) | 0.9 |
| Close (diff ≤ 0.1) | 0.8 |
| Good (diff ≤ 0.15) | 0.7 |
| Partial (diff ≤ 0.2) | 0.6 |
| Weak (diff ≤ 0.3) | 0.4 |
| Poor (diff ≤ 0.4) | 0.2 |
| Very poor (diff ≤ 0.5) | 0.1 |
| Wrong (diff > 0.5) | 0.0 |

**Additional signals:**
- Penalty of -0.4 for over-grading (giving ≥0.9 when correct is <0.5)
- Penalty of -0.4 for under-grading (giving ≤0.1 when correct is >0.5)
- Bonus of up to +0.1 for aligning with semantic and concept signals
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

| Task | Avg Reward | Steps | Success |
|---|---|---|---|
| Task 1 — Easy | 0.97 | 3 | ✅ true |
| Task 2 — Medium | 0.88 | 3 | ✅ true |
| Task 3 — Hard | 0.77 | 3 | ✅ true |

## 🔗 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task_id=1` | POST | Start new episode, returns first observation |
| `/step?task_id=1` | POST | Submit grading action, returns next observation + reward |
| `/state?task_id=1` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

## 🏗️ Project Structure

```
edueval-grading-env/
├── inference.py              # Baseline inference script
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
├── openenv.yaml             # OpenEnv metadata
├── server/app.py            # Entry point
└── src/
    ├── data/sample_papers/  # Question banks (10 questions each)
    │   ├── task1_easy.json
    │   ├── task2_medium.json
    │   └── task3_hard.json
    └── envs/grading_env/
        ├── models.py        # Pydantic models
        └── server/
            ├── app.py       # FastAPI application
            └── environment.py # Core RL environment logic
```