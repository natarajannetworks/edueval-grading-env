---
title: Grading Env
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: other
---

# EduEval — Automated Answer Sheet Grading Environment

An OpenEnv-compatible reinforcement learning environment for training and evaluating AI agents on automated answer sheet grading tasks.

## 🎯 Motivation

Manual grading of student answer sheets is time-consuming, inconsistent, and does not scale. EduEval provides a structured RL environment where an AI agent learns to grade student answers accurately — matching human expert graders — across varying difficulty levels.

## 🌍 Environment Description

EduEval simulates a real-world educational grading pipeline. At each step, the agent receives a student's answer along with the correct answer key, semantic similarity score, and concept coverage score. The agent must award marks between 0.0 and 1.0. The environment rewards accurate grading and penalizes over-grading or under-grading.

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
| semantic_similarity | float | Similarity score between student and key answer |
| concept_coverage | float | How well key concepts are covered |
| question_number | int | Current question index |
| total_questions | int | Total questions in episode |
| max_marks | float | Maximum marks for this question |
| task_id | int | Task difficulty level |
| done | bool | Whether episode is complete |

## 📋 Tasks

| Task | Difficulty | Description |
|---|---|---|
| Task 1 | Easy | Basic factual questions with simple answers |
| Task 2 | Medium | Multi-concept questions requiring partial credit |
| Task 3 | Hard | Complex descriptive answers requiring deep evaluation |

## 🏆 Reward Function

- Exact match (diff = 0.0): reward = 1.0
- Close match (diff ≤ 0.1): reward = 0.8
- Good match (diff ≤ 0.2): reward = 0.6
- Partial match (diff ≤ 0.3): reward = 0.4
- Weak match (diff ≤ 0.5): reward = 0.2
- Poor match (diff > 0.5): reward = 0.0
- Penalty for over-grading or under-grading extreme cases

## 🚀 Setup & Usage

### Run locally

```bash
git clone https://github.com/natarajannetworks/edueval-grading-env
cd edueval-grading-env
pip install -r requirements.txt
export PYTHONPATH=src
uvicorn src.envs.grading_env.server.app:app --reload
```

### Run inference

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Run with Docker

```bash
docker build -t edueval .
docker run -p 7860:7860 edueval
```

## 📊 Baseline Scores

| Task | Score |
|---|---|
| Task 1 (Easy) | 1.00 |
| Task 2 (Medium) | 1.00 |
| Task 3 (Hard) | 1.00 |

## 🔗 API Endpoints

- `POST /reset?task_id=1` — Start new episode
- `POST /step?task_id=1` — Submit grading action
- `GET /state?task_id=1` — Get current state
- `GET /health` — Health check
- `GET /docs` — Interactive API documentation