import json
import uuid
import random
from pathlib import Path
from typing import Tuple

from ..models import GradingAction, GradingObservation, GradingState


class GradingEnvironment:

    DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/sample_papers"

    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.all_questions = []
        self.questions = []
        self.current_index = 0
        self.episode_id = ""
        self.cumulative_marks = 0.0
        self.consecutive_accurate = 0
        self._load_task(task_id)

    def reset(self) -> GradingObservation:
        self.current_index = 0
        self.cumulative_marks = 0.0
        self.consecutive_accurate = 0
        self.episode_id = str(uuid.uuid4())
        self.questions = random.sample(self.all_questions, min(3, len(self.all_questions)))
        return self._make_observation()

    def step(self, action: GradingAction) -> Tuple[GradingObservation, float, bool]:
        current_q = self.questions[self.current_index]
        manual_mark = current_q["manual_mark"]
        agent_mark = action.marks_awarded
        question_type = current_q.get("question_type", "factual")

        reward = self._compute_reward(agent_mark, manual_mark, current_q, question_type)

        self.cumulative_marks += agent_mark
        self.current_index += 1
        done = self.current_index >= len(self.questions)
        obs = self._make_observation(done=done)

        return obs, reward, done

    def state(self) -> GradingState:
        return GradingState(
            episode_id=self.episode_id,
            step_count=self.current_index,
            task_id=self.task_id,
            total_questions=len(self.questions),
            current_question=self.current_index,
            cumulative_marks=self.cumulative_marks,
            max_possible_marks=float(len(self.questions)),
            is_complete=self.current_index >= len(self.questions)
        )

    def _compute_reward(self, agent_mark: float, manual_mark: float, question: dict, question_type: str) -> float:
        diff = abs(agent_mark - manual_mark)

        if question_type == "factual":
            if diff <= 0.05:
                base_reward = 0.93
            elif diff <= 0.1:
                base_reward = 0.73
            elif diff <= 0.2:
                base_reward = 0.43
            elif diff <= 0.3:
                base_reward = 0.23
            elif diff <= 0.5:
                base_reward = 0.13
            else:
                base_reward = 0.07

        elif question_type == "conceptual":
            if diff <= 0.05:
                base_reward = 0.91
            elif diff <= 0.1:
                base_reward = 0.81
            elif diff <= 0.15:
                base_reward = 0.71
            elif diff <= 0.2:
                base_reward = 0.57
            elif diff <= 0.3:
                base_reward = 0.37
            elif diff <= 0.4:
                base_reward = 0.21
            elif diff <= 0.5:
                base_reward = 0.11
            else:
                base_reward = 0.07

        else:
            if diff <= 0.05:
                base_reward = 0.87
            elif diff <= 0.1:
                base_reward = 0.77
            elif diff <= 0.15:
                base_reward = 0.67
            elif diff <= 0.2:
                base_reward = 0.57
            elif diff <= 0.25:
                base_reward = 0.47
            elif diff <= 0.3:
                base_reward = 0.37
            elif diff <= 0.4:
                base_reward = 0.21
            elif diff <= 0.5:
                base_reward = 0.11
            else:
                base_reward = 0.07

        if agent_mark >= 0.9 and manual_mark < 0.5:
            base_reward = max(0.07, base_reward - 0.3)
        if agent_mark <= 0.1 and manual_mark > 0.5:
            base_reward = max(0.07, base_reward - 0.3)

        if diff <= 0.1:
            self.consecutive_accurate += 1
        else:
            self.consecutive_accurate = 0

        final = round(base_reward, 4)
        final = max(0.07, min(0.93, final))
        return final

    def _make_observation(self, done: bool = False) -> GradingObservation:
        idx = min(self.current_index, len(self.questions) - 1)
        q = self.questions[idx]

        keyword_match = self._check_keyword_match(
            q["student_answer"],
            q["answer_key"]
        )

        concept = q.get("concept_coverage", 0.5)
        concept = max(0.05, min(0.95, concept))

        return GradingObservation(
            question_text=q["question"],
            student_answer=q["student_answer"],
            answer_summary=q.get("summary", q["student_answer"]),
            answer_key=q["answer_key"],
            semantic_similarity=keyword_match,
            concept_coverage=concept,
            question_number=self.current_index + 1,
            total_questions=len(self.questions),
            max_marks=q["max_marks"],
            task_id=self.task_id,
            done=done
        )

    def _check_keyword_match(self, student_answer: str, answer_key: str) -> float:
        student_lower = student_answer.lower()
        key_phrases = [p.strip() for p in answer_key.split('.') if p.strip()]
        if not key_phrases:
            return 0.5

        matches = 0
        for phrase in key_phrases:
            terms = [t.strip() for t in phrase.split() if len(t.strip()) > 2]
            for term in terms:
                if term.lower() in student_lower:
                    matches += 1
                    break

        raw = matches / len(key_phrases)
        return round(max(0.05, min(0.95, raw)), 4)

    def _load_task(self, task_id: int):
        file_map = {
            1: "task1_easy.json",
            2: "task2_medium.json",
            3: "task3_hard.json"
        }
        path = self.DATA_PATH / file_map[task_id]
        with open(path, "r") as f:
            self.all_questions = json.load(f)
        self.questions = random.sample(self.all_questions, min(3, len(self.all_questions)))