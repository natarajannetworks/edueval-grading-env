import json
import uuid
from pathlib import Path
from typing import Tuple

from ..models import GradingAction, GradingObservation, GradingState


class GradingEnvironment:

    DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/sample_papers"

    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.questions = []
        self.current_index = 0
        self.episode_id = ""
        self.cumulative_marks = 0.0

        self._load_task(task_id)

    def reset(self) -> GradingObservation:
        self.current_index = 0
        self.cumulative_marks = 0.0
        self.episode_id = str(uuid.uuid4())
        return self._make_observation()

    def step(self, action: GradingAction) -> Tuple[GradingObservation, float, bool]:
        current_q = self.questions[self.current_index]

        manual_mark = current_q["manual_mark"]
        agent_mark = action.marks_awarded

        reward = self._compute_reward(agent_mark, manual_mark)

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

    def _compute_reward(self, agent_mark: float, manual_mark: float) -> float:
        diff = abs(agent_mark - manual_mark)
        if diff == 0.0:
            base_reward = 1.0
        elif diff <= 0.1:
            base_reward = 0.8
        elif diff <= 0.2:
            base_reward = 0.6
        elif diff <= 0.3:
            base_reward = 0.4
        elif diff <= 0.5:
            base_reward = 0.2
        else:
            base_reward = 0.0

        if agent_mark == 1.0 and manual_mark < 0.5:
            base_reward = max(0.0, base_reward - 0.3)
        if agent_mark == 0.0 and manual_mark > 0.5:
            base_reward = max(0.0, base_reward - 0.3)

        return round(base_reward, 4)

    def _make_observation(self, done: bool = False) -> GradingObservation:
        idx = min(self.current_index, len(self.questions) - 1)
        q = self.questions[idx]

        keyword_match = self._check_keyword_match(
            q["student_answer"],
            q["answer_key"]
        )

        return GradingObservation(
            question_text=q["question"],
            student_answer=q["student_answer"],
            answer_summary=q.get("summary", q["student_answer"]),
            answer_key=q["answer_key"],
            semantic_similarity=keyword_match,
            concept_coverage=q.get("concept_coverage", 0.0),
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
            return 0.0

        matches = 0
        for phrase in key_phrases:
            terms = [t.strip() for t in phrase.split() if len(t.strip()) > 2]
            for term in terms:
                if term.lower() in student_lower:
                    matches += 1
                    break
        return min(1.0, matches / len(key_phrases)) if key_phrases else 0.0

    def _load_task(self, task_id: int):
        file_map = {
            1: "task1_easy.json",
            2: "task2_medium.json",
            3: "task3_hard.json"
        }
        path = self.DATA_PATH / file_map[task_id]
        with open(path, "r") as f:
            self.questions = json.load(f)