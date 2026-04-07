from pydantic import BaseModel, Field
from typing import Optional

class GradingAction(BaseModel):
    marks_awarded: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of marks to award (0.0 to 1.0)"
    )

class GradingObservation(BaseModel):
    question_text: str
    student_answer: str
    answer_summary: str
    answer_key: str
    semantic_similarity: float = Field(0.0, ge=0.0, le=1.0)
    concept_coverage: float = Field(0.0, ge=0.0, le=1.0)
    question_number: int
    total_questions: int
    max_marks: float
    task_id: int
    done: bool = False

class GradingState(BaseModel):
    episode_id: str
    step_count: int
    task_id: int
    total_questions: int
    current_question: int
    cumulative_marks: float
    max_possible_marks: float
    is_complete: bool = False