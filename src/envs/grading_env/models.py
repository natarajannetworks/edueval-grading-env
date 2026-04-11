from pydantic import BaseModel, Field
from typing import Optional


class GradingAction(BaseModel):
    """Action taken by the grading agent - awarding marks to a student answer."""
    marks_awarded: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of marks to award (0.0 = wrong, 1.0 = perfect)"
    )


class GradingObservation(BaseModel):
    """Observation returned by the environment after each step."""
    question_text: str = Field(..., description="The exam question being graded")
    student_answer: str = Field(..., description="The student's response")
    answer_summary: str = Field(..., description="Summary of student answer quality")
    answer_key: str = Field(..., description="The reference correct answer")
    semantic_similarity: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Keyword-based similarity between student and reference answer"
    )
    concept_coverage: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How well key concepts are covered in student answer"
    )
    question_number: int = Field(..., description="Current question number in episode")
    total_questions: int = Field(..., description="Total questions in this episode")
    max_marks: float = Field(..., description="Maximum marks available for this question")
    task_id: int = Field(..., description="Task difficulty level (1=easy, 2=medium, 3=hard)")
    done: bool = Field(False, description="Whether the episode is complete")


class GradingState(BaseModel):
    """Current state of the grading environment."""
    episode_id: str = Field(..., description="Unique identifier for this episode")
    step_count: int = Field(..., description="Number of steps taken so far")
    task_id: int = Field(..., description="Task difficulty level")
    total_questions: int = Field(..., description="Total questions in episode")
    current_question: int = Field(..., description="Current question index")
    cumulative_marks: float = Field(..., description="Total marks awarded so far")
    max_possible_marks: float = Field(..., description="Maximum possible marks for episode")
    is_complete: bool = Field(False, description="Whether episode is complete")