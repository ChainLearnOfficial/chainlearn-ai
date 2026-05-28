"""Quiz-related Pydantic models."""

from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    """A single multiple-choice question."""

    prompt: str = Field(..., description="The question text")
    options: list[str] = Field(..., min_length=2, max_length=6, description="Answer choices")
    correct_index: int = Field(..., ge=0, description="Index of the correct option")
    explanation: str = Field(default="", description="Why the correct answer is right")


class Quiz(BaseModel):
    """A quiz composed of multiple-choice questions."""

    quiz_id: str = Field(..., description="Unique quiz identifier")
    course_id: str = Field(default="", description="Associated course ID")
    module_id: str = Field(default="", description="Associated module ID")
    questions: list[QuizQuestion] = Field(default_factory=list)
    difficulty: str = Field(default="beginner")


class GenerateQuizRequest(BaseModel):
    """Request body for POST /generate-quiz."""

    user_id: str = Field(..., description="User requesting the quiz")
    course_id: str = Field(default="", description="Course context")
    module_id: str = Field(default="", description="Module context")
    difficulty: str = Field(
        default="beginner",
        pattern=r"^(beginner|intermediate|advanced)$",
    )
    num_questions: int = Field(default=5, ge=1, le=20)
    topic: str = Field(default="", description="Specific topic to quiz on")
