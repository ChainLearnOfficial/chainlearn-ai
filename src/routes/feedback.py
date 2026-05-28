"""Feedback generation endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.quiz import QuizQuestion
from src.services.feedback_engine import get_feedback_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-feedback", tags=["feedback"])


class GenerateFeedbackRequest(BaseModel):
    """Request body for POST /generate-feedback."""

    quiz_id: str = Field(..., description="ID of the quiz being reviewed")
    questions: list[QuizQuestion] = Field(..., description="The quiz questions")
    user_answers: list[int] = Field(..., description="User's selected answer indices")
    correct_answers: list[int] = Field(..., description="Correct answer indices")
    score: float | None = Field(default=None, description="Score percentage (0-100)")
    topic: str = Field(default="", description="Quiz topic for context")


class FeedbackResponse(BaseModel):
    """Response body for feedback endpoint."""

    feedback: str = Field(..., description="Detailed feedback text")
    weak_areas: list[str] = Field(default_factory=list, description="Topics needing review")
    recommendations: list[str] = Field(default_factory=list, description="Study recommendations")


@router.post("", response_model=FeedbackResponse)
async def generate_feedback(request: GenerateFeedbackRequest) -> FeedbackResponse:
    """Generate personalized feedback for a quiz attempt.

    Analyzes the user's answers, identifies weak areas, and provides
    targeted study recommendations.

    Args:
        request: Quiz results including questions, answers, and score.

    Returns:
        FeedbackResponse with feedback text, weak areas, and recommendations.

    Raises:
        500: If generation fails.
    """
    logger.info(
        "Generating feedback: quiz=%s questions=%d",
        request.quiz_id,
        len(request.questions),
    )

    try:
        engine = get_feedback_engine()
        feedback = await engine.generate_feedback(
            questions=request.questions,
            user_answers=request.user_answers,
            correct_answers=request.correct_answers,
            score=request.score,
            topic=request.topic,
        )
        return FeedbackResponse(
            feedback=feedback.feedback_text,
            weak_areas=feedback.weak_areas,
            recommendations=feedback.recommendations,
        )
    except RuntimeError as exc:
        logger.error("Cohere API error: %s", exc)
        raise HTTPException(status_code=502, detail="AI service unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected error generating feedback")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
