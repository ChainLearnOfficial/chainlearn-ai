"""Quiz generation endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from src.models.quiz import GenerateQuizRequest, Quiz
from src.services.quiz_generator import get_quiz_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-quiz", tags=["quizzes"])


@router.post("", response_model=Quiz)
async def generate_quiz(request: GenerateQuizRequest) -> Quiz:
    """Generate a quiz on a given topic or module.

    Produces multiple-choice questions using Cohere with optional
    RAG context from the knowledge base.

    Args:
        request: Quiz generation parameters.

    Returns:
        A Quiz object with questions, options, and correct answers.

    Raises:
        500: If generation fails.
    """
    logger.info(
        "Generating quiz: user=%s topic=%s difficulty=%s questions=%d",
        request.user_id,
        request.topic or request.module_id,
        request.difficulty,
        request.num_questions,
    )

    try:
        generator = get_quiz_generator()
        quiz = await generator.generate_quiz(request)
        logger.info("Generated quiz %s with %d questions", quiz.quiz_id, len(quiz.questions))
        return quiz
    except ValueError as exc:
        logger.error("Failed to parse quiz response: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate quiz") from exc
    except RuntimeError as exc:
        logger.error("Cohere API error: %s", exc)
        raise HTTPException(status_code=502, detail="AI service unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected error generating quiz")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
