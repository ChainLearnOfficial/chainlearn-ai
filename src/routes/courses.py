"""Course generation endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.models.course import Course, GenerateCourseRequest
from src.services.generation import get_generation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-course", tags=["courses"])


@router.post("", response_model=Course)
async def generate_course(request: GenerateCourseRequest) -> Course:
    """Generate a course on a given topic.

    Uses Cohere's command model with optional RAG context from the
    knowledge base to produce structured course modules.

    Args:
        request: Course generation parameters including topic, difficulty,
                 and learner background.

    Returns:
        A fully populated Course with modules.

    Raises:
        500: If generation fails.
    """
    logger.info(
        "Generating course: topic=%s difficulty=%s modules=%d",
        request.topic,
        request.difficulty,
        request.num_modules,
    )

    try:
        service = get_generation_service()
        course = await service.generate_course(request)
        logger.info("Generated course '%s' with %d modules", course.title, len(course.modules))
        return course
    except ValueError as exc:
        logger.error("Failed to parse generation response: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to generate course content") from exc
    except RuntimeError as exc:
        logger.error("Cohere API error: %s", exc)
        raise HTTPException(status_code=502, detail="AI service unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected error generating course")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
