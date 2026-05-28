"""Health check endpoint."""

from fastapi import APIRouter

from src.config import settings
from src.knowledge.vectorstore import get_vectorstore

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Return service health status.

    Returns:
        JSON object with service status, version, and knowledge base info.
    """
    store = get_vectorstore()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "0.1.0",
        "knowledge_base": {
            "loaded": not store.is_empty(),
            "chunks": store.count,
        },
    }
