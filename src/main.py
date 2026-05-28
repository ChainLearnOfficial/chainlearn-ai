"""ChainLearn AI — FastAPI application entry point.

Provides endpoints for course generation, quiz creation, and
personalized feedback powered by Cohere LLMs.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routes import courses, feedback, health, quizzes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup: configure logging and load knowledge base
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s (debug=%s)", settings.app_name, settings.debug)

    # Attempt to load the pre-built knowledge index
    from src.knowledge.vectorstore import get_vectorstore

    store = get_vectorstore()
    if not store.is_empty():
        logger.info("Knowledge base loaded: %d chunks", store.count)
    else:
        logger.warning("Knowledge base is empty — RAG context will be unavailable")

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered learning content generation for blockchain education",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router)
    app.include_router(courses.router)
    app.include_router(quizzes.router)
    app.include_router(feedback.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
