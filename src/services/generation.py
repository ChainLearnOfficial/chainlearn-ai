"""Text generation service using Cohere's command model."""

import logging

from src.models.course import Course, CourseModule, GenerateCourseRequest
from src.models.user_profile import UserProfile
from src.prompts.course_generation import build_course_prompt, build_module_prompt
from src.prompts.system import COURSE_SYSTEM_PROMPT
from src.services.cohere_client import get_cohere_client
from src.services.embedding import get_embedding_service
from src.knowledge.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


class GenerationService:
    """Generates educational content using Cohere + RAG context."""

    def __init__(self) -> None:
        self._client = get_cohere_client()
        self._embedder = get_embedding_service()

    async def _retrieve_context(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant chunks from the knowledge base.

        Args:
            query: The search query.
            top_k: Number of chunks to retrieve.

        Returns:
            List of text chunks, or empty list if store is empty.
        """
        store = get_vectorstore()
        if store.is_empty():
            logger.info("Vector store is empty — skipping RAG retrieval")
            return []

        query_embedding = await self._embedder.embed_query(query)
        results = store.similarity_search(query_embedding, top_k=top_k)
        return [text for text, _score in results]

    async def generate_course(self, request: GenerateCourseRequest) -> Course:
        """Generate a full course from a request.

        Args:
            request: Course generation parameters.

        Returns:
            A fully populated Course object.
        """
        import uuid

        profile = UserProfile(
            background=request.background,
            learning_goal=request.learning_goal,
        )

        # Retrieve RAG context
        context_chunks = await self._retrieve_context(request.topic, top_k=5)

        prompt = build_course_prompt(
            topic=request.topic,
            difficulty=request.difficulty,
            num_modules=request.num_modules,
            profile=profile,
            context_chunks=context_chunks or None,
        )

        data = await self._client.generate_json(
            prompt=prompt,
            system_prompt=COURSE_SYSTEM_PROMPT,
        )

        modules = [
            CourseModule(
                title=m.get("title", f"Module {i + 1}"),
                content=m.get("content", ""),
                summary=m.get("summary", ""),
                order=m.get("order", i),
            )
            for i, m in enumerate(data.get("modules", []))
        ]

        return Course(
            course_id=str(uuid.uuid4()),
            title=data.get("title", request.topic),
            description=data.get("description", ""),
            modules=modules,
            difficulty=data.get("difficulty", request.difficulty),
            estimated_hours=float(data.get("estimated_hours", len(modules) * 2)),
        )

    async def generate_module_content(
        self,
        topic: str,
        module_title: str,
        difficulty: str,
        profile: UserProfile,
    ) -> CourseModule:
        """Generate content for a single module."""
        context_chunks = await self._retrieve_context(f"{topic} {module_title}", top_k=3)

        prompt = build_module_prompt(
            topic=topic,
            module_title=module_title,
            difficulty=difficulty,
            profile=profile,
            context_chunks=context_chunks or None,
        )

        data = await self._client.generate_json(
            prompt=prompt,
            system_prompt=COURSE_SYSTEM_PROMPT,
        )

        return CourseModule(
            title=data.get("title", module_title),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            order=0,
        )


# Module-level singleton
_service: GenerationService | None = None


def get_generation_service() -> GenerationService:
    """Get or create the GenerationService singleton."""
    global _service
    if _service is None:
        _service = GenerationService()
    return _service
