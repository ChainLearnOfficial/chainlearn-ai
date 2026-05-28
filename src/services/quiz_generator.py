"""Quiz generation service using Cohere."""

import logging
import uuid

from src.models.quiz import GenerateQuizRequest, Quiz, QuizQuestion
from src.prompts.quiz_generation import build_quiz_prompt, build_question_from_content_prompt
from src.prompts.system import QUIZ_SYSTEM_PROMPT
from src.services.cohere_client import get_cohere_client
from src.services.embedding import get_embedding_service
from src.knowledge.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Generates quizzes using Cohere and optional RAG context."""

    def __init__(self) -> None:
        self._client = get_cohere_client()
        self._embedder = get_embedding_service()

    async def _retrieve_context(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant knowledge-base chunks."""
        store = get_vectorstore()
        if store.is_empty():
            return []

        query_embedding = await self._embedder.embed_query(query)
        results = store.similarity_search(query_embedding, top_k=top_k)
        return [text for text, _score in results]

    async def generate_quiz(self, request: GenerateQuizRequest) -> Quiz:
        """Generate a quiz from a request.

        Args:
            request: Quiz generation parameters.

        Returns:
            A populated Quiz object.

        Raises:
            ValueError: If the Cohere response cannot be parsed.
        """
        topic = request.topic or f"course {request.course_id} module {request.module_id}"
        context_chunks = await self._retrieve_context(topic, top_k=3)

        prompt = build_quiz_prompt(
            topic=topic,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            context_chunks=context_chunks or None,
        )

        data = await self._client.generate_json(
            prompt=prompt,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )

        questions = [
            QuizQuestion(
                prompt=q.get("prompt", ""),
                options=q.get("options", []),
                correct_index=q.get("correct_index", 0),
                explanation=q.get("explanation", ""),
            )
            for q in data.get("questions", [])
        ]

        quiz_id = str(uuid.uuid4())
        logger.info("Generated quiz %s with %d questions", quiz_id, len(questions))

        return Quiz(
            quiz_id=quiz_id,
            course_id=request.course_id,
            module_id=request.module_id,
            questions=questions,
            difficulty=request.difficulty,
        )

    async def generate_from_content(
        self,
        content: str,
        difficulty: str,
        num_questions: int,
    ) -> list[QuizQuestion]:
        """Generate questions from specific content text.

        Args:
            content: The source content.
            difficulty: Target difficulty.
            num_questions: Number of questions.

        Returns:
            List of QuizQuestion objects.
        """
        prompt = build_question_from_content_prompt(
            content=content,
            difficulty=difficulty,
            num_questions=num_questions,
        )

        data = await self._client.generate_json(
            prompt=prompt,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )

        return [
            QuizQuestion(
                prompt=q.get("prompt", ""),
                options=q.get("options", []),
                correct_index=q.get("correct_index", 0),
                explanation=q.get("explanation", ""),
            )
            for q in data.get("questions", [])
        ]


# Module-level singleton
_generator: QuizGenerator | None = None


def get_quiz_generator() -> QuizGenerator:
    """Get or create the QuizGenerator singleton."""
    global _generator
    if _generator is None:
        _generator = QuizGenerator()
    return _generator
