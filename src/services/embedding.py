"""Embedding service for generating and querying vector embeddings."""

import logging

from src.services.cohere_client import get_cohere_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    """High-level interface for embedding operations."""

    def __init__(self) -> None:
        self._client = get_cohere_client()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks for storage.

        Args:
            texts: Document text chunks.

        Returns:
            List of embedding vectors.
        """
        # Cohere embed API supports batching; process in chunks of 96
        batch_size = 96
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._client.embed(
                texts=batch,
                input_type="search_document",
            )
            all_embeddings.extend(embeddings)

        logger.info("Embedded %d documents in %d batches", len(texts), (len(texts) - 1) // batch_size + 1)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Args:
            query: The search query string.

        Returns:
            Embedding vector for the query.
        """
        result = await self._client.embed(
            texts=[query],
            input_type="search_query",
        )
        return result[0]


# Module-level singleton
_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the EmbeddingService singleton."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
