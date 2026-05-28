"""In-memory vector store for embeddings with similarity search.

Stores document chunks and their embeddings in a dict-based structure.
Supports cosine similarity search and persistence to/from JSON.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        # Each entry: {"text": str, "embedding": list[float], "metadata": dict}
        self._entries: list[dict] = []
        self._index_loaded = False

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents with their embeddings to the store.

        Args:
            texts: Document text chunks.
            embeddings: Corresponding embedding vectors.
            metadatas: Optional metadata dicts for each document.
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._entries.append(
                {
                    "text": text,
                    "embedding": emb,
                    "metadata": meta,
                }
            )

        logger.info("Added %d entries to vector store (total: %d)", len(texts), len(self._entries))

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for the most similar documents.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of results to return.
            threshold: Minimum similarity score to include.

        Returns:
            List of (text, similarity_score) tuples, sorted by score descending.
        """
        if not self._entries:
            return []

        scored: list[tuple[str, float]] = []
        for entry in self._entries:
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim >= threshold:
                scored.append((entry["text"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def is_empty(self) -> bool:
        """Check if the store has any entries."""
        return len(self._entries) == 0

    @property
    def count(self) -> int:
        """Number of entries in the store."""
        return len(self._entries)

    def save(self, path: str | Path | None = None) -> None:
        """Persist the store to a JSON file.

        Args:
            path: File path. Defaults to settings.index_path.
        """
        path = Path(path or settings.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": self._entries,
            "count": len(self._entries),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved %d entries to %s", len(self._entries), path)

    def load(self, path: str | Path | None = None) -> bool:
        """Load the store from a JSON file.

        Args:
            path: File path. Defaults to settings.index_path.

        Returns:
            True if the file was loaded, False if it doesn't exist.
        """
        path = Path(path or settings.index_path)
        if not path.exists():
            logger.warning("Index file not found: %s", path)
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._entries = data.get("entries", [])
            self._index_loaded = True
            logger.info("Loaded %d entries from %s", len(self._entries), path)
            return True
        except Exception as exc:
            logger.error("Failed to load index from %s: %s", path, exc)
            return False

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._entries = []
        logger.info("Cleared vector store")


# Module-level singleton
_store: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    """Get or create the VectorStore singleton.

    Automatically loads from disk on first access if the index file exists.
    """
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store


def reset_vectorstore() -> VectorStore:
    """Create a fresh VectorStore (useful for testing)."""
    global _store
    _store = VectorStore()
    return _store
