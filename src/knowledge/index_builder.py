"""Index builder that orchestrates ingestion, embedding, and storage.

Reads documents from the knowledge base, generates embeddings,
and populates the vector store.
"""

import asyncio
import logging
import time
from pathlib import Path

from src.config import settings
from src.knowledge.ingestion import ingest_knowledge_base
from src.knowledge.vectorstore import VectorStore, get_vectorstore
from src.services.embedding import get_embedding_service

logger = logging.getLogger(__name__)

# Default knowledge base directory
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent.parent / "knowledge-base"


async def build_index(
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> VectorStore:
    """Build the vector index from knowledge-base documents.

    Args:
        source_dir: Directory containing source Markdown files.
                    Defaults to the project's knowledge-base/ directory.
        output_path: Where to save the index JSON. Defaults to settings.index_path.

    Returns:
        The populated VectorStore.
    """
    source_dir = Path(source_dir) if source_dir else KNOWLEDGE_BASE_DIR
    output_path = output_path or settings.index_path

    logger.info("Building index from %s", source_dir)
    start = time.time()

    # Step 1: Ingest and chunk documents
    chunks = ingest_knowledge_base(source_dir)
    if not chunks:
        logger.warning("No chunks found in %s — creating empty index", source_dir)
        store = get_vectorstore()
        store.save(output_path)
        return store

    logger.info("Ingested %d chunks", len(chunks))

    # Step 2: Generate embeddings
    embedder = get_embedding_service()
    texts = [c["text"] for c in chunks]

    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = await embedder.embed_documents(texts)

    # Step 3: Build vector store
    store = get_vectorstore()
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

    # Step 4: Persist
    store.save(output_path)

    elapsed = time.time() - start
    logger.info(
        "Index built: %d chunks, %d embeddings, saved to %s (%.1fs)",
        len(chunks),
        len(embeddings),
        output_path,
        elapsed,
    )
    return store


def build_index_sync(
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> VectorStore:
    """Synchronous wrapper for build_index().

    Convenient for scripts and CLI usage.
    """
    return asyncio.run(build_index(source_dir, output_path))
