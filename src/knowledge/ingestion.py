"""Document loading and chunking for the knowledge base.

Handles loading Markdown files from disk, splitting them into
overlapping chunks suitable for embedding and retrieval.
"""

import logging
import os
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


def load_markdown_files(directory: str | Path) -> list[dict[str, str]]:
    """Recursively load all Markdown files from a directory.

    Args:
        directory: Root directory to scan.

    Returns:
        List of dicts with keys 'path', 'filename', and 'content'.
    """
    directory = Path(directory)
    documents: list[dict[str, str]] = []

    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return documents

    for md_file in sorted(directory.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            documents.append(
                {
                    "path": str(md_file),
                    "filename": md_file.name,
                    "content": content,
                }
            )
            logger.debug("Loaded %s (%d chars)", md_file, len(content))
        except Exception as exc:
            logger.error("Failed to load %s: %s", md_file, exc)

    logger.info("Loaded %d Markdown files from %s", len(documents), directory)
    return documents


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    Attempts to split on paragraph boundaries (double newlines) when possible,
    falling back to character-level splitting.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Split into paragraphs first
    paragraphs = text.split("\n\n")

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If a single paragraph exceeds chunk_size, split it by sentences
        if para_len > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep overlap
                overlap_text = "\n\n".join(current_chunk)
                overlap_chars = overlap_text[-chunk_overlap:] if len(overlap_text) > chunk_overlap else overlap_text
                current_chunk = [overlap_chars] if overlap_chars.strip() else []
                current_length = len(overlap_chars)

            # Split long paragraph by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_length + len(sentence) + 2 > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Overlap: keep last portion
                    last = current_chunk[-1]
                    overlap_text = last[-chunk_overlap:] if len(last) > chunk_overlap else last
                    current_chunk = [overlap_text] if overlap_text.strip() else []
                    current_length = len(overlap_text)
                current_chunk.append(sentence)
                current_length += len(sentence) + 2
        elif current_length + para_len + 2 > chunk_size:
            # Flush current chunk
            chunks.append("\n\n".join(current_chunk))
            # Overlap: keep tail of the last paragraph
            last = current_chunk[-1] if current_chunk else ""
            overlap_text = last[-chunk_overlap:] if len(last) > chunk_overlap else last
            current_chunk = [overlap_text] if overlap_text.strip() else []
            current_length = len(overlap_text)
            current_chunk.append(para)
            current_length += para_len + 2
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    # Flush remaining
    if current_chunk:
        final = "\n\n".join(current_chunk).strip()
        if final:
            chunks.append(final)

    logger.debug("Chunked %d chars into %d chunks", len(text), len(chunks))
    return chunks


def ingest_knowledge_base(base_dir: str | Path) -> list[dict[str, str]]:
    """Load and chunk all documents from the knowledge base.

    Args:
        base_dir: Root directory of the knowledge base.

    Returns:
        List of dicts with keys 'text', 'source', and 'chunk_index'.
    """
    documents = load_markdown_files(base_dir)
    all_chunks: list[dict[str, str]] = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source": doc["path"],
                    "chunk_index": i,
                }
            )

    logger.info(
        "Ingested %d total chunks from %d documents",
        len(all_chunks),
        len(documents),
    )
    return all_chunks
