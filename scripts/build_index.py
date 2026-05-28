#!/usr/bin/env python3
"""Build the vector index from the knowledge base.

Usage:
    python scripts/build_index.py [--source-dir PATH] [--output PATH]

Examples:
    python scripts/build_index.py
    python scripts/build_index.py --source-dir ./my-knowledge --output ./my-index.json
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.index_builder import build_index  # noqa: E402


async def main(source_dir: str | None, output: str | None) -> None:
    """Run the index builder."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger = logging.getLogger("build_index")
    logger.info("Starting index build...")

    try:
        store = await build_index(
            source_dir=source_dir,
            output_path=output,
        )
        logger.info("Index build complete: %d chunks stored", store.count)
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Directory containing knowledge-base Markdown files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the index JSON file",
    )
    args = parser.parse_args()

    asyncio.run(main(args.source_dir, args.output))
