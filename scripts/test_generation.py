#!/usr/bin/env python3
"""Manual test script for verifying Cohere integration.

Sends a test prompt to Cohere and prints the response.
Useful for validating API key and model access before running the server.

Usage:
    python scripts/test_generation.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.cohere_client import get_cohere_client  # noqa: E402
from src.services.embedding import get_embedding_service  # noqa: E402


async def test_generation() -> None:
    """Test text generation via Cohere."""
    client = get_cohere_client()

    print("\n=== Testing Text Generation ===")
    prompt = "Explain what a smart contract is in 2-3 sentences, suitable for a beginner."
    print(f"Prompt: {prompt}\n")

    try:
        response = await client.generate(prompt=prompt, max_tokens=200)
        print(f"Response:\n{response}")
        print("[PASS] Generation successful")
    except Exception as exc:
        print(f"[FAIL] Generation error: {exc}")


async def test_json_generation() -> None:
    """Test JSON generation via Cohere."""
    client = get_cohere_client()

    print("\n=== Testing JSON Generation ===")
    prompt = """Return a JSON object with exactly these keys: "concept" (string), "difficulty" (integer 1-5), "topics" (list of 3 strings).
The concept should be about blockchain consensus mechanisms."""
    print(f"Prompt: {prompt}\n")

    try:
        result = await client.generate_json(prompt=prompt, max_tokens=300)
        print(f"Parsed JSON:\n{result}")
        print("[PASS] JSON generation successful")
    except Exception as exc:
        print(f"[FAIL] JSON generation error: {exc}")


async def test_embedding() -> None:
    """Test embedding generation via Cohere."""
    embedder = get_embedding_service()

    print("\n=== Testing Embedding ===")
    texts = [
        "Blockchain is a distributed ledger technology.",
        "Smart contracts are self-executing programs on a blockchain.",
        "DeFi stands for decentralized finance.",
    ]
    print(f"Embedding {len(texts)} texts...")

    try:
        embeddings = await embedder.embed_documents(texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Dimension: {len(embeddings[0])}")
        print(f"Sample (first 5 values): {embeddings[0][:5]}")
        print("[PASS] Embedding successful")
    except Exception as exc:
        print(f"[FAIL] Embedding error: {exc}")


async def test_query_embedding() -> None:
    """Test query embedding and similarity."""
    embedder = get_embedding_service()
    from src.knowledge.vectorstore import cosine_similarity

    print("\n=== Testing Query Similarity ===")
    docs = [
        "Bitcoin uses proof of work for consensus.",
        "Ethereum transitioned to proof of stake.",
        "Stellar uses the Stellar Consensus Protocol.",
    ]
    query = "How does Stellar achieve consensus?"

    try:
        doc_embeddings = await embedder.embed_documents(docs)
        query_embedding = await embedder.embed_query(query)

        print(f"Query: {query}\n")
        for doc, emb in zip(docs, doc_embeddings):
            sim = cosine_similarity(query_embedding, emb)
            print(f"  Similarity {sim:.4f}: {doc}")

        print("[PASS] Similarity search successful")
    except Exception as exc:
        print(f"[FAIL] Similarity error: {exc}")


async def main() -> None:
    """Run all tests."""
    logging.basicConfig(level=logging.WARNING)

    print("ChainLearn AI — Integration Test")
    print("=" * 40)

    await test_generation()
    await test_json_generation()
    await test_embedding()
    await test_query_embedding()

    print("\n" + "=" * 40)
    print("All tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
