# Mastering Blockchain Knowledge Base

This directory contains source material used to build the RAG (Retrieval-Augmented Generation)
knowledge index for the ChainLearn AI service.

## Purpose

The knowledge base provides domain-specific context for:

- **Course generation**: Grounding generated course content in authoritative blockchain material
- **Quiz generation**: Ensuring quiz questions reference real concepts and terminology
- **Feedback generation**: Providing accurate explanations when users answer incorrectly

## Adding Content

Place `.md` (Markdown) files in this directory. The ingestion pipeline will:

1. Load all Markdown files recursively
2. Split them into overlapping chunks (default 500 tokens, 50 overlap)
3. Generate embeddings using Cohere `embed-english-v3.0`
4. Store chunks and embeddings in the vector store

## Recommended Structure

```
mastering-blockchain/
├── 01-introduction.md
├── 02-cryptography.md
├── 03-consensus-mechanisms.md
├── 04-smart-contracts.md
├── 05-stellar-network.md
└── 06-defi-fundamentals.md
```

## Rebuilding the Index

After adding or modifying content, rebuild the index:

```bash
python scripts/build_index.py
```

The index is stored in `src/knowledge/data/index.json` and loaded at application startup.
