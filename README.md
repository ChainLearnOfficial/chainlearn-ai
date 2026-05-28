# ChainLearn AI

AI-powered content generation service for the ChainLearn blockchain learning platform.
Generates courses, quizzes, and personalized feedback using Cohere LLMs with RAG
(Retrieval-Augmented Generation) from a blockchain knowledge base.

## Architecture

```
FastAPI App
    |
    +-- Routes (courses, quizzes, feedback, health)
    |       |
    |       +-- Services (generation, quiz_generator, feedback_engine)
    |               |
    |               +-- CohereClient (async API wrapper)
    |               +-- EmbeddingService (embed-english-v3.0)
    |               +-- VectorStore (in-memory + JSON persistence)
    |
    +-- Knowledge Pipeline (ingestion -> chunking -> embedding -> storage)
```

## Prerequisites

- Python 3.11+
- A Cohere API key ([get one here](https://dashboard.cohere.com/api-keys))

## Setup

### 1. Clone and install

```bash
cd chainlearn-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Required variables:

```env
COHERE_API_KEY=your-cohere-api-key-here
```

Optional variables (with defaults):

```env
COHERE_EMBED_MODEL=embed-english-v3.0
COHERE_GENERATE_MODEL=command-nightly
DEBUG=false
LOG_LEVEL=INFO
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

### 3. Build the knowledge index (optional)

If you have content in `knowledge-base/`, build the vector index:

```bash
python scripts/build_index.py
```

This processes all Markdown files, generates embeddings, and saves the index
to `src/knowledge/data/index.json`. The index is loaded automatically at startup.

### 4. Run the server

```bash
uvicorn src.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### GET /health

Service health check.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "chainlearn-ai",
  "version": "0.1.0",
  "knowledge_base": {
    "loaded": true,
    "chunks": 42
  }
}
```

### POST /generate-course

Generate a structured course on a topic.

```bash
curl -X POST http://localhost:8000/generate-course \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Smart Contracts on Stellar",
    "difficulty": "intermediate",
    "background": "beginner",
    "learning_goal": "Build and deploy Soroban smart contracts",
    "num_modules": 4
  }'
```

Response:
```json
{
  "course_id": "uuid-here",
  "title": "Smart Contracts on Stellar",
  "description": "Learn to build...",
  "difficulty": "intermediate",
  "estimated_hours": 8.0,
  "modules": [
    {
      "title": "Introduction to Soroban",
      "content": "Full markdown content...",
      "summary": "Overview of Soroban...",
      "order": 0
    }
  ]
}
```

### POST /generate-quiz

Generate a quiz on a topic or module.

```bash
curl -X POST http://localhost:8000/generate-quiz \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "topic": "Blockchain Consensus Mechanisms",
    "difficulty": "beginner",
    "num_questions": 5
  }'
```

Response:
```json
{
  "quiz_id": "uuid-here",
  "course_id": "",
  "module_id": "",
  "difficulty": "beginner",
  "questions": [
    {
      "prompt": "What consensus mechanism does Bitcoin use?",
      "options": ["Proof of Stake", "Proof of Work", "Delegated PoS", "PBFT"],
      "correct_index": 1,
      "explanation": "Bitcoin uses Proof of Work..."
    }
  ]
}
```

### POST /generate-feedback

Get personalized feedback on quiz answers.

```bash
curl -X POST http://localhost:8000/generate-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "quiz_id": "quiz-123",
    "questions": [
      {
        "prompt": "What is DeFi?",
        "options": ["A bank", "Decentralized Finance", "A protocol", "A token"],
        "correct_index": 1,
        "explanation": "DeFi stands for Decentralized Finance."
      }
    ],
    "user_answers": [0],
    "correct_answers": [1],
    "score": 0.0,
    "topic": "DeFi Fundamentals"
  }'
```

Response:
```json
{
  "feedback": "You answered 0 out of 1 questions correctly...",
  "weak_areas": ["DeFi basics"],
  "recommendations": [
    "Review the fundamentals of decentralized finance",
    "Study how DeFi protocols differ from traditional finance"
  ]
}
```

## Knowledge Base

Place Markdown files in `knowledge-base/` to provide RAG context for content
generation. The ingestion pipeline:

1. Recursively loads all `.md` files
2. Splits into overlapping chunks (configurable size/overlap)
3. Generates embeddings with Cohere `embed-english-v3.0`
4. Stores in an in-memory vector index with JSON persistence

See `knowledge-base/mastering-blockchain/README.md` for content guidelines.

## Testing

Run the test suite:

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Run the manual integration test (requires valid API key):

```bash
python scripts/test_generation.py
```

## Docker

Build and run with Docker:

```bash
docker build -t chainlearn-ai .
docker run -p 8000:8000 --env-file .env chainlearn-ai
```

## Project Structure

```
chainlearn-ai/
├── src/
│   ├── main.py                  # FastAPI app with CORS, lifespan, routers
│   ├── config.py                # Pydantic Settings (env vars)
│   ├── services/
│   │   ├── cohere_client.py     # Async Cohere SDK wrapper
│   │   ├── embedding.py         # Embedding service (embed-english-v3.0)
│   │   ├── generation.py        # Course generation with RAG
│   │   ├── quiz_generator.py    # Quiz creation from content
│   │   └── feedback_engine.py   # Intelligent quiz feedback
│   ├── knowledge/
│   │   ├── ingestion.py         # Markdown loading + chunking
│   │   ├── vectorstore.py       # In-memory vector store + cosine search
│   │   └── index_builder.py     # Build/update knowledge index
│   ├── prompts/
│   │   ├── system.py            # System prompts for Cohere
│   │   ├── course_generation.py # Course generation prompts
│   │   ├── quiz_generation.py   # Quiz generation prompts
│   │   └── feedback.py          # Feedback prompts
│   ├── models/
│   │   ├── course.py            # Course, CourseModule schemas
│   │   ├── quiz.py              # Quiz, QuizQuestion schemas
│   │   └── user_profile.py      # UserProfile schema
│   └── routes/
│       ├── courses.py           # POST /generate-course
│       ├── quizzes.py           # POST /generate-quiz
│       ├── feedback.py          # POST /generate-feedback
│       └── health.py            # GET /health
├── knowledge-base/              # Source material for RAG
├── scripts/
│   ├── build_index.py           # CLI index builder
│   └── test_generation.py       # Integration test script
├── tests/                       # Unit tests
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```
