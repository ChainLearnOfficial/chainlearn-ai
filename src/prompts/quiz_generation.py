"""Prompt templates for quiz generation."""


def build_quiz_prompt(
    topic: str,
    difficulty: str,
    num_questions: int,
    context_chunks: list[str] | None = None,
) -> str:
    """Build a prompt for generating a quiz.

    Args:
        topic: The topic to quiz on.
        difficulty: Target difficulty level.
        num_questions: Number of questions to generate.
        context_chunks: Optional retrieved knowledge-base passages.

    Returns:
        Formatted prompt string.
    """
    context_section = ""
    if context_chunks:
        joined = "\n---\n".join(context_chunks)
        context_section = f"""

REFERENCE MATERIAL (base questions on this):
<reference>
{joined}
</reference>
"""

    return f"""Generate {num_questions} multiple-choice quiz questions about "{topic}"
at {difficulty} difficulty.{context_section}

Requirements:
- Each question must test understanding, not just recall
- Provide 4 options per question (A, B, C, D)
- Exactly one option is correct
- Include a brief explanation for the correct answer
- Vary question types: conceptual, application, and scenario-based

Respond with a JSON object:
{{
  "questions": [
    {{
      "prompt": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_index": 0,
      "explanation": "Why option A is correct"
    }}
  ]
}}

Generate exactly {num_questions} questions."""


def build_question_from_content_prompt(
    content: str,
    difficulty: str,
    num_questions: int,
) -> str:
    """Build a prompt for generating questions from specific content."""
    return f"""Based on the following content, generate {num_questions} multiple-choice
questions at {difficulty} difficulty.

CONTENT:
{content}

Respond with a JSON object:
{{
  "questions": [
    {{
      "prompt": "Question text?",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "Explanation"
    }}
  ]
}}"""
