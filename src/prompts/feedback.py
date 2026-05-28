"""Prompt templates for quiz feedback generation."""

from src.models.quiz import QuizQuestion


def build_feedback_prompt(
    questions: list[QuizQuestion],
    user_answers: list[int],
    correct_answers: list[int],
    score: float,
    topic: str = "",
) -> str:
    """Build a prompt for generating personalized quiz feedback.

    Args:
        questions: The quiz questions.
        user_answers: Indices of user's selected answers.
        correct_answers: Indices of correct answers.
        score: Score as a percentage (0-100).
        topic: Optional topic context.

    Returns:
        Formatted prompt string.
    """
    question_lines: list[str] = []
    for i, q in enumerate(questions):
        user_ans = user_answers[i] if i < len(user_answers) else -1
        correct_ans = correct_answers[i] if i < len(correct_answers) else -1
        is_correct = user_ans == correct_ans

        question_lines.append(
            f"Q{i + 1}: {q.prompt}\n"
            f"  Options: {', '.join(q.options)}\n"
            f"  User answered: {user_ans} ({'CORRECT' if is_correct else 'WRONG'})\n"
            f"  Correct answer: {correct_ans}"
        )

    questions_text = "\n\n".join(question_lines)
    topic_line = f"Topic: {topic}\n" if topic else ""

    return f"""Analyze this quiz performance and provide detailed, constructive feedback.

{topic_line}Score: {score:.1f}%

QUESTIONS AND ANSWERS:
{questions_text}

Respond with a JSON object:
{{
  "feedback": "Detailed paragraph of overall feedback (encouraging but honest)",
  "weak_areas": ["topic area 1", "topic area 2"],
  "recommendations": [
    "Specific study recommendation 1",
    "Specific study recommendation 2",
    "Specific study recommendation 3"
  ]
}}

Guidelines:
- If score >= 80%: praise strengths, suggest advanced topics
- If score 50-79%: identify specific gaps, suggest focused review
- If score < 50%: be encouraging, recommend revisiting fundamentals
- Always include at least 2 recommendations
- Reference specific questions when discussing weak areas"""
