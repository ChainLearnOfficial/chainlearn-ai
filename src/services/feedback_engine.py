"""Intelligent feedback engine for quiz results."""

import logging

from src.models.quiz import QuizQuestion
from src.prompts.feedback import build_feedback_prompt
from src.prompts.system import FEEDBACK_SYSTEM_PROMPT
from src.services.cohere_client import get_cohere_client

logger = logging.getLogger(__name__)


class Feedback:
    """Structured feedback from the feedback engine."""

    def __init__(
        self,
        feedback_text: str,
        weak_areas: list[str],
        recommendations: list[str],
    ) -> None:
        self.feedback_text = feedback_text
        self.weak_areas = weak_areas
        self.recommendations = recommendations

    def to_dict(self) -> dict:
        return {
            "feedback": self.feedback_text,
            "weak_areas": self.weak_areas,
            "recommendations": self.recommendations,
        }


class FeedbackEngine:
    """Generates personalized feedback on quiz performance."""

    def __init__(self) -> None:
        self._client = get_cohere_client()

    def _compute_score(self, user_answers: list[int], correct_answers: list[int]) -> float:
        """Compute score as a percentage."""
        if not correct_answers:
            return 0.0
        correct_count = sum(
            1 for u, c in zip(user_answers, correct_answers) if u == c
        )
        return (correct_count / len(correct_answers)) * 100.0

    async def generate_feedback(
        self,
        questions: list[QuizQuestion],
        user_answers: list[int],
        correct_answers: list[int],
        score: float | None = None,
        topic: str = "",
    ) -> Feedback:
        """Generate personalized feedback for a quiz attempt.

        Args:
            questions: The quiz questions.
            user_answers: User's selected answer indices.
            correct_answers: Correct answer indices.
            score: Pre-computed score percentage. If None, computed from answers.
            topic: Optional topic context.

        Returns:
            Feedback object with text, weak areas, and recommendations.
        """
        if score is None:
            score = self._compute_score(user_answers, correct_answers)

        prompt = build_feedback_prompt(
            questions=questions,
            user_answers=user_answers,
            correct_answers=correct_answers,
            score=score,
            topic=topic,
        )

        data = await self._client.generate_json(
            prompt=prompt,
            system_prompt=FEEDBACK_SYSTEM_PROMPT,
        )

        feedback = Feedback(
            feedback_text=data.get("feedback", "Keep studying and you will improve!"),
            weak_areas=data.get("weak_areas", []),
            recommendations=data.get("recommendations", []),
        )

        logger.info(
            "Generated feedback for score %.1f%% — %d weak areas identified",
            score,
            len(feedback.weak_areas),
        )
        return feedback

    async def quick_feedback(
        self,
        question: QuizQuestion,
        user_answer: int,
    ) -> str:
        """Generate brief feedback for a single question.

        Args:
            question: The quiz question.
            user_answer: User's selected answer index.

        Returns:
            Brief feedback string.
        """
        is_correct = user_answer == question.correct_index
        if is_correct:
            return f"Correct! {question.explanation}"

        correct_option = question.options[question.correct_index]
        user_option = question.options[user_answer] if 0 <= user_answer < len(question.options) else "N/A"
        return (
            f"Incorrect. You chose '{user_option}', but the correct answer is '{correct_option}'. "
            f"{question.explanation}"
        )


# Module-level singleton
_engine: FeedbackEngine | None = None


def get_feedback_engine() -> FeedbackEngine:
    """Get or create the FeedbackEngine singleton."""
    global _engine
    if _engine is None:
        _engine = FeedbackEngine()
    return _engine
