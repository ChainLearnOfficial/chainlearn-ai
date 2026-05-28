"""Tests for the feedback engine service."""

import pytest

from src.models.quiz import QuizQuestion
from src.services.feedback_engine import FeedbackEngine


class TestFeedbackScoreComputation:
    """Test score calculation in the feedback engine."""

    @pytest.fixture
    def engine(self):
        return FeedbackEngine()

    def test_perfect_score(self, engine):
        user = [0, 1, 2]
        correct = [0, 1, 2]
        assert engine._compute_score(user, correct) == 100.0

    def test_zero_score(self, engine):
        user = [2, 0, 1]
        correct = [0, 1, 2]
        assert engine._compute_score(user, correct) == 0.0

    def test_partial_score(self, engine):
        user = [0, 0, 2]
        correct = [0, 1, 2]
        assert engine._compute_score(user, correct) == pytest.approx(66.67, abs=0.1)

    def test_empty_answers(self, engine):
        assert engine._compute_score([], []) == 0.0

    def test_single_question(self, engine):
        assert engine._compute_score([1], [1]) == 100.0
        assert engine._compute_score([0], [1]) == 0.0


class TestQuickFeedback:
    """Test the quick_feedback method (no LLM call)."""

    @pytest.fixture
    def engine(self):
        return FeedbackEngine()

    @pytest.fixture
    def sample_question(self):
        return QuizQuestion(
            prompt="What is DeFi?",
            options=["A bank", "Decentralized Finance", "A protocol", "A token"],
            correct_index=1,
            explanation="DeFi stands for Decentralized Finance.",
        )

    @pytest.mark.asyncio
    async def test_correct_answer_feedback(self, engine, sample_question):
        feedback = await engine.quick_feedback(sample_question, user_answer=1)
        assert "Correct" in feedback
        assert "Decentralized Finance" in feedback

    @pytest.mark.asyncio
    async def test_wrong_answer_feedback(self, engine, sample_question):
        feedback = await engine.quick_feedback(sample_question, user_answer=0)
        assert "Incorrect" in feedback
        assert "bank" in feedback  # User's wrong choice
        assert "Decentralized Finance" in feedback  # Correct answer

    @pytest.mark.asyncio
    async def test_out_of_range_answer(self, engine, sample_question):
        feedback = await engine.quick_feedback(sample_question, user_answer=99)
        assert "Incorrect" in feedback
        assert "N/A" in feedback


class TestFeedbackDataClass:
    """Test the Feedback data class."""

    def test_to_dict(self):
        from src.services.feedback_engine import Feedback

        fb = Feedback(
            feedback_text="Good job!",
            weak_areas=["consensus", "hashing"],
            recommendations=["Review chapter 2", "Practice more"],
        )
        d = fb.to_dict()
        assert d["feedback"] == "Good job!"
        assert len(d["weak_areas"]) == 2
        assert len(d["recommendations"]) == 2
