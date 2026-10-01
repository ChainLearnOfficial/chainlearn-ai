"""Tests for feedback request list-length validation."""

import pytest
from pydantic import ValidationError

from src.routes.feedback import GenerateFeedbackRequest


def _make_question(prompt: str = "What is 1+1?") -> dict:
    """Return a minimal QuizQuestion dict."""
    return {
        "prompt": prompt,
        "options": ["1", "2", "3", "4"],
        "correct_index": 1,
        "explanation": "Basic arithmetic.",
    }


class TestFeedbackRequestListLengthValidation:
    """Validate that mismatched list lengths are rejected."""

    def test_matching_lengths_accepted(self):
        """Equal-length lists should pass validation."""
        req = GenerateFeedbackRequest(
            quiz_id="q1",
            questions=[_make_question(), _make_question()],
            user_answers=[0, 1],
            correct_answers=[1, 1],
        )
        assert len(req.questions) == 2
        assert len(req.user_answers) == 2
        assert len(req.correct_answers) == 2

    def test_user_answers_shorter_rejected(self):
        """Fewer user_answers than questions must raise."""
        with pytest.raises(ValidationError, match="List length mismatch"):
            GenerateFeedbackRequest(
                quiz_id="q2",
                questions=[_make_question(), _make_question(), _make_question()],
                user_answers=[0],
                correct_answers=[1, 1, 1],
            )

    def test_correct_answers_shorter_rejected(self):
        """Fewer correct_answers than questions must raise."""
        with pytest.raises(ValidationError, match="List length mismatch"):
            GenerateFeedbackRequest(
                quiz_id="q3",
                questions=[_make_question(), _make_question()],
                user_answers=[0, 1],
                correct_answers=[1],
            )

    def test_extra_user_answers_rejected(self):
        """More user_answers than questions must raise."""
        with pytest.raises(ValidationError, match="List length mismatch"):
            GenerateFeedbackRequest(
                quiz_id="q4",
                questions=[_make_question()],
                user_answers=[0, 1, 2],
                correct_answers=[1],
            )

    def test_all_three_different_lengths_rejected(self):
        """All three lists with different lengths must raise."""
        with pytest.raises(ValidationError, match="List length mismatch"):
            GenerateFeedbackRequest(
                quiz_id="q5",
                questions=[_make_question()],
                user_answers=[0, 1],
                correct_answers=[1, 1, 1],
            )

    def test_empty_lists_accepted(self):
        """Three empty lists have matching length (0) and should pass."""
        req = GenerateFeedbackRequest(
            quiz_id="q6",
            questions=[],
            user_answers=[],
            correct_answers=[],
        )
        assert len(req.questions) == 0

    def test_single_question_matching_accepted(self):
        """Single-element lists should pass validation."""
        req = GenerateFeedbackRequest(
            quiz_id="q7",
            questions=[_make_question()],
            user_answers=[1],
            correct_answers=[1],
        )
        assert len(req.questions) == 1

    def test_error_message_includes_lengths(self):
        """Rejection message should report actual lengths for debugging."""
        with pytest.raises(ValidationError) as exc_info:
            GenerateFeedbackRequest(
                quiz_id="q8",
                questions=[_make_question(), _make_question()],
                user_answers=[0],
                correct_answers=[1, 1],
            )
        error_text = str(exc_info.value)
        assert "questions=2" in error_text
        assert "user_answers=1" in error_text
        assert "correct_answers=2" in error_text
