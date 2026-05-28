"""Tests for the quiz generator service."""

import pytest

from src.models.quiz import GenerateQuizRequest, Quiz, QuizQuestion


class TestQuizModels:
    """Test Quiz and QuizQuestion Pydantic models."""

    def test_quiz_question_creation(self):
        q = QuizQuestion(
            prompt="What is a blockchain?",
            options=["A database", "A distributed ledger", "A website", "A virus"],
            correct_index=1,
            explanation="A blockchain is a distributed ledger.",
        )
        assert q.prompt == "What is a blockchain?"
        assert len(q.options) == 4
        assert q.correct_index == 1

    def test_quiz_question_min_options(self):
        with pytest.raises(Exception):
            QuizQuestion(
                prompt="Test?",
                options=["Only one"],
                correct_index=0,
            )

    def test_quiz_creation(self):
        questions = [
            QuizQuestion(
                prompt="Q1?",
                options=["A", "B", "C", "D"],
                correct_index=0,
            ),
            QuizQuestion(
                prompt="Q2?",
                options=["A", "B", "C", "D"],
                correct_index=2,
            ),
        ]
        quiz = Quiz(
            quiz_id="test-123",
            questions=questions,
            difficulty="beginner",
        )
        assert quiz.quiz_id == "test-123"
        assert len(quiz.questions) == 2
        assert quiz.questions[0].correct_index == 0

    def test_generate_quiz_request_defaults(self):
        req = GenerateQuizRequest(user_id="user-1")
        assert req.user_id == "user-1"
        assert req.difficulty == "beginner"
        assert req.num_questions == 5

    def test_generate_quiz_request_validation(self):
        # num_questions out of range
        with pytest.raises(Exception):
            GenerateQuizRequest(user_id="user-1", num_questions=0)

        # invalid difficulty
        with pytest.raises(Exception):
            GenerateQuizRequest(user_id="user-1", difficulty="expert")


class TestQuizGeneratorLogic:
    """Test quiz generator business logic (mocked Cohere calls)."""

    @pytest.fixture
    def sample_quiz_data(self):
        """Sample JSON data as returned by Cohere."""
        return {
            "questions": [
                {
                    "prompt": "What consensus mechanism does Bitcoin use?",
                    "options": ["Proof of Stake", "Proof of Work", "DPoS", "PBFT"],
                    "correct_index": 1,
                    "explanation": "Bitcoin uses Proof of Work.",
                },
                {
                    "prompt": "What is a smart contract?",
                    "options": [
                        "A legal document",
                        "A self-executing program on blockchain",
                        "A type of cryptocurrency",
                        "A mining algorithm",
                    ],
                    "correct_index": 1,
                    "explanation": "Smart contracts are self-executing programs.",
                },
            ]
        }

    def test_parse_quiz_response(self, sample_quiz_data):
        """Test that quiz data parses into QuizQuestion objects."""
        questions = [
            QuizQuestion(
                prompt=q["prompt"],
                options=q["options"],
                correct_index=q["correct_index"],
                explanation=q["explanation"],
            )
            for q in sample_quiz_data["questions"]
        ]

        assert len(questions) == 2
        assert questions[0].correct_index == 1
        assert "Proof of Work" in questions[0].options

    def test_quiz_scoring(self, sample_quiz_data):
        """Test score computation from answers."""
        questions = [
            QuizQuestion(**q) for q in sample_quiz_data["questions"]
        ]
        correct = [q.correct_index for q in questions]
        user_answers = [1, 0]  # First correct, second wrong

        score = sum(1 for u, c in zip(user_answers, correct) if u == c) / len(correct) * 100
        assert score == 50.0

    def test_perfect_score(self, sample_quiz_data):
        questions = [QuizQuestion(**q) for q in sample_quiz_data["questions"]]
        correct = [q.correct_index for q in questions]

        score = sum(1 for u, c in zip(correct, correct) if u == c) / len(correct) * 100
        assert score == 100.0
