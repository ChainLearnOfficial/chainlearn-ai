"""User profile model for personalizing learning content."""

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Learner profile used to tailor generated content."""

    background: str = Field(
        default="beginner",
        description="Current knowledge level",
        pattern=r"^(beginner|intermediate|advanced)$",
    )
    learning_goal: str = Field(
        default="",
        description="What the user wants to learn or achieve",
    )
    pace: str = Field(
        default="moderate",
        description="Preferred learning pace",
        pattern=r"^(fast|moderate|slow)$",
    )
    language: str = Field(
        default="en",
        description="Preferred language code (e.g. en, es, fr)",
    )
