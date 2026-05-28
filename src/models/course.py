"""Course-related Pydantic models."""

from pydantic import BaseModel, Field


class CourseModule(BaseModel):
    """A single module within a course."""

    title: str = Field(..., description="Module title")
    content: str = Field(..., description="Full module content in Markdown")
    summary: str = Field(..., description="Brief summary of the module")
    order: int = Field(..., ge=0, description="Zero-based ordering index")


class Course(BaseModel):
    """A complete course composed of modules."""

    course_id: str = Field(..., description="Unique course identifier")
    title: str = Field(..., description="Course title")
    description: str = Field(..., description="Course overview")
    modules: list[CourseModule] = Field(default_factory=list)
    difficulty: str = Field(..., description="beginner / intermediate / advanced")
    estimated_hours: float = Field(..., gt=0, description="Estimated completion time in hours")


class GenerateCourseRequest(BaseModel):
    """Request body for POST /generate-course."""

    topic: str = Field(..., min_length=3, description="Course topic")
    difficulty: str = Field(
        default="beginner",
        pattern=r"^(beginner|intermediate|advanced)$",
    )
    background: str = Field(
        default="beginner",
        description="Learner background level",
        pattern=r"^(beginner|intermediate|advanced)$",
    )
    learning_goal: str = Field(default="", description="What the learner wants to achieve")
    num_modules: int = Field(default=5, ge=1, le=10, description="Number of modules to generate")
