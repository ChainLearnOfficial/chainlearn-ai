"""Prompt templates for course content generation."""

from src.models.user_profile import UserProfile


def build_course_prompt(
    topic: str,
    difficulty: str,
    num_modules: int,
    profile: UserProfile,
    context_chunks: list[str] | None = None,
) -> str:
    """Build a prompt for generating a full course.

    Args:
        topic: The course topic.
        difficulty: Target difficulty level.
        num_modules: How many modules to generate.
        profile: The learner's profile for personalization.
        context_chunks: Optional retrieved knowledge-base passages.

    Returns:
        Formatted prompt string.
    """
    context_section = ""
    if context_chunks:
        joined = "\n---\n".join(context_chunks)
        context_section = f"""

REFERENCE MATERIAL (use this to ground your content):
<reference>
{joined}
</reference>
"""

    return f"""Generate a {num_modules}-module course on "{topic}" at {difficulty} level.

LEARNER PROFILE:
- Background: {profile.background}
- Learning goal: {profile.learning_goal or "General understanding"}
- Pace: {profile.pace}
- Language: {profile.language}
{context_section}
Respond with a JSON object matching this schema:
{{
  "title": "Course Title",
  "description": "2-3 sentence course overview",
  "difficulty": "{difficulty}",
  "estimated_hours": 10.0,
  "modules": [
    {{
      "title": "Module Title",
      "content": "Full module content in Markdown (at least 3 paragraphs)",
      "summary": "One-paragraph summary",
      "order": 0
    }}
  ]
}}

Generate exactly {num_modules} modules. Make content substantive — each module should be
at least 300 words. Order modules from foundational to advanced."""


def build_module_prompt(
    topic: str,
    module_title: str,
    difficulty: str,
    profile: UserProfile,
    context_chunks: list[str] | None = None,
) -> str:
    """Build a prompt for generating or expanding a single module."""
    context_section = ""
    if context_chunks:
        joined = "\n---\n".join(context_chunks)
        context_section = f"\nREFERENCE:\n{joined}\n"

    return f"""Write a detailed course module titled "{module_title}" for a course on "{topic}".
Difficulty: {difficulty}. Learner background: {profile.background}.
{context_section}
Respond with JSON:
{{
  "title": "{module_title}",
  "content": "Full Markdown content (at least 300 words)",
  "summary": "One-paragraph summary"
}}"""
