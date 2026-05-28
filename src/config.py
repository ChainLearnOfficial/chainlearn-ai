"""Application configuration via Pydantic Settings.

Reads from environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Cohere
    cohere_api_key: str
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_generate_model: str = "command-nightly"

    # Application
    app_name: str = "chainlearn-ai"
    debug: bool = False
    log_level: str = "INFO"

    # Knowledge base
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_dimensions: int = 1024
    index_path: str = "src/knowledge/data/index.json"

    # Generation defaults
    default_max_tokens: int = 2048
    default_temperature: float = 0.7

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


settings = Settings()
