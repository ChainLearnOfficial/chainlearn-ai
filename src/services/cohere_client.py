"""Async wrapper around the Cohere Python SDK.

Provides a unified interface for text generation and embeddings,
handling retries, errors, and response parsing.
"""

import json
import logging
from typing import Any

import cohere

from src.config import settings

logger = logging.getLogger(__name__)


class CohereClient:
    """Async Cohere API client with generate and embed methods."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.cohere_api_key
        self._client = cohere.ClientV2(api_key=self._api_key)
        self._embed_model = settings.cohere_embed_model
        self._generate_model = settings.cohere_generate_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from a prompt using Cohere's chat endpoint.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If the API call fails.
        """
        max_tokens = max_tokens or settings.default_max_tokens
        temperature = temperature if temperature is not None else settings.default_temperature

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self._generate_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.message.content[0].text
            logger.debug("Generated %d chars with Cohere", len(text))
            return text
        except Exception as exc:
            logger.error("Cohere generation failed: %s", exc)
            raise RuntimeError(f"Cohere generation failed: {exc}") from exc

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate text and parse it as JSON.

        Strips markdown code fences if present before parsing.

        Returns:
            Parsed JSON as a dict.

        Raises:
            ValueError: If the response is not valid JSON.
        """
        raw = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Strip markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Cohere response as JSON: %s", exc)
            logger.debug("Raw response: %s", raw[:500])
            raise ValueError(f"Invalid JSON response from Cohere: {exc}") from exc

    async def embed(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            input_type: One of "search_document", "search_query", "classification", "clustering".

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            RuntimeError: If the API call fails.
        """
        if not texts:
            return []

        try:
            response = self._client.embed(
                model=self._embed_model,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
            )
            embeddings = response.embeddings.float
            logger.debug("Embedded %d texts, dim=%d", len(texts), len(embeddings[0]))
            return embeddings
        except Exception as exc:
            logger.error("Cohere embedding failed: %s", exc)
            raise RuntimeError(f"Cohere embedding failed: {exc}") from exc


# Module-level singleton (lazy initialization)
_client: CohereClient | None = None


def get_cohere_client() -> CohereClient:
    """Get or create the CohereClient singleton."""
    global _client
    if _client is None:
        _client = CohereClient()
    return _client
