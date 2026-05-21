"""
Embedding abstraction layer.

Defines an `Embedder` Protocol and two concrete implementations:
  - `LocalEmbedder`  — HuggingFace sentence-transformers (runs locally)
  - `OpenAIEmbedder` — OpenAI-compatible /embeddings API via httpx

Use `get_embedder()` to get the singleton instance configured via settings.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from typing import Protocol, runtime_checkable

import httpx

from deepdive.core.config import settings

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Structural protocol for all embedding backends."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        ...


# ---------------------------------------------------------------------------
# Local (HuggingFace sentence-transformers)
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """
    Runs a sentence-transformers model locally.

    `SentenceTransformer.encode()` is CPU/GPU-bound and synchronous, so we
    offload it to a thread pool to keep the async event loop unblocked.

    The underlying ``_encode`` method is LRU-cached so the full model
    inference is skipped when the same text is embedded more than once.
    """

    def __init__(self, model_name: str) -> None:
        # Import here so the heavy library isn't loaded unless LocalEmbedder is used
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @functools.lru_cache(maxsize=2048)
    def _encode(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode, text)


# ---------------------------------------------------------------------------
# OpenAI-compatible API
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """
    Calls an OpenAI-compatible /embeddings endpoint via httpx.
    Works with OpenAI, Azure OpenAI, or any local proxy (LiteLLM, vLLM, etc.).
    """

    def __init__(self, client: httpx.AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post(
                "/embeddings",
                json={"input": text, "model": self._model},
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding request failed: {e}") from e


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

# Module-level singleton — populated by get_embedder() on first call or
# explicitly at app startup via initialise_embedder().
_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def initialise_embedder() -> Embedder:
    """
    Build and cache the embedder singleton.  Call this once at app startup
    (e.g. inside FastAPI's lifespan context) so the model is warm before
    the first request arrives.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        backend = settings.embedding_backend
        model = settings.embedding_model

        if backend == "local":
            _embedder = LocalEmbedder(model_name=model)
        elif backend == "openai":
            _http_client = httpx.AsyncClient(
                base_url=settings.llm_api_base,
                timeout=30.0,
            )
            _embedder = OpenAIEmbedder(client=_http_client, model=model)
        else:
            raise ValueError(f"Unknown embedding backend: {backend!r}")

        return _embedder


def get_embedder() -> Embedder:
    """
    Return the active embedder singleton.

    Prefer calling `initialise_embedder()` at startup. This function will
    lazy-initialise as a fallback (e.g. in tests or CLI scripts).
    """
    if _embedder is None:
        return initialise_embedder()
    return _embedder
