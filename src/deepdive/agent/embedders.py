"""
Embedding abstraction layer.

Embeddings are computed by the external Embitious service (Quarkus/DJL),
exposing ``POST /embed`` with ``{"text": ...}`` and returning
``{"embedding": [...]}``. This module provides an async client for that
endpoint plus a singleton factory used at app startup.
"""

from __future__ import annotations

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
# Remote (Embitious) embedding API
# ---------------------------------------------------------------------------


class RemoteEmbedder:
    """
    Calls the Embitious embedding service via httpx.

    The service is a drop-in embedding endpoint: ``POST /embed`` with a JSON
    body ``{"text": ...}`` responds with ``{"embedding": [...]}``.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post("/embed", json={"text": text})
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Remote embedding request failed: {e}") from e


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

# Module-level singleton — populated by get_embedder() on first call or
# explicitly at app startup via initialise_embedder().
_embedder: Embedder | None = None
_client: httpx.AsyncClient | None = None
_embedder_lock = threading.Lock()


def initialise_embedder() -> Embedder:
    """
    Build and cache the embedder singleton.  Call this once at app startup
    (inside FastAPI's lifespan context) so the HTTP client is warm before
    the first request arrives.
    """
    global _embedder, _client
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        _client = httpx.AsyncClient(
            base_url=settings.embedding_api_url,
            timeout=settings.embedding_timeout,
        )
        _embedder = RemoteEmbedder(client=_client)
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


async def close_embedder() -> None:
    """Dispose of the embedder singleton and its HTTP client."""
    global _embedder, _client
    with _embedder_lock:
        if _client is not None:
            await _client.aclose()
        _client = None
        _embedder = None
