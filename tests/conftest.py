"""
Shared fixtures for the DeepDive test suite.

Provides reusable async client, mock embedder, and embedding helpers
to reduce duplication and improve test consistency.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock

from deepdive.main import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # nomic-embed-text default dimension
FAKE_VECTOR = [0.1] * EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def async_client():
    """Reusable async HTTP client bound to the FastAPI test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def fake_vector():
    """A 768-dimension fake embedding vector."""
    return FAKE_VECTOR.copy()


@pytest.fixture
def mock_embed_text_success():
    """An async mock for embed_text that returns a valid 768-dim vector."""
    async def _mock(text: str):
        return FAKE_VECTOR
    return _mock


@pytest.fixture
def mock_embed_text_failure():
    """An async mock for embed_text that raises RuntimeError."""
    async def _mock(text: str):
        raise RuntimeError("Embedding service unavailable")
    return _mock


@pytest.fixture
def mock_http_embedder_client():
    """A mock httpx.AsyncClient pre-configured for embedding API responses."""
    client = AsyncMock()
    response = MagicMock()
    response.json.return_value = {"data": [{"embedding": FAKE_VECTOR}]}
    response.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client
