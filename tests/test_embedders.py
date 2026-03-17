"""
Tests for the embedding abstraction layer.

- test_local_embedder: real model load (downloads ~90MB on first run)
- test_openai_embedder_mocked: mocked httpx, no real API call
- test_get_embedder_*: verifies factory returns the right type
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_embedder_returns_float_list():
    """LocalEmbedder.embed() should return a non-empty list of floats."""
    from deepdive.agent.embedders import LocalEmbedder

    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    vector = await embedder.embed("test medical query")

    assert isinstance(vector, list), "embed() must return a list"
    assert len(vector) > 0, "embedding vector must be non-empty"
    assert all(isinstance(v, float) for v in vector), "all elements must be floats"


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embedder_mocked():
    """OpenAIEmbedder.embed() should parse the API response and return a vector."""
    import httpx
    from deepdive.agent.embedders import OpenAIEmbedder

    fake_vector = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": fake_vector}]}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    embedder = OpenAIEmbedder(client=mock_client, model="text-embedding-3-small")
    vector = await embedder.embed("aspirin contraindications")

    assert vector == fake_vector
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "/embeddings"
    assert call_kwargs[1]["json"]["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_openai_embedder_raises_on_http_error():
    """OpenAIEmbedder should raise RuntimeError on HTTP failure."""
    import httpx
    from deepdive.agent.embedders import OpenAIEmbedder

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    embedder = OpenAIEmbedder(client=mock_client, model="text-embedding-3-small")

    with pytest.raises(RuntimeError, match="OpenAI embedding request failed"):
        await embedder.embed("test query")


# ---------------------------------------------------------------------------
# get_embedder / initialise_embedder factory
# ---------------------------------------------------------------------------


def test_get_embedder_local(monkeypatch):
    """get_embedder() returns a LocalEmbedder when backend=local."""
    import deepdive.agent.embedders as emb_module
    from deepdive.agent.embedders import LocalEmbedder

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module.settings, "embedding_backend", "local")
    monkeypatch.setattr(emb_module.settings, "embedding_model", "all-MiniLM-L6-v2")

    embedder = emb_module.initialise_embedder()
    assert isinstance(embedder, LocalEmbedder)

    # Clean up singleton for isolation
    monkeypatch.setattr(emb_module, "_embedder", None)


def test_get_embedder_openai(monkeypatch):
    """get_embedder() returns an OpenAIEmbedder when backend=openai."""
    import deepdive.agent.embedders as emb_module
    from deepdive.agent.embedders import OpenAIEmbedder

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module.settings, "embedding_backend", "openai")
    monkeypatch.setattr(emb_module.settings, "embedding_model", "text-embedding-3-small")

    embedder = emb_module.initialise_embedder()
    assert isinstance(embedder, OpenAIEmbedder)

    monkeypatch.setattr(emb_module, "_embedder", None)


def test_get_embedder_invalid_backend(monkeypatch):
    """get_embedder() raises ValueError for an unknown backend."""
    import deepdive.agent.embedders as emb_module

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module.settings, "embedding_backend", "unknown")

    with pytest.raises(ValueError, match="Unknown embedding backend"):
        emb_module.initialise_embedder()

    monkeypatch.setattr(emb_module, "_embedder", None)
