"""
Tests for the embedding abstraction layer.

All HTTP calls are mocked via httpx AsyncClient mocks, so no live
embedding service is required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# RemoteEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_embedder_returns_vector():
    """RemoteEmbedder.embed() should parse the API response and return a vector."""
    import httpx
    from deepdive.agent.embedders import RemoteEmbedder

    fake_vector = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": fake_vector}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    embedder = RemoteEmbedder(client=mock_client)
    vector = await embedder.embed("aspirin contraindications")

    assert vector == fake_vector
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "/embed"
    assert call_args[1]["json"] == {"text": "aspirin contraindications"}


@pytest.mark.asyncio
async def test_remote_embedder_raises_on_http_error():
    """RemoteEmbedder should raise RuntimeError on HTTP failure."""
    import httpx
    from deepdive.agent.embedders import RemoteEmbedder

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    embedder = RemoteEmbedder(client=mock_client)

    with pytest.raises(RuntimeError, match="Remote embedding request failed"):
        await embedder.embed("test query")


@pytest.mark.asyncio
async def test_remote_embedder_raises_on_connect_error():
    """RemoteEmbedder should raise RuntimeError when the service is unreachable."""
    import httpx
    from deepdive.agent.embedders import RemoteEmbedder

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    embedder = RemoteEmbedder(client=mock_client)

    with pytest.raises(RuntimeError, match="Remote embedding request failed"):
        await embedder.embed("test query")


@pytest.mark.asyncio
async def test_remote_embedder_raises_on_malformed_response():
    """RemoteEmbedder should raise RuntimeError if 'embedding' key is missing."""
    import httpx
    from deepdive.agent.embedders import RemoteEmbedder

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"unexpected": "payload"}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    embedder = RemoteEmbedder(client=mock_client)

    with pytest.raises(RuntimeError, match="Remote embedding request failed"):
        await embedder.embed("test query")


# ---------------------------------------------------------------------------
# get_embedder / initialise_embedder / close_embedder factory
# ---------------------------------------------------------------------------


def test_initialise_embedder_returns_remote_embedder(monkeypatch):
    """initialise_embedder() returns a RemoteEmbedder singleton."""
    import deepdive.agent.embedders as emb_module
    from deepdive.agent.embedders import RemoteEmbedder

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module, "_client", None)

    embedder = emb_module.initialise_embedder()
    assert isinstance(embedder, RemoteEmbedder)

    # Second call returns the same instance (singleton)
    assert emb_module.initialise_embedder() is embedder

    # Clean up singleton for isolation
    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module, "_client", None)


def test_get_embedder_lazy_initialises(monkeypatch):
    """get_embedder() lazy-initialises when the singleton is unset."""
    import deepdive.agent.embedders as emb_module
    from deepdive.agent.embedders import RemoteEmbedder

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module, "_client", None)

    embedder = emb_module.get_embedder()
    assert isinstance(embedder, RemoteEmbedder)

    monkeypatch.setattr(emb_module, "_embedder", None)
    monkeypatch.setattr(emb_module, "_client", None)


@pytest.mark.asyncio
async def test_close_embedder_disposes_singleton(monkeypatch):
    """close_embedder() closes the HTTP client and resets the singleton."""
    import deepdive.agent.embedders as emb_module

    fake_client = AsyncMock()
    monkeypatch.setattr(emb_module, "_embedder", object())
    monkeypatch.setattr(emb_module, "_client", fake_client)

    await emb_module.close_embedder()

    fake_client.aclose.assert_awaited_once()
    assert emb_module._embedder is None
    assert emb_module._client is None
