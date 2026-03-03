"""
Unit tests for LocalEmbedder and HTTPEmbedder classes.

These tests verify the embedder implementations directly without mocking
the entire embed_text function. LocalEmbedder tests bypass __init__ to
avoid importing sentence_transformers (which would hit HuggingFace).
The mock model and batch_size are set directly on the instance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from conftest import EMBEDDING_DIM, FAKE_VECTOR


class TestLocalEmbedder:
    """Tests for the LocalEmbedder class.

    We bypass LocalEmbedder.__init__ (which imports sentence_transformers
    and hits HuggingFace) by constructing instances via __new__ and manually
    setting the .model and .batch_size attributes. This isolates the
    embed_text / embed_texts logic we actually want to test.
    """

    def _create_embedder(self, mock_model, batch_size=32):
        """Create a LocalEmbedder without calling __init__."""
        from deepdive.agent.rag import LocalEmbedder

        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.model = mock_model
        embedder.batch_size = batch_size
        return embedder

    def _create_mock_model(self, return_value):
        """Helper to create a mock SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=MagicMock(tolist=lambda: return_value)
        )
        return mock_model

    @pytest.mark.asyncio
    async def test_embed_text_single(self):
        """Test single text embedding returns a list of floats."""
        mock_model = self._create_mock_model([[0.1] * EMBEDDING_DIM])
        embedder = self._create_embedder(mock_model)

        result = await embedder.embed_text("test text")

        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self):
        """Test batch text embedding returns list of embeddings."""
        mock_model = self._create_mock_model(
            [[0.1] * EMBEDDING_DIM, [0.2] * EMBEDDING_DIM]
        )
        embedder = self._create_embedder(mock_model)

        result = await embedder.embed_texts(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, list) for e in result)
        assert all(len(e) == EMBEDDING_DIM for e in result)

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self):
        """Test that errors are wrapped in RuntimeError."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Model error"))
        embedder = self._create_embedder(mock_model)

        with pytest.raises(
            RuntimeError, match="Failed to generate embeddings locally"
        ):
            await embedder.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_text_batch_size_passed(self):
        """Test that batch_size is passed to the model encode call."""
        mock_model = self._create_mock_model([[0.1] * EMBEDDING_DIM])
        embedder = self._create_embedder(mock_model, batch_size=64)

        await embedder.embed_texts(["test"])

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("batch_size") == 64

    @pytest.mark.asyncio
    async def test_init_with_mocked_sentence_transformer(self):
        """Test that __init__ correctly calls SentenceTransformer with settings."""
        mock_model = MagicMock()
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": mock_st, "torch": MagicMock()},
        ):
            with patch("deepdive.agent.rag.settings") as mock_settings:
                mock_settings.embedding_device = "cpu"
                mock_settings.embedding_batch_size = 32
                mock_settings.embedding_trust_remote_code = False

                from deepdive.agent.rag import LocalEmbedder

                embedder = LocalEmbedder("nomic-embed-text")

                mock_st.SentenceTransformer.assert_called_once_with(
                    "nomic-embed-text",
                    device="cpu",
                    trust_remote_code=False,
                )
                assert embedder.batch_size == 32


class TestHTTPEmbedder:
    """Tests for the HTTPEmbedder class."""

    @pytest.mark.asyncio
    async def test_embed_text_single(self):
        """Test single text embedding via HTTP."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * EMBEDDING_DIM}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "nomic-embed-text")

        result = await embedder.embed_text("test text")

        assert result == [0.1] * EMBEDDING_DIM
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self):
        """Test batch text embedding via HTTP."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * EMBEDDING_DIM},
                {"embedding": [0.2] * EMBEDDING_DIM},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "nomic-embed-text")

        result = await embedder.embed_texts(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1] * EMBEDDING_DIM
        assert result[1] == [0.2] * EMBEDDING_DIM
        mock_client.post.assert_called_once()

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["input"] == ["text1", "text2"]
        assert call_args.kwargs["json"]["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self):
        """Test that HTTP errors are wrapped in RuntimeError."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("HTTP 502")
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "nomic-embed-text")

        with pytest.raises(
            RuntimeError, match="Failed to generate embeddings via HTTP"
        ):
            await embedder.embed_text("test")


class TestGetEmbedder:
    """Tests for the get_embedder() factory function."""

    @pytest.mark.asyncio
    async def test_get_embedder_returns_http_when_local_disabled(self):
        """Test that get_embedder returns HTTPEmbedder when use_local_embeddings is False."""
        import deepdive.agent.rag as rag_module

        original_embedder = rag_module._embedder
        rag_module._embedder = None

        try:
            with patch.object(
                rag_module, "settings"
            ) as mock_settings:
                mock_settings.use_local_embeddings = False
                mock_settings.embedding_model = "nomic-embed-text"
                mock_settings.llm_api_base = "http://localhost:8000/v1"

                embedder = rag_module.get_embedder()

                assert isinstance(embedder, rag_module.HTTPEmbedder)
                assert embedder.model_name == "nomic-embed-text"
        finally:
            rag_module._embedder = original_embedder

    @pytest.mark.asyncio
    async def test_get_embedder_returns_local_when_enabled(self):
        """Test that get_embedder returns LocalEmbedder when use_local_embeddings is True."""
        import deepdive.agent.rag as rag_module

        original_embedder = rag_module._embedder

        mock_model = MagicMock()
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        rag_module._embedder = None

        try:
            with patch.dict(
                "sys.modules",
                {"sentence_transformers": mock_st, "torch": MagicMock()},
            ):
                with patch.object(
                    rag_module, "settings"
                ) as mock_settings:
                    mock_settings.use_local_embeddings = True
                    mock_settings.embedding_model = "nomic-embed-text"
                    mock_settings.embedding_device = "cpu"
                    mock_settings.embedding_batch_size = 32
                    mock_settings.embedding_trust_remote_code = False

                    embedder = rag_module.get_embedder()

                    assert isinstance(embedder, rag_module.LocalEmbedder)
        finally:
            rag_module._embedder = original_embedder

    @pytest.mark.asyncio
    async def test_get_embedder_returns_singleton(self):
        """Test that get_embedder returns the same instance on repeated calls."""
        import deepdive.agent.rag as rag_module

        original_embedder = rag_module._embedder
        rag_module._embedder = None

        try:
            with patch.object(
                rag_module, "settings"
            ) as mock_settings:
                mock_settings.use_local_embeddings = False
                mock_settings.embedding_model = "nomic-embed-text"
                mock_settings.llm_api_base = "http://localhost:8000/v1"

                first = rag_module.get_embedder()
                second = rag_module.get_embedder()

                assert first is second
        finally:
            rag_module._embedder = original_embedder


class TestCleanupEmbedder:
    """Tests for the cleanup_embedder function."""

    @pytest.mark.asyncio
    async def test_cleanup_none_embedder(self):
        """Test cleanup when no embedder is initialized."""
        import deepdive.agent.rag as rag_module

        original_embedder = rag_module._embedder
        rag_module._embedder = None

        from deepdive.agent.rag import cleanup_embedder

        await cleanup_embedder()

        assert rag_module._embedder is None

        rag_module._embedder = original_embedder

    @pytest.mark.asyncio
    async def test_cleanup_with_model(self):
        """Test cleanup properly deletes model."""
        import deepdive.agent.rag as rag_module

        mock_embedder = MagicMock()
        mock_embedder.model = MagicMock()
        rag_module._embedder = mock_embedder

        from deepdive.agent.rag import cleanup_embedder

        await cleanup_embedder()

        assert rag_module._embedder is None
