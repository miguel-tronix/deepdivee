"""
Unit tests for LocalEmbedder and HTTPEmbedder classes.

These tests verify the embedder implementations directly without mocking
the entire embed_text function.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


class TestLocalEmbedder:
    """Tests for the LocalEmbedder class."""

    @pytest.mark.asyncio
    async def test_embed_text_single(self):
        """Test single text embedding returns a list of floats."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=MagicMock(tolist=lambda: [[0.1] * 768])
        )

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            with patch("deepdive.agent.rag.settings") as mock_settings:
                mock_settings.embedding_device = "cpu"
                mock_settings.embedding_batch_size = 32
                mock_settings.embedding_trust_remote_code = False

                from deepdive.agent.rag import LocalEmbedder

                embedder = LocalEmbedder("test-model")

                result = await embedder.embed_text("test text")

                assert isinstance(result, list)
                assert len(result) == 768
                mock_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self):
        """Test batch text embedding returns list of embeddings."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=MagicMock(tolist=lambda: [[0.1] * 768, [0.2] * 768])
        )

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            with patch("deepdive.agent.rag.settings") as mock_settings:
                mock_settings.embedding_device = "cpu"
                mock_settings.embedding_batch_size = 32
                mock_settings.embedding_trust_remote_code = False

                from deepdive.agent.rag import LocalEmbedder

                embedder = LocalEmbedder("test-model")

                result = await embedder.embed_texts(["text1", "text2"])

                assert isinstance(result, list)
                assert len(result) == 2
                assert all(isinstance(e, list) for e in result)

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self):
        """Test that errors are wrapped in RuntimeError."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Model error"))

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            with patch("deepdive.agent.rag.settings") as mock_settings:
                mock_settings.embedding_device = "cpu"
                mock_settings.embedding_batch_size = 32
                mock_settings.embedding_trust_remote_code = False

                from deepdive.agent.rag import LocalEmbedder

                embedder = LocalEmbedder("test-model")

                with pytest.raises(
                    RuntimeError, match="Failed to generate embeddings locally"
                ):
                    await embedder.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_text_batch_size_passed(self):
        """Test that batch_size is passed to the model encode call."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=MagicMock(tolist=lambda: [[0.1] * 768])
        )

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            with patch("deepdive.agent.rag.settings") as mock_settings:
                mock_settings.embedding_device = "cpu"
                mock_settings.embedding_batch_size = 64
                mock_settings.embedding_trust_remote_code = False

                from deepdive.agent.rag import LocalEmbedder

                embedder = LocalEmbedder("test-model")

                await embedder.embed_texts(["test"])

                mock_model.encode.assert_called_once()
                call_kwargs = mock_model.encode.call_args.kwargs
                assert call_kwargs.get("batch_size") == 64


class TestHTTPEmbedder:
    """Tests for the HTTPEmbedder class."""

    @pytest.mark.asyncio
    async def test_embed_text_single(self):
        """Test single text embedding via HTTP."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "test-model")

        result = await embedder.embed_text("test text")

        assert result == [0.1] * 768
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self):
        """Test batch text embedding via HTTP."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 768}, {"embedding": [0.2] * 768}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "test-model")

        result = await embedder.embed_texts(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1] * 768
        assert result[1] == [0.2] * 768
        mock_client.post.assert_called_once()

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["input"] == ["text1", "text2"]

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self):
        """Test that HTTP errors are wrapped in RuntimeError."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(side_effect=Exception("HTTP 502"))
        mock_client.post = AsyncMock(return_value=mock_response)

        from deepdive.agent.rag import HTTPEmbedder

        embedder = HTTPEmbedder(mock_client, "test-model")

        with pytest.raises(
            RuntimeError, match="Failed to generate embeddings via HTTP"
        ):
            await embedder.embed_text("test")


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
