"""
Unit tests for the two new RAG ingestion and retrieval endpoints.

All external dependencies (embedding API, database) are monkeypatched so
these tests run without a live stack.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from deepdive.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR = [0.1] * 768  # matches Vector(768) dimension


# ---------------------------------------------------------------------------
# POST /api/embeddings — create_pubmed_embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_pubmed_embedding_success(monkeypatch):
    """
    Verify that a well-formed POST to /api/embeddings returns HTTP 200 with the
    expected pmid, with both the embedding call and DB insert monkeypatched.
    """
    async def mock_embed_text(text: str):
        return FAKE_VECTOR

    async def mock_insert(db, pmid, title, content, embedding):
        from deepdive.db.models import PubMedAbstract
        obj = PubMedAbstract(id=1, pmid=pmid, title=title, content=content, embedding=embedding)
        return obj

    import deepdive.api.routes as routes_module
    import deepdive.db.repository as repo_module

    monkeypatch.setattr(routes_module, "embed_text", mock_embed_text)
    monkeypatch.setattr(repo_module, "insert_pubmed_embedding", mock_insert)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/embeddings",
            json={
                "pmid": "37000001",
                "title": "Aspirin and GI bleeding risk",
                "abstract": "Aspirin is associated with increased GI bleeding especially in elderly patients.",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "ok"
    assert data["pmid"] == "37000001"


@pytest.mark.asyncio
async def test_create_pubmed_embedding_embed_failure(monkeypatch):
    """
    Verify that an embedding API failure is surfaced as HTTP 502.
    """
    async def mock_embed_text_fail(text: str):
        raise RuntimeError("Embedding service unavailable")

    import deepdive.api.routes as routes_module
    monkeypatch.setattr(routes_module, "embed_text", mock_embed_text_fail)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/embeddings",
            json={"pmid": "123", "title": "Test", "abstract": "Test abstract."},
        )

    assert response.status_code == 502


# ---------------------------------------------------------------------------
# GET /api/indications — augment_indications_query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_augment_indications_query_success(monkeypatch):
    """
    Verify that GET /api/indications returns the top-10 cosine matches with
    the expected schema, with embed_text and cosine_similarity_search mocked.
    """
    async def mock_embed_text(text: str):
        return FAKE_VECTOR

    async def mock_cosine_search(db, query_vector, top_k=10):
        return [
            {
                "pmid": f"37{i:06d}",
                "title": f"Study {i}",
                "content": f"Abstract content {i}",
                "similarity_score": round(0.99 - i * 0.01, 6),
            }
            for i in range(top_k)
        ]

    import deepdive.api.routes as routes_module
    import deepdive.db.repository as repo_module

    monkeypatch.setattr(routes_module, "embed_text", mock_embed_text)
    monkeypatch.setattr(repo_module, "cosine_similarity_search", mock_cosine_search)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/api/indications",
            params={"question": "What are the contraindications of aspirin in pregnancy?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    # Validate schema of first result
    first = data[0]
    assert "pmid" in first
    assert "title" in first
    assert "content" in first
    assert "similarity_score" in first
    assert isinstance(first["similarity_score"], float)


@pytest.mark.asyncio
async def test_augment_indications_query_missing_param():
    """
    Verify that omitting the required `question` query param returns HTTP 422.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/indications")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_augment_indications_query_embed_failure(monkeypatch):
    """
    Verify that an embedding failure surfaces as HTTP 502.
    """
    async def mock_embed_text_fail(text: str):
        raise RuntimeError("Embedding service unavailable")

    import deepdive.api.routes as routes_module
    monkeypatch.setattr(routes_module, "embed_text", mock_embed_text_fail)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/indications", params={"question": "aspirin risks"})

    assert response.status_code == 502
