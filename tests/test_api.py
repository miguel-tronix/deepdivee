import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock
from deepdive.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "DeepDive RAG Agent"}


@pytest.mark.asyncio
async def test_contraindications_endpoint_mocked(monkeypatch):
    """
    Test the endpoint with mocked agent logic to avoid hitting the actual DB/LLM
    during basic API unit testing.
    """

    async def mock_analyze_with_agent(intervention: str) -> str:
        return "Mocked Contraindication Analysis"

    # Patch the agent logic
    import deepdive.api.routes

    monkeypatch.setattr(
        deepdive.api.routes, "analyze_with_agent", mock_analyze_with_agent
    )
    monkeypatch.setattr(
        deepdive.api.routes.memory_store, "get_cached_analysis", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        deepdive.api.routes.memory_store, "cache_analysis", AsyncMock()
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/contraindications", json={"intervention": "Aspirin"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intervention"] == "Aspirin"
    assert data["analysis"] == "Mocked Contraindication Analysis"
