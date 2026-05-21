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
    Test the endpoint with mocked service dependencies to avoid hitting
    the actual DB/LLM during basic API unit testing.
    """

    async def mock_analyze_with_agent(intervention: str) -> str:
        return "Mocked Contraindication Analysis"

    import deepdive.api.services as svc_module

    monkeypatch.setattr(svc_module, "analyze_with_agent", mock_analyze_with_agent)
    monkeypatch.setattr(
        svc_module.memory_store,
        "get_cached_analysis",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        svc_module.memory_store, "cache_analysis", AsyncMock()
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
