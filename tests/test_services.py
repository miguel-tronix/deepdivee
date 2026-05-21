"""
Unit tests for the service layer.

Tests each concrete service implementation in isolation by monkeypatching
all external dependencies.  These tests validate the business logic and
error-to-exception mapping without touching the API layer.
"""

import pytest
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from deepdive.api.schemas import (
    IndicationMatch,
    PubMedEmbeddingRequest,
    PubMedEmbeddingResponse,
    RAGRequest,
    RAGResponse,
)
from deepdive.api.exceptions import (
    AnalysisError,
    DatabaseError,
    EmbeddingServiceError,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# PubMedEmbeddingService
# ---------------------------------------------------------------------------


class TestPubMedEmbeddingService:
    async def test_create_embedding_success(self, monkeypatch):
        import deepdive.api.services as svc_module

        fake_vector = [0.1] * 384
        monkeypatch.setattr(
            svc_module, "embed_text", AsyncMock(return_value=fake_vector)
        )
        monkeypatch.setattr(
            svc_module.repository, "insert_pubmed_embedding", AsyncMock()
        )

        from deepdive.api.services import PubMedEmbeddingService

        service = PubMedEmbeddingService()
        request = PubMedEmbeddingRequest(
            pmid="123", title="Test", abstract="Test abstract"
        )
        db = AsyncMock(spec=AsyncSession)
        response = await service.create_embedding(request, db)

        assert isinstance(response, PubMedEmbeddingResponse)
        assert response.message == "ok"
        assert response.pmid == "123"

    async def test_create_embedding_runtime_error_raises_embedding_service_error(
        self, monkeypatch
    ):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module,
            "embed_text",
            AsyncMock(side_effect=RuntimeError("API down")),
        )

        from deepdive.api.services import PubMedEmbeddingService

        service = PubMedEmbeddingService()
        request = PubMedEmbeddingRequest(
            pmid="123", title="Test", abstract="Test"
        )
        db = AsyncMock(spec=AsyncSession)

        with pytest.raises(EmbeddingServiceError, match="API down"):
            await service.create_embedding(request, db)

    async def test_create_embedding_generic_error_raises_database_error(
        self, monkeypatch
    ):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module,
            "embed_text",
            AsyncMock(side_effect=ValueError("bad data")),
        )

        from deepdive.api.services import PubMedEmbeddingService

        service = PubMedEmbeddingService()
        request = PubMedEmbeddingRequest(
            pmid="123", title="Test", abstract="Test"
        )
        db = AsyncMock(spec=AsyncSession)

        with pytest.raises(DatabaseError, match="bad data"):
            await service.create_embedding(request, db)


# ---------------------------------------------------------------------------
# PubMedIndicationService
# ---------------------------------------------------------------------------


class TestPubMedIndicationService:
    async def test_search_success(self, monkeypatch):
        import deepdive.api.services as svc_module

        fake_vector = [0.1] * 384
        fake_matches = [
            {
                "pmid": "37000001",
                "title": "Study 1",
                "content": "Abstract 1",
                "similarity_score": 0.95,
            },
        ]
        monkeypatch.setattr(
            svc_module, "embed_text", AsyncMock(return_value=fake_vector)
        )
        monkeypatch.setattr(
            svc_module.repository,
            "cosine_similarity_search",
            AsyncMock(return_value=fake_matches),
        )

        from deepdive.api.services import PubMedIndicationService

        service = PubMedIndicationService()
        db = AsyncMock(spec=AsyncSession)
        results = await service.search("aspirin risks", db)

        assert len(results) == 1
        match = results[0]
        assert isinstance(match, IndicationMatch)
        assert match.pmid == "37000001"
        assert match.similarity_score == 0.95

    async def test_search_runtime_error_raises_embedding_service_error(
        self, monkeypatch
    ):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module,
            "embed_text",
            AsyncMock(side_effect=RuntimeError("API down")),
        )

        from deepdive.api.services import PubMedIndicationService

        service = PubMedIndicationService()
        db = AsyncMock(spec=AsyncSession)

        with pytest.raises(EmbeddingServiceError, match="API down"):
            await service.search("test query", db)

    async def test_search_generic_error_raises_database_error(
        self, monkeypatch
    ):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module,
            "embed_text",
            AsyncMock(side_effect=ValueError("bad data")),
        )

        from deepdive.api.services import PubMedIndicationService

        service = PubMedIndicationService()
        db = AsyncMock(spec=AsyncSession)

        with pytest.raises(DatabaseError, match="bad data"):
            await service.search("test query", db)


# ---------------------------------------------------------------------------
# AgentContraindicationService
# ---------------------------------------------------------------------------


class TestAgentContraindicationService:
    async def test_analyze_returns_cached_result(self, monkeypatch):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module.memory_store,
            "get_cached_analysis",
            AsyncMock(return_value="cached result"),
        )

        from deepdive.api.services import AgentContraindicationService

        service = AgentContraindicationService()
        request = RAGRequest(intervention="Aspirin")
        response = await service.analyze(request)

        assert isinstance(response, RAGResponse)
        assert response.intervention == "Aspirin"
        assert response.analysis == "cached result"

    async def test_analyze_runs_agent_when_no_cache(self, monkeypatch):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module.memory_store,
            "get_cached_analysis",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            svc_module,
            "analyze_with_agent",
            AsyncMock(return_value="fresh analysis"),
        )
        monkeypatch.setattr(
            svc_module.memory_store, "cache_analysis", AsyncMock()
        )

        from deepdive.api.services import AgentContraindicationService

        service = AgentContraindicationService()
        request = RAGRequest(intervention="Ibuprofen")
        response = await service.analyze(request)

        assert response.analysis == "fresh analysis"
        svc_module.memory_store.cache_analysis.assert_awaited_once_with(
            "Ibuprofen", "fresh analysis"
        )

    async def test_analyze_raises_analysis_error_on_failure(self, monkeypatch):
        import deepdive.api.services as svc_module

        monkeypatch.setattr(
            svc_module.memory_store,
            "get_cached_analysis",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            svc_module,
            "analyze_with_agent",
            AsyncMock(side_effect=Exception("LLM failed")),
        )

        from deepdive.api.services import AgentContraindicationService

        service = AgentContraindicationService()
        request = RAGRequest(intervention="Paracetamol")

        with pytest.raises(AnalysisError, match="LLM failed"):
            await service.analyze(request)


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestServiceFactoryFunctions:
    def test_get_embedding_service_returns_pubmed_embedding_service(self):
        from deepdive.api.services import (
            PubMedEmbeddingService,
            get_embedding_service,
        )

        service = get_embedding_service()
        assert isinstance(service, PubMedEmbeddingService)

    def test_get_indication_service_returns_pubmed_indication_service(self):
        from deepdive.api.services import (
            PubMedIndicationService,
            get_indication_service,
        )

        service = get_indication_service()
        assert isinstance(service, PubMedIndicationService)

    def test_get_contraindication_service_returns_agent_contraindication_service(
        self,
    ):
        from deepdive.api.services import (
            AgentContraindicationService,
            get_contraindication_service,
        )

        service = get_contraindication_service()
        assert isinstance(service, AgentContraindicationService)


# ---------------------------------------------------------------------------
# Protocol conformance (structural subtyping)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_embedding_service_conforms_to_protocol(self):
        from deepdive.api.services import (
            EmbeddingService,
            PubMedEmbeddingService,
        )

        assert isinstance(PubMedEmbeddingService(), EmbeddingService)

    def test_indication_service_conforms_to_protocol(self):
        from deepdive.api.services import (
            IndicationService,
            PubMedIndicationService,
        )

        assert isinstance(PubMedIndicationService(), IndicationService)

    def test_contraindication_service_conforms_to_protocol(self):
        from deepdive.api.services import (
            AgentContraindicationService,
            ContraindicationService,
        )

        assert isinstance(AgentContraindicationService(), ContraindicationService)
