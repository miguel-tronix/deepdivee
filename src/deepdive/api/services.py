from typing import Protocol, runtime_checkable

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
from deepdive.agent.agent import analyze_with_agent
from deepdive.agent.memory import memory_store
from deepdive.agent.rag import embed_text
from deepdive.db import repository


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingService(Protocol):
    async def create_embedding(
        self,
        request: PubMedEmbeddingRequest,
        db: AsyncSession,
    ) -> PubMedEmbeddingResponse:
        ...


@runtime_checkable
class IndicationService(Protocol):
    async def search(
        self,
        question: str,
        db: AsyncSession,
    ) -> list[IndicationMatch]:
        ...


@runtime_checkable
class ContraindicationService(Protocol):
    async def analyze(self, request: RAGRequest) -> RAGResponse:
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class PubMedEmbeddingService:
    async def create_embedding(
        self,
        request: PubMedEmbeddingRequest,
        db: AsyncSession,
    ) -> PubMedEmbeddingResponse:
        try:
            vector = await embed_text(request.abstract)
            await repository.insert_pubmed_embedding(
                db=db,
                pmid=request.pmid,
                title=request.title,
                content=request.abstract,
                embedding=vector,
            )
            return PubMedEmbeddingResponse(message="ok", pmid=request.pmid)
        except RuntimeError as e:
            raise EmbeddingServiceError(str(e))
        except Exception as e:
            raise DatabaseError(str(e))


class PubMedIndicationService:
    async def search(
        self,
        question: str,
        db: AsyncSession,
    ) -> list[IndicationMatch]:
        try:
            query_vector = await embed_text(question)
            matches = await repository.cosine_similarity_search(
                db=db,
                query_vector=query_vector,
                top_k=10,
            )
            return [IndicationMatch(**m) for m in matches]
        except RuntimeError as e:
            raise EmbeddingServiceError(str(e))
        except Exception as e:
            raise DatabaseError(str(e))


class AgentContraindicationService:
    async def analyze(self, request: RAGRequest) -> RAGResponse:
        intervention = request.intervention
        cached = await memory_store.get_cached_analysis(intervention)
        if cached is not None:
            return RAGResponse(intervention=intervention, analysis=cached)
        try:
            analysis = await analyze_with_agent(intervention)
            await memory_store.cache_analysis(intervention, analysis)
            return RAGResponse(intervention=intervention, analysis=analysis)
        except Exception as e:
            raise AnalysisError(str(e))


# ---------------------------------------------------------------------------
# Factory functions — used as FastAPI Depends
# ---------------------------------------------------------------------------


def get_embedding_service() -> EmbeddingService:
    return PubMedEmbeddingService()


def get_indication_service() -> IndicationService:
    return PubMedIndicationService()


def get_contraindication_service() -> ContraindicationService:
    return AgentContraindicationServicea
