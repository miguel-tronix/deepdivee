"""
API routes for the DeepDive RAG agent.

Routes
------
POST /api/embeddings          create_pubmed_embedding
GET  /api/indications         augment_indications_query
POST /api/contraindications   get_contraindications
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from deepdive.api.schemas import (
    IndicationMatch,
    PubMedEmbeddingRequest,
    PubMedEmbeddingResponse,
    RAGRequest,
    RAGResponse,
)
from deepdive.api.services import (
    ContraindicationService,
    EmbeddingService,
    IndicationService,
    get_contraindication_service,
    get_embedding_service,
    get_indication_service,
)
from deepdive.db.session import get_db

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /embeddings — ingest a PubMed abstract
# ---------------------------------------------------------------------------


@router.post("/embeddings", response_model=PubMedEmbeddingResponse, status_code=200)
async def create_pubmed_embedding(
    request: PubMedEmbeddingRequest,
    db: AsyncSession = Depends(get_db),
    service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Ingest a PubMed abstract by creating its vector embedding and persisting
    both the text and vector to PostgreSQL via pgvector.

    - **pmid**: PubMed identifier (e.g. ``"37123456"``)
    - **title**: Title of the abstract
    - **abstract**: Full abstract text to embed and store
    """
    return await service.create_embedding(request, db)


# ---------------------------------------------------------------------------
# GET /indications — semantic search over stored abstracts
# ---------------------------------------------------------------------------


@router.get("/indications", response_model=list[IndicationMatch])
async def augment_indications_query(
    question: str = Query(
        ...,
        description="A medical intervention question to semantically match against stored PubMed abstracts.",
        min_length=3,
    ),
    db: AsyncSession = Depends(get_db),
    service: IndicationService = Depends(get_indication_service),
):
    """
    Embed the given question and return the top-10 PubMed abstracts ranked by
    cosine similarity.  Results are ordered from most to least similar.

    - **question**: Free-text query (e.g. ``"contraindications of aspirin in pregnancy"``)
    """
    return await service.search(question, db)


# ---------------------------------------------------------------------------
# POST /contraindications — existing RAG + LLM synthesis endpoint
# ---------------------------------------------------------------------------


@router.post("/contraindications", response_model=RAGResponse)
async def get_contraindications(
    request: RAGRequest,
    service: ContraindicationService = Depends(get_contraindication_service),
):
    """
    RAG endpoint for contra-indication synthesis.

    Given a medical intervention, the agent generates a precise search query,
    retrieves relevant PubMed abstracts via pgvector cosine similarity, and
    synthesises an analysis using the LLM — all in a multi-step ReAct loop.
    Results are cached in Redis (TTL 1 hour) to avoid redundant LLM calls.
    """
    return await service.analyze(request)
