"""
API routes for the DeepDive RAG agent.

Routes
------
POST /api/embeddings          create_pubmed_embedding
GET  /api/indications         augment_indications_query
POST /api/contraindications   get_contraindications  (existing)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from deepdive.db.session import get_db
from deepdive.db import repository
from deepdive.agent.rag import embed_text
from deepdive.agent.memory import memory_store
from deepdive.agent.agent import analyze_with_agent

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared Pydantic schemas
# ---------------------------------------------------------------------------


class PubMedEmbeddingRequest(BaseModel):
    """Request body for ingesting a PubMed abstract."""

    pmid: str
    title: str
    abstract: str


class PubMedEmbeddingResponse(BaseModel):
    """Success response after storing an embedding."""

    message: str
    pmid: str


class IndicationMatch(BaseModel):
    """A single cosine-similarity match returned by the query endpoint."""

    pmid: str
    title: str
    content: str
    similarity_score: float


class RAGRequest(BaseModel):
    intervention: str


class RAGResponse(BaseModel):
    intervention: str
    analysis: str


# ---------------------------------------------------------------------------
# POST /embeddings — ingest a PubMed abstract
# ---------------------------------------------------------------------------


@router.post("/embeddings", response_model=PubMedEmbeddingResponse, status_code=200)
async def create_pubmed_embedding(
    request: PubMedEmbeddingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest a PubMed abstract by creating its vector embedding and persisting
    both the text and vector to PostgreSQL via pgvector.

    - **pmid**: PubMed identifier (e.g. ``"37123456"``)
    - **title**: Title of the abstract
    - **abstract**: Full abstract text to embed and store
    """
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
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
):
    """
    Embed the given question and return the top-10 PubMed abstracts ranked by
    cosine similarity.  Results are ordered from most to least similar.

    - **question**: Free-text query (e.g. ``"contraindications of aspirin in pregnancy"``)
    """
    try:
        query_vector = await embed_text(question)
        matches = await repository.cosine_similarity_search(
            db=db,
            query_vector=query_vector,
            top_k=10,
        )
        return [IndicationMatch(**m) for m in matches]
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /contraindications — existing RAG + LLM synthesis endpoint
# ---------------------------------------------------------------------------


@router.post("/contraindications", response_model=RAGResponse)
async def get_contraindications(request: RAGRequest):
    """
    RAG endpoint for contra-indication synthesis.

    Given a medical intervention, the agent generates a precise search query,
    retrieves relevant PubMed abstracts via pgvector cosine similarity, and
    synthesises an analysis using the LLM — all in a multi-step ReAct loop.
    Results are cached in Redis (TTL 1 hour) to avoid redundant LLM calls.
    """
    intervention = request.intervention

    cached = await memory_store.get_cached_analysis(intervention)
    if cached is not None:
        return RAGResponse(intervention=intervention, analysis=cached)

    try:
        analysis = await analyze_with_agent(intervention)

        await memory_store.cache_analysis(intervention, analysis)

        return RAGResponse(intervention=intervention, analysis=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
