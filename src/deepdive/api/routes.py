from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from deepdive.db.session import get_db
from deepdive.agent.rag import retrieve_context, analyze_contraindications

router = APIRouter()


class RAGRequest(BaseModel):
    intervention: str


class RAGResponse(BaseModel):
    intervention: str
    analysis: str


@router.post("/contraindications", response_model=RAGResponse)
async def get_contraindications(
    request: RAGRequest, db: AsyncSession = Depends(get_db)
):
    """
    RAG Endpoint for Contra-indications.
    Given a medical intervention, fetches PubMed abstracts using `pgvector`
    and returns a synthesized analysis.
    """
    intervention = request.intervention

    # Needs to be hooked up to Redis caching
    # e.g., cached_response = await redis.get(intervention)

    try:
        context = await retrieve_context(intervention, db)
        if not context:
            # Optionally just return an analysis stating no context is available,
            # or hit the LLM directly as a fallback
            return RAGResponse(
                intervention=intervention,
                analysis="No specific PubMed context found for this intervention's contra-indications.",
            )

        analysis = await analyze_contraindications(intervention, context)

        # e.g., await redis.set(intervention, analysis, ex=3600)

        return RAGResponse(intervention=intervention, analysis=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
