import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from deepdive.db.models import PubMedAbstract
from deepdive.core.config import settings

# This would ideally be injected or instanced as a lifecycle singleton
_http_client = httpx.AsyncClient(base_url=settings.llm_api_base)

async def _embed_query(query: str) -> list[float]:
    """
    Calls the external embedding model API to vectorize the user's query.
    Dummy implementation using an assumed /v1/embeddings endpoint.
    """
    try:
        response = await _http_client.post(
            "/embeddings", 
            json={"input": query, "model": "text-embedding-3-small"}
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        # Fallback or strict error handling here
        raise RuntimeError(f"Failed to generate embeddings: {e}")

async def retrieve_context(query: str, db: AsyncSession, top_k: int = 5) -> str:
    """
    1. Embeds the medical intervention query.
    2. Searches Postgres (`pgvector`) for nearest neighbors in PubMed abstracts.
    """
    query_vector = await _embed_query(query)
    
    # L2 distance (<->) via pgvector
    stmt = (
        select(PubMedAbstract)
        .order_by(PubMedAbstract.embedding.l2_distance(query_vector))
        .limit(top_k)
    )
    result = await db.execute(stmt)
    abstracts = result.scalars().all()
    
    if not abstracts:
        return ""
    
    context_chunks = [
        f"PMID: {a.pmid}\nTitle: {a.title}\nAbstract: {a.content}"
        for a in abstracts
    ]
    return "\n\n---\n\n".join(context_chunks)

async def analyze_contraindications(intervention: str, context: str) -> str:
    """
    Calls the LLM to generate an analysis based on the retrieved context.
    """
    prompt = f"""
    You are an expert medical AI specializing in identifying contra-indications.
    Based strictly on the provided PubMed literature context, analyze the following intervention
    and list its major contra-indications.

    Intervention: {intervention}
    
    Context:
    {context}
    
    Response format: Provide a concise summary followed by bullet points of contra-indications.
    """
    try:
        # Dummy implementation utilizing a typically openAI-compatible completion endpoint
        response = await _http_client.post(
            "/chat/completions",
            json={
                "model": "gpt-4o",  # or claude-3, llama-3 based on what's running locally
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")
