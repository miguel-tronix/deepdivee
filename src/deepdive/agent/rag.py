"""
RAG agent core: embedding and LLM completion helpers.

``embed_text`` is public and shared between the ingestion and query
routes to avoid duplication.
"""

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from deepdive.db import repository
from deepdive.core.config import settings
from deepdive.agent.embedders import get_embedder

# Dedicated HTTP client for LLM completion calls only
_llm_client = httpx.AsyncClient(base_url=settings.llm_api_base)


async def embed_text(text: str) -> list[float]:
    """
    Embed a single piece of text using the configured embedder.

    Shared helper for both ingestion (``create_pubmed_embedding``) and
    query (``retrieve_context`` / ``augment_indications_query``) paths.
    """
    embedder = get_embedder()
    return await embedder.embed(text)


async def retrieve_context(query: str, db: AsyncSession, top_k: int = 5) -> str:
    """
    1. Embeds the medical intervention query using the configured embedder.
    2. Searches Postgres (`pgvector`) for nearest neighbours in PubMed abstracts.
    """
    query_vector = await embed_text(query)
    matches = await repository.cosine_similarity_search(
        db=db,
        query_vector=query_vector,
        top_k=top_k,
    )

    if not matches:
        return ""

    context_chunks = [
        f"PMID: {m['pmid']}\nTitle: {m['title']}\nAbstract: {m['content']}"
        for m in matches
    ]
    return "\n\n---\n\n".join(context_chunks)


async def analyze_contraindications(intervention: str, context: str) -> str:
    """
    Call the LLM to produce a contra-indication analysis grounded in retrieved
    PubMed context.

    Args:
        intervention: The medical intervention being queried.
        context: Formatted PubMed context string from ``retrieve_context``.

    Returns:
        LLM-generated analysis string.

    Raises:
        RuntimeError: If the LLM API call fails.
    """
    prompt = f"""
    You are an expert medical AI specialising in identifying contra-indications.
    Based strictly on the provided PubMed literature context, analyse the following
    intervention and list its major contra-indications.

    Intervention: {intervention}

    Context:
    {context}

    Response format: Provide a concise summary followed by bullet points of
    contra-indications.
    """
    try:
        response = await _llm_client.post(
            "/chat/completions",
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
