"""
RAG agent core: embedding and LLM completion helpers.

``embed_text`` is public and shared between the ingestion and query
routes to avoid duplication.
"""

from any_llm import AnyLLM
from sqlalchemy.ext.asyncio import AsyncSession
from deepdive.db import repository
from deepdive.core.config import settings
from deepdive.agent.embedders import get_embedder

_llm: AnyLLM | None = None


def get_llm() -> AnyLLM:
    global _llm
    if _llm is None:
        _llm = AnyLLM.create(
            settings.llm_provider,
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
        )
    return _llm


async def embed_text(text: str) -> list[float]:
    embedder = get_embedder()
    return await embedder.embed(text)


async def retrieve_context(query: str, db: AsyncSession, top_k: int = 5) -> str:
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
    llm = get_llm()
    try:
        response = await llm.acompletion(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
