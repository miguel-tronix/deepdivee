"""
RAG agent core: embedding and LLM completion helpers.

``embed_text`` is public and shared between the ingestion and query
routes to avoid duplication.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from deepdive.db import repository
from deepdive.agent.embedders import get_embedder
from deepdive.agent.templating import render

# Simple async-safe LRU cache for embedding results.
# functools.lru_cache cannot be used directly on async functions (it caches
# the coroutine object, not the result).  This dict-based cache with a
# max-size eviction policy avoids repeated model-inference or API calls
# when the same text is embedded more than once.
_embed_cache: dict[str, list[float]] = {}
_MAX_EMBED_CACHE = 2048

async def embed_text(text: str) -> list[float]:
    try:
        return _embed_cache[text]
    except KeyError:
        embedder = get_embedder()
        vector = await embedder.embed(text)
        if len(_embed_cache) >= _MAX_EMBED_CACHE:
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[text] = vector
        return vector


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
        render("context_chunk.jinja2", pmid=m["pmid"], title=m["title"], content=m["content"])
        for m in matches
    ]
    return "\n\n---\n\n".join(context_chunks)


async def retrieve_pubmed_context(search_query: str) -> str:
    """Search PubMed abstracts via vector similarity and return relevant context.

    Use this tool to retrieve medical literature context about a specific
    intervention, drug, or condition. Performs a semantic search over stored
    PubMed abstracts and returns the most relevant results.

    Args:
        search_query: A detailed search query describing the medical intervention,
                     drug, or condition to look up.

    Returns:
        Formatted context with PMIDs, titles, and abstracts, or a message
        indicating no relevant literature was found.
    """
    from deepdive.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await retrieve_context(search_query, db, top_k=5)
    return result or "No relevant PubMed literature found for this query."



