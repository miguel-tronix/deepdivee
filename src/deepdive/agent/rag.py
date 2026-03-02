"""
RAG agent core: embedding and LLM completion helpers.

``embed_text`` is public and shared between the ingestion and query
routes to avoid duplication.
"""

import asyncio
from typing import Protocol
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from deepdive.db import repository
from deepdive.core.config import settings

# Lifecycle singleton — avoids spawning a new client per request.
_http_client = httpx.AsyncClient(base_url=settings.llm_api_base)


class Embedder(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...


class HTTPEmbedder:
    """Uses an external OpenAI-compatible HTTP API to generate embeddings."""

    def __init__(self, client: httpx.AsyncClient, model_name: str):
        self.client = client
        self.model_name = model_name

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post(
                "/embeddings",
                json={"input": texts, "model": self.model_name},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings via HTTP: {e}") from e


class LocalEmbedder:
    """Uses a local Hugging Face model via sentence-transformers to generate embeddings."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        import torch

        device = settings.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=settings.embedding_trust_remote_code,
        )
        self.batch_size = settings.embedding_batch_size

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
            )
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings locally: {e}") from e


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if settings.use_local_embeddings:
            _embedder = LocalEmbedder(settings.embedding_model)
        else:
            _embedder = HTTPEmbedder(_http_client, settings.embedding_model)
    return _embedder


async def cleanup_embedder():
    """Cleanup embedder resources on app shutdown."""
    global _embedder
    if _embedder is not None:
        if hasattr(_embedder, "model"):
            del _embedder.model
        _embedder = None


async def embed_text(text: str) -> list[float]:
    """
    Generate the float vector representation of the text.

    Uses either a local or HTTP embedder as configured in settings.
    """
    embedder = get_embedder()
    return await embedder.embed_text(text)


async def retrieve_context(query: str, db: AsyncSession, top_k: int = 5) -> str:
    """
    Embed the query, then retrieve the top-K nearest PubMed abstracts using
    cosine similarity and format them into a single context string for the LLM.

    Args:
        query: Medical intervention query string.
        db: Async SQLAlchemy session.
        top_k: Number of nearest neighbours to fetch.

    Returns:
        Newline-separated context string, or empty string if no matches.
    """
    query_vector = await embed_text(query)
    abstracts = await repository.cosine_similarity_search(db, query_vector, top_k=top_k)

    if not abstracts:
        return ""

    chunks = [
        f"PMID: {a['pmid']}\nTitle: {a['title']}\nAbstract: {a['content']}"
        for a in abstracts
    ]
    return "\n\n---\n\n".join(chunks)


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
        response = await _http_client.post(
            "/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
