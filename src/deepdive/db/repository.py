"""
Database repository for PubMed abstract embeddings.

Provides async helpers for inserting new embeddings and querying
the pgvector store using cosine similarity.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from deepdive.db.models import PubMedAbstract


async def insert_pubmed_embedding(
    db: AsyncSession,
    pmid: str,
    title: str,
    content: str,
    embedding: list[float],
) -> PubMedAbstract:
    """
    Persist a new PubMed abstract with its pre-computed embedding vector.

    Args:
        db: Async SQLAlchemy session.
        pmid: PubMed identifier string.
        title: Title of the abstract.
        content: Full abstract text.
        embedding: Pre-computed float vector (must match Vector(768)).

    Returns:
        The newly created and committed PubMedAbstract ORM object.
    """
    record = PubMedAbstract(
        pmid=pmid,
        title=title,
        content=content,
        embedding=embedding,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def cosine_similarity_search(
    db: AsyncSession,
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve the top-K PubMed abstracts closest to the query vector using
    cosine distance (pgvector ``<=>`` operator). Results are ordered from
    most to least similar.

    Args:
        db: Async SQLAlchemy session.
        query_vector: Embedded query as a list of floats.
        top_k: Maximum number of results to return (default 10).

    Returns:
        List of dicts with keys ``pmid``, ``title``, ``content``, and
        ``similarity_score`` (1 - cosine_distance, range [0, 1]).
    """
    stmt = (
        select(
            PubMedAbstract,
            PubMedAbstract.embedding.cosine_distance(query_vector).label("distance"),
        )
        .order_by(PubMedAbstract.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "pmid": row.PubMedAbstract.pmid,
            "title": row.PubMedAbstract.title,
            "content": row.PubMedAbstract.content,
            "similarity_score": round(1.0 - float(row.distance), 6),
        }
        for row in rows
    ]
