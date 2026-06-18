from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, Integer, String
from pgvector.sqlalchemy import Vector

from deepdive.core.config import settings


class Base(DeclarativeBase):
    pass


class PubMedAbstract(Base):
    """
    Represents a chunk extracted from a PubMed abstract regarding
    medical interventions and their contra-indications.
    """

    __tablename__ = "pubmed_abstracts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pmid: Mapped[str] = mapped_column(String, index=True)  # PubMed ID
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)

    embedding: Mapped[Vector] = mapped_column(Vector(settings.embedding_dimension))
