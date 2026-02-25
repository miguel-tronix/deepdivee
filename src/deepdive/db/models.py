from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Text, Integer, String
from pgvector.sqlalchemy import Vector

Base = declarative_base()

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
    
    # 768 is a common embedding dimension (e.g. for some BERT/nomic models), 
    # adjust as needed based on the LLM ingestion pipeline.
    embedding: Mapped[Vector] = mapped_column(Vector(768)) 
