from pydantic import BaseModel


class PubMedEmbeddingRequest(BaseModel):
    pmid: str
    title: str
    abstract: str


class PubMedEmbeddingResponse(BaseModel):
    message: str
    pmid: str


class IndicationMatch(BaseModel):
    pmid: str
    title: str
    content: str
    similarity_score: float


class RAGRequest(BaseModel):
    intervention: str


class RAGResponse(BaseModel):
    intervention: str
    analysis: str
