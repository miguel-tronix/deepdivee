from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application Settings Configured via Environment Variables.
    """

    project_name: str = "DeepDive RAG Agent"
    version: str = "0.1.0"

    # Postgres pgvector Settings
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")
    postgres_host: str = Field(default="localhost")
    postgres_port: str = Field(default="5432")
    postgres_db: str = Field(default="deepdive")

    # Redis Settings
    redis_url: str = Field(default="redis://localhost:6379/0")

    # any-llm provider (maps to any_llm.LLMProvider)
    llm_provider: str = Field(default="openai")

    # External LLM / Ingestion API
    llm_api_base: str = Field(default="http://localhost:8000/v1")
    llm_api_key: str = Field(default="dummy")

    # Local Embedding Settings
    embedding_device: str = Field(default="auto")  # "auto", "cpu", "cuda"
    embedding_batch_size: int = Field(default=32)
    embedding_trust_remote_code: bool = Field(default=False)

    # Embedding settings
    embedding_backend: Literal["local", "openai"] = Field(default="local")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Embedding dimension (must match the model in use)
    # all-MiniLM-L6-v2 → 384
    embedding_dimension: int = Field(default=384)

    # LLM completion settings
    llm_model: str = Field(default="gpt-4o")

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
