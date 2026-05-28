"""
config.py — Centralized configuration using pydantic-settings.

All settings are read from environment variables or .env file.
Add new settings here; never hardcode values elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    chat_model: str = Field("gpt-4o-mini", description="Chat completion model")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature")

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = Field(1000, gt=0, description="Text chunk size in characters")
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks")

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = Field(5, gt=0, le=20, description="Number of retrieved chunks")

    # ── Storage ──────────────────────────────────────────────
    vector_db_path: str = Field("faiss_index", description="Local FAISS index path")
    upload_dir: str = Field("uploads", description="Directory for uploaded PDFs")

    # ── Logging ──────────────────────────────────────────────
    log_level: str = Field("INFO", description="Logging level")

    # ── API ──────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", description="FastAPI host")
    api_port: int = Field(8000, description="FastAPI port")
    allowed_origins: str = Field(
        "http://localhost:3000,http://localhost:8501",
        description="Comma-separated CORS origins",
    )

    # ── Analytics ────────────────────────────────────────────
    enable_analytics: bool = Field(True, description="Enable query analytics")
    analytics_db_path: str = Field("analytics.db", description="SQLite analytics DB")

    # ── Derived properties ───────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def vector_db_dir(self) -> Path:
        return Path(self.vector_db_path)

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()  # type: ignore[call-arg]
