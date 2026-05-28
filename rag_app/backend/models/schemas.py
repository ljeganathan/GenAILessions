"""
schemas.py — Pydantic models for all FastAPI request/response bodies.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversational mode")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override retrieval top-k")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Override LLM temperature")

    model_config = {"json_schema_extra": {"example": {"question": "What is this document about?"}}}


class SourceCitation(BaseModel):
    source: str
    page: Any
    chunk_id: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = []
    session_id: Optional[str] = None
    latency_ms: int = 0
    chunks_used: int = 0


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int
    index_path: str


class DocumentListResponse(BaseModel):
    documents: list[str]
    total: int


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    file: str
    pages: int
    chunks: int
    index_path: str


# ── Rebuild ───────────────────────────────────────────────────────────────────

class RebuildResponse(BaseModel):
    message: str
    files: int
    pages: int
    chunks: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    vector_db_exists: bool = False


# ── Analytics ────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_queries: int
    total_uploads: int
    avg_latency_ms: float
    top_questions: list[dict[str, Any]]
