"""
routers/chat.py — Chat and document management API endpoints.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.core.config import Settings, get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    RebuildResponse,
    SourceCitation,
    UploadResponse,
)
from backend.services import analytics as analytics_svc
from backend.services.ingestion import ingest_folder, ingest_pdf
from backend.services.rag_chain import ConversationalRAGChain, RAGChain

logger = get_logger(__name__)
router = APIRouter()

# In-process session store (swap for Redis in production)
_sessions: dict[str, ConversationalRAGChain] = {}


def _get_or_create_session(
    session_id: Optional[str], settings: Settings
) -> tuple[str, ConversationalRAGChain]:
    sid = session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = ConversationalRAGChain(
            top_k=settings.top_k, temperature=settings.temperature
        )
    return sid, _sessions[sid]


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Ask a question")
async def chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Answer a question using the RAG pipeline.

    If `session_id` is provided, conversation history is retained.
    Pass a new `session_id` to start a fresh conversation.
    """
    try:
        if body.session_id:
            # Conversational mode
            sid, chain = _get_or_create_session(body.session_id, settings)
            result = chain.ask(body.question)
        else:
            # Stateless mode
            chain = RAGChain(top_k=body.top_k, temperature=body.temperature)
            result = chain.ask(body.question)
            sid = None

        # Record analytics
        if settings.enable_analytics:
            analytics_svc.record_query(
                question=body.question,
                answer=result["answer"],
                latency_ms=result["latency_ms"],
                chunks_used=result["chunks_used"],
                session_id=sid,
            )

        sources = [SourceCitation(**s) for s in result["sources"]]
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            session_id=sid,
            latency_ms=result["latency_ms"],
            chunks_used=result["chunks_used"],
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question.",
        )


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, summary="Upload a PDF")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """
    Upload a PDF, chunk it, embed it, and merge into the FAISS index.
    Accepts only PDF files.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    save_path = settings.upload_dir_path / (file.filename or "upload.pdf")
    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info("Saved uploaded file: %s (%d bytes)", save_path.name, len(content))

        result = ingest_pdf(save_path, append=True)

        if settings.enable_analytics:
            analytics_svc.record_upload(
                filename=result["file"],
                pages=result["pages"],
                chunks=result["chunks"],
            )

        return UploadResponse(
            message=f"Successfully ingested '{file.filename}'.",
            **result,
        )
    except ValueError as exc:
        logger.warning("Upload rejected for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Upload error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process '{file.filename}': {exc}",
        )


# ── Documents ─────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse, summary="List uploaded documents")
async def list_documents(settings: Settings = Depends(get_settings)) -> DocumentListResponse:
    """Return the names of all uploaded PDFs."""
    upload_dir = settings.upload_dir_path
    pdfs = sorted(p.name for p in upload_dir.glob("*.pdf"))
    return DocumentListResponse(documents=pdfs, total=len(pdfs))


# ── Rebuild ───────────────────────────────────────────────────────────────────

@router.post("/rebuild", response_model=RebuildResponse, summary="Rebuild vector DB")
async def rebuild_vector_db(settings: Settings = Depends(get_settings)) -> RebuildResponse:
    """
    Re-ingest all PDFs in the upload directory and rebuild the FAISS index.
    Use after adding/removing documents manually.
    """
    try:
        result = ingest_folder(settings.upload_dir_path)
        return RebuildResponse(
            message="Vector database rebuilt successfully.",
            **result,
        )
    except Exception as exc:
        logger.exception("Rebuild error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )