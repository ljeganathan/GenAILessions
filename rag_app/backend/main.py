"""
main.py — FastAPI application entry point.

Registers:
  • Middleware (CORS, request logging, timing)
  • Routers (chat, health, analytics)
  • Startup / shutdown lifecycle hooks
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.logging_config import get_logger, setup_logging
from backend.models.schemas import AnalyticsSummary, HealthResponse
from backend.routers.chat import router as chat_router
from backend.services import analytics as analytics_svc
from backend.services.ingestion import load_vector_store

# ── Bootstrap ─────────────────────────────────────────────────────────────────
setup_logging()
settings = get_settings()
logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Application API",
    description=(
        "Production-style Retrieval-Augmented Generation API built with "
        "FastAPI, LangChain, OpenAI, and FAISS."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging + Timing Middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    logger.info("→ %s %s [req:%s]", request.method, request.url.path, request_id)
    response: Response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "← %s %s [req:%s] status=%d %dms",
        request.method,
        request.url.path,
        request_id,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting RAG Application API…")
    analytics_svc.init_db()

    # Warm-up: try loading the vector store (non-fatal if absent)
    try:
        load_vector_store()
        logger.info("Vector store loaded successfully on startup")
    except FileNotFoundError:
        logger.warning(
            "No vector store found — upload PDFs via POST /upload to begin."
        )


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("Shutting down RAG Application API")


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/v1", tags=["RAG"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    db_exists = settings.vector_db_dir.exists()
    return HealthResponse(status="ok", version="1.0.0", vector_db_exists=db_exists)


@app.get("/api/v1/analytics", response_model=AnalyticsSummary, tags=["Analytics"])
async def analytics_summary() -> AnalyticsSummary:
    """Return query and upload analytics."""
    data = analytics_svc.get_summary()
    return AnalyticsSummary(**data)


@app.get("/api/v1/analytics/trend", tags=["Analytics"])
async def analytics_trend(days: int = 7) -> dict:
    """Return query trend for the last N days."""
    return {"trend": analytics_svc.get_query_trend(days)}
