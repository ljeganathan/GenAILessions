"""
retriever.py — Phase 2: Retrieval layer.

Wraps FAISS similarity search with:
  • Configurable top-k
  • Metadata filtering
  • Source deduplication
  • Formatted context builder for prompts
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.services.ingestion import load_vector_store

logger = get_logger(__name__)
settings = get_settings()


# ── Core Retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: Optional[int] = None,
    store: Optional[FAISS] = None,
    filter_metadata: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """
    Perform similarity search and return the top-k most relevant chunks.

    Args:
        query:           The user's question.
        top_k:           Number of chunks to retrieve (defaults to settings.top_k).
        store:           Pre-loaded FAISS store (loads from disk if None).
        filter_metadata: Optional dict to filter on metadata fields.

    Returns:
        List of Document chunks ranked by relevance.
    """
    k = top_k or settings.top_k
    vs = store or load_vector_store()

    logger.debug("Retrieving top-%d chunks for query: '%s'", k, query[:80])

    if filter_metadata:
        docs = vs.similarity_search(query, k=k, filter=filter_metadata)
    else:
        docs = vs.similarity_search(query, k=k)

    logger.debug("  → %d chunk(s) retrieved", len(docs))
    return docs


def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    store: Optional[FAISS] = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve chunks with their similarity scores (lower = more similar for L2).

    Args:
        query:  The user's question.
        top_k:  Number of results.
        store:  Pre-loaded FAISS store.

    Returns:
        List of (Document, score) tuples.
    """
    k = top_k or settings.top_k
    vs = store or load_vector_store()
    return vs.similarity_search_with_score(query, k=k)


# ── Context Building ──────────────────────────────────────────────────────────

def format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a numbered context block for the prompt.

    Each chunk includes its source file and page for citation.

    Args:
        docs: Retrieved document chunks.

    Returns:
        Multi-line string ready for injection into the prompt template.
    """
    if not docs:
        return "No relevant documents found."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        parts.append(f"[{i}] Source: {source} | Page: {page}\n{text}")

    return "\n\n---\n\n".join(parts)


def extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """
    Extract deduplicated source citation objects from retrieved chunks.

    Args:
        docs: Retrieved document chunks.

    Returns:
        List of source dicts: {source, page, chunk_id}.
    """
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []

    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "")
        key = f"{src}::{page}"

        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": src,
                    "page": page,
                    "chunk_id": chunk_id,
                    "snippet": doc.page_content[:200].strip() + "…",
                }
            )

    return sources
