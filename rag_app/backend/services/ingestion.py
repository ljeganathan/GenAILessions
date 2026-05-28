"""
ingestion.py — Phase 1: Document Processing Pipeline.

Responsibilities:
  • Load single or multiple PDFs with metadata extraction
  • Split text into overlapping chunks
  • Generate OpenAI embeddings
  • Persist / update FAISS vector store
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Embeddings singleton ──────────────────────────────────────────────────────

def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


# ── PDF Loading ───────────────────────────────────────────────────────────────

def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a single PDF and attach rich metadata to every page-document.
    Filters out pages with no extractable text (scanned/image pages).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF: %s", path.name)
    loader = PyPDFLoader(str(path))
    pages: list[Document] = loader.load()

    # Filter pages that have actual text content
    pages = [p for p in pages if p.page_content and p.page_content.strip()]

    if not pages:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "This may be a scanned/image-based PDF. "
            "Please use a text-based PDF or run OCR on it first."
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    for i, doc in enumerate(pages):
        doc.metadata.update(
            {
                "source": path.name,
                "source_path": str(path),
                "page": doc.metadata.get("page", i),
                "total_pages": len(pages),
                "ingested_at": timestamp,
            }
        )

    logger.info("  → %d pages with text loaded from '%s'", len(pages), path.name)
    return pages


def load_pdfs_from_folder(folder: str | Path) -> list[Document]:
    """Recursively load all PDFs found inside a folder."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    pdf_files = list(folder.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in: %s", folder)
        return []

    logger.info("Found %d PDF(s) in '%s'", len(pdf_files), folder)
    all_docs: list[Document] = []
    for pdf in pdf_files:
        try:
            all_docs.extend(load_pdf(pdf))
        except Exception as exc:
            logger.error("Failed to load '%s': %s", pdf.name, exc)

    return all_docs


# ── Text Splitting ────────────────────────────────────────────────────────────

def split_documents(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """Split documents into overlapping chunks and assign unique chunk IDs."""
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Filter empty chunks
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    if not chunks:
        raise ValueError("No text chunks could be created from the document.")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = i

    logger.info("Split %d document(s) → %d chunk(s)", len(documents), len(chunks))
    return chunks


# ── FAISS Vector Store ────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> FAISS:
    """Create a new FAISS vector store from document chunks."""
    if not chunks:
        raise ValueError("Cannot build vector store: no chunks provided.")

    logger.info("Building FAISS index from %d chunk(s)…", len(chunks))
    t0 = time.perf_counter()
    embeddings = _make_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0
    logger.info("  → FAISS index built in %.2fs", elapsed)
    return store


def save_vector_store(store: FAISS, path: Optional[str | Path] = None) -> Path:
    """Persist a FAISS vector store to disk."""
    save_path = Path(path or settings.vector_db_path)
    save_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(save_path))
    logger.info("FAISS index saved to '%s'", save_path)
    return save_path


def load_vector_store(path: Optional[str | Path] = None) -> FAISS:
    """Load a persisted FAISS vector store from disk."""
    load_path = Path(path or settings.vector_db_path)
    if not load_path.exists():
        raise FileNotFoundError(
            f"No FAISS index at '{load_path}'. Ingest documents first."
        )

    embeddings = _make_embeddings()
    store = FAISS.load_local(
        str(load_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from '%s'", load_path)
    return store


def append_to_vector_store(
    new_chunks: list[Document],
    path: Optional[str | Path] = None,
) -> FAISS:
    """Add new chunks to an existing FAISS index (or create if absent)."""
    if not new_chunks:
        raise ValueError("Cannot append: no chunks provided.")

    load_path = Path(path or settings.vector_db_path)
    if load_path.exists():
        store = load_vector_store(load_path)
        store.add_documents(new_chunks)
        logger.info("Appended %d chunk(s) to existing index", len(new_chunks))
    else:
        logger.info("No existing index — creating new one")
        store = build_vector_store(new_chunks)

    save_vector_store(store, load_path)
    return store


# ── High-level Convenience ────────────────────────────────────────────────────

def ingest_pdf(file_path: str | Path, append: bool = True) -> dict:
    """Full ingestion pipeline for a single PDF."""
    pages = load_pdf(file_path)          # raises ValueError if no text
    chunks = split_documents(pages)      # raises ValueError if no chunks

    if append:
        store = append_to_vector_store(chunks)
    else:
        store = build_vector_store(chunks)
        save_vector_store(store)

    return {
        "file": Path(file_path).name,
        "pages": len(pages),
        "chunks": len(chunks),
        "index_path": str(settings.vector_db_path),
    }


def ingest_folder(folder: str | Path) -> dict:
    """Full ingestion pipeline for a folder of PDFs."""
    docs = load_pdfs_from_folder(folder)
    if not docs:
        return {"files": 0, "pages": 0, "chunks": 0}

    chunks = split_documents(docs)
    store = build_vector_store(chunks)
    save_vector_store(store)

    sources = {d.metadata.get("source") for d in docs}
    return {
        "files": len(sources),
        "pages": len(docs),
        "chunks": len(chunks),
        "index_path": str(settings.vector_db_path),
    }