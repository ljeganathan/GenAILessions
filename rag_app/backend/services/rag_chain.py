"""
rag_chain.py — Phase 2: RAG Chain Orchestration.

Combines retriever + prompt + LLM into a runnable pipeline.
Supports both stateless Q&A and conversational (memory-aware) modes.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.services.ingestion import load_vector_store
from backend.services.prompt_templates import (
    condense_question_prompt,
    conversational_rag_prompt,
    rag_chat_prompt,
)
from backend.services.retriever import extract_sources, format_context, retrieve

logger = get_logger(__name__)
settings = get_settings()


# ── LLM singleton factory ─────────────────────────────────────────────────────

def _make_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=temperature if temperature is not None else settings.temperature,
        openai_api_key=settings.openai_api_key,
    )


# ── Stateless RAG ─────────────────────────────────────────────────────────────

class RAGChain:
    """
    Single-turn (stateless) Retrieval-Augmented Generation chain.

    Usage:
        chain = RAGChain()
        result = chain.ask("What is the refund policy?")
    """

    def __init__(self, top_k: Optional[int] = None, temperature: Optional[float] = None):
        self._top_k = top_k or settings.top_k
        self._llm = _make_llm(temperature)
        self._store = None  # lazy-loaded

    @property
    def store(self):
        if self._store is None:
            self._store = load_vector_store()
        return self._store

    def ask(self, question: str) -> dict[str, Any]:
        """
        Answer a question using retrieved document context.

        Args:
            question: Natural language question.

        Returns:
            Dict with keys: answer, sources, latency_ms, chunks_used.
        """
        t0 = time.perf_counter()

        # 1. Retrieve relevant chunks
        docs = retrieve(question, top_k=self._top_k, store=self.store)

        if not docs:
            return {
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": [],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "chunks_used": 0,
            }

        # 2. Build context string
        context = format_context(docs)

        # 3. Run LLM
        chain = (
            rag_chat_prompt
            | self._llm
            | StrOutputParser()
        )
        answer = chain.invoke({"context": context, "question": question})

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "RAGChain.ask | latency=%dms | chunks=%d | model=%s",
            latency_ms,
            len(docs),
            settings.chat_model,
        )

        return {
            "answer": answer,
            "sources": extract_sources(docs),
            "latency_ms": latency_ms,
            "chunks_used": len(docs),
        }


# ── Conversational RAG ────────────────────────────────────────────────────────

class ConversationalRAGChain:
    """
    Multi-turn RAG chain with in-session conversation memory.

    Usage:
        chain = ConversationalRAGChain()
        r1 = chain.ask("What is the return policy?")
        r2 = chain.ask("What about international orders?")  # uses history
        chain.clear_history()
    """

    def __init__(self, top_k: Optional[int] = None, temperature: Optional[float] = None):
        self._top_k = top_k or settings.top_k
        self._llm = _make_llm(temperature)
        self._store = None
        self._history: list[dict[str, str]] = []  # [{role, content}]

    @property
    def store(self):
        if self._store is None:
            self._store = load_vector_store()
        return self._store

    def _format_history(self) -> str:
        if not self._history:
            return "No prior conversation."
        lines = []
        for msg in self._history[-6:]:  # Keep last 3 exchanges (6 messages)
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up into a standalone retrieval query."""
        if not self._history:
            return question

        chain = condense_question_prompt | self._llm | StrOutputParser()
        standalone = chain.invoke(
            {
                "chat_history": self._format_history(),
                "question": question,
            }
        )
        logger.debug("Condensed question: '%s' → '%s'", question[:60], standalone[:60])
        return standalone

    def ask(self, question: str) -> dict[str, Any]:
        """
        Answer a question using conversation history + retrieved context.

        Args:
            question: Current user question.

        Returns:
            Dict with keys: answer, sources, latency_ms, chunks_used.
        """
        t0 = time.perf_counter()

        # 1. Rewrite question for better retrieval
        retrieval_query = self._condense_question(question)

        # 2. Retrieve
        docs = retrieve(retrieval_query, top_k=self._top_k, store=self.store)
        context = format_context(docs) if docs else "No relevant documents found."

        # 3. Generate
        chain = conversational_rag_prompt | self._llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context,
                "chat_history": self._format_history(),
                "question": question,
            }
        )

        # 4. Update memory
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "ConversationalRAGChain.ask | latency=%dms | turns=%d",
            latency_ms,
            len(self._history) // 2,
        )

        return {
            "answer": answer,
            "sources": extract_sources(docs),
            "latency_ms": latency_ms,
            "chunks_used": len(docs),
        }

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._history = []
        logger.info("Conversation history cleared")

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)