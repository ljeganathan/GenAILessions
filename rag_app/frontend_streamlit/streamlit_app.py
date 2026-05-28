"""
streamlit_app.py — Phase 4: Streamlit UI for the RAG chatbot.

Features:
  • PDF upload with progress feedback
  • Conversational Q&A with session persistence
  • Source citations panel
  • Chat history display
  • Sidebar settings (model, top-k, temperature)
"""

from __future__ import annotations

import uuid
from io import BytesIO
from typing import Optional

import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"

# ── Session State Init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_chat(question: str, session_id: str, top_k: int, temperature: float) -> dict:
    resp = requests.post(
        f"{API_BASE}/chat",
        json={
            "question": question,
            "session_id": session_id,
            "top_k": top_k,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def api_upload(file_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, BytesIO(file_bytes), "application/pdf")},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_documents() -> list[str]:
    resp = requests.get(f"{API_BASE}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json().get("documents", [])


def api_analytics() -> dict:
    resp = requests.get(f"{API_BASE}/analytics", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_rebuild() -> dict:
    resp = requests.post(f"{API_BASE}/rebuild", timeout=120)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 RAG Settings")
    st.divider()

    st.subheader("🔧 Retrieval Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and st.button("📤 Ingest PDF", use_container_width=True):
        with st.spinner("Ingesting PDF…"):
            try:
                result = api_upload(uploaded_file.read(), uploaded_file.name)
                st.success(
                    f"✅ Ingested **{result['file']}**\n"
                    f"• {result['pages']} pages\n"
                    f"• {result['chunks']} chunks"
                )
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("📁 Indexed Documents")
    if st.button("🔄 Refresh list", use_container_width=True):
        try:
            docs = api_documents()
            if docs:
                for d in docs:
                    st.write(f"• {d}")
            else:
                st.info("No documents indexed yet.")
        except Exception as exc:
            st.error(f"Could not fetch documents: {exc}")

    st.divider()
    if st.button("🔁 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding…"):
            try:
                r = api_rebuild()
                st.success(r.get("message", "Done"))
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_sources = []
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ── Main Chat Area ────────────────────────────────────────────────────────────

col_chat, col_sources = st.columns([3, 1])

with col_chat:
    st.title("💬 Document Q&A Chat")
    st.caption("Ask questions about your uploaded documents.")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("latency_ms"):
                st.caption(
                    f"⏱ {msg['latency_ms']}ms · 📎 {msg.get('chunks_used', 0)} chunks"
                )

    # Chat input
    if prompt := st.chat_input("Ask something about your documents…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    result = api_chat(
                        prompt,
                        st.session_state.session_id,
                        top_k,
                        temperature,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    latency = result.get("latency_ms", 0)
                    chunks = result.get("chunks_used", 0)

                    st.markdown(answer)
                    st.caption(f"⏱ {latency}ms · 📎 {chunks} chunks retrieved")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "latency_ms": latency,
                            "chunks_used": chunks,
                        }
                    )
                    st.session_state.last_sources = sources

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API. Is the backend running?")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ── Sources Panel ─────────────────────────────────────────────────────────────

with col_sources:
    st.subheader("📌 Sources")
    sources = st.session_state.last_sources

    if sources:
        for i, src in enumerate(sources, 1):
            with st.expander(f"{i}. {src['source']} · p.{src['page']}"):
                st.markdown(f"*{src['snippet']}*")
                st.caption(f"Chunk ID: `{src['chunk_id'][:8]}…`")
    else:
        st.info("Sources will appear here after you ask a question.")

    st.divider()

    # Mini analytics
    st.subheader("📊 Analytics")
    if st.button("📈 Load Stats"):
        try:
            stats = api_analytics()
            st.metric("Total Queries", stats["total_queries"])
            st.metric("Total Uploads", stats["total_uploads"])
            st.metric("Avg Latency", f"{stats['avg_latency_ms']}ms")
        except Exception:
            st.warning("Analytics unavailable")
