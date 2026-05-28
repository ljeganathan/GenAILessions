# 📚 RAG Application — Production-Style Document Q&A

A full-stack Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, **OpenAI**, and **FAISS**, with both a **Streamlit** and **React** frontend.

---

## 🏗 Architecture

```
rag_app/
├── backend/
│   ├── core/
│   │   ├── config.py           ← Pydantic settings (env-driven)
│   │   └── logging_config.py   ← Structured logging
│   ├── services/
│   │   ├── ingestion.py        ← Phase 1: PDF → chunks → FAISS
│   │   ├── retriever.py        ← Phase 2: Similarity search
│   │   ├── prompt_templates.py ← Phase 2: Prompt engineering
│   │   ├── rag_chain.py        ← Phase 2: RAG orchestration
│   │   └── analytics.py        ← Phase 5: SQLite analytics
│   ├── routers/
│   │   └── chat.py             ← Phase 3: FastAPI endpoints
│   ├── models/
│   │   └── schemas.py          ← Pydantic request/response models
│   └── main.py                 ← FastAPI app + middleware
├── frontend_streamlit/
│   └── streamlit_app.py        ← Phase 4: Streamlit UI
├── frontend_react/
│   └── src/
│       ├── App.jsx             ← Phase 4: React chat UI
│       └── api/ragApi.js       ← API client
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ⚡ Quick Start

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload
# → http://localhost:8000
# → Swagger docs: http://localhost:8000/docs
```

### 4a. Start the Streamlit UI

```bash
streamlit run frontend_streamlit/streamlit_app.py
# → http://localhost:8501
```

### 4b. Start the React UI

```bash
cd frontend_react
npm install
npm run dev
# → http://localhost:3000
```

---

## 🐳 Docker

```bash
docker-compose up --build
# Backend:   http://localhost:8000
# Streamlit: http://localhost:8501
```

---

## 🔌 API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload & ingest a PDF |
| `POST` | `/api/v1/chat` | Ask a question |
| `GET` | `/api/v1/documents` | List indexed documents |
| `POST` | `/api/v1/rebuild` | Rebuild FAISS from all uploads |
| `GET` | `/api/v1/analytics` | Analytics summary |
| `GET` | `/api/v1/analytics/trend` | Query trend (last N days) |

### Chat Request

```json
{
  "question": "What is the refund policy?",
  "session_id": "optional-uuid-for-memory",
  "top_k": 5,
  "temperature": 0.2
}
```

### Chat Response

```json
{
  "answer": "According to page 3 of policy.pdf, refunds are...",
  "sources": [
    {
      "source": "policy.pdf",
      "page": 3,
      "chunk_id": "abc123...",
      "snippet": "Refunds must be requested within 30 days..."
    }
  ],
  "session_id": "uuid",
  "latency_ms": 842,
  "chunks_used": 5
}
```

---

## ⚙️ Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | Your OpenAI key |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `5` | Retrieved chunks per query |
| `TEMPERATURE` | `0.2` | LLM creativity (0–1) |
| `VECTOR_DB_PATH` | `faiss_index` | FAISS index directory |
| `UPLOAD_DIR` | `uploads` | PDF upload directory |

---

## 🧠 RAG Pipeline Explained

```
User Question
     ↓
[Condense Question]   ← Only in conversational mode
     ↓
[Embed Question]      ← OpenAI Embeddings
     ↓
[FAISS Search]        ← Top-K similarity search
     ↓
[Format Context]      ← Source + page citations
     ↓
[Prompt + LLM]        ← GPT with grounded context
     ↓
[Answer + Sources]
```

---

## 🛡 Hallucination Reduction

- **Explicit grounding**: "Answer ONLY from context"
- **Fallback instruction**: "Say I don't know if not in context"
- **Low temperature** (0.2 default)
- **Source citation requirement** in every prompt
- **Standalone question reformulation** for follow-ups
- **Chunk overlap** prevents boundary truncation

---

## 🚀 Production Improvements

- **Redis** for session storage and response caching
- **PostgreSQL** for document metadata and analytics
- **Hybrid search** (BM25 + FAISS) for better recall
- **Re-ranking** with a cross-encoder model
- **Authentication** via JWT / OAuth2
- **Rate limiting** per user/API key
- **Kubernetes** deployment with autoscaling
- **Monitoring** via Prometheus + Grafana
- **OCR support** via Tesseract for scanned PDFs

---

## 🐛 Common Issues

**`FileNotFoundError: No FAISS index`** — Upload at least one PDF first via `POST /api/v1/upload`.

**`AuthenticationError`** — Double-check `OPENAI_API_KEY` in `.env`.

**`chunk_overlap must be less than chunk_size`** — Reduce `CHUNK_OVERLAP` in `.env`.

**React: Cannot connect to API** — Ensure backend is running on `localhost:8000` and CORS allows `localhost:3000`.
