"""
prompt_templates.py — Phase 2: Reusable prompt templates for the RAG pipeline.

Best practices applied:
  • Explicit "use only the context" instruction → reduces hallucination
  • Instruction to say "I don't know" rather than fabricate
  • System / human roles separated for chat models
  • Templates are parametric so they can be swapped via config
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_HELPFUL_ASSISTANT = """You are a helpful, knowledgeable assistant.
Answer the user's question clearly and concisely.
If you are unsure, say so honestly instead of guessing."""

SYSTEM_DOCUMENT_QA = """You are a precise document analysis assistant.
Your sole source of truth is the context below, extracted from the user's documents.
Rules:
  1. Answer ONLY from the provided context.
  2. If the answer is not in the context, reply: "I don't have enough information in the provided documents to answer that question."
  3. Always cite the source document and page number when possible.
  4. Do NOT add knowledge from outside the context.
  5. Keep answers concise but complete."""

SYSTEM_STRICT_CONTEXT = """You are a strict, context-only assistant.
You must answer exclusively from the CONTEXT block below.
If the context does not contain the answer, respond with exactly:
"The provided documents do not contain this information."
Never speculate or use external knowledge."""

# ── RAG Chat Prompt (primary) ─────────────────────────────────────────────────

RAG_SYSTEM_TEMPLATE = SYSTEM_DOCUMENT_QA

RAG_HUMAN_TEMPLATE = """CONTEXT (extracted from documents):
{context}

---
QUESTION: {question}

Answer (cite source and page when possible):"""

rag_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human", RAG_HUMAN_TEMPLATE),
    ]
)

# ── Conversational RAG Prompt (with history) ──────────────────────────────────

CONVERSATIONAL_SYSTEM = """You are a helpful document Q&A assistant with memory of the conversation.
Answer from the retrieved CONTEXT only. If the context lacks the answer, say so.
Always maintain conversation continuity — refer to prior exchanges when relevant."""

CONVERSATIONAL_HUMAN = """CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER: {question}

ASSISTANT:"""

conversational_rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONVERSATIONAL_SYSTEM),
        ("human", CONVERSATIONAL_HUMAN),
    ]
)

# ── Standalone Question Reformulator ─────────────────────────────────────────
# Rewrites a follow-up question into a self-contained query for retrieval.

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a STANDALONE question that captures all required context.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

# ── Hallucination Reduction Tips (documentation) ──────────────────────────────

HALLUCINATION_REDUCTION_NOTES = """
Best practices embedded in the prompts above
──────────────────────────────────────────────
1. Explicit grounding — "Answer ONLY from the context" prevents the model from
   filling gaps with training data.

2. Fallback instruction — "If not in context, say I don't know" is more reliable
   than hoping the model will volunteer ignorance.

3. Low temperature (0.0–0.3) — Reduces creative/random completions. Set via
   TEMPERATURE in .env.

4. Source citation requirement — Forces the model to anchor answers to real
   retrieved chunks, making hallucinations detectable.

5. Chunk overlap — Ensures sentences near chunk boundaries are captured by at
   least one chunk, preventing mid-sentence truncation that leads to incomplete
   context.

6. Top-K tuning — Too few chunks → missing context. Too many → diluted signal
   and higher cost. Start with TOP_K=5.

7. Condense-then-retrieve — The standalone question reformulation step prevents
   retrieval failures caused by pronouns or context-dependent follow-ups
   ("What about the second point?" → full rewritten query).
"""
