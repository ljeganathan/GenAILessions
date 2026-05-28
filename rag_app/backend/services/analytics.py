"""
analytics.py — Phase 5: Analytics tracking with SQLite.

Tracks queries, uploads, latency, and token usage.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── DB Setup ──────────────────────────────────────────────────────────────────

DB_PATH = Path(settings.analytics_db_path)

CREATE_QUERIES_TABLE = """
CREATE TABLE IF NOT EXISTS queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT,
    latency_ms  INTEGER,
    chunks_used INTEGER,
    session_id  TEXT,
    created_at  TEXT NOT NULL
);
"""

CREATE_UPLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS uploads (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    filename   TEXT NOT NULL,
    pages      INTEGER,
    chunks     INTEGER,
    created_at TEXT NOT NULL
);
"""

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER REFERENCES queries(id),
    score      INTEGER CHECK(score BETWEEN 1 AND 5),
    comment    TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create analytics tables if they don't exist."""
    with _conn() as con:
        con.execute(CREATE_QUERIES_TABLE)
        con.execute(CREATE_UPLOADS_TABLE)
        con.execute(CREATE_FEEDBACK_TABLE)
    logger.info("Analytics DB initialised at '%s'", DB_PATH)


# ── Write helpers ─────────────────────────────────────────────────────────────

def record_query(
    question: str,
    answer: str,
    latency_ms: int,
    chunks_used: int,
    session_id: str | None = None,
) -> int:
    """Insert a query record; returns the new row ID."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO queries (question, answer, latency_ms, chunks_used, session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (question, answer, latency_ms, chunks_used, session_id, now),
        )
        return cur.lastrowid or 0


def record_upload(filename: str, pages: int, chunks: int) -> None:
    """Insert an upload record."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "INSERT INTO uploads (filename, pages, chunks, created_at) VALUES (?, ?, ?, ?)",
            (filename, pages, chunks, now),
        )


def record_feedback(query_id: int, score: int, comment: str = "") -> None:
    """Insert user feedback for a query."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "INSERT INTO feedback (query_id, score, comment, created_at) VALUES (?, ?, ?, ?)",
            (query_id, score, comment, now),
        )


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_summary() -> dict[str, Any]:
    """Return high-level analytics summary."""
    with _conn() as con:
        total_q = con.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        total_u = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        avg_lat = con.execute("SELECT AVG(latency_ms) FROM queries").fetchone()[0] or 0.0

        top_q = con.execute(
            """SELECT question, COUNT(*) as cnt
               FROM queries GROUP BY question ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()

    return {
        "total_queries": total_q,
        "total_uploads": total_u,
        "avg_latency_ms": round(avg_lat, 1),
        "top_questions": [{"question": r["question"], "count": r["cnt"]} for r in top_q],
    }


def get_query_trend(days: int = 7) -> list[dict[str, Any]]:
    """Return query counts per day for the last N days."""
    with _conn() as con:
        rows = con.execute(
            """SELECT DATE(created_at) as day, COUNT(*) as cnt
               FROM queries
               WHERE created_at >= DATE('now', ?)
               GROUP BY day ORDER BY day""",
            (f"-{days} days",),
        ).fetchall()
    return [{"day": r["day"], "count": r["cnt"]} for r in rows]
