"""
interview_history.py
--------------------
SQLite-backed per-candidate interview history.

Responsibilities:
  - Store questions asked to each candidate (keyed by a session ID).
  - Check for duplicate / substantially similar questions.
  - Ensure complete isolation between different candidates.
  - Never store API credentials or sensitive personal information.

The database file is stored at: interview_history.db (project root).
It is created automatically on first use.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
import hashlib
from datetime import datetime
from typing import Optional

# Path to the SQLite database file (relative to this module)
DB_PATH = os.path.join(os.path.dirname(__file__), "interview_history.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS interview_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    candidate_name TEXT   NOT NULL,
    job_role      TEXT    NOT NULL,
    interview_type TEXT   NOT NULL,
    experience_level TEXT NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS asked_questions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    question_hash TEXT    NOT NULL,
    asked_at      TEXT    NOT NULL,
    UNIQUE(session_id, question_hash)
);

CREATE INDEX IF NOT EXISTS idx_asked_session  ON asked_questions(session_id);
CREATE INDEX IF NOT EXISTS idx_asked_hash     ON asked_questions(question_hash);
CREATE INDEX IF NOT EXISTS idx_session_id     ON interview_sessions(session_id);
"""


def _get_conn() -> sqlite3.Connection:
    """Open and return a connection to the history database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent readers
    return conn


def initialise_db() -> None:
    """Create tables if they do not exist. Safe to call multiple times."""
    with _get_conn() as conn:
        conn.executescript(_DDL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_question(text: str) -> str:
    """
    Normalise a question string for similarity comparison.
    Lowercases, removes punctuation, collapses whitespace.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)        # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def _question_hash(text: str) -> str:
    """SHA-256 hash of the normalised question text (first 16 hex chars used)."""
    norm = _normalise_question(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def _similarity_ratio(a: str, b: str) -> float:
    """
    Overlap-coefficient similarity for near-duplicate detection.
    Uses intersection / min(|a|, |b|) so short questions score higher
    than Jaccard would give them.
    Returns a value in [0, 1]. >= 0.6 is treated as "substantially similar".
    """
    tokens_a = set(_normalise_question(a).split())
    tokens_b = set(_normalise_question(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / min(len(tokens_a), len(tokens_b))


_SIMILARITY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_session(
    session_id: str,
    candidate_name: str,
    job_role: str,
    interview_type: str,
    experience_level: str,
) -> None:
    """
    Record the start of a new interview session.
    Safe to call multiple times with the same session_id (no duplicate rows).
    """
    initialise_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO interview_sessions
                (session_id, candidate_name, job_role, interview_type,
                 experience_level, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                candidate_name[:80],     # cap length — no sensitive data stored
                job_role[:100],
                interview_type[:60],
                experience_level[:60],
                datetime.utcnow().isoformat(),
            ),
        )


def record_question(session_id: str, question: str) -> None:
    """
    Record a question as having been asked in this session.
    Silently ignores if the exact question was already recorded.
    """
    initialise_db()
    q_hash = _question_hash(question)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO asked_questions
                (session_id, question, question_hash, asked_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, question[:600], q_hash, datetime.utcnow().isoformat()),
        )


def get_asked_questions(session_id: str) -> list[str]:
    """
    Return all questions asked in a given session.
    Only returns questions for the requested session_id — no cross-leakage.
    """
    initialise_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT question FROM asked_questions WHERE session_id = ? ORDER BY asked_at",
            (session_id,),
        ).fetchall()
    return [row["question"] for row in rows]


def is_duplicate(session_id: str, candidate_question: str) -> bool:
    """
    Return True if the given question is an exact or near-duplicate of a
    previously asked question in this session.
    """
    asked = get_asked_questions(session_id)
    # 1. Exact hash match
    q_hash = _question_hash(candidate_question)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM asked_questions WHERE session_id=? AND question_hash=?",
            (session_id, q_hash),
        ).fetchone()
    if row:
        return True

    # 2. Token-overlap near-duplicate check
    for asked_q in asked:
        if _similarity_ratio(candidate_question, asked_q) >= _SIMILARITY_THRESHOLD:
            return True

    return False


def filter_new_questions(
    session_id: str,
    questions: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split a list of questions into new (not yet asked) and duplicates.

    Returns
    -------
    (new_questions, duplicate_questions) — both as lists of strings.
    """
    new_qs:  list[str] = []
    dupe_qs: list[str] = []
    for q in questions:
        if is_duplicate(session_id, q):
            dupe_qs.append(q)
        else:
            new_qs.append(q)
    return new_qs, dupe_qs


def get_session_stats(session_id: str) -> dict:
    """Return basic statistics for a session."""
    initialise_db()
    with _get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM asked_questions WHERE session_id=?",
            (session_id,),
        ).fetchone()["cnt"]
        meta = conn.execute(
            "SELECT * FROM interview_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return {
        "session_id":       session_id,
        "total_asked":      total,
        "candidate_name":   meta["candidate_name"] if meta else "Unknown",
        "job_role":         meta["job_role"]        if meta else "Unknown",
        "interview_type":   meta["interview_type"]  if meta else "Unknown",
    }


def clear_session(session_id: str) -> None:
    """
    Delete all records for a session.
    Used when a candidate starts a completely fresh interview.
    Does NOT affect any other session.
    """
    initialise_db()
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM asked_questions WHERE session_id=?", (session_id,)
        )
        conn.execute(
            "DELETE FROM interview_sessions WHERE session_id=?", (session_id,)
        )
