"""
kb_loader.py
------------
Parses the structured .txt interview Q&A files inside knowledge_base/ and
provides random question sampling for RAG context enrichment.

Each .txt file uses this repeating block format:
  ID       : <id>
  TOPIC    : <topic>
  DIFFICULTY: <Beginner|Intermediate|Advanced>
  QUESTION :
  <question text>
  ANSWER :
  <answer text>
  KEYWORDS : <comma-separated keywords>

This module performs NO network calls — it is pure local file parsing.
"""

import os
import re
import random
from functools import lru_cache
from typing import Optional

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

# ---------------------------------------------------------------------------
# File → subject tag mapping
# Used to route a job role / interview type to the right file(s)
# ---------------------------------------------------------------------------
FILE_TAGS: dict[str, list[str]] = {
    "Python_Interview_QA.txt":        ["python", "django", "flask", "fastapi", "backend"],
    "Django_Interview_QA.txt":        ["django", "python", "web", "rest", "api"],
    "SQL_Interview_QA.txt":           ["sql", "database", "postgresql", "mysql", "data engineer"],
    "HR_Interview_QA.txt":            ["hr", "human resources", "behavioral", "soft skills", "general"],
    "DSA_Interview_QA.txt":           ["dsa", "algorithms", "data structures", "coding", "technical"],
    "System_Design_Interview_QA.txt": ["system design", "architecture", "scalability", "design"],
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_file(filepath: str) -> list[dict]:
    """
    Parse a single .txt Q&A file into a list of question dicts.

    File format — each entry is a contiguous block that starts with:
        ID       : <id>
        TOPIC    : <topic>
        DIFFICULTY: <level>
        (separator line of dashes)
        QUESTION :
        <question text>

        ANSWER :
        <answer text>

        KEYWORDS : <comma-separated>

    Entries are separated by lines of '=' (80+ chars) or '-' (80+ chars).

    Each returned dict has keys:
        id, topic, difficulty, question, answer, keywords (list[str])
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    entries = []

    # A full entry begins at an ID line and ends just before the next ID line
    # or end-of-file. We locate every ID line and slice between them.
    id_positions = [m.start() for m in re.finditer(r"^ID\s*:", text, re.MULTILINE)]

    for i, start in enumerate(id_positions):
        end   = id_positions[i + 1] if i + 1 < len(id_positions) else len(text)
        block = text[start:end]

        entry: dict = {}

        # ID
        m = re.search(r"^ID\s*:\s*(.+)$", block, re.MULTILINE)
        if m:
            entry["id"] = m.group(1).strip()

        # TOPIC
        m = re.search(r"^TOPIC\s*:\s*(.+)$", block, re.MULTILINE)
        if m:
            entry["topic"] = m.group(1).strip()

        # DIFFICULTY
        m = re.search(r"^DIFFICULTY\s*:\s*(.+)$", block, re.MULTILINE)
        if m:
            entry["difficulty"] = m.group(1).strip()

        # QUESTION — text after "QUESTION :" until the next major section marker
        m = re.search(
            r"QUESTION\s*:\s*\n(.*?)"
            r"(?=\n(?:ANSWER|PROBLEM STATEMENT|REQUIREMENTS|PYTHON SOLUTION"
            r"|DESIGN EXPLANATION|HIGH-LEVEL ARCH)\s*:|\Z)",
            block,
            re.DOTALL,
        )
        if m:
            entry["question"] = m.group(1).strip()

        # ANSWER — multiple possible answer markers depending on question type
        # Priority: ANSWER → DESIGN EXPLANATION → PYTHON SOLUTION
        for answer_marker in (
            r"\nANSWER\s*:\s*\n",
            r"\nDESIGN EXPLANATION\s*:\s*\n",
            r"\nPYTHON SOLUTION\s*[^:]*:\s*\n",
        ):
            m = re.search(
                answer_marker + r"(.*?)(?=\nKEYWORDS\s*:|\nTIME COMPLEXITY\s*:|\nSCALABILITY\s*:|\Z)",
                block,
                re.DOTALL,
            )
            if m:
                entry["answer"] = m.group(1).strip()
                break

        # KEYWORDS
        m = re.search(r"^KEYWORDS\s*:\s*(.+)$", block, re.MULTILINE)
        if m:
            entry["keywords"] = [kw.strip() for kw in m.group(1).split(",")]
        else:
            entry["keywords"] = []

        # Only add if both id and question are present
        if entry.get("id") and entry.get("question"):
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Cached loader — parses each file once per process lifetime
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_all_files() -> dict[str, list[dict]]:
    """
    Load and parse all .txt files in knowledge_base/.
    Returns {filename: [entry, ...]}
    Cached so files are read from disk only once.
    """
    result: dict[str, list[dict]] = {}
    if not os.path.isdir(KB_DIR):
        return result

    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            entries = _parse_file(path)
            if entries:
                result[fname] = entries
        except Exception:
            # Silently skip unreadable files — app must not crash on load
            pass

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_questions(
    filename: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """
    Return all parsed questions, optionally filtered by file and/or difficulty.

    Parameters
    ----------
    filename   : e.g. "Python_Interview_QA.txt" — if None, returns from ALL files
    difficulty : "Beginner", "Intermediate", or "Advanced" — case-insensitive

    Returns
    -------
    list[dict] — each dict has: id, topic, difficulty, question, answer, keywords
    """
    all_files = _load_all_files()

    if filename:
        pool = all_files.get(filename, [])
    else:
        pool = [entry for entries in all_files.values() for entry in entries]

    if difficulty:
        diff_lower = difficulty.lower()
        pool = [e for e in pool if e.get("difficulty", "").lower() == diff_lower]

    return pool


def sample_questions(
    interview_type: str,
    job_role: str,
    difficulty: Optional[str] = None,
    n: int = 5,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Sample n random questions relevant to the given interview type and job role.

    Selection strategy:
      1. Score every loaded file against the interview_type and job_role.
      2. Collect all questions from the top-matching file(s).
      3. Optionally filter by difficulty.
      4. Return n randomly sampled questions (without replacement).

    Parameters
    ----------
    interview_type : e.g. "Technical Interview", "HR Interview"
    job_role       : e.g. "Python Developer"
    difficulty     : optional filter — "Beginner" / "Intermediate" / "Advanced"
    n              : number of questions to sample (default 5)
    seed           : optional random seed for reproducibility in tests

    Returns
    -------
    list[dict] — sampled questions, each with id, topic, difficulty,
                 question, answer, keywords
    """
    all_files = _load_all_files()
    if not all_files:
        return []

    itype_lower  = interview_type.lower()
    role_lower   = job_role.lower()
    search_terms = itype_lower.split() + role_lower.split()

    # Score each file by how many of its tags match the search terms
    scores: dict[str, int] = {}
    for fname, tags in FILE_TAGS.items():
        if fname not in all_files:
            continue
        score = sum(
            1 for tag in tags
            if any(tag in term or term in tag for term in search_terms)
        )
        scores[fname] = score

    # Pick files with the highest score; fall back to all if no matches
    max_score = max(scores.values(), default=0)
    if max_score > 0:
        best_files = [f for f, s in scores.items() if s == max_score]
    else:
        best_files = list(all_files.keys())

    # Collect question pool from best-matching files
    pool: list[dict] = []
    for fname in best_files:
        pool.extend(all_files.get(fname, []))

    # Filter by difficulty if requested
    if difficulty:
        diff_lower  = difficulty.lower()
        filtered    = [e for e in pool if e.get("difficulty", "").lower() == diff_lower]
        # Fall back to full pool if filter leaves too few
        if len(filtered) >= n:
            pool = filtered
        elif filtered:
            pool = filtered + random.sample(
                [e for e in pool if e not in filtered],
                min(n - len(filtered), len(pool) - len(filtered)),
            )

    if not pool:
        return []

    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def get_questions_as_text_list(
    interview_type: str,
    job_role: str,
    difficulty: Optional[str] = None,
    n: int = 5,
) -> list[str]:
    """
    Convenience wrapper: returns just the question strings (no metadata).
    Used to inject into RAG context for Granite prompt enrichment.
    """
    samples = sample_questions(
        interview_type=interview_type,
        job_role=job_role,
        difficulty=difficulty,
        n=n,
    )
    return [entry["question"] for entry in samples]


def get_file_stats() -> dict[str, int]:
    """Return {filename: question_count} for all loaded files."""
    return {fname: len(entries) for fname, entries in _load_all_files().items()}
