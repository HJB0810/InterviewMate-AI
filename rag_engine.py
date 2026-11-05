"""
rag_engine.py
-------------
Retrieves relevant context from two sources:
  1. knowledge_base.json  — structured topic/concept metadata (fast JSON lookup)
  2. knowledge_base/*.txt — rich Q&A documents (random sampling via kb_loader)

This module performs NO network calls — it is pure local file access.
"""

import json
import os
import re
from typing import Optional

import kb_loader

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

# Maximum length of any user-supplied string injected into context
_MAX_INPUT_LEN = 200


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

def _sanitise(text: str) -> str:
    """
    Strip characters that could cause prompt injection or break JSON parsing.
    Truncates to _MAX_INPUT_LEN characters after stripping.
    """
    if not isinstance(text, str):
        return ""
    # Remove null bytes and ASCII control characters (except newline/tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse sequences of whitespace-only lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Truncate
    return cleaned[:_MAX_INPUT_LEN].strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_kb() -> dict:
    """Load and return the knowledge base JSON."""
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_role(job_role: str) -> Optional[str]:
    """
    Map a free-text job role to a key that exists in the knowledge base.
    Returns None if no match is found.
    """
    role_lower = job_role.lower()
    mapping = {
        # Python
        "python": "python",
        "django": "python",
        "flask": "python",
        "fastapi": "python",
        "backend python": "python",
        # JavaScript
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "nodejs": "javascript",
        "typescript": "javascript",
        # React
        "react": "react",
        "next.js": "react",
        "nextjs": "react",
        "frontend react": "react",
        # Java
        "java": "java",
        "spring": "java",
        "spring boot": "java",
        "android": "java",
        "kotlin": "java",
        # Data Science / ML
        "data science": "data_science",
        "data scientist": "data_science",
        "machine learning": "data_science",
        "ml engineer": "data_science",
        "ai engineer": "data_science",
        "deep learning": "data_science",
        "nlp": "data_science",
        "data analyst": "data_science",
        # Web Development
        "web development": "web_development",
        "web developer": "web_development",
        "frontend": "web_development",
        "front-end": "web_development",
        "full stack": "web_development",
        "fullstack": "web_development",
        "html": "web_development",
        "css": "web_development",
        # DevOps / Cloud
        "devops": "devops",
        "sre": "devops",
        "site reliability": "devops",
        "cloud": "devops",
        "aws": "devops",
        "azure": "devops",
        "gcp": "devops",
        "kubernetes": "devops",
        "docker": "devops",
        "platform engineer": "devops",
        # SQL / Database
        "sql": "sql_database",
        "database": "sql_database",
        "dba": "sql_database",
        "data engineer": "sql_database",
        "postgresql": "sql_database",
        "mysql": "sql_database",
        # System Design
        "system design": "web_development",
        "architect": "web_development",
        # DSA
        "dsa": "python",
        "algorithms": "python",
    }
    for keyword, kb_key in mapping.items():
        if keyword in role_lower:
            return kb_key
    return None


def _normalise_level(experience_level: str) -> str:
    """
    Map an experience level string to beginner / intermediate / senior.
    Defaults to beginner when uncertain.
    """
    level_lower = experience_level.lower()
    if any(w in level_lower for w in ["senior", "lead", "principal", "expert", "5+"]):
        return "senior"
    if any(w in level_lower for w in ["mid", "intermediate", "2", "3", "4"]):
        return "intermediate"
    return "beginner"


def _level_to_difficulty(level: str) -> str:
    """Map normalised level to .txt file difficulty label."""
    return {"senior": "Advanced", "intermediate": "Intermediate"}.get(level, "Beginner")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_context(
    interview_type: str,
    job_role: str,
    experience_level: str,
    skills: str,
    num_sample_questions: int = 5,
) -> dict:
    """
    Retrieve relevant context from both the JSON knowledge base and the
    .txt Q&A files.

    Parameters
    ----------
    interview_type       : "Technical Interview", "HR Interview",
                           "Behavioral Interview", or "Soft Skills Assessment"
    job_role             : Free-text job role entered by the candidate
    experience_level     : Free-text experience level (e.g. "2 years", "Senior")
    skills               : Comma-separated list of skills entered by the candidate
    num_sample_questions : How many random .txt questions to sample for context

    Returns
    -------
    A dict with keys:
        - questions          : list[str]  — randomly sampled questions from .txt files
        - topics             : list[str]  — relevant topics from JSON KB
        - key_concepts       : list[str]  — concepts to probe
        - preparation_tips   : list[str]  — actionable preparation advice
        - sampled_qa         : list[dict] — full Q&A dicts from .txt files
        - matched_role       : str        — JSON KB key that was matched
        - level              : str        — normalised level used
        - interview_type     : str        — normalised interview type
    """
    # ── Sanitise all user inputs before they touch any data ──
    interview_type   = _sanitise(interview_type)
    job_role         = _sanitise(job_role)
    experience_level = _sanitise(experience_level)
    skills           = _sanitise(skills)

    kb          = _load_kb()
    level       = _normalise_level(experience_level)
    difficulty  = _level_to_difficulty(level)
    itype_lower = interview_type.lower()

    # ── Sample random questions from .txt files ──
    sampled_qa = kb_loader.sample_questions(
        interview_type=interview_type,
        job_role=job_role,
        difficulty=difficulty,
        n=num_sample_questions,
    )
    sampled_questions = [entry["question"] for entry in sampled_qa]

    # ── Behavioral interview ──
    if "behavioral" in itype_lower:
        data = kb.get("behavioral", {}).get("general", {}).get("all_levels", {})
        return {
            "questions":        sampled_questions or data.get("questions", []),
            "topics":           data.get("topics", []),
            "key_concepts":     data.get("evaluation_criteria", []),
            "preparation_tips": data.get("preparation_tips", []),
            "sampled_qa":       sampled_qa,
            "matched_role":     "general",
            "level":            "all levels",
            "interview_type":   "Behavioral Interview",
        }

    # ── Soft skills assessment ──
    if "soft" in itype_lower:
        data = kb.get("soft_skills", {}).get("general", {}).get("all_levels", {})
        return {
            "questions":        sampled_questions or data.get("questions", []),
            "topics":           data.get("topics", []),
            "key_concepts":     data.get("evaluation_criteria", []),
            "preparation_tips": data.get("preparation_tips", []),
            "sampled_qa":       sampled_qa,
            "matched_role":     "general",
            "level":            "all levels",
            "interview_type":   "Soft Skills Assessment",
        }

    # ── HR interview ──
    if "hr" in itype_lower:
        data = kb.get("hr", {}).get("general", {}).get("all_levels", {})
        return {
            "questions":        sampled_questions or data.get("questions", []),
            "topics":           data.get("topics", []),
            "key_concepts":     data.get("evaluation_criteria", []),
            "preparation_tips": data.get("preparation_tips", []),
            "sampled_qa":       sampled_qa,
            "matched_role":     "general",
            "level":            "all levels",
            "interview_type":   "HR Interview",
        }

    # ── Technical interview ──
    tech_kb      = kb.get("technical", {})
    matched_role = _normalise_role(job_role)

    if matched_role and matched_role in tech_kb:
        role_data = tech_kb[matched_role].get(level, {})
        if not role_data:
            role_data = tech_kb[matched_role].get("beginner", {})
    else:
        role_data    = {}
        matched_role = "general"
        for role_content in tech_kb.values():
            level_data = role_content.get(level, role_content.get("beginner", {}))
            role_data.setdefault("questions", []).extend(level_data.get("questions", [])[:2])
            role_data.setdefault("topics", []).extend(level_data.get("topics", [])[:2])
            role_data.setdefault("key_concepts", []).extend(level_data.get("key_concepts", [])[:2])
            role_data.setdefault("preparation_tips", []).extend(level_data.get("preparation_tips", [])[:1])

    # Prefer randomly sampled .txt questions; fall back to JSON KB list
    final_questions = sampled_questions if sampled_questions else role_data.get("questions", [])

    return {
        "questions":        final_questions,
        "topics":           role_data.get("topics", []),
        "key_concepts":     role_data.get("key_concepts", []),
        "preparation_tips": role_data.get("preparation_tips", []),
        "sampled_qa":       sampled_qa,
        "matched_role":     matched_role,
        "level":            level,
        "interview_type":   "Technical Interview",
    }
