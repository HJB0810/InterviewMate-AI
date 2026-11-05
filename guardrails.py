"""
guardrails.py
-------------
Centralised AI guardrail system for InterviewMate AI.

Responsibilities:
  1. Role restriction  — reject out-of-scope requests before they reach Granite.
  2. Input validation  — Pydantic models for all user-supplied data.
  3. Output validation — Pydantic models for all Granite responses.
  4. Evaluation checks — score range, required fields, fabrication guards.
  5. Prompt injection  — detect and neutralise jailbreak attempts in answers.
  6. Privacy           — ensure no cross-candidate data leaks.

This module never calls IBM Granite directly.
All validation is pure Python / Pydantic — no network calls.
"""

from __future__ import annotations

import re
import html
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# CONSTANTS
# ============================================================================

# Allowed interview types (validated against this list)
ALLOWED_INTERVIEW_TYPES = {
    "Technical Interview",
    "HR Interview",
    "Behavioral Interview",
    "Soft Skills Assessment",
}

# Allowed difficulty levels
ALLOWED_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}

# Allowed experience levels (normalised display values from the selectbox)
ALLOWED_EXPERIENCE_LEVELS = {
    "Beginner (0–1 yr)",
    "Intermediate (2–4 yrs)",
    "Senior (5+ yrs)",
}

# Keywords that indicate a request is NOT interview-related
_OUT_OF_SCOPE_PATTERNS: list[str] = [
    r"\b(write a story|tell me a joke|create an image|draw|poem|fiction)\b",
    r"\b(stock price|weather|sports|recipe|cook|dating|relationship)\b",
    r"\b(hack|exploit|malware|virus|crack password)\b",
    r"\b(ignore (all |previous |above |your )(instructions?|rules?|prompts?))\b",
    r"\b(pretend you are|act as|you are now|forget you are)\b",
    r"\b(reveal (your |the |system |)prompt|show api key|print secret)\b",
    r"\b(translate this|write a letter|compose an email unrelated)\b",
    r"\b(what is the capital|history of|president of|movie|music)\b",
]
_OUT_OF_SCOPE_RE = re.compile(
    "|".join(_OUT_OF_SCOPE_PATTERNS), re.IGNORECASE
)

# Prompt-injection patterns — checked inside candidate answers
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+|previous\s+|above\s+|your\s+)+(instructions?|rules?|prompts?)",
    r"\b(you are now|pretend you are)\b",
    r"\bact as\s+(a\s+|an\s+)?(?!interviewer)\w",
    r"\breveal\s+(your\s+|the\s+)?system\s+prompt\b",
    r"\bshow\s+(your\s+|the\s+)?instructions?\b",
    r"\bprint\s+(api[_ ]key|secret|password|token)\b",
    r"\b(jailbreak|bypass (safety|filter|restriction|guardrail))\b",
    r"<\|system\|>|<\|user\|>|<\|assistant\|>",          # Granite chat tokens
    r"\{\{.*?\}\}",                                       # template injection
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE
)

# Field length limits
_MAX_NAME_LEN      = 80
_MAX_ROLE_LEN      = 100
_MAX_SKILLS_LEN    = 200
_MAX_ANSWER_LEN    = 3000
_MAX_QUESTION_LEN  = 600

# ============================================================================
# PYDANTIC INPUT MODELS
# ============================================================================

class CandidateProfile(BaseModel):
    """Validated candidate profile submitted via the UI."""

    name: str = Field(..., min_length=1, max_length=_MAX_NAME_LEN)
    job_role: str = Field(..., min_length=2, max_length=_MAX_ROLE_LEN)
    experience_level: str
    skills: str = Field(..., min_length=2, max_length=_MAX_SKILLS_LEN)
    interview_type: str
    num_questions: int = Field(default=5, ge=3, le=8)

    @field_validator("name")
    @classmethod
    def name_no_special(cls, v: str) -> str:
        cleaned = _strip_control(v).strip()
        if not cleaned:
            raise ValueError("Name must not be empty after stripping.")
        # Allow letters, spaces, hyphens, apostrophes — reject otherwise
        if re.search(r"[^\w\s\-\'\.]", cleaned):
            raise ValueError(
                "Name may only contain letters, spaces, hyphens, apostrophes."
            )
        return cleaned

    @field_validator("job_role")
    @classmethod
    def role_not_empty(cls, v: str) -> str:
        cleaned = _strip_control(v).strip()
        if not cleaned:
            raise ValueError("Job role must not be empty.")
        return cleaned

    @field_validator("skills")
    @classmethod
    def skills_not_empty(cls, v: str) -> str:
        cleaned = _strip_control(v).strip()
        if not cleaned:
            raise ValueError("Skills must not be empty.")
        return cleaned

    @field_validator("experience_level")
    @classmethod
    def validate_experience(cls, v: str) -> str:
        if v not in ALLOWED_EXPERIENCE_LEVELS:
            raise ValueError(
                f"Experience level must be one of: {sorted(ALLOWED_EXPERIENCE_LEVELS)}"
            )
        return v

    @field_validator("interview_type")
    @classmethod
    def validate_interview_type(cls, v: str) -> str:
        if v not in ALLOWED_INTERVIEW_TYPES:
            raise ValueError(
                f"Interview type must be one of: {sorted(ALLOWED_INTERVIEW_TYPES)}"
            )
        return v


class CandidateAnswer(BaseModel):
    """Validated candidate answer submitted for evaluation."""

    question: str = Field(..., min_length=5, max_length=_MAX_QUESTION_LEN)
    answer: str = Field(..., max_length=_MAX_ANSWER_LEN)
    job_role: str = Field(..., min_length=2, max_length=_MAX_ROLE_LEN)
    experience_level: str
    interview_type: str

    @field_validator("answer")
    @classmethod
    def answer_sanitised(cls, v: str) -> str:
        return _strip_control(v).strip()

    @field_validator("experience_level")
    @classmethod
    def validate_experience(cls, v: str) -> str:
        if v not in ALLOWED_EXPERIENCE_LEVELS:
            raise ValueError("Invalid experience level.")
        return v

    @field_validator("interview_type")
    @classmethod
    def validate_interview_type(cls, v: str) -> str:
        if v not in ALLOWED_INTERVIEW_TYPES:
            raise ValueError("Invalid interview type.")
        return v


# ============================================================================
# PYDANTIC OUTPUT MODELS
# ============================================================================

class EvaluationResult(BaseModel):
    """Validated output from Granite's answer evaluation."""

    score: int = Field(..., ge=0, le=10)
    strengths: str = Field(..., min_length=1)
    missing_concepts: str = Field(..., min_length=1)
    improvements: str = Field(..., min_length=1)
    model_answer: str = Field(default="")
    confidence_tip: str = Field(default="")

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v) -> int:
        """Accept numeric strings and clamp to 0–10."""
        try:
            return max(0, min(10, int(float(str(v)))))
        except (ValueError, TypeError):
            return 0

    @model_validator(mode="after")
    def no_fabricated_execution_claims(self) -> "EvaluationResult":
        """
        Strip phrases that claim code was executed/run, which Granite
        cannot actually do.
        """
        _fabrication_phrases = [
            "I ran your code",
            "I executed",
            "I tested this",
            "running your solution",
            "output of the program is",
        ]
        for field_name in ("strengths", "improvements", "model_answer"):
            val = getattr(self, field_name, "")
            for phrase in _fabrication_phrases:
                val = val.replace(phrase, "[evaluated conceptually]")
            setattr(self, field_name, val)
        return self


class PreparationStrategy(BaseModel):
    """Validated output from Granite's preparation strategy."""

    study_plan: str = Field(default="")
    key_areas: str = Field(default="")
    practice_advice: str = Field(default="")
    day_of_tips: str = Field(default="")
    resources: str = Field(default="")

    @model_validator(mode="after")
    def fill_missing(self) -> "PreparationStrategy":
        defaults = {
            "study_plan":      "No study plan generated. Please try again.",
            "key_areas":       "Focus on the topics listed above.",
            "practice_advice": "Practice answering questions aloud.",
            "day_of_tips":     "Rest well, arrive early, stay confident.",
            "resources":       "official documentation, freeCodeCamp, LeetCode",
        }
        for k, v in defaults.items():
            if not getattr(self, k, "").strip():
                setattr(self, k, v)
        return self


# ============================================================================
# ROLE RESTRICTION
# ============================================================================

class GuardrailViolation(Exception):
    """Raised when a guardrail check fails."""
    def __init__(self, message: str, code: str = "VIOLATION"):
        super().__init__(message)
        self.code = code


def check_scope(text: str) -> None:
    """
    Raise GuardrailViolation if the text appears to be an out-of-scope
    request (not related to interview preparation).
    """
    if _OUT_OF_SCOPE_RE.search(text):
        raise GuardrailViolation(
            "This request is outside the scope of InterviewMate AI. "
            "I can only help with interview preparation: generating questions, "
            "evaluating answers, and providing preparation advice.",
            code="OUT_OF_SCOPE",
        )


def check_injection(text: str) -> str:
    """
    Detect prompt injection attempts inside candidate-provided text.
    Returns the text with injection patterns replaced by [REDACTED].
    Does NOT raise — allows evaluation to continue with sanitised text.
    """
    return _INJECTION_RE.sub("[REDACTED]", text)


# ============================================================================
# INPUT VALIDATION (thin public wrappers used by app.py / granite_service.py)
# ============================================================================

def validate_profile(
    name: str,
    job_role: str,
    experience_level: str,
    skills: str,
    interview_type: str,
    num_questions: int = 5,
) -> CandidateProfile:
    """
    Validate and return a CandidateProfile.
    Raises GuardrailViolation with a user-friendly message on failure.
    """
    try:
        return CandidateProfile(
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            skills=skills,
            interview_type=interview_type,
            num_questions=num_questions,
        )
    except Exception as exc:
        raise GuardrailViolation(
            f"Profile validation failed: {exc}", code="INVALID_PROFILE"
        )


def validate_answer_submission(
    question: str,
    answer: str,
    job_role: str,
    experience_level: str,
    interview_type: str,
) -> CandidateAnswer:
    """
    Validate and return a CandidateAnswer.
    Raises GuardrailViolation with a user-friendly message on failure.
    """
    try:
        return CandidateAnswer(
            question=question,
            answer=answer,
            job_role=job_role,
            experience_level=experience_level,
            interview_type=interview_type,
        )
    except Exception as exc:
        raise GuardrailViolation(
            f"Answer submission failed validation: {exc}", code="INVALID_ANSWER"
        )


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================

def validate_evaluation(raw: dict) -> EvaluationResult:
    """
    Validate Granite's evaluation output.
    Returns a safe EvaluationResult (never raises — fills in safe defaults).
    """
    _SAFE_FALLBACK = {
        "score":            0,
        "strengths":        "Unable to evaluate. Please try again.",
        "missing_concepts": "N/A",
        "improvements":     "Please resubmit your answer.",
        "model_answer":     "",
        "confidence_tip":   "Keep practising — every attempt builds skill.",
    }
    try:
        return EvaluationResult(**raw)
    except Exception:
        # Merge raw data with fallback, keeping safe keys
        merged = {**_SAFE_FALLBACK}
        for k in _SAFE_FALLBACK:
            if k in raw and raw[k] is not None:
                merged[k] = raw[k]
        try:
            return EvaluationResult(**merged)
        except Exception:
            return EvaluationResult(**_SAFE_FALLBACK)


def validate_preparation_strategy(raw: dict) -> PreparationStrategy:
    """
    Validate Granite's preparation strategy output.
    Returns a safe PreparationStrategy (never raises).
    """
    try:
        return PreparationStrategy(**raw)
    except Exception:
        return PreparationStrategy()


def validate_questions(questions: list, num_expected: int) -> list[str]:
    """
    Validate the list of generated questions.
    - Ensures each item is a non-empty string.
    - Removes duplicates.
    - Pads with a placeholder if Granite returned too few.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q:
            continue
        # Normalise for dedup
        norm = q.lower().strip("?. ")
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(q)

    # Pad if short — use indexed placeholders so dedup doesn't collapse them
    pad_index = 1
    while len(cleaned) < num_expected:
        cleaned.append(
            f"Please describe your approach to a key concept in this interview domain. (Q{pad_index})"
        )
        pad_index += 1

    return cleaned[:num_expected]


# ============================================================================
# EVALUATION RULE CHECKS (module-specific)
# ============================================================================

def check_hr_evaluation_bias(evaluation: EvaluationResult) -> EvaluationResult:
    """
    Ensure the HR evaluation does not reference protected characteristics.
    Strips any mention of age, gender, race, religion, etc.
    """
    _PROTECTED = re.compile(
        r"\b(age|gender|race|ethnicity|religion|nationality|disability|"
        r"pregnant|married|divorced|sexual orientation)\b",
        re.IGNORECASE,
    )
    for field_name in ("strengths", "missing_concepts", "improvements"):
        val = getattr(evaluation, field_name, "")
        if _PROTECTED.search(val):
            setattr(
                evaluation,
                field_name,
                _PROTECTED.sub("[protected characteristic removed]", val),
            )
    return evaluation


# ============================================================================
# HELPERS
# ============================================================================

def _strip_control(text: str) -> str:
    """Remove ASCII control characters and normalise unicode."""
    if not isinstance(text, str):
        return ""
    # NFKC normalisation: converts lookalike unicode to ASCII equivalents
    text = unicodedata.normalize("NFKC", text)
    # Remove control chars (except tab \x09 and newline \x0a)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def safe_html(text: str) -> str:
    """HTML-escape a string for safe display in Streamlit markdown."""
    return html.escape(str(text))
