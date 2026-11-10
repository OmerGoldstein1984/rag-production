"""
src/security/prompt_guard - Prompt injection / jailbreak defenses for the RAG
generation path.

This is the canonical fail-closed module used by src/generation/answer.py. It
provides `PromptInjectionBlocked`, injection detection, input framing, and
grounding verification. It is self-contained (no import from src.security.injection)
to avoid a circular import; injection.py re-exports from this module.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PromptInjectionBlocked(Exception):
    """Raised when a user query matches a high-confidence injection pattern."""


# Patterns that indicate an attempt to override system instructions or exfiltrate
# the prompt. Compiled once at import.
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all)\s+above\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a\s+)?\w+", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(a\s+)?\w+", re.IGNORECASE),
    re.compile(r"\b(repeat|reveal|print|output)\s+(the\s+)?(system|prompt|instructions)\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|dan|do\s+anything\s+now)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(the\s+)?(system|prompt)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<<\s*(system|user|assistant)\s*>>", re.IGNORECASE),
]

# Delimiters that frame untrusted user input inside the trusted prompt.
USER_INPUT_START = "<<USER_INPUT>>"
USER_INPUT_END = "<<END_USER_INPUT>>"


def is_injection_attempt(query: str) -> bool:
    """Return True if the query matches any known jailbreak / override pattern."""
    if not query:
        return False
    for rx in _PROMPT_INJECTION_PATTERNS:
        if rx.search(query):
            logger.warning("Prompt injection attempt detected: %s", query[:80])
            return True
    return False


def sanitize_user_input(query: str) -> str:
    """Fail-closed sanitization: raise PromptInjectionBlocked on injection.

    Unlike a neutralizing sanitizer, this canonical gate used by answer.py raises
    PromptInjectionBlocked so the caller refuses the request rather than
    neutralize-and-proceed.
    """
    if not query:
        return query
    if is_injection_attempt(query):
        logger.warning("Blocked prompt injection attempt: %s", query[:80])
        raise PromptInjectionBlocked("Request contains disallowed instruction-override patterns.")
    # Strip raw delimiter-like sequences so a user cannot forge our framing.
    cleaned = query.replace(USER_INPUT_START, "").replace(USER_INPUT_END, "")
    return cleaned.strip()


def frame_user_input(query: str) -> str:
    """Wrap untrusted user input in explicit delimiters."""
    if not query:
        return query
    return f"{USER_INPUT_START}\n{query}\n{USER_INPUT_END}"


def _stopwords() -> set[str]:
    return {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with", "this", "but", "they",
        "have", "had", "what", "which", "who", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "can", "just", "should", "now", "i", "me", "my",
        "myself", "we", "our", "ours", "you", "your", "yours", "him",
        "his", "she", "her", "hers", "it", "its", "them", "their", "theirs",
    }


def answer_overlaps_context(answer: str, context: str, min_overlap: int = 2) -> bool:
    """Check if key terms from `answer` appear in `context`.

    Returns True if at least `min_overlap` significant words from the answer
    are found in the context (case-insensitive, stopwords removed). This
    provides weak grounding evidence for answers that lack explicit citations
    but are factually traceable to retrieved context.
    """
    if not answer or not context:
        return False
    sw = _stopwords()

    def words(text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()[]{}")
            for w in text.lower().split()
            if w.strip(".,!?;:\"'()[]{}") and w.strip(".,!?;:\"'()[]{}") not in sw
        }

    answer_words = words(answer)
    context_words = words(context)
    if not answer_words:
        return False
    return len(answer_words & context_words) >= min_overlap


def is_grounded(answer: str, context: str) -> bool:
    """Heuristic: answer appears grounded if it cites sources, states
    insufficient evidence, or shares key terms with the retrieved context —
    and does not echo instruction-override patterns."""
    if not answer:
        return False
    low = answer.lower()
    # (a) Insufficiency: model correctly refused due to missing evidence.
    if "i don't have enough information" in low:
        return True
    # (b) Source citation: answer references sources with brackets.
    if "[" in answer and "]" in answer:
        return True
    # (c) Context overlap: answer shares key terms with retrieved context.
    if answer_overlaps_context(answer, context):
        return True
    # (d) Injection echo: reject if answer contains override patterns.
    for rx in _PROMPT_INJECTION_PATTERNS:
        if rx.search(answer):
            return False
    return False


def validate_answer(answer: str, context: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, reason).

    If ok is False, the answer is clearly ungrounded or echoes an injection
    pattern and should be replaced with a safe refusal.

    Weakly grounded answers (share context terms but lack citations) are
    allowed through with ok=True so callers can log a warning instead of
    blocking legitimate responses.
    """
    if not answer:
        return False, "Empty answer."
    low = answer.lower()
    # Reject injection echoes outright.
    for rx in _PROMPT_INJECTION_PATTERNS:
        if rx.search(answer):
            return False, "Answer echoes instruction-override pattern."
    # Allow insufficiency messages.
    if "i don't have enough information" in low:
        return True, None
    # Allow citation-backed answers.
    if "[" in answer and "]" in answer:
        return True, None
    # Allow context-overlapping answers (weak grounding).
    if answer_overlaps_context(answer, context):
        return True, None
    # Otherwise ungrounded.
    return False, "Answer failed grounding validation."
