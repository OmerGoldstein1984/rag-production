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


def is_grounded(answer: str, context: str) -> bool:
    """Heuristic: answer appears grounded if it cites sources or states
    insufficient evidence, and does not contain an instruction-override echo."""
    if not answer:
        return False
    low = answer.lower()
    if "i don't have enough information" in low:
        return True
    if "[" in answer and "]" in answer:
        return True
    for rx in _PROMPT_INJECTION_PATTERNS:
        if rx.search(answer):
            return False
    return bool(context and context.strip())


def validate_answer(answer: str, context: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, reason). If not ok, the answer should be replaced with a
    safe refusal rather than returned to the client."""
    if not is_grounded(answer, context):
        return False, "Answer failed grounding validation."
    return True, None
