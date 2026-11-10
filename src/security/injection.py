"""
src.security.injection - compatibility shim.

Prompt-injection defenses now live in src.security.prompt_guard (the canonical
module: fail-closed detection, input framing, grounding verification). This module
re-exports its public API so any existing `from src.security.injection import ...`
imports keep working without duplicating logic.
"""

from src.security.prompt_guard import (  # noqa: F401
    PromptInjectionBlocked,
    frame_user_input,
    is_grounded,
    is_injection_attempt,
    sanitize_user_input,
    validate_answer,
)

# Legacy aliases kept for backwards compatibility.
delimit_user_input = frame_user_input

__all__ = [
    "PromptInjectionBlocked",
    "sanitize_user_input",
    "frame_user_input",
    "delimit_user_input",
    "is_grounded",
    "validate_answer",
    "is_injection_attempt",
]
