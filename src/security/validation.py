"""
src.security.validation - input validation, request limits, and upload allowlists.

THREAT: Unbounded inputs (giant request bodies, huge top_k/chunk_size values,
traversal filenames, disallowed upload extensions) enable denial-of-service and
path-traversal attacks.

CONTROL:
- Bounded Pydantic field constraints.
- MAX_REQUEST_BODY_SIZE middleware.
- Upload extension + size allowlist.
- Filename validation (no traversal, no double extensions, no executables).
"""

from __future__ import annotations

import os
import re
from typing import Optional

MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}

# Dangerous filename patterns: path traversal, double extensions, executables.
_TRAVERSAL_PATTERN = re.compile(r"(\.\.|/|\\)")
_DOUBLE_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,4}\.(exe|sh|bat|cmd|ps1|py|pl|rb|php|jsp|asp|aspx|cgi|com)$", re.IGNORECASE)
_EXECUTABLE_EXTENSIONS = {".exe", ".sh", ".bat", ".cmd", ".ps1", ".py", ".pl", ".rb", ".php", ".jsp", ".asp", ".aspx", ".cgi", ".com"}


def bounded_int(value: int, lo: int, hi: int, name: str) -> int:
    """Clamp/validate an integer into [lo, hi]; raise ValueError if out of range."""
    if value is None:
        return lo
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if v < lo or v > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def validate_query_length(query: str, max_chars: int = 2000) -> str:
    """Reject queries longer than max_chars."""
    if query is None:
        return ""
    if len(query) > max_chars:
        raise ValueError(f"query exceeds maximum length of {max_chars} characters")
    return query


def validate_filename(filename: str) -> str:
    """Validate an upload filename: no traversal, no double extension, no executable."""
    if not filename:
        raise ValueError("filename is required")
    base = os.path.basename(filename)  # strip any directory components
    if base != filename:
        raise ValueError("filename must not contain path components")
    if _TRAVERSAL_PATTERN.search(base):
        raise ValueError("filename must not contain path traversal sequences")
    lower = base.lower()
    if _DOUBLE_EXTENSION.search(lower):
        raise ValueError("filename with double extension is not allowed")
    ext = os.path.splitext(base)[1].lower()
    if ext in _EXECUTABLE_EXTENSIONS:
        raise ValueError(f"extension {ext} is not allowed")
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"extension {ext} is not allowed")
    return base


def request_body_size_middleware_max() -> int:
    return MAX_REQUEST_BODY_SIZE
