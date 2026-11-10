"""
src.security.redaction - PII and secret redaction.

THREAT: Telecom documents and user queries may contain personally identifying
information (emails, phone numbers, IMSI/IMEI, customer IDs, MAC addresses) and
secrets (API keys). These must not leak into LLM answers, logs, traces, or API
responses.

CONTROL:
- PII_REDACTION_PATTERNS: regexes for common telecom/PII identifiers.
- redact_pii(text): replace PII with [REDACTED:<type]> placeholders.
- redact_dict(d): recursively redact PII in dict values (for traces/metrics/logs).
- redact_query(text): redact PII from user queries before they reach the LLM/logs.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# PII / secret patterns. Ordered so specific telecom identifiers run BEFORE the
# greedy phone pattern, otherwise phone swallows IMSI/IMEI/MSISDN digits first.
# Each entry: (name, compiled regex).
_PII_PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("imsi", re.compile(r"\bIMSI\s*[:=]?\s*\d{5,15}\b", re.IGNORECASE)),
    ("imei", re.compile(r"\bIMEI\s*[:=]?\s*\d{10,20}\b", re.IGNORECASE)),
    ("msisdn", re.compile(r"\bMSISDN\s*[:=]?\s*\d{5,15}\b", re.IGNORECASE)),
    ("customer_id", re.compile(r"\bcustomer[-_]?id\s*[:=]?\s*[A-Za-z0-9-]{6,}\b", re.IGNORECASE)),
    ("mac", re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")),
    # phone: (?<!\d) avoids matching digits already consumed by an identifier pattern.
    ("phone", re.compile(r"(?<!\d)\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b")),
    ("api_key", re.compile(r"\b(nvapi|sk|pk|ak)-[A-Za-z0-9_\-]{16,}\b")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9_\.\-]{20,}\b", re.IGNORECASE)),
]


def redact_pii(text: str) -> str:
    """Replace PII and secret-looking tokens in `text` with [REDACTED:<type]>."""
    if not isinstance(text, str):
        return text
    result = text
    for name, pattern in _PII_PATTERNS:
        result = pattern.sub(f"[REDACTED:{name.upper()}]", result)
    return result


# Backwards-compatible alias used by the observability logging layer.
redact_text = redact_pii


def redact_value(value: Any) -> Any:
    """Redact a single value (string or non-string)."""
    if isinstance(value, str):
        return redact_pii(value)
    return value


def redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact PII in all string values of a dict (for traces/logs)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = redact_pii(v)
        elif isinstance(v, dict):
            out[k] = redact_dict(v)
        elif isinstance(v, list):
            out[k] = [redact_pii(i) if isinstance(i, str) else i for i in v]
        else:
            out[k] = v
    return out


def redact_query(text: str) -> str:
    """Redact PII from a user query before logging or sending to the LLM."""
    return redact_pii(text)
