"""
Structured security audit logging.

Emits one JSON audit event per security-relevant action (auth success/failure,
rate-limit blocks, prompt-injection blocks, PII redactions, document ingests /
deletes, tool calls) with request_id, client/key identity, event, outcome, and a
redacted detail. Intended to be shipped to a log aggregator / SIEM in production.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from src.observability import get_request_id, redact


class AuditLogger:
    """Thread-safe structured audit logger.

    WHAT: Records security events as JSON lines with correlation context.
    WHY: Gives operators an immutable trail of who did what, which is essential
    for investigating abuse (injection attempts, authz bypasses, bulk ingestion).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._logger = logging.getLogger("rag.audit")

    def log_event(
        self,
        event: str,
        *,
        outcome: str = "ok",
        key_id: Optional[str] = None,
        client_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.time(),
            "request_id": request_id or get_request_id(),
            "event": event,
            "outcome": outcome,
            "key_id": key_id,
            "client_id": client_id,
            "detail": redact(value) if (detail := detail) else {},
        }
        # Emit as a single JSON log line so it is captured by the structured
        # logging pipeline (which itself redacts secrets).
        with self._lock:
            self._logger.info(json.dumps(record, default=str))


# Singleton instance; enabled/disabled via config.
audit_logger = AuditLogger()
