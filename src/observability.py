"""
Lightweight observability: structured JSON logs, request/correlation IDs,
per-stage tracing, and in-memory metrics exposed at /metrics.

Design rules:
- JSON logs only in "structured" mode; secrets are redacted by a logging filter
  (API keys / bearer tokens never reach the logs).
- Every request gets a request_id (correlation id); traces record per-stage
  timings: query_transform, embed, retrieve, rerank, context_build, llm, total.
- Metrics are simple in-memory counters/histograms — no heavy stack.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- correlation id -------------------------------------------------------

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    return request_id_ctx.get()


# --- secret redaction ------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\.\-]{10,}", re.IGNORECASE),
]


def redact(text: Any) -> str:
    """Scrub secret-looking tokens from a string."""
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    for rx in _SECRET_PATTERNS:
        s = rx.sub("[REDACTED]", s)
    return s


class RedactFilter(logging.Filter):
    """Logging filter that redacts secrets from every record.

    Only string args are redacted; non-string args (e.g. numbers for %d)
    are left untouched so %-formatting keeps working.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (redact(v) if isinstance(v, str) else v)
                               for k, v in record.args.items()}
            else:
                record.args = tuple(redact(a) if isinstance(a, str) else a
                                    for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter with request_id."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "msg": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    """Configure root logging: JSON + redaction when structured, else plain."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # remove existing handlers to avoid duplicates on re-init
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.addFilter(RedactFilter())
    handler.setFormatter(JsonFormatter() if structured else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)


# --- tracing ---------------------------------------------------------------

@dataclass
class Span:
    stage: str
    started_ms: float
    duration_ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """One request's end-to-end trace of pipeline stages."""

    request_id: str
    started_at: float = field(default_factory=time.time)
    spans: List[Span] = field(default_factory=list)
    error: Optional[str] = None

    def stage(self, name: str, **detail: Any) -> "_StageCtx":
        return _StageCtx(self, name, detail)

    def add_span(self, name: str, duration_ms: float, detail: Dict[str, Any]) -> None:
        self.spans.append(Span(stage=name, started_ms=0.0, duration_ms=duration_ms, detail=detail))

    @property
    def total_ms(self) -> float:
        return (time.time() - self.started_at) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_ms": round(self.total_ms, 2),
            "error": self.error,
            "stages": [
                {"stage": s.stage, "duration_ms": round(s.duration_ms, 2), **s.detail}
                for s in self.spans
            ],
        }


class _StageCtx:
    """Context manager: `with trace.stage('retrieve', k=5): ...`."""

    def __init__(self, trace: Trace, name: str, detail: Dict[str, Any]):
        self.trace, self.name, self.detail = trace, name, detail
        self.t0 = 0.0

    def __enter__(self) -> "_StageCtx":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dur = (time.perf_counter() - self.t0) * 1000.0
        if exc is not None and self.trace.error is None:
            self.trace.error = f"{self.name}: {exc}"
        self.trace.add_span(self.name, dur, self.detail)


# --- metrics ---------------------------------------------------------------

class Metrics:
    """Thread-safe in-memory metrics: counters, latencies, token usage, errors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = {}
        self.latency_ms: Dict[str, List[float]] = {}
        self.tokens: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self.errors: Dict[str, int] = {}

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def observe_latency(self, name: str, ms: float) -> None:
        with self._lock:
            self.latency_ms.setdefault(name, []).append(ms)

    def add_tokens(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.tokens["prompt"] += prompt
            self.tokens["completion"] += completion
            self.tokens["total"] += prompt + completion

    def inc_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] = self.errors.get(kind, 0) + 1

    @staticmethod
    def _pct(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return s[k]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lat = {
                name: {
                    "count": len(v),
                    "p50_ms": round(self._pct(v, 50), 2),
                    "p95_ms": round(self._pct(v, 95), 2),
                    "mean_ms": round(sum(v) / len(v), 2),
                }
                for name, v in self.latency_ms.items()
            }
            total_requests = self.counters.get("requests_total", 0)
            total_errors = sum(self.errors.values())
            return {
                "counters": dict(self.counters),
                "latency": lat,
                "tokens": dict(self.tokens),
                "errors": dict(self.errors),
                "error_rate": round(total_errors / total_requests, 4) if total_requests else 0.0,
            }


# Global registry
metrics = Metrics()

# recent traces ring buffer (for diagnostics endpoint)
_trace_lock = threading.Lock()
_traces: List[Dict[str, Any]] = []
_TRACE_LIMIT = 100


def record_trace(trace: Trace) -> None:
    with _trace_lock:
        _traces.append(trace.to_dict())
        if len(_traces) > _TRACE_LIMIT:
            del _traces[:-_TRACE_LIMIT]


def get_recent_traces(limit: int = 20) -> List[Dict[str, Any]]:
    with _trace_lock:
        return list(_traces[-limit:])
