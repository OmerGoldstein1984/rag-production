"""
Lightweight observability: structured JSON logs, request/correlation IDs,
per-stage tracing, and in-memory metrics exposed at /metrics.

BUSINESS PROBLEM SOLVED: A production RAG system needs to answer "what happened
during this request?" -- how long each pipeline stage took, whether it errored,
and which request a log line belongs to -- without adding a heavy monitoring
stack. This file provides JSON-structured logs with secret redaction, per-request
correlation IDs, per-stage timing traces, and in-memory counters/histograms, all
safe to expose over an HTTP /metrics endpoint.

WHERE IT SITS IN THE DATA FLOW: Cross-cutting layer used by the request path.
src/api/main.py calls setup_logging() at startup, new_request_id()/get_request_id()
to tag each request, Trace/Trace.stage() to time pipeline stages
(query_transform, embed, retrieve, rerank, context_build, llm, total), and
Metrics/record_trace() to record counters and traces. Every log record carries the
current request_id so logs, metrics, and traces can be correlated.

WHAT CALLS IT: src/api/main.py (startup + per-request tracing/metrics), plus
tests/test_observability.py. Any module can use get_request_id() or Metrics.

WHAT IT CALLS: Standard library only (json, logging, re, threading, time, uuid,
contextvars, dataclasses, typing). No network, no external monitoring SDK.

WHAT BREAKS IF REMOVED: No structured logs, no correlation IDs, no per-stage
timing, and no /metrics data. The API would still run but operators could not
correlate logs across services, measure stage latency, or see error rates --
critical for debugging production telecom QA issues.

Design rules (preserved):
- JSON logs only in "structured" mode; secrets are redacted by a logging filter
  (API keys / bearer tokens never reach the logs).
- Every request gets a request_id (correlation id); traces record per-stage
  timings: query_transform, embed, retrieve, rerank, context_build, llm, total.
- Metrics are simple in-memory counters/histograms — no heavy stack.

# GAP: Structured logging + correlation IDs. Learners from web/automation often
grep unstructured text files; here every log is JSON with a request_id so logs
from many concurrent requests can be filtered and correlated.
# GAP: Secret redaction at the logging layer. A RedactFilter scrubs API keys and
bearer tokens from every log record so credentials never reach logs, even if a
developer accidentally logs them.
# BUSINESS: Protects uptime and trust -- fast incident diagnosis (correlated
traces/metrics) and compliance (no secrets in logs).
# INTERVIEW: Say it out loud -- "observability gives us JSON logs with
request IDs, per-stage timing traces, and in-memory metrics, with secrets
redacted at the logging filter so nothing secret reaches the logs."
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

from src.security.redaction import redact_dict, redact_text as redact_pii

# --- correlation id -------------------------------------------------------

# ContextVar holding the current request (correlation) ID for the lifetime of a
# request. ContextVars work across async/await and threads so every log line and
# trace span in one request shares the same ID; default "-" when no request is
# in flight (e.g. at startup).
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """Generate and set the current request (correlation) ID.

    WHAT: Creates a 12-hex-char request ID, stores it in request_id_ctx, and
    returns it.
    WHY: Every request needs a stable correlation ID so its logs, trace spans,
    and metrics can be tied together across the distributed pipeline.
    INPUTS: None.
    OUTPUTS: str -- a 12-char hex ID, e.g. "a3f7c2e9b1d4".
    SIDE EFFECTS: Sets request_id_ctx for the current context (thread/async
    task).
    FAILURE MODES: None -- uuid4 is effectively failure-free.
    ALTERNATIVE: Pass request IDs explicitly through every function signature --
    rejected because ContextVar avoids threading the ID through every call while
    still making it available to logging and tracing.
    # GAP: Correlation IDs / ContextVar. Learners from web backgrounds often
    thread IDs through every function; a ContextVar lets any code in the request
    context read the ID without changing signatures.
    # BUSINESS: Protects debuggability -- correlating logs across services is
    impossible without a shared request ID.
    # INTERVIEW: Say it out loud -- "new_request_id creates a short ID and stores
    it in a ContextVar so every log/trace in the request shares it."
    """
    rid = uuid.uuid4().hex[:12]   # short, unique-enough ID for a single request
    request_id_ctx.set(rid)        # publish it to the current async/thread context
    return rid


def get_request_id() -> str:
    """Return the current request (correlation) ID.

    WHAT: Reads request_id_ctx, returning "-" if no request ID has been set.
    WHY: Lets loggers and trace code attach the request ID to every record.
    INPUTS: None.
    OUTPUTS: str -- the current request ID, or "-".
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: Thread a request_id parameter through every call -- rejected
    (ContextVar is less invasive).
    """
    return request_id_ctx.get()


# --- secret redaction ------------------------------------------------------

# Regexes matching secret-like tokens so they can be scrubbed from logs:
# NVIDIA keys (nvapi-...), OpenAI-style keys (sk-...), and Bearer tokens.
# Compiled once at import for efficiency.
_SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\.\-]{10,}", re.IGNORECASE),
]


def redact(text: Any) -> str:
    """Scrub secret-looking tokens from a string.

    WHAT: Converts `text` to a string (JSON-serializing non-strings), then
    replaces every secret-pattern match with "[REDACTED]".
    WHY: Prevents API keys and bearer tokens from ever being written to logs,
    even if a developer accidentally logs a response or header containing one.
    INPUTS: text (Any) -- a string or any JSON-serializable object.
    OUTPUTS: str with secrets replaced by "[REDACTED]".
    SIDE EFFECTS: None -- pure function.
    FAILURE MODES: None for normal input; json.dumps with default=str handles
    non-serializable objects gracefully.
    ALTERNATIVE: Redact at each log call site -- rejected because a single
    logging filter (RedactFilter) guarantees redaction everywhere, including
    third-party library logs.
    # GAP: Secret redaction. Learners may think "I won't log the key"; but keys
    hide inside error responses, headers, and exception text, so a centralized
    regex-based redaction layer is the reliable defense.
    # BUSINESS: Protects credentials and compliance -- no secrets reach logs,
    avoiding key theft and audit failures.
    # INTERVIEW: Say it out loud -- "redact() scrubs NVIDIA keys, sk- keys, and
    Bearer tokens from any value before it's logged, so secrets never reach log
    output."
    """
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    for rx in _SECRET_PATTERNS:
        s = rx.sub("[REDACTED]", s)
    # Also scrub PII (emails, phones, IMSI/IMEI, customer IDs, IP addresses).
    s = redact_pii(s)
    return s


class RedactFilter(logging.Filter):
    """Logging filter that redacts secrets from every log record.

    WHAT: A logging.Filter that runs redact() on each record's message and
    arguments before the record is formatted and emitted.
    WHY: Guarantees secrets are scrubbed from ALL log output (including logs from
    third-party libraries) at one central place, so no API key or bearer token
    ever reaches a log file or stdout.
    INPUTS: record (logging.LogRecord) -- mutated in place (msg and args
    redacted).
    OUTPUTS: bool -- always True (the record is kept; we only transform it).
    SIDE EFFECTS: Mutates record.msg and record.args in place.
    FAILURE MODES: None for normal records; redact() is failure-tolerant.
    ALTERNATIVE: Call redact() at every logger.info/debug site -- rejected
    because it's easy to forget a site; a filter covers everything uniformly.
    # GAP: Logging filters. Learners often only know logger.setLevel; a Filter
    lets us transform every record centrally, which is the right place for
    cross-cutting concerns like redaction.
    # BUSINESS: Protects credentials and compliance -- a single missed log line
    containing a key is a security incident.
    # INTERVIEW: Say it out loud -- "RedactFilter runs redact() on every log
    record's message and args, always returns True (keeps the record), and never
    lets a full key reach the logs."
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the main message.
        record.msg = redact(record.msg)
        # Redact any string arguments (positional or keyword), leaving numbers
        # and other types untouched so %-formatting keeps working.
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (redact(v) if isinstance(v, str) else v)
                               for k, v in record.args.items()}
            else:
                record.args = tuple(redact(a) if isinstance(a, str) else a
                                    for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter with request_id.

    WHAT: Formats each log record as a single JSON object with timestamp, level,
    logger name, request_id, the (redacted) message, and optional exception info.
    WHY: JSON logs are machine-parseable, so log aggregation tools can index,
    filter, and correlate them by request_id across services.
    INPUTS: record (logging.LogRecord).
    OUTPUTS: str -- one JSON line per record.
    SIDE EFFECTS: Calls get_request_id() and redact() (no mutation of the record).
    FAILURE MODES: None for normal records; json.dumps is given ensure_ascii=False
    so non-ASCII logs survive intact.
    ALTERNATIVE: Send logs to an external APM vendor -- rejected because in-process
    JSON logging has zero external dependency and is trivial to ship to any log
    collector.
    # GAP: Structured logging. Learners used to printf-style logs find JSON logs
    harder to read by hand but far easier to query and correlate in production.
    # BUSINESS: Protects debuggability and compliance -- machine-parseable logs
    with redacted messages are fast to search and safe to retain.
    # INTERVIEW: Say it out loud -- "JsonFormatter emits one JSON object per log
    line with ts, level, logger, request_id, and a redacted msg."
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "msg": redact(record.getMessage()),   # scrub secrets before emitting
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    """Configure root logging: JSON + redaction when structured, else plain.

    WHAT: Sets the root logger level, removes any existing handlers (to avoid
    duplicates on re-init), adds a StreamHandler with RedactFilter, and uses
    JsonFormatter (structured) or a plain formatter otherwise.
    WHY: Centralizes logging configuration so the whole application emits
    consistent, redacted logs; called once at API startup by src/api/main.py.
    INPUTS: level (str, default "INFO"), structured (bool, default True).
    OUTPUTS: None.
    SIDE EFFECTs: Mutates the root logger (level, handlers) globally.
    FAILURE MODES: getattr(logging, level.upper(), logging.INFO) falls back to
    INFO for an invalid level name rather than crashing.
    ALTERNATIVE: Use logging.basicConfig() -- rejected because we need to remove
    existing handlers on re-init and add a custom RedactFilter, which
    basicConfig() doesn't support.
    # GAP: Handler deduplication. Removing existing handlers before adding a new
    one prevents duplicate log lines when setup_logging() is called more than
    once (e.g. in tests or reloads).
    # BUSINESS: Protects compliance and debuggability -- consistent, redacted,
    non-duplicated logs across the application.
    # INTERVIEW: Say it out loud -- "setup_logging configures the root logger
    with JSON formatting and a RedactFilter, removing old handlers to avoid
    duplicate logs."
    """
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
    """One timed pipeline stage within a trace.

    WHAT: Records a stage name, start time, duration in ms, and optional detail
    (e.g. k=5 for retrieve, model name for llm).
    WHY: Breaking a request into per-stage spans lets operators see which part of
    the RAG pipeline is slow (retrieve vs llm vs embed) rather than only total
    latency.
    INPUTS: stage (str), started_ms (float), duration_ms (float, default 0.0),
    detail (Dict[str, Any]).
    OUTPUTS: None (data holder).
    SIDE EFFECTS: None.
    FAILURE MODES: None -- plain data.
    ALTERNATIVE: Use a third-party APM trace object -- rejected because in-memory
    spans have zero external dependency and are simple to serialize.
    # GAP: Distributed-tracing lite. A Span is the smallest unit of a trace;
    learners should understand that timing each stage separately is how you find
    bottlenecks in a multi-step pipeline.
    # BUSINESS: Protects latency insight -- per-stage timing shows which component
    to optimize instead of guessing.
    # INTERVIEW: Say it out loud -- "a Span records one pipeline stage's name,
    duration, and detail so we can see which stage is slow."
    """

    stage: str
    started_ms: float
    duration_ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """One request's end-to-end trace of pipeline stages.

    WHAT: Holds a request_id, a start timestamp, a list of Span objects (one per
    pipeline stage), and an optional error string.
    WHY: A Trace is the complete observability record for a single request --
    stages, total duration, and any error -- serializable to the /metrics or
    /traces endpoint.
    INPUTS: request_id (str), started_at (float, default time.time()), spans
    (List[Span]), error (Optional[str]).
    OUTPUTS: None (data holder); to_dict() serializes to a JSON-friendly dict.
    SIDE EFFECTS: stage() returns a context manager that mutates the trace on
    exit; add_span() appends a span.
    FAILURE MODES: None -- plain data; errors are recorded, not raised.
    ALTERNATIVE: Export to OpenTelemetry/Jaeger -- rejected because an in-memory
    trace is sufficient for a lightweight system and needs no collector.
    # GAP: End-to-end tracing. A Trace groups spans by request_id so operators
    can replay exactly what happened during one request and where it failed.
    # BUSINESS: Protects uptime -- knowing which stage failed and how long each
    took speeds incident resolution.
    # INTERVIEW: Say it out loud -- "a Trace holds one request's request_id,
    start time, per-stage spans, and error; to_dict() serializes it for the
    diagnostics endpoint."
    """

    request_id: str
    started_at: float = field(default_factory=time.time)
    spans: List[Span] = field(default_factory=list)
    error: Optional[str] = None

    def stage(self, name: str, **detail: Any) -> "_StageCtx":
        """Begin a timed stage: `with trace.stage('retrieve', k=5): ...`."""
        return _StageCtx(self, name, detail)

    def add_span(self, name: str, duration_ms: float, detail: Dict[str, Any]) -> None:
        """Append a completed span to the trace (called by _StageCtx on exit)."""
        self.spans.append(Span(stage=name, started_ms=0.0, duration_ms=duration_ms, detail=detail))

    @property
    def total_ms(self) -> float:
        """Wall-clock milliseconds elapsed since the trace started."""
        return (time.time() - self.started_at) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trace to a JSON-friendly dict."""
        return {
            "request_id": self.request_id,
            "total_ms": round(self.total_ms, 2),
            "error": self.error,
            "stages": [
                {"stage": s.stage, "duration_ms": round(s.duration_ms, 2), **redact_dict(s.detail)}
                for s in self.spans
            ],
        }


class _StageCtx:
    """Context manager: `with trace.stage('retrieve', k=5): ...`."""

    def __init__(self, trace: Trace, name: str, detail: Dict[str, Any]):
        self.trace, self.name, self.detail = trace, name, detail
        self.t0 = 0.0

    def __enter__(self) -> "_StageCtx":
        self.t0 = time.perf_counter()   # high-resolution start time
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dur = (time.perf_counter() - self.t0) * 1000.0  # stage duration in ms
        # Record the first error encountered so later stages don't overwrite it.
        if exc is not None and self.trace.error is None:
            self.trace.error = f"{self.name}: {exc}"
        self.trace.add_span(self.name, dur, self.detail)


# --- metrics ---------------------------------------------------------------

class Metrics:
    """Thread-safe in-memory metrics: counters, latencies, token usage, errors.

    WHAT: Maintains counters (e.g. requests_total), per-name latency samples,
    token usage (prompt/completion/total), and error counts by kind, all guarded
    by a lock for thread safety.
    WHY: The /metrics endpoint exposes these so operators can see request volume,
    latency percentiles, token consumption, and error rate without an external
    monitoring stack.
    INPUTS: None (constructs empty structures).
    OUTPUTS: None (stateful object); inc/observe_latency/add_tokens/inc_error
    mutate state; snapshot() returns a JSON-friendly dict.
    SIDE EFFECTS: All mutating methods take self._lock.
    FAILURE MODES: None for normal input; snapshot divides by total_requests and
    guards the zero case with error_rate=0.0.
    ALTERNATIVE: Push metrics to Prometheus/Pushgateway -- rejected because an
    in-memory store has no external dependency and is easy to expose over HTTP.
    # GAP: In-memory metrics vs a metrics backend. Learners may expect Prometheus;
    here we keep counters/histograms in memory and expose them via an endpoint,
    which is fine for a single-instance service but not multi-instance.
    # GAP: Percentile computation. _pct() computes p50/p95/mean from a list of
    samples -- learners should understand percentiles (not just averages) for
    latency SLOs.
    # BUSINESS: Protects latency insight and cost visibility -- token counts show
    LLM spend; error_rate shows reliability.
    # INTERVIEW: Say it out loud -- "Metrics is a thread-safe in-memory store of
    counters, latency samples, token totals, and error counts, exposed via
    snapshot()."
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()                       # guards all mutable state
        self.counters: Dict[str, int] = {}                  # named integer counters
        self.latency_ms: Dict[str, List[float]] = {}        # latency samples per stage
        self.tokens: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self.errors: Dict[str, int] = {}                    # error counts by kind

    def inc(self, name: str, amount: int = 1) -> None:
        """Increment a named counter by amount (default 1)."""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def observe_latency(self, name: str, ms: float) -> None:
        """Record a latency sample (ms) under a named stage."""
        with self._lock:
            self.latency_ms.setdefault(name, []).append(ms)

    def add_tokens(self, prompt: int, completion: int) -> None:
        """Add prompt and completion token counts, updating the running total."""
        with self._lock:
            self.tokens["prompt"] += prompt
            self.tokens["completion"] += completion
            self.tokens["total"] += prompt + completion

    def inc_error(self, kind: str) -> None:
        """Increment the error count for a given error kind."""
        with self._lock:
            self.errors[kind] = self.errors.get(kind, 0) + 1

    @staticmethod
    def _pct(values: List[float], p: float) -> float:
        """Return the p-th percentile of a sorted list of values (0 if empty)."""
        if not values:
            return 0.0
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return s[k]

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary: counters, latency percentiles, tokens,
        errors, and error_rate."""
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


# Global, process-wide metrics singleton (imported by src/api/main.py).
metrics = Metrics()

# recent traces ring buffer (for diagnostics endpoint) -- keeps the last N traces
# so operators can inspect recent requests without unbounded memory growth.
_trace_lock = threading.Lock()
_traces: List[Dict[str, Any]] = []
_TRACE_LIMIT = 100


def record_trace(trace: Trace) -> None:
    """Append a completed trace to the in-memory ring buffer (bounded to _TRACE_LIMIT)."""
    with _trace_lock:
        _traces.append(trace.to_dict())
        if len(_traces) > _TRACE_LIMIT:
            del _traces[:-_TRACE_LIMIT]   # drop oldest traces to keep memory bounded


def get_recent_traces(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent `limit` traces as dicts (newest last)."""
    with _trace_lock:
        return list(_traces[-limit:])


# --- persistent observability --------------------------------------------------

# Persistent storage paths (loaded lazily from settings).
_metrics_path: str = "data/metrics.json"
_traces_path: str = "data/traces.jsonl"


def _configure_storage_paths() -> None:
    """Read storage paths from settings if available."""
    global _metrics_path, _traces_path
    try:
        settings = get_settings()
        _metrics_path = settings.metrics_storage_path
        _traces_path = settings.traces_storage_path
    except Exception:
        pass  # settings not configured yet; use defaults


def flush_metrics() -> None:
    """Persist current metrics snapshot to disk as JSON."""
    _configure_storage_paths()
    data = {
        "counters": dict(metrics.counters),
        "latency": {name: list(v) for name, v in metrics.latency_ms.items()},
        "tokens": dict(metrics.tokens),
        "errors": dict(metrics.errors),
    }
    try:
        with open(_metrics_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # best-effort persistence


def load_persistent_metrics() -> None:
    """Load metrics from disk and merge into current metrics."""
    _configure_storage_paths()
    try:
        with open(_metrics_path, "r") as f:
            data = json.load(f)
        with metrics._lock:
            metrics.counters.update(data.get("counters", {}))
            for name, values in data.get("latency", {}).items():
                metrics.latency_ms.setdefault(name, []).extend(values)
            metrics.tokens.update(data.get("tokens", {}))
            metrics.errors.update(data.get("errors", {}))
    except (OSError, json.JSONDecodeError):
        pass  # no existing metrics file; start fresh


def persistent_record_trace(trace: Trace) -> None:
    """Append a completed trace to the persistent JSONL file."""
    _configure_storage_paths()
    try:
        with open(_traces_path, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")
    except OSError:
        pass  # best-effort persistence


def get_persistent_traces(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent `limit` traces from the persistent JSONL file."""
    _configure_storage_paths()
    try:
        with open(_traces_path, "r") as f:
            lines = f.readlines()
        traces = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
        return traces
    except (OSError, json.JSONDecodeError):
        return []



# --- security audit logging --------------------------------------------------

@dataclass
class AuditEvent:
    """One security-relevant event: who did what, when, where, with what outcome."""

    timestamp: float
    actor: str            # client_id / key_id / role
    action: str           # e.g. "auth_failure", "rate_limit_block", "ingest", "delete"
    endpoint: str         # e.g. "/ask", "/ingest"
    detail: Dict[str, Any]  # redacted context (query hash, tool, etc.)
    outcome: str          # "allowed", "blocked", "error"


class AuditLogger:
    """Bounded ring buffer of security audit events."""

    def __init__(self, limit: int = 1000):
        self.limit = limit
        self.events: List[AuditEvent] = []
        self._lock = threading.Lock()

    def log(self, actor: str, action: str, endpoint: str,
            detail: Optional[Dict[str, Any]] = None, outcome: str = "allowed") -> None:
        event = AuditEvent(
            timestamp=time.time(),
            actor=actor,
            action=action,
            endpoint=endpoint,
            detail=redact_dict(detail or {}),
            outcome=outcome,
        )
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.limit:
                del self.events[: len(self.events) - self.limit]

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "timestamp": round(e.timestamp, 3),
                    "actor": e.actor,
                    "action": e.action,
                    "endpoint": e.endpoint,
                    "detail": e.detail,
                    "outcome": e.outcome,
                }
                for e in self.events
            ]


# Global audit logger (imported by src/api/main.py).
audit_logger = AuditLogger()


def log_security_event(actor: str, action: str, endpoint: str,
                       detail: Optional[Dict[str, Any]] = None,
                       outcome: str = "allowed") -> None:
    """Record a security-relevant event (auth failure, rate-limit block, ingest,
    delete, prompt-injection block, PII redaction)."""
    audit_logger.log(actor, action, endpoint, detail, outcome)


def get_recent_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent `limit` audit events."""
    return audit_logger.snapshot()[-limit:]
