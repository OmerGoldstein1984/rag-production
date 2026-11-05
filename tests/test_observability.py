"""
Tests for observability: secret redaction, structured JSON logs, tracing,
metrics, and the /metrics, /traces, /provider endpoints.
"""

import json
import logging
import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.observability import (
    Metrics,
    Trace,
    get_recent_traces,
    metrics,
    new_request_id,
    redact,
    record_trace,
    setup_logging,
    JsonFormatter,
)


# --- redaction ------------------------------------------------------------

def test_redact_scrubs_api_keys_and_bearer():
    s = "error with nvapi-abcdef1234567890 and Bearer abcdef1234567890XYZ"
    out = redact(s)
    assert "nvapi-abcdef1234567890" not in out
    assert "Bearer abcdef" not in out
    assert "[REDACTED]" in out


def test_json_formatter_emits_json_with_request_id():
    new_request_id()
    rec = logging.LogRecord("test.obs", logging.INFO, __file__, 10,
                            "hello nvapi-secretkey9999999 %d", (5,), None)
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "hello [REDACTED] 5"  # arg formatting still works
    assert "request_id" in payload and payload["request_id"] != "-"
    assert payload["level"] == "INFO"


# --- tracing --------------------------------------------------------------

def test_trace_stage_records_duration():
    t = Trace(request_id="abc")
    with t.stage("retrieve", k=5):
        pass
    d = t.to_dict()
    assert d["stages"][0]["stage"] == "retrieve"
    assert d["stages"][0]["k"] == 5
    assert d["stages"][0]["duration_ms"] >= 0
    assert "total_ms" in d


def test_trace_records_error():
    t = Trace(request_id="abc")
    with pytest.raises(ValueError):
        with t.stage("llm"):
            raise ValueError("boom")
    assert "llm" in (t.error or "")


def test_trace_ring_buffer():
    for i in range(5):
        record_trace(Trace(request_id=f"r{i}"))
    traces = get_recent_traces(limit=3)
    assert len(traces) == 3
    assert traces[-1]["request_id"] == "r4"


# --- metrics --------------------------------------------------------------

def test_metrics_counters_latency_tokens_errors():
    m = Metrics()
    m.inc("requests_total", 2)
    m.observe_latency("request_ms", 10)
    m.observe_latency("request_ms", 20)
    m.observe_latency("request_ms", 30)
    m.add_tokens(prompt=100, completion=50)
    m.inc_error("http_500")
    snap = m.snapshot()
    assert snap["counters"]["requests_total"] == 2
    assert snap["latency"]["request_ms"]["count"] == 3
    assert snap["latency"]["request_ms"]["p50_ms"] == 20
    assert snap["tokens"]["total"] == 150
    assert snap["errors"]["http_500"] == 1
    assert snap["error_rate"] == 0.5


# --- endpoints ------------------------------------------------------------

def test_metrics_traces_provider_endpoints():
    os.environ["LLM_PROVIDER"] = "mock"
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    initialize_rag_pipeline("data/raw")
    c = TestClient(app)

    r = c.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "counters" in body and "latency" in body and "tokens" in body

    r = c.get("/traces?limit=5")
    assert r.status_code == 200
    assert "traces" in r.json()

    r = c.get("/provider")
    assert r.status_code == 200
    p = r.json()
    assert p["provider"] == "mock"
    assert p["is_real"] is False  # mock must be visible, not hidden

    # /ask should add a trace and metrics
    r = c.post("/ask", json={"query": "remote days?", "top_k": 2}, headers={"X-Access-Level": "1"})
    assert r.status_code == 200
    assert "X-Request-Id" in r.headers
    m = c.get("/metrics").json()
    assert m["counters"].get("answers_generated", 0) >= 1
    t = c.get("/traces?limit=10").json()["traces"]
    all_stages = {s["stage"] for tr in t for s in tr["stages"]}
    assert "retrieve" in all_stages and "llm" in all_stages


def test_no_secret_in_metrics_or_traces(caplog):
    os.environ["LLM_PROVIDER"] = "mock"
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    initialize_rag_pipeline("data/raw")
    c = TestClient(app)
    c.post("/ask", json={"query": "x", "top_k": 1}, headers={"X-Access-Level": "1"})
    blob = c.get("/metrics").text + c.get("/traces").text
    assert "nvapi-" not in blob
