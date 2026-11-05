"""
Tests for streaming /ask (SSE) and provider/model/usage in the normal response.
"""

import json
import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ask_stream_emits_tokens_then_done():
    os.environ["LLM_PROVIDER"] = "mock"
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    initialize_rag_pipeline("data/raw")
    c = TestClient(app)
    with c.stream("POST", "/ask/stream", json={"query": "How many remote days?", "top_k": 2},
                  headers={"X-Access-Level": "1"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()

    events = []
    for line in body.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    token_events = [e for e in events if e.get("type") == "token"]
    done = [e for e in events if e.get("type") == "done"]
    assert token_events, "expected at least one token event"
    assert done, "expected a final done event"
    assert done[0]["provider"] == "mock"
    assert "sources" in done[0] and isinstance(done[0]["sources"], list)
    assert "latency_ms" in done[0]
    # reassembled answer equals concatenated tokens
    assert "".join(e["text"] for e in token_events) == done[0]["answer"]


def test_ask_response_includes_provider_model_usage():
    os.environ["LLM_PROVIDER"] = "mock"
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    initialize_rag_pipeline("data/raw")
    c = TestClient(app)
    r = c.post("/ask", json={"query": "remote days?", "top_k": 2}, headers={"X-Access-Level": "1"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock"
    assert body["model"]
    # mock has no provider-reported usage -> usage may be None, that's fine
    assert "usage" in body
