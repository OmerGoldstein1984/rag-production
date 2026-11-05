"""
Unit tests for the NVIDIA NIM provider — all HTTP mocked, no network, no key.

Covers M1 acceptance criteria: auth header, payload, usage tracking, latency,
timeout/retry, 429 rate-limit retry, 401 auth error (key never leaked),
streaming, structured-output validation + correction retry, and config
fail-fast when NVIDIA_API_KEY is missing.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from providers.llm_nvidia import (
    NVIDIAProvider,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.config import MissingConfigError, get_settings

TEST_KEY = "nvapi-test-key-12345"


def make_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://x"))


def ok_payload(text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "meta/llama-3.1-70b-instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_provider(handler) -> NVIDIAProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return NVIDIAProvider(api_key=TEST_KEY, client=client, backoff_base_s=0.0, backoff_max_s=0.0)


# --- config / fail-fast ---------------------------------------------------

def test_config_fails_fast_without_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(MissingConfigError):
        get_settings()


def test_config_ok_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-abc")
    s = get_settings()
    assert s.nvidia_api_key == "nvapi-abc"
    assert s.is_real_provider


def test_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(MissingConfigError):
        get_settings()


# --- happy path -----------------------------------------------------------

def test_chat_success_returns_text_usage_latency():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return make_response(ok_payload("Berlin"))

    p = make_provider(handler)
    result = p.chat([{"role": "user", "content": "Where?"}], temperature=0.0)
    assert result.text == "Berlin"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.latency_ms >= 0
    assert result.provider == "nvidia"
    assert seen["auth"] == f"Bearer {TEST_KEY}"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "meta/llama-3.1-70b-instruct"
    assert seen["body"]["stream"] is False
    assert seen["body"]["temperature"] == 0.0


def test_generate_interface_returns_text():
    p = make_provider(lambda req: make_response(ok_payload("hi there")))
    assert p.generate("say hi") == "hi there"


# --- error handling -------------------------------------------------------

def test_401_raises_auth_error_without_leaking_key():
    def handler(request):
        return make_response({"error": {"message": "invalid api key nvapi-test-key-12345"}}, status=401)

    p = make_provider(handler)
    with pytest.raises(ProviderAuthError) as exc_info:
        p.chat([{"role": "user", "content": "hi"}])
    assert TEST_KEY not in str(exc_info.value)


def test_429_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return make_response({"error": "rate limited"}, status=429)
        return make_response(ok_payload("ok"))

    p = make_provider(handler)
    result = p.chat([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert calls["n"] == 3


def test_429_exhausts_retries_raises_rate_limit():
    def handler(request):
        return make_response({"error": "rate limited"}, status=429)

    p = make_provider(handler)
    with pytest.raises(ProviderRateLimitError):
        p.chat([{"role": "user", "content": "hi"}])


def test_500_retries_then_server_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return make_response({"error": "boom"}, status=500)

    p = make_provider(handler)
    with pytest.raises(ProviderServerError):
        p.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 4  # max_retries=3 -> 4 total attempts


def test_timeout_retries_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out", request=request)

    p = make_provider(handler)
    with pytest.raises(ProviderTimeoutError):
        p.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 4


def test_empty_key_rejected():
    with pytest.raises(ProviderAuthError):
        NVIDIAProvider(api_key="  ")


# --- streaming ------------------------------------------------------------

def test_stream_chat_yields_deltas():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    def handler(request):
        body = "\n".join(lines) + "\n"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", "https://x"),
        )

    p = make_provider(handler)
    chunks = list(p.stream_chat([{"role": "user", "content": "hi"}]))
    assert chunks == ["Hel", "lo"]


# --- structured output ----------------------------------------------------

class Answer(BaseModel):
    title: str = Field(min_length=1)
    location: str | None = None


def test_chat_json_validates_schema():
    payload = ok_payload(json.dumps({"title": "Backend Engineer", "location": "Berlin"}))
    p = make_provider(lambda req: make_response(payload))
    obj = p.chat_json([{"role": "user", "content": "extract"}], Answer)
    assert obj.title == "Backend Engineer"
    assert obj.location == "Berlin"


def test_chat_json_retries_on_invalid_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return make_response(ok_payload("not json at all"))
        return make_response(ok_payload(json.dumps({"title": "X"})))

    p = make_provider(handler)
    obj = p.chat_json([{"role": "user", "content": "extract"}], Answer, max_retries=1)
    assert obj.title == "X"
    assert calls["n"] == 2


def test_chat_json_raises_after_exhausting_retries():
    p = make_provider(lambda req: make_response(ok_payload("nope")))
    with pytest.raises(Exception):
        p.chat_json([{"role": "user", "content": "x"}], Answer, max_retries=1)


# --- model info -----------------------------------------------------------

def test_model_info():
    p = make_provider(lambda req: make_response(ok_payload()))
    info = p.get_model_info()
    assert info["provider"] == "nvidia"
    assert info["supports_streaming"] is True
