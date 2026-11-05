"""
NVIDIA NIM LLM provider (OpenAI-compatible API).

Endpoint: {NVIDIA_BASE_URL}/chat/completions
Auth:     Authorization: Bearer $NVIDIA_API_KEY

Responsibilities: auth, model selection, chat completion, streaming,
structured (JSON) output with Pydantic validation, token-usage collection,
timeout + retry with backoff, rate-limit (429) handling, latency measurement,
and secret redaction (the API key never appears in logs or exceptions).

This module contains ONLY provider-specific code. Business logic (retrieval,
prompts, answer assembly, evaluation) must not import from here directly —
use the LLMProvider interface in providers/llm.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .llm import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Base error for provider failures. Never contains the API key."""


class ProviderAuthError(ProviderError):
    """401/403 — the API key is missing, invalid, or lacks permission."""


class ProviderRateLimitError(ProviderError):
    """429 — rate limited. Retryable with backoff."""


class ProviderServerError(ProviderError):
    """5xx — NVIDIA-side error. Retryable with backoff."""


class ProviderTimeoutError(ProviderError):
    """Request timed out. Retryable."""


@dataclass
class Usage:
    """Token usage reported by the provider (actual, not estimated)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    """One completed chat call, with metadata for observability."""

    text: str
    model: str
    provider: str = "nvidia"
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _redact(text: Any, api_key: Optional[str]) -> str:
    """Convert to str and scrub the API key if present."""
    s = text if isinstance(text, str) else repr(text)
    if api_key:
        s = s.replace(api_key, "[REDACTED]")
    return s


class NVIDIAProvider(LLMProvider):
    """NVIDIA NIM provider over the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        client: Optional[httpx.Client] = None,
        key_pool: Optional[Any] = None,
    ) -> None:
        # key_pool (src.key_pool.KeyPool) takes precedence: multi-key with failover.
        self._key_pool = key_pool
        if key_pool is not None:
            self._api_key = key_pool.current_key()
        else:
            if not api_key or not api_key.strip():
                raise ProviderAuthError("NVIDIA API key is empty. Set NVIDIA_API_KEY.")
            self._api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    # -- internal helpers -------------------------------------------------

    def _redact_fp(self) -> str:
        """Fingerprint of the current key for logs (never the full key)."""
        k = self._api_key
        return f"{k[:4]}...{k[-4:]}" if len(k) >= 12 else "****"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, stream: bool = False) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(self, messages: List[Dict[str, str]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        # Pass through common optional params (temperature, max_tokens, ...)
        for key in ("temperature", "max_tokens", "top_p", "stop", "tools", "tool_choice", "response_format"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        return payload

    def _sleep(self, attempt: int) -> None:
        delay = min(self.backoff_base_s * (2 ** attempt), self.backoff_max_s)
        time.sleep(delay)

    def _classify_error(self, exc: Exception) -> ProviderError:
        """Wrap an httpx error into a typed ProviderError without the key."""
        msg = _redact(str(exc), self._api_key)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(f"NVIDIA request timed out after {self.timeout_s}s: {msg}")
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return ProviderAuthError(f"NVIDIA auth failed (HTTP {status}): {msg}")
            if status == 429:
                return ProviderRateLimitError(f"NVIDIA rate limited (HTTP 429): {msg}")
            if status >= 500:
                return ProviderServerError(f"NVIDIA server error (HTTP {status}): {msg}")
            return ProviderError(f"NVIDIA request failed (HTTP {status}): {msg}")
        return ProviderError(f"NVIDIA request failed: {msg}")

    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        """POST with retries on timeout / 429 / 5xx, plus key-pool failover on 401/403.

        Key rotation does not consume the retry budget: with a pool of N keys we
        may try up to N keys on auth failures, and max_retries on transient errors.
        Raises typed ProviderError.
        """
        last_exc: Optional[Exception] = None
        attempt = 0
        rotations = 0
        max_attempts = self.max_retries + (len(self._key_pool.keys) if self._key_pool else 0)
        while attempt <= max_attempts:
            try:
                resp = self._client.post(self._url(stream=False), headers=self._headers(), json=payload)
                if resp.status_code in (401, 403) and self._key_pool is not None:
                    nxt = self._key_pool.mark_bad(self._api_key, f"HTTP {resp.status_code}")
                    if nxt:
                        rotations += 1
                        logger.warning("key %s failed auth; rotating to next key in pool", self._redact_fp())
                        self._api_key = nxt
                        continue
                    resp.raise_for_status()
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    if attempt < self.max_retries:
                        logger.warning("NVIDIA HTTP %s, retrying (attempt %d/%d)", resp.status_code, attempt + 1, self.max_retries)
                        self._sleep(attempt)
                        attempt += 1
                        continue
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning("NVIDIA timeout, retrying (attempt %d/%d)", attempt + 1, self.max_retries)
                    self._sleep(attempt)
                    attempt += 1
                    continue
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Non-retryable statuses raise immediately
                if status != 429 and status < 500:
                    raise self._classify_error(exc)
                last_exc = exc
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    attempt += 1
                    continue
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    attempt += 1
                    continue
            attempt += 1
        raise self._classify_error(last_exc or RuntimeError("unknown error"))

    @staticmethod
    def _parse_usage(data: Dict[str, Any]) -> Usage:
        u = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0)),
            completion_tokens=int(u.get("completion_tokens", 0)),
            total_tokens=int(u.get("total_tokens", 0)),
        )

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> tuple[str, str]:
        choices = data.get("choices") or []
        if not choices:
            return "", ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or ""), str(choices[0].get("finish_reason") or "")

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Non-streaming chat completion. Returns text + usage + latency."""
        payload = self._payload(messages, stream=False, temperature=temperature, max_tokens=max_tokens, **kwargs)
        t0 = time.perf_counter()
        resp = self._post(payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        text, finish = self._extract_text(data)
        result = ChatResult(
            text=text,
            model=str(data.get("model", self.model)),
            usage=self._parse_usage(data),
            latency_ms=round(latency_ms, 2),
            finish_reason=finish,
            raw=data,
        )
        self.last_result = result  # exposed for usage/observability in callers
        logger.info(
            "nvidia chat ok model=%s latency_ms=%.1f tokens=%d",
            result.model, result.latency_ms, result.usage.total_tokens,
        )
        return result

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """LLMProvider interface: single-prompt completion -> text."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs).text

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Streaming chat completion. Yields text deltas as they arrive."""
        payload = self._payload(messages, stream=True, temperature=temperature, max_tokens=max_tokens, **kwargs)
        t0 = time.perf_counter()
        with self._client.stream(
            "POST", self._url(stream=True), headers=self._headers(), json=payload
        ) as resp:
            if resp.status_code in (401, 403):
                resp.read()
                raise self._classify_error(httpx.HTTPStatusError("auth", request=resp.request, response=resp))
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.read()
                raise self._classify_error(httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp))
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        logger.info("nvidia stream done in %.1fms", (time.perf_counter() - t0) * 1000.0)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        *,
        max_retries: int = 1,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> T:
        """Structured output: ask for JSON, validate against a Pydantic schema.

        Retries once with a correction prompt if validation fails.
        Raises ValidationError (or JSONDecodeError) on final failure.
        """
        sys = {"role": "system", "content": "Respond ONLY with JSON matching the required schema. No prose."}
        last_raw = ""
        for attempt in range(max_retries + 1):
            try:
                result = self.chat([sys, *messages], temperature=temperature, **kwargs)
                last_raw = result.text
                parsed = json.loads(last_raw)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt < max_retries:
                    logger.warning("structured output invalid (attempt %d), retrying with correction", attempt + 1)
                    messages = messages + [
                        {"role": "assistant", "content": last_raw},
                        {"role": "user", "content": f"That was not valid JSON matching the schema: {exc}. Return ONLY corrected JSON."},
                    ]
                    continue
                raise
        raise ValidationError.from_exception_data(schema.__name__, [])  # unreachable

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "nvidia",
            "model_name": self.model,
            "base_url": self.base_url,
            "type": "llm",
            "supports_streaming": True,
            "max_retries": self.max_retries,
            "timeout_s": self.timeout_s,
        }

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:
            pass
