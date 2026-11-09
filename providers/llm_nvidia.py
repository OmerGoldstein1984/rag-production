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

BUSINESS PROBLEM: This file is the production LLM backend. It talks to NVIDIA
NIM (an OpenAI-compatible chat-completions API) to generate answers from
retrieved telecom context, and it is hardened for production: retries with
exponential backoff on timeouts/429/5xx, multi-key failover (key pool) on
401/403, token-usage tracking for cost observability, per-call latency timing,
and API-key redaction so secrets never leak into logs or error messages.

PLACE IN DATA FLOW: Called via the LLMProvider interface from
src/generation/answer.py (AnswerGenerator) and src/agent/agent.py (LLM planner).
It is the leaf node of the generation stack — retrieval hands it context+prompt,
it returns text + usage + latency. Upstream (retrieval/prompts) never imports
this module directly; they go through providers/llm.py.

WHAT CALLS IT: providers/llm.py (create_llm_provider, lazy import),
src/generation/answer.py (via the LLMProvider interface: chat, generate,
stream_chat, chat_json, get_model_info, last_result),
src/agent/agent.py (provider.generate for the LLM planner).

WHAT IT CALLS: httpx (sync Client + .stream for SSE), pydantic (BaseModel,
ValidationError for structured output), providers.llm (LLMProvider base class),
standard library (json, logging, time).

WHAT BREAKS IF REMOVED: Production LLM generation stops entirely. With
LLM_PROVIDER=nvidia (the production default path) there is no fallback provider
in this module — answers cannot be generated without it.

# GAP: Uses a blocking httpx.Client and time.sleep for backoff — synchronous I/O
inside what may be an async caller; no async client, no circuit breaker, no
jitter on backoff (thundering-herd risk on 429).
# BUSINESS: Protects against NVIDIA NIM outages and rate limits (retries,
backoff, key rotation) and prevents secret leakage (redaction), which is a
compliance requirement for any production LLM integration.
# INTERVIEW: Say it out loud — "We never log the API key; we redact it. On 429 we
back off exponentially and retry; on 401/403 with a key pool we rotate to the
next key without burning the retry budget."
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
    """
    Base error for provider failures. Never contains the API key.

    WHAT: Base exception type for all NVIDIA provider failures.
    WHY: Callers catch ProviderError (or its subclasses) to distinguish provider
    problems from bugs; the message is redacted so the API key never escapes.
    INPUTS: A message string passed to RuntimeError.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: None beyond being raised.
    ALTERNATIVE: Use a plain Exception hierarchy without a base marker class.
    """


class ProviderAuthError(ProviderError):
    """
    401/403 — the API key is missing, invalid, or lacks permission.

    WHAT: Raised when NVIDIA rejects the API key (authentication/authorization).
    WHY: Auth failures are non-retryable by backoff alone; with a key pool they
    trigger key rotation instead.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Raised by _classify_error for HTTP 401/403 and by _post when
    key rotation is exhausted.
    ALTERNATIVE: Treat auth failures as retryable (incorrect — retrying with a
    bad key just wastes time and quota).
    """


class ProviderRateLimitError(ProviderError):
    """
    429 — rate limited. Retryable with backoff.

    WHAT: Raised when NVIDIA returns HTTP 429 (too many requests).
    WHY: Signals the caller should back off and retry rather than fail fast.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Raised by _classify_error/_post for HTTP 429; _post retries
    up to max_retries with exponential backoff before re-raising.
    ALTERNATIVE: Treat 429 as a hard failure (loses resilience under load).
    """


class ProviderServerError(ProviderError):
    """
    5xx — NVIDIA-side error. Retryable with backoff.

    WHAT: Raised when NVIDIA returns HTTP 500+ (server error).
    WHY: Server errors are transient; retrying with backoff often succeeds.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Raised for HTTP >= 500; retried up to max_retries.
    ALTERNATIVE: Fail fast on 5xx (sacrifices resilience for speed).
    """


class ProviderTimeoutError(ProviderError):
    """
    Request timed out. Retryable.

    WHAT: Raised when an httpx request exceeds timeout_s.
    WHY: Timeouts are transient; retrying after a backoff often completes.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Raised by _classify_error for httpx.TimeoutException; retried
    up to max_retries.
    ALTERNATIVE: Increase timeout instead of retrying (costs latency per call).
    """


@dataclass
class Usage:
    """
    Token usage reported by the provider (actual, not estimated).

    WHAT: Holds prompt_tokens, completion_tokens, total_tokens as reported by
    NVIDIA NIM.
    WHY: Actual usage lets us bill/cost-track accurately instead of guessing
    with chars-per-token estimation (src/generation/answer.py:57).
    INPUTS (constructor): prompt_tokens: int (default 0), completion_tokens: int
        (default 0), total_tokens: int (default 0).
    OUTPUTS: None (data holder).
    SIDE EFFECTS: None.
    FAILURE MODES: int() conversion in _parse_usage (providers/llm_nvidia.py:231)
    could raise ValueError if usage values are non-numeric strings.
    ALTERNATIVE: Estimate tokens from character count (less accurate).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    """
    One completed chat call, with metadata for observability.

    WHAT: Bundles the generated text, model name, usage, latency, finish reason,
    and the raw response for a single non-streaming chat call.
    WHY: Callers (answer generator) need usage + latency for observability and
    cost tracking, not just the raw text.
    INPUTS (constructor): text: str (e.g. "5G uses OFDM..."), model: str,
        provider: str (default "nvidia"), usage: Usage, latency_ms: float,
        finish_reason: str, raw: Dict[str, Any].
    OUTPUTS: None (data holder).
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: Return a plain tuple (text, usage, latency) — less self-docu-
    menting and easier to misuse positionally.
    """

    text: str
    model: str
    provider: str = "nvidia"
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _redact(text: Any, api_key: Optional[str]) -> str:
    """
    Convert to str and scrub the API key if present.

    WHAT: Turn any value into a string and replace any occurrence of the API key
    with "[REDACTED]".
    WHY: Ensures the secret never appears in logs or exception messages (defense
    in depth for credential handling).
    INPUTS: text: Any — e.g. an httpx exception message containing "Bearer
        sk-1234..."; api_key: Optional[str] — the key to scrub.
    OUTPUTS: str — e.g. "NVIDIA request failed: Bearer [REDACTED]...".
    SIDE EFFECTS: None.
    FAILURE MODES: None — repr() fallback for non-str values.
    ALTERNATIVE: Use a logging filter that scrubs secrets globally instead of
    redacting at each call site.
    # GAP: Redaction only happens where _redact is explicitly called; any log
    line that interpolates the raw exception or response without _redact could
    leak the key.
    """
    s = text if isinstance(text, str) else repr(text)
    if api_key:
        s = s.replace(api_key, "[REDACTED]")
    return s


class NVIDIAProvider(LLMProvider):
    """
    NVIDIA NIM provider over the OpenAI-compatible chat completions API.

    WHAT: Concrete LLMProvider implementation that calls NVIDIA NIM's
    /chat/completions endpoint, with retries, backoff, key-pool failover,
    streaming, structured JSON output, usage tracking, and secret redaction.
    WHY: Production-grade LLM backend — resilient to transient failures, able to
    rotate API keys on auth failure, and observable (latency + token usage).
    INPUTS (__init__): api_key: str (default ""), model: str (default
        DEFAULT_MODEL), base_url: str (default DEFAULT_BASE_URL),
        timeout_s: float (default 60.0), max_retries: int (default 3),
        backoff_base_s: float (default 1.0), backoff_max_s: float (default 30.0),
        client: Optional[httpx.Client], key_pool: Optional[Any] (KeyPool).
    OUTPUTS: None (instance).
    SIDE EFFECTS: Constructs an httpx.Client (owns it unless one is injected);
        reads key_pool.current_key() if a pool is given; raises ProviderAuthError
        if no key is available.
    FAILURE MODES: __init__ raises ProviderAuthError when api_key is empty and
        no key_pool is provided (providers/llm_nvidia.py:108).
    ALTERNATIVE: Use the OpenAI Python SDK pointed at NVIDIA's base URL instead
    of a raw httpx client.
    """

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
        """
        Fingerprint of the current key for logs (never the full key).

        WHAT: Return a masked prefix/suffix of the API key for log correlation.
        WHY: Lets operators correlate logs to a key without exposing the secret.
        INPUTS: None (uses self._api_key).
        OUTPUTS: str — e.g. "abcd...1234" or "****" for short keys.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Log the full key (unacceptable — secret leakage).
        # GAP: If the key is shorter than 12 chars it returns "****" — two
        different short keys become indistinguishable in logs.
        """
        k = self._api_key
        return f"{k[:4]}...{k[-4:]}" if len(k) >= 12 else "****"

    def _headers(self) -> Dict[str, str]:
        """
        Build HTTP request headers including the Bearer auth.

        WHAT: Construct the Authorization (Bearer API key), Content-Type, and
        Accept headers for NVIDIA NIM.
        WHY: Every request needs the secret key in the Authorization header; this
        centralizes header construction so the key is always present and never
        accidentally omitted.
        INPUTS: None (uses self._api_key).
        OUTPUTS: Dict[str, str] — e.g.
            {"Authorization": "Bearer sk-...", "Content-Type":
             "application/json", "Accept": "application/json"}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Build headers inline at each call site (duplication risk).
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, stream: bool = False) -> str:
        """
        Return the chat-completions endpoint URL.

        WHAT: Compose the full URL for the NVIDIA NIM /chat/completions endpoint.
        WHY: Single source of truth for the endpoint path; stream flag is
        accepted for API symmetry though the URL is the same.
        INPUTS: stream: bool (unused in URL construction, kept for interface
        symmetry).
        OUTPUTS: str — e.g. "https://integrate.api.nvidia.com/v1/chat/completions".
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Hard-code the URL at each call site.
        """
        return f"{self.base_url}/chat/completions"

    def _payload(self, messages: List[Dict[str, str]], stream: bool,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Build the JSON request body for chat/completions.

        WHAT: Assemble model, messages, stream flag, and any passed-through
        optional params (temperature, max_tokens, top_p, stop, tools,
        tool_choice, response_format).
        WHY: Centralizes payload construction so optional generation parameters
        are forwarded consistently and None values are dropped.
        INPUTS: messages: List[Dict[str, str]] — e.g.
            [{"role": "system", "content": "..."}, {"role": "user", "content":
            "..."}]; stream: bool; **kwargs: optional generation params.
        OUTPUTS: Dict[str, Any] — e.g.
            {"model": "meta/llama-3.1-70b-instruct", "messages": [...],
             "stream": False, "temperature": 0.2}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Build the dict inline at each caller (duplication).
        """
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
        """
        Sleep with exponential backoff (capped) before a retry.

        WHAT: Compute delay = min(backoff_base_s * 2**attempt, backoff_max_s)
        and block with time.sleep.
        WHY: Exponential backoff reduces load on NVIDIA during transient
        failures/rate limits; the cap prevents unbounded waits.
        INPUTS: attempt: int — the retry attempt index (0-based).
        OUTPUTS: None (blocks the calling thread).
        SIDE EFFECTS: Blocks the thread (synchronous sleep) — see GAP note below.
        FAILURE MODES: None.
        ALTERNATIVE: asyncio.sleep for async callers; add jitter to avoid
        thundering herds.
        # GAP: Blocking time.sleep in a sync client; if this provider is ever
        used from an async event loop it blocks the loop. No jitter means
        retries from many instances can collide (thundering herd).
        """
        delay = min(self.backoff_base_s * (2 ** attempt), self.backoff_max_s)
        time.sleep(delay)

    def _classify_error(self, exc: Exception) -> ProviderError:
        """
        Wrap an httpx error into a typed ProviderError without the key.

        WHAT: Convert httpx exceptions (timeouts, HTTP status errors) into typed
        ProviderError subclasses, with the API key redacted from the message.
        WHY: Callers can catch specific error types (auth vs rate limit vs server
        vs timeout) and decide retry/rotate/fail-fast accordingly — without the
        secret leaking into the message.
        INPUTS: exc: Exception — an httpx.TimeoutException or httpx.HTTPStatusError.
        OUTPUTS: ProviderError — e.g. ProviderAuthError for 401/403,
            ProviderRateLimitError for 429, ProviderServerError for 5xx,
            ProviderTimeoutError for timeouts.
        SIDE EFFECTS: Calls _redact to scrub the key.
        FAILURE MODES: None (returns a ProviderError for any input).
        ALTERNATIVE: Map status codes with a dict instead of if/elif chains.
        """
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
        """
        POST with retries on timeout / 429 / 5xx, plus key-pool failover on 401/403.

        WHAT: Send the chat request, retrying transient failures (timeout, 429,
        5xx) with exponential backoff, and rotating API keys on 401/403 when a
        key pool is configured.
        WHY: Production resilience — transient NVIDIA outages and rate limits
        shouldn't fail the whole request; compromised/expired keys are rotated
        automatically without burning the retry budget.
        INPUTS: payload: Dict[str, Any] — the JSON request body.
        OUTPUTS: httpx.Response on success (2xx, or 401/403 when no pool).
        SIDE EFFECTs: Makes network calls; sleeps between retries; mutates
        self._api_key on key rotation; logs warnings on retry/rotation.
        FAILURE MODES: Raises typed ProviderError (via _classify_error) when all
        retries/rotations are exhausted; non-429/non-5xx 4xx errors raise
        immediately (not retried).
        # GAP: Key rotation does not consume the retry budget — with a pool of N
        keys we may try up to N keys on auth failures plus max_retries on
        transients, so total attempts can exceed max_retries. Callers should not
        assume a hard attempt ceiling.
        # BUSINESS: Key-pool failover keeps the service available when one NVIDIA
        API key is revoked or rate-limited, without operator intervention.
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
        """
        Parse the usage block from the NVIDIA JSON response.

        WHAT: Extract prompt_tokens, completion_tokens, total_tokens from the
        "usage" field of the response, defaulting to 0.
        WHY: Actual token usage from NVIDIA (not estimates) is what we report for
        cost/observability.
        INPUTS: data: Dict[str, Any] — the parsed JSON response body.
        OUTPUTS: Usage — e.g. Usage(prompt_tokens=120, completion_tokens=45,
            total_tokens=165).
        SIDE EFFECTS: None.
        FAILURE MODES: int() on non-numeric values could raise ValueError if
        NVIDIA returns strings.
        ALTERNATIVE: Estimate from character count instead of trusting the API.
        """
        u = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0)),
            completion_tokens=int(u.get("completion_tokens", 0)),
            total_tokens=int(u.get("total_tokens", 0)),
        )

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> tuple[str, str]:
        """
        Extract generated text and finish reason from the first choice.

        WHAT: Pull the assistant message content and finish_reason from
        data["choices"][0].
        WHY: The text is the actual answer; finish_reason tells us whether the
        model stopped naturally or was cut off (length, content filter).
        INPUTS: data: Dict[str, Any] — the parsed JSON response body.
        OUTPUTS: tuple[str, str] — (text, finish_reason), e.g. ("5G uses OFDM",
            "stop"); ("", "") when there are no choices.
        SIDE EFFECTS: None.
        FAILURE MODES: Returns ("", "") for missing/empty choices rather than
        raising — a missing answer is silently treated as empty.
        ALTERNATIVE: Raise on empty choices (stricter, fails fast).
        """
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
        """
        Non-streaming chat completion. Returns text + usage + latency.

        WHAT: Send a full chat request to NVIDIA NIM, measure latency, parse
        text/usage/finish_reason, and return a ChatResult.
        WHY: The primary generation path used by answer generation; latency and
        usage are captured for observability and stored on self.last_result.
        INPUTS: messages: List[Dict[str, str]]; temperature: Optional[float];
            max_tokens: Optional[int]; **kwargs: extra payload params.
        OUTPUTS: ChatResult — e.g.
            ChatResult(text="5G uses OFDM...", model="meta/llama-3.1-70b-instruct",
            usage=Usage(120, 45, 165), latency_ms=842.5, finish_reason="stop").
        SIDE EFFECTs: Makes a network POST; sets self.last_result; logs an INFO
        line with model/latency/tokens.
        FAILURE MODES: Raises ProviderError (auth/rate-limit/server/timeout) via
        _post when all retries are exhausted.
        ALTERNATIVE: Use stream_chat() when you need incremental tokens instead
        of waiting for the full answer.
        # BUSINESS: Latency + usage captured per call = the observability data
        needed to answer "how fast and how much did this answer cost?".
        """
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
        """
        LLMProvider interface: single-prompt completion -> text.

        WHAT: Wrap a single prompt as a user message and return just the text.
        WHY: Satisfies the LLMProvider.generate contract for callers (planner in
        agent.py, fallback in answer.py) that only need a string back.
        INPUTS: prompt: str — e.g. "Context: ...\n\nQuestion: What is 5G?";
            **kwargs: forwarded to chat().
        OUTPUTS: str — e.g. "5G uses orthogonal frequency-division multiplexing."
        SIDE EFFECTs: Calls chat() (network + sets self.last_result).
        FAILURE MODES: Propagates ProviderError from chat() on failure.
        ALTERNATIVE: Call chat() directly when you need usage/latency metadata.
        """
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
        """
        Streaming chat completion. Yields text deltas as they arrive.

        WHAT: Open a streaming POST to NVIDIA NIM and yield each SSE "data:"
        chunk's delta content as it arrives.
        WHY: Enables incremental answer display in the Streamlit UI and real-time
        feedback instead of waiting for the full response.
        INPUTS: messages: List[Dict[str, str]]; temperature, max_tokens,
            **kwargs: forwarded to _payload.
        OUTPUTS (yields): str deltas — e.g. "5G", " uses ", "OFDM".
        SIDE EFFECTs: Opens a streaming HTTP connection held open until done;
        reads the full response body on auth/rate-limit/server errors before
        raising.
        FAILURE MODES: Raises ProviderError for auth (401/403), rate limit
        (429), or server (5xx) errors; malformed JSON lines are skipped silently
        (a corrupt chunk is dropped, not failed).
        ALTERNATIVE: Use chat() (non-streaming) when you need the complete answer
        before showing anything.
        # GAP: Malformed SSE lines are skipped with continue — a corrupt chunk
        is silently dropped, which could truncate an answer without signaling an
        error.
        """
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
        """
        Structured output: ask for JSON, validate against a Pydantic schema.

        WHAT: Instruct the model to return JSON only, parse it, and validate
        against a Pydantic model; retry once with a correction prompt on failure.
        WHY: Lets the agent/programmatic callers get typed, schema-valid output
        (e.g. GroundedAnswer in src/generation/answer.py:363) instead of
        free-form text.
        INPUTS: messages: List[Dict[str, str]]; schema: Type[T] (a Pydantic
            BaseModel subclass); max_retries: int (default 1); temperature: float
            (default 0.0); **kwargs: forwarded to chat().
        OUTPUTS: T — an instance of schema, e.g. GroundedAnswer(answer="...",
            citations=["lte.md"], insufficient_evidence=False).
        SIDE EFFECTs: May make up to max_retries+1 chat calls; appends
        correction messages to the conversation on validation failure.
        FAILURE MODES: Raises ValidationError (or json.JSONDecodeError) on final
        failure after exhausting retries.
        ALTERNATIVE: Ask the model for free text and parse with regex (fragile,
        no schema guarantee).
        # GAP: Retries once (configurable) then re-raises; if the model
        persistently refuses JSON the whole call fails rather than degrading to
        text.
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
        """
        Return metadata describing the NVIDIA model.

        WHAT: Build a dict with provider, model name, base URL, type, streaming
        support, timeout, and max_retries.
        WHY: Callers (answer generator) log which model produced each answer for
        auditing and cost tracking.
        INPUTS: None (uses instance attributes).
        OUTPUTS: Dict[str, Any] — e.g.
            {"provider": "nvidia", "model_name": "meta/llama-3.1-70b-instruct",
             "base_url": "https://integrate.api.nvidia.com/v1", "type": "llm",
             "supports_streaming": True, "max_retries": 3, "timeout_s": 60.0}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Read model info from a static config map.
        """
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
        """
        Close the underlying httpx client if this instance owns it.

        WHAT: Release the httpx.Client connection pool when the provider is done.
        WHY: Prevents file-descriptor/connection leaks in long-running processes
        (API server, Streamlit) that create many providers.
        INPUTS: None.
        OUTPUTS: None.
        SIDE EFFECTs: Closes self._client if self._owns_client.
        FAILURE MODES: None (best-effort; __del__ swallows errors).
        ALTERNATIVE: Rely on garbage collection (unpredictable timing).
        # GAP: Users must remember to call close() or rely on __del__; there is
        no context-manager (__enter__/__exit__) support, so leaked clients are
        possible.
        """
        if self._owns_client:
            self._client.close()

    def __del__(self) -> None:  # best-effort
        """
        Best-effort cleanup on garbage collection.

        WHAT: Try to close the client when the object is being collected.
        WHY: Safety net so connections are released even if close() was forgotten.
        INPUTS: None.
        OUTPUTS: None.
        SIDE EFFECTs: Calls close(); any exception is swallowed.
        FAILURE MODES: __del__ timing is non-deterministic (Python GC); not a
        substitute for explicit close().
        ALTERNATIVE: Context manager for deterministic cleanup.
        """
        try:
            self.close()
        except Exception:
            pass
