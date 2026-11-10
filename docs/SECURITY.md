# AI Production-Grade Security — RAG Academy

This document explains the AI security controls added to the rag-project RAG
system: what each term means, why it matters, what changed in the code, how to
configure the controls, and how to verify them.

## Purpose

The RAG Academy system answers telecom-domain questions (LTE/5G, Roaming, SIM
Security, eSIM) from ingested documents using an LLM. Because the model is reachable
over HTTP and takes untrusted user queries, it is exposed to AI-specific threats
(prompt injection, jailbreaking, data exfiltration) on top of classic web risks.
These controls provide defense in depth so no single layer is trusted alone.

## Threat model

Attackers may try to:

- **Prompt injection / jailbreak**: override the system prompt ("ignore previous
  instructions", "reveal your system prompt", "you are now DAN") to exfiltrate
  retrieved telecom documents or force ungrounded answers.
- **Authorization bypass**: self-assert an admin role (e.g. spoof the
  `X-Access-Level` header) to ingest or delete documents.
- **Denial of service**: flood the LLM/embedding pipeline with rapid or huge
  requests, oversized uploads, or unbounded `top_k` / `chunk_size`.
- **Data exfiltration / PII leakage**: return customer or network identifiers
  (emails, phone numbers, IMSI/IMEI) in answers, logs, or traces.
- **Output XSS**: return HTML/JavaScript that executes in the Streamlit UI.
- **Path traversal**: point ingestion at arbitrary host files.

## Controls — what each term means

| Control | What it means | Where it lives |
|---------|---------------|----------------|
| **Scoped API keys** | Rotatable, revocable bearer keys with least-privilege scopes (`ask`, `ingest`, `delete`, `admin`). Replaces the single shared `API_TOKEN`. | `src/security/auth.py` |
| **Server-side authorization** | Roles are derived from the verified key's scopes — never from client-controlled headers. A caller cannot self-assert `admin`. | `src/security/auth.py`, `src/api/main.py` |
| **Rate limiting** | Sliding-window limit per (client, endpoint). Redis backend shared across uvicorn workers (or in-memory fallback); `Retry-After` header on 429; idle-client eviction to bound memory. Stricter limits on `/ingest` and `/agent/ask`. | `src/security/rate_limit.py`, `src/api/main.py` |
| **Prompt injection defense** | Fail-closed detection of jailbreak / instruction-override patterns. User input is framed in explicit delimiters so the model can distinguish trusted system instructions from untrusted text. | `src/security/prompt_guard.py`, `src/security/injection.py`, `src/generation/prompt.py`, `src/generation/answer.py` |
| **Output grounding verification** | Generated answers are checked for a source citation or an insufficiency message; ungrounded answers (possible injection/hallucination) are flagged. | `src/generation/answer.py` |
| **PII / secret redaction** | Emails, phones, IMSI/IMEI, customer IDs, API keys, and bearer tokens are replaced with `[REDACTED:<type>]` in answers, logs, and traces. | `src/security/redaction.py`, `src/api/main.py` |
| **Input validation & request limits** | Bounded `top_k`, `chunk_size`, `chunk_overlap`, and query length; max request-body and upload size; upload extension allowlist (no `.exe`, `.sh`, `.py`, etc.). | `src/security/validation.py`, `src/api/main.py` |
| **Resilience** | Per-LLM-call timeout, exponential-backoff + jitter retries, circuit breaker, and an agent wall-clock timeout. | `src/generation/answer.py`, `src/api/main.py` |
| **Security headers + CORS** | HSTS, CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`; CORS restricted to known frontends. | `src/api/main.py` |
| **Audit logging** | Structured security events (auth successes/failures, rate-limit blocks, injection blocks, PII redactions). | `src/security/audit.py` |

## What changed in the code

- `src/config.py` — added settings for API keys, rate limiting, PII redaction,
  security headers, CORS, audit logging, input limits, and resilience timeouts.
- `src/security/auth.py` (**new**) — scoped API key model, constant-time
  verification, server-side role derivation, `require_scope` dependency,
  `get_current_user`.
- `src/security/audit.py` (**new**) — structured audit logger.
- `src/security/rate_limit.py` (**new**) — distributed sliding-window rate
  limiter (Redis + in-memory fallback), per-endpoint rules, `Retry-After`.
- `src/security/validation.py` (**new**) — input validation, body/upload limits,
  filename validation.
- `src/security/injection.py` (**new**) — prompt injection patterns, detection,
  sanitization, delimiter framing (compatibility shim over `prompt_guard`).
- `src/security/prompt_guard.py` (**new**) — canonical fail-closed injection
  guard plus grounding verification.
- `src/security/redaction.py` — PII/secret redaction (enhanced with typed
  `[REDACTED:<type>]` labels).
- `src/api/main.py` — scoped bearer auth on all endpoints, per-endpoint rate
  limiting with `Retry-After`, CORS, body-size limit, security headers, Pydantic
  field constraints, upload validation, PII redaction of answers/context, agent
  wall-clock timeout.
- `src/generation/prompt.py` — user input wrapped in delimiter framing.
- `src/generation/answer.py` — query sanitization, output grounding verification,
  retry/backoff, circuit breaker, per-call timeout.

## Configuration (environment variables)

```bash
# Auth
API_KEYS=key1:secret1,key2:secret2     # comma-separated "id:secret" scoped keys
API_KEY_REQUIRED=true
API_KEY_EXPIRY_DAYS=90

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_BACKEND=memory               # or "redis"
REDIS_URL=redis://localhost:6379/0

# PII / headers / audit
PII_REDACTION_ENABLED=true
SECURITY_HEADERS_ENABLED=true
CORS_ALLOWED_ORIGINS=http://localhost:8501
AUDIT_LOG_ENABLED=true

# Limits / resilience
MAX_QUERY_LENGTH=2000
MAX_TOP_K=20
MAX_CHUNK_SIZE=2000
MAX_UPLOAD_BYTES=10485760
LLM_REQUEST_TIMEOUT_S=30
LLM_MAX_RETRIES=3
AGENT_WALL_CLOCK_TIMEOUT_S=60
```

## How to verify

Run the adversarial (red-team) test suites:

```bash
pytest tests/test_ai_security_redteam.py -v          # prompt injection, rate limiting, authZ, validation, PII redaction, grounding, path traversal
pytest tests/test_security_adversarial.py -v         # parallel-agent adversarial suite
pytest tests/test_adversarial_security.py -v         # parallel-agent adversarial suite
pytest tests/test_security_api.py                    # existing security API tests
```

The suites attempt prompt injection, authorization bypass, rate-limit bypass,
path traversal, oversized inputs, disallowed uploads, output XSS, and PII
leakage, and assert that every control holds.
