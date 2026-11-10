"""
src.security - AI production-grade security controls for the RAG Academy telecom QA system.

Modules:
- auth: scoped API keys / JWT, server-side authorization (replaces client-controlled headers)
- rate_limit: distributed sliding-window rate limiter (Redis-backed, in-memory fallback)
- prompt_guard: prompt injection / jailbreak detection, input framing, grounding verification
- redaction: PII and secret redaction for answers, logs, and traces
- validation: input validation, request size limits, upload allowlists

Design principles:
- Defense in depth: every control is a layer; no single layer is trusted alone.
- Fail closed: when in doubt, reject the request.
- Never trust client input: roles, scopes, and access levels are derived server-side
  from verified credentials, never from client-controlled headers.
- Redact secrets and PII everywhere they could be logged or returned.
"""
