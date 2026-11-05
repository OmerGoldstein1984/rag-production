# Security design

## Secrets
- `NVIDIA_API_KEYS` / `NVIDIA_API_KEY`, `API_TOKEN`, `SMTP_PASSWORD` come **only from env**. `.env` is gitignored and `.dockerignore`d (CI asserts no `.env` in the image).
- Only **fingerprints** (`first4...last4`) of keys are ever logged, stored in the registry, or emailed. Tests assert full keys never appear in logs/errors/registry/email.
- Provider errors are typed and the key is scrubbed from messages (`_redact`).
- Logging has a redaction filter (nvapi-/sk-/Bearer patterns → `[REDACTED]`).

## API
- If `API_TOKEN` is set, every endpoint requires `Authorization: Bearer <token>` (constant-time compare). Without it, local dev is open.
- Role checks via `X-Access-Level` (admin-only for upload/reindex/delete); header is only trustworthy behind the token.
- `/ingest` & reindex confine paths to the project `data/` or `uploads/` (traversal blocked).
- Rate limiting (per client) and an in-memory query cache.
- File upload: extension allowlist + 10 MB cap + plain filename.

## Untrusted data
- Retrieved chunks and LLM output are **data, never instructions**: prompts wrap them in delimiters; tool inputs are Pydantic-validated; the agent allow-list prevents arbitrary execution; errors are generic to clients.
- Structured outputs validated with Pydantic (`GroundedAnswer`) with a correction retry.

## Document lifecycle
- Content-hash dedupe; version supersession (`is_current=False` on old chunks); delete cascades to vector store + file; ingest status log records failures (retryable).
