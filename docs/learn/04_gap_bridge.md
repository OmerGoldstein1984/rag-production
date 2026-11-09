# 04 — Gap Bridge: Web-App Patterns vs. Production RAG System Patterns

**Audience:** senior QA / automation engineer moving into AI Solutions Engineering (telecom/eSIM, IoT).
You know program flow, not line-by-line code. You are weak on production architecture,
concurrency/async, auth/multi-tenancy, observability, evals, deployment, cost/latency
tradeoffs, and customer-facing communication.

**Purpose of this file.** For each non-trivial pattern in the repo, it states:

> In a typical web app you did X; in a production system this repo does Y, because Z.

Every claim is grounded in `file:line` so you can verify it yourself. Analogies use
telecom / provisioning / test-pipeline language throughout.

**Constraint honored:** no source code was changed to write this doc. Only this new file
was added under `docs/learn/`.

---

## How to read this file

Each section follows the same template:

| Element | Meaning |
|---------|---------|
| **Typical web app (X)** | What a junior/automation engineer usually builds — the "naive" version. |
| **This repo (Y)** | What the rag-academy production pattern actually does. |
| **Because (Z)** | The production reason — correctness, cost, safety, or operability. |
| **file:line** | Where to look in the repo. |
| **Analogy** | Telecom / provisioning / test-pipeline framing. |

# GAP: marks a concept the learner is likely to lack (2–3 sentences).
# BUSINESS: marks what the pattern protects (uptime, budget, compliance, trust).
# INTERVIEW: a say-it-out-loud point for FDE interviews.

---

## 0. Orientation — what this system is

This is a RAG (Retrieval-Augmented Generation) system for FDE interview prep.
Ingestion: load → clean → chunk → embed → store. Query: retrieve → (rerank) → generate
a grounded answer from retrieved chunks only. Served by FastAPI (`src/api/main.py`)
with a Streamlit UI (`ui/streamlit_app.py`) and a Docker deployment.

The production patterns below are what separate a toy "it works on my laptop"
prototype from a system you could hand to a customer and keep running.

---

# ============================================================
# 1. FAILURE HANDLING
# ============================================================

## 1.1 Fail-fast at startup for missing secrets

**Typical web app (X):** Read config lazily. If `NVIDIA_API_KEY` is missing, the app
starts fine and only fails with a cryptic 500 when the first user asks a question.

**This repo (Y):** `get_settings()` validates required secrets at import/startup and
crashes immediately with a clear, actionable message if `LLM_PROVIDER=nvidia` but no
key is set.

**Because (Z):** A RAG system without an LLM key cannot answer anything. A clear
startup crash beats intermittent 500s that waste user time and hide misconfiguration.

**file:line:** `src/config.py:42` (`MissingConfigError`), `src/config.py:177-183`
(nvidia key required check), `src/config.py:70-71` (frozen Settings so nothing can
mutate config mid-flight).

**# INTERVIEW:** "Required secrets are validated at startup; if NVIDIA_API_KEY is
missing we crash fast with a clear message rather than failing silently per request."

**Analogy:** Like a cell-site commissioning checklist — if the backhaul fiber isn't
lit, you don't power up the radio and hope; you fail the commissioning step with a
clear ticket so it gets fixed before customers are affected.

---

## 1.2 Retry with exponential backoff on transient failure

**Typical web app (X):** Call an external API once; on timeout or 500, let the
exception bubble to a generic error handler and return 500 to the user.

**This repo (Y):** `retry_with_backoff` retries up to `max_retries` times with
exponential delay (1s, 2s, 4s… capped at `max_delay`) before re-raising.

**Because (Z):** Embedding calls and NVIDIA NIM calls can fail transiently (flaky
model load, network blip). Retrying keeps the ingest pipeline alive instead of
aborting on a blip.

**file:line:** `src/embedding/embedder.py:73-113` (`retry_with_backoff` decorator),
applied at `src/embedding/embedder.py:286`.

**# GAP:** Hand-rolled retry (stdlib `time.sleep`, no jitter) instead of a battle-
tested library like `tenacity`. No circuit breaker either — after retries are
exhausted the error still propagates.

**# BUSINESS:** Protects ingestion uptime and prevents a transient NVIDIA blip from
failing a whole document upload.

**Analogy:** Like retrying a failed SIM provisioning attempt on the HLR a few times
with backoff before declaring the activation failed — network hiccups shouldn't
doom a legitimate activation.

---

## 1.3 Graceful degradation — mock fallbacks instead of crashing

**Typical web app (X):** A missing optional dependency (e.g. the ML model package)
crashes the process at import time.

**This repo (Y):** If `sentence-transformers` isn't installed or the model fails to
load, `self.model` stays `None` and the mock embedder is used — no exception escapes
the constructor. Similarly, the reranker degrades to passthrough scores when its
cross-encoder is unavailable, and ChromaDB falls back to an in-memory store.

**Because (Z):** Tests, demos, and local dev must run offline without downloading
hundreds of MB of model weights. The system stays available at reduced quality
rather than dying entirely.

**file:line:** `src/embedding/embedder.py:229-238` (model load failure → `None`),
`src/embedding/embedder.py:346-350` (embed error → mock vector fallback),
`src/store/vector_store.py:67-73` (ChromaDB → in-memory fallback),
`src/retrieval/reranker.py:47-50` (CrossEncoder optional, passthrough fallback),
`src/generation/answer.py:208-215` (LLM provider failure → insufficiency message).

**# GAP:** Swallowed errors can mask real problems (a bad model silently producing
useless vectors). Production monitoring should alert when fallback mode is active.

**Analogy:** Like a provisioning gateway that falls back to a cached subscriber
record when the HLR is unreachable — service continues at reduced fidelity instead
of returning "system unavailable" to every subscriber.

---

## 1.4 Step caps and repeat caps on the agent

**Typical web app (X):** An LLM agent loops until it produces an answer, with no
bound on tool calls — a misbehaving planner can loop forever, burning cost and
time.

**This repo (Y):** `RAGAgent.run` caps total steps at `MAX_STEPS = 5` and caps
repeated identical tool calls at `MAX_REPEAT_SAME_CALL = 2`. On tool failure it
stops and reports deterministically.

**Because (Z):** Bounded execution protects cost (each tool call may hit the LLM),
latency, and predictability. A tool failure is observable, not an infinite loop.

**file:line:** `src/agent/agent.py:29-30`, `src/agent/agent.py:108-151` (`run` loop),
`src/agent/tools.py:102-136` (`ToolRegistry.execute` wraps validation, auth,
timeout, and records every call).

**# INTERVIEW:** "We cap steps and repeat calls so an agent can't spiral; every tool
call is validated, authorized, timed, and audited."

**Analogy:** Like a field technician's work order that auto-closes after N attempted
fixes so one stubborn fault doesn't tie up the tech indefinitely.

---

# ============================================================
# 2. SCALE
# ============================================================

## 2.1 Key pool with round-robin rotation and failover

**Typical web app (X):** One API key stored in an env var. If that key is
rate-limited or revoked, every request fails until someone rotates it manually.

**This repo (Y):** `KeyPool` holds multiple NVIDIA keys from `NVIDIA_API_KEYS`
(comma-separated), selects them round-robin, and on a 401/403 marks the key bad and
rotates to the next one — without consuming the retry budget.

**Because (Z):** Production LLM traffic must survive a single key being revoked or
rate-limited without operator intervention. Rotation spreads usage so no single key
burns its quota first.

**file:line:** `src/key_pool.py:54-174` (`KeyPool`), `src/key_pool.py:149-157`
(`mark_bad` + `rotate`), `providers/llm_nvidia.py:451-457` (key rotation on 401/403),
`providers/llm.py:345-354` (factory wires pool into NVIDIAProvider).

**# BUSINESS:** Key-pool failover keeps the QA service available when one NVIDIA key
is revoked, without a page to on-call.

**Analogy:** Like a SIM provisioning cluster with multiple provisioning keys — if
one key is rate-limited by the clearinghouse, the system routes the next activation
through a different key instead of queuing all activations.

---

## 2.2 Embedding cache keyed by content hash

**Typical web app (X):** Embed every document chunk on every ingest run, even if
the text hasn't changed — paying the model cost and latency repeatedly.

**This repo (Y):** `Embedder` caches vectors on disk under `data/embedding_cache/`,
keyed by an MD5 of the text. Identical text hits the cache; only changed text is
re-embedded.

**Because (Z):** Embedding is the most expensive per-call step in ingestion.
Caching cuts both cost (API dollars if remote) and latency (no model forward pass
for repeats).

**file:line:** `src/embedding/embedder.py:240-270` (cache read/write),
`src/embedding/embedder.py:300-350` (`embed_text` checks cache first),
`src/embedding/embedder.py:352-422` (`embed_batch` splits hits/misses).

**# GAP:** Cache is per-process, disk-backed JSON files with no TTL and no
eviction policy beyond manual `clear_cache`. Stale embeddings persist if the model
changes.

**Analogy:** Like provisioning a SIM profile once and caching the result — if the
same subscriber record is re-uploaded, you don't re-run the entire provisioning
workflow; you serve the cached profile.

---

## 2.3 Query cache and rate limiter at the API edge

**Typical web app (X):** Every identical user query re-runs the full pipeline
(retrieve → rerank → LLM), and there's no protection against a client flooding the
endpoint.

**This repo (Y):** `QueryCache` caches `QueryResponse` by a hash of all request
parameters (query + flags). `RateLimiter` tracks per-client timestamps in a
sliding window and returns 429 when exceeded.

**Because (Z):** Repeated telecom questions shouldn't re-run the whole pipeline —
caching cuts LLM token cost and latency. Rate limiting protects the LLM budget and
keeps the service available to all users under load.

**file:line:** `src/api/main.py:126-189` (`QueryCache`), `src/api/main.py:195-250`
(`RateLimiter`), enforced at `src/api/main.py:527-528` and `src/api/main.py:663-664`.

**# GAP:** Both are in-memory only — state resets on restart and isn't shared
across uvicorn workers (each worker has its own counter). No Redis backing store.

**Analogy:** Like a provisioning system that caches recently-activated SIM profiles
and rate-limits a single dealer's activation requests so one bulk script can't
exhaust the day's activation quota.

---

## 2.4 Document dedupe and versioning at ingest

**Typical web app (X):** Re-uploading the same file creates duplicate entries, and
there's no notion of "this version supersedes that one."

**This repo (Y):** `DocumentManager.upload` dedupes by SHA-256 content hash
(identical bytes → `duplicate` status, no-op) and versions by filename: re-uploading
a changed file supersedes the old version (`is_current=False` on old chunks) and
bumps `doc_version`.

**Because (Z):** Prevents stale or duplicate chunks from accumulating, which would
otherwise make retrieval return contradictory answers for the same question.

**file:line:** `src/ingestion/manager.py:114-148` (`upload`: dedupe + versioning),
`src/ingestion/manager.py:31-32` (allowed extensions + 10 MB cap),
`src/store/vector_store.py:519-578` (`delete_by_source` for cascade delete before
rewrite).

**# INTERVIEW:** "How do you update an already-ingested document? Delete all old
chunks for that source, then write the fresh ones — delete-then-write, not upsert."

**Analogy:** Like a subscription database where re-shipping the same SIM card is a
no-op, but issuing a replacement SIM marks the old one inactive and activates the
new one.

---

## 2.5 Vector store does not scale beyond a modest corpus

**Typical web app (X):** A managed vector database with approximate-nearest
indexes (HNSW), sharding, and replication.

**This repo (Y):** ChromaDB embedded (single-process) with an in-memory fallback.
Search is a linear scan over all vectors; no HNSW index, no result caching, no
latency histograms on search.

**Because (Z):** At demo/eval scale (tens of documents), linear scan is fast
enough and keeps the system runnable offline with zero infrastructure. Re-evaluate
at scale.

**file:line:** `src/store/vector_store.py:340-478` (`search` + `_fallback_search`),
`src/store/vector_store.py:363-366` (GAP note on linear scan).

**# GAP:** No concurrency control, no locking, no transactions — concurrent writers
can interleave state. `store.get_stats()` (`src/store/vector_store.py:620-667`)
reports `storage_type` so you can tell ChromaDB vs fallback.

**Analogy:** Like a local card-catalog index at a small field office — fine for a
few thousand records, but you'd migrate to a proper database (HLR-scale) before
rolling out nationwide.

---

# ============================================================
# 3. SECURITY
# ============================================================

## 3.1 API bearer token (authentication)

**Typical web app (X):** Session cookies or no auth at all in local dev; secrets
sometimes checked into source.

**This repo (Y):** If `API_TOKEN` is set, every endpoint requires
`Authorization: Bearer <token>`, compared in constant time (`hmac.compare_digest`).
Without a configured token, local dev is open. The token is never logged.

**Because (Z):** Mutating endpoints (ingest, delete, reindex, upload) must not be
callable by anyone who can reach the network. Constant-time compare prevents
timing attacks; never logging the token prevents leakage.

**file:line:** `src/api/main.py:268-284` (`verify_api_token`),
`src/config.py:121` (`api_token` setting), `src/api/main.py:451-456` (`/metrics`
and `/traces` also require the token).

**Analogy:** Like requiring a provisioning engineer's badge (bearer token) before
allowing any line activation — anyone on the network can read public info, but
only badged staff can change subscriber records.

---

## 3.2 Secret redaction (defense in depth)

**Typical web app (X):** Exception messages and log lines interpolate raw API keys
and bearer tokens, so a single 500 leaks credentials to logs/Sentry.

**This repo (Y):** `redact()` scrubs `nvapi-`, `sk-`, and `Bearer` patterns from
every log record via `RedactFilter`; `NVIDIAProvider._redact` scrubs the key from
error messages; only key fingerprints (`first4…last4`) are stored in the registry
and emailed.

**Because (Z):** Credential leakage is a compliance failure. Redaction at every
log/exception call site means a key never reaches logs, errors, or alert emails.

**file:line:** `src/observability.py:42-56` (`redact`), `src/observability.py:57-73`
(`RedactFilter`), `src/observability.py:76-89` (`JsonFormatter` redacts `getMessage`),
`providers/llm_nvidia.py:201-223` (`_redact`), `src/key_pool.py:28-33`
(`fingerprint`), `src/key_monitor.py:146-160` (alert body shows fingerprints only).

**# GAP:** Redaction only happens where `_redact` / `redact` is explicitly called;
a log line interpolating a raw exception without redaction could still leak the key.

**Analogy:** Like a provisioning log that records `SIM ICCID: ****1234` instead of
the full ICCID — operators can correlate records without exposing the secret.

---

## 3.3 Allow-listed tools with Pydantic validation

**Typical web app (X):** An LLM agent can call arbitrary functions or even shell
commands; LLM output is trusted directly as a function call.

**This repo (Y):** `ToolRegistry.execute` validates tool input against a Pydantic
schema (`model_validate`), checks authorization, runs with a timeout, and records
the outcome. The LLM may *propose* a tool; the app *validates + authorizes +
executes*. The LLM never gets raw filesystem/DB/OS access.

**Because (Z):** LLM output is untrusted. Pydantic validation rejects malformed or
malicious arguments before they reach a handler; timeouts prevent a hung tool from
stalling the agent; the audit history (`ToolCall.to_dict`) gives a tamper-evident
trace of every call.

**file:line:** `src/agent/tools.py:58-136` (`ToolRegistry.execute`),
`src/agent/tools.py:140-162` (Pydantic input schemas: `SearchInput`, `FilenameInput`,
`CompareVersionsInput`, `SectionsInput`, `ReportInput`),
`src/agent/tools.py:165-250` (`build_default_tools`),
`src/agent/agent.py:89-104` (`_plan_with_llm` validates the picked tool is in the
allow-list).

**# INTERVIEW:** "The LLM proposes, the application disposes — every tool call is
validated against a schema, authorized, timed, and audited before execution."

**Analogy:** Like a field technician who can request a line test or a swap through
the provisioning system, but can't run arbitrary commands on the switch — the system
validates each request against an approved menu.

---

## 3.4 Admin-only tools and role-based access

**Typical web app (X):** Every authenticated user can upload, delete, and reindex
documents; roles are either absent or not enforced.

**This repo (Y):** `USER_ROLES` defines admin/manager/employee/guest with
`can_ingest` and `can_delete` flags. `generate_report` is `requires_admin=True`.
Endpoints check `user_info["can_ingest"]` / `user_info["can_delete"]` and return 403
for insufficient privileges. The `X-Access-Level` header is only trustworthy when
bearer-token auth is enabled.

**Because (Z):** Document ingestion and deletion are mutating operations that
affect the shared corpus. Not every user should be able to rewrite the knowledge
base. The caveat: without `API_TOKEN`, the access-level header is client-controlled
(so it's advisory in local dev).

**file:line:** `src/api/main.py:261-266` (`USER_ROLES`),
`src/api/main.py:287-305` (`verify_user_role`),
`src/api/main.py:804-808` (`/ingest` requires `can_ingest`),
`src/api/main.py:882-886` (delete requires `can_delete`),
`src/api/main.py:962-964` (reindex requires `can_ingest`),
`src/api/main.py:911-912` (upload requires `can_ingest`).

**# GAP:** Multi-tenancy is absent — there's one shared corpus and no per-customer
isolation. Department metadata is inferred from filenames (brittle keyword matching),
not enforced access control.

**file:line (GAP):** `src/ingestion/loaders.py:118-153` (`_infer_department`),
`src/api/main.py:290-291` (header trustworthiness caveat).

**Analogy:** Like a provisioning OSS where only provisioning engineers (admin) can
activate or deactivate lines, customer-support staff (employee) can only read
subscriber records, and the access badge only counts if the turnstile (bearer
token) is actually checked.

---

## 3.5 Path traversal guard and file-upload allowlist

**Typical web app (X):** A file-upload endpoint takes a client-supplied path and
writes it to disk, allowing `../../etc/passwd` or arbitrary file types.

**This repo (Y):** `resolve_safe_ingest_dir` confines ingestion to the project's
`data/` or `uploads/` directories, blocking path traversal. `DocumentManager.validate`
allows only `.pdf/.docx/.md/.txt`, requires a plain filename (no path separators),
rejects empty files, and caps size at 10 MB.

**Because (Z):** Prevents attackers from writing outside the intended directory or
uploading executable/bomb files that could compromise the host.

**file:line:** `src/api/main.py:309-330` (`resolve_safe_ingest_dir`),
`src/api/main.py:811` (used by `/ingest`),
`src/ingestion/manager.py:98-110` (`validate`),
`src/ingestion/manager.py:31-32` (`ALLOWED_EXTENSIONS`, `MAX_FILE_SIZE_BYTES`).

**Analogy:** Like a SIM card intake process that only accepts factory-sealed SIMs
of the right form factor and logs each one — no unlabeled cards, no writing outside
the intake cage.

---

# ============================================================
# 4. COST
# ============================================================

## 4.1 Key-pool rotation protects the LLM budget

**Typical web app (X):** No cost awareness; one key used until it's rate-limited,
then all requests fail (or a new key is bought).

**This repo (Y):** Round-robin spreads calls across all keys in `NVIDIA_API_KEYS`;
a failing key is marked bad and rotated. Token usage is tracked per call from
provider-reported usage.

**Because (Z):** LLM tokens cost money. Rotation avoids burning one key's quota and
keeps the service available; usage tracking answers "how much did this answer cost?"

**file:line:** `src/key_pool.py:54-174`, `providers/llm_nvidia.py:589-593` (chat logs
latency + tokens), `src/generation/answer.py:269-282` (captures provider usage),
`src/api/main.py:593-595` (adds tokens to metrics).

**Analogy:** Like rotating provisioning requests across multiple clearinghouse
connections so no single connection hits its monthly transaction cap, and logging
each transaction's cost for billing.

---

## 4.2 Token budget on context

**Typical web app (X):** Stuffs all retrieved chunks into the prompt, risking
context-window overflow and paying for unnecessary tokens.

**This repo (Y):** `_build_context_within_budget` bounds context to `max_tokens`
(default 2000), taking highest-ranked chunks first and stopping before the budget.
Prompt tokens are estimated (chars / 4) and recorded.

**Because (Z):** Keeps prompts within the model's window and bounds per-answer token
cost. The tradeoff: character-based estimation is imprecise vs. a real tokenizer.

**file:line:** `src/generation/answer.py:118-186` (`_build_context_within_budget`),
`src/generation/answer.py:96-116` (`_estimate_tokens`),
`src/generation/answer.py:141-143` (GAP: bounds context, not full prompt).

**# GAP:** `max_context_length` truncation in `format_context` (`src/generation/
prompt.py:306-307`) is by character count, not token count — a poor proxy for the
model's actual token budget.

**Analogy:** Like a provisioning request form with a fixed field count — you include
the most important subscriber attributes first and stop before the form overflows,
rather than attaching every detail and paying for extra pages.

---

## 4.3 Embedding cache avoids redundant model cost

(See section 2.2.) Re-embedding identical text is avoided entirely, which is the
single biggest cost and latency win in ingestion.

---

# ============================================================
# 5. OBSERVABILITY
# ============================================================

## 5.1 Structured JSON logs with secret redaction

**Typical web app (X):** Plain-text `print`/`logging` output; secrets sometimes in
log messages; hard to parse programmatically.

**This repo (Y):** `JsonFormatter` emits one JSON object per log line with `ts`,
`level`, `logger`, `request_id`, and `msg`. `RedactFilter` scrubs secrets from every
record.

**Because (Z):** JSON logs are ingestible by log analysis tools (ELK, CloudWatch,
Datadog). Redaction prevents credential leakage. `request_id` lets you correlate all
logs for one request across stages.

**file:line:** `src/observability.py:76-89` (`JsonFormatter`),
`src/observability.py:57-73` (`RedactFilter`),
`src/api/main.py:440-446` (startup calls `setup_logging(structured=True)`).

**Analogy:** Like a provisioning system that emits structured CDR (call detail
record) events — machine-parsable, with sensitive fields masked — instead of
free-text operator notes.

---

## 5.2 Correlation IDs (request IDs)

**Typical web app (X):** No request ID; troubleshooting a failed user query requires
grepping logs by timestamp and guessing which lines belong to the same request.

**This repo (Y):** Every request gets a `request_id` via a `ContextVar`
(`new_request_id()` / `get_request_id()`). It's included in every JSON log line and
returned in the `X-Request-Id` response header. Traces carry it too.

**Because (Z):** Correlating logs, traces, and metrics for a single request is
impossible without a stable ID that travels through all stages (middleware →
retrieve → rerank → LLM → response).

**file:line:** `src/observability.py:27-37` (ContextVar + `new_request_id`),
`src/api/main.py:411-438` (middleware assigns ID, sets `X-Request-Id` header).

**# INTERVIEW:** "Every request gets a correlation ID; it's in the logs and the
response header, so you can trace one user's query end to end."

**Analogy:** Like every provisioning ticket carrying a ticket number that appears in
the HLR log, the provisioning gateway log, and the SIM inventory log — one ID
follows the work end to end.

---

## 5.3 Per-stage tracing

**Typical web app (X):** You know the total request time, but not which stage
(retrieve, rerank, LLM) dominated it.

**This repo (Y):** `Trace` records per-stage spans with timings: retrieve, rerank,
llm, plus total. `trace.stage(name, **detail)` is a context manager that times the
block and records any error. Traces are stored in a ring buffer (last 100) and
exposed at `/traces`.

**Because (Z):** Per-stage timing tells you where to optimize — is it retrieval,
reranking, or the LLM call that's slow? Errors are captured per stage.

**file:line:** `src/observability.py:107-162` (`Trace`, `Span`, `_StageCtx`),
`src/observability.py:224-242` (ring buffer + `get_recent_traces`),
`src/api/main.py:562-597` (`/ask` uses `trace.stage` for retrieve/rerank/llm),
`src/api/main.py:459-464` (`/traces` endpoint).

**Analogy:** Like a provisioning workflow that timestamps each substage (HLR query,
HSS update, SIM inventory, activation) so you know exactly where a slow activation
stalls.

---

## 5.4 Metrics endpoint (`/metrics`)

**Typical web app (X):** Metrics only at the infrastructure level (CPU, memory) or
not at all; application-level counters (answers generated, tokens used, errors)
aren't tracked.

**This repo (Y):** `Metrics` tracks counters (`requests_total`, `answers_generated`,
`retrieved_chunks`), latency histograms (p50/p95/mean), token usage
(prompt/completion/total), and errors. Exposed at `/metrics` (bearer-token
protected). `error_rate` is computed.

**Because (Z):** Application metrics answer "is the system healthy?" and "how much
is it costing?" — independent of infrastructure metrics. Latency percentiles (not
just averages) reveal tail latency.

**file:line:** `src/observability.py:166-221` (`Metrics`),
`src/api/main.py:451-456` (`/metrics` endpoint),
`src/api/main.py:419-433` (middleware increments counters and observes latency).

**Analogy:** Like a network operations dashboard showing activation success rate,
activation latency p95, and provisioning API tokens consumed — not just server CPU.

---

## 5.5 `/provider` endpoint exposes the active model

**Typical web app (X):** No way to tell from outside whether a mock or real model
is answering; a mock silently deployed to production would go unnoticed.

**This repo (Y):** `/provider` returns `{provider, model, is_real}` — making it
impossible to hide a mock. Combined with `/health`, operators can confirm the
production path is live.

**file:line:** `src/api/main.py:467-475` (`/provider`).

**# INTERVIEW:** "The `/provider` endpoint makes a mock impossible to hide — it
tells you exactly which model is generating answers."

**Analogy:** Like a provisioning gateway status page that shows which clearinghouse
connection is active — you can tell at a glance whether you're on the real network
or a test simulator.

---

# ============================================================
# 6. TESTING / EVALS
# ============================================================

## 6.1 Golden set of questions with expected answers

**Typical web app (X):** Tests assert status codes and response shapes; there's no
notion of "does this answer actually match the source documents?"

**This repo (Y):** `golden_set.json` holds 30 questions, each with
`expected_answer_contains`, `expected_sources`, `question_type`, and `hop_count`.
Includes "unanswerable" questions (e.g. CEO salary) whose expected answer is
"cannot be determined" — testing the refusal behavior, not just retrieval.

**Because (Z):** A RAG system is only as good as its grounding. The golden set
lets you regress-test that new chunking, retrieval, or prompt changes don't degrade
answer quality, and that the system says "I don't know" when it should.

**file:line:** `src/evaluation/golden_set.json` (all 30 questions),
`src/evaluation/run_eval.py:334-396` (`evaluate_golden_set` runs the whole set),
`src/evaluation/run_eval.py:170-278` (`evaluate_single_question`).

**# INTERVIEW:** "We test with a golden set including unanswerable questions, so a
regression that makes the model hallucinate shows up as a test failure."

**Analogy:** Like a regression test suite that runs 30 known-good provisioning
scenarios every night — including scenarios that should fail (invalid ICCID, barred
subscriber) to prove the system rejects them correctly.

---

## 6.2 Retrieval metrics implemented from scratch

**Typical web app (X):** No retrieval evaluation; you eyeball whether search
results look reasonable.

**This repo (Y):** `RetrievalMetrics` implements hit-rate@k, recall@k, precision@k,
MRR, nDCG@k, and average precision — all from scratch (no external metrics library).

**Because (Z):** Retrieval quality drives answer quality. These metrics quantify
whether the right chunks surface for each golden-set question, letting you compare
dense vs. hybrid vs. RRF objectively.

**file:line:** `src/evaluation/metrics.py:16-241` (`RetrievalMetrics`),
`src/evaluation/metrics.py:484-508` (`RAGEvaluator.evaluate_retrieval`).

**Analogy:** Like measuring provisioning accuracy: of 100 activations, how many
had the right plan (precision), how many of the activations that should have
succeeded did (recall), and how fast was the first correct activation (MRR).

---

## 6.3 LLM-as-judge for generation quality

**Typical web app (X):** Generation quality isn't measured automatically.

**This repo (Y):** `GenerationMetrics` uses an LLM judge (the env-selected provider,
mock in CI) to score answer relevance, faithfulness, and contradiction on a 0-10
rubric, normalized to 0-1. Falls back to keyword-overlap heuristics when no judge
is configured.

**Because (Z):** "Is the answer faithful to the retrieved context?" can't be
checked by exact string match. LLM-as-judge approximates human evaluation at scale,
with heuristic fallbacks so CI stays cheap and deterministic.

**file:line:** `src/evaluation/metrics.py:244-474` (`GenerationMetrics`),
`src/evaluation/metrics.py:510-551` (`RAGEvaluator.evaluate_generation`),
`src/evaluation/run_eval.py:32-50` (`make_llm_judge`).

**# GAP:** The LLM judge is not ground truth — it's another model with its own
biases. Documented as a limitation.

**Analogy:** Like having a senior engineer review each activation ticket for
correctness (LLM judge), backed by a checklist of must-have fields (heuristic
fallback) when the senior engineer isn't available.

---

## 6.4 Ablation studies compare pipeline configurations

**Typical web app (X):** You change one thing, redeploy, and hope metrics improve.

**This repo (Y):** `run_ablation_study` runs the golden set under multiple configs
(baseline, chunking improvement, hybrid retrieval, hybrid+RRF, reranking, advanced
prompting) and compares aggregate metrics side by side.

**Because (Z):** You need to know *which* component actually moved the metric —
was it reranking, or the new chunk size? Ablation isolates each variable.

**file:line:** `src/evaluation/run_eval.py:469-590` (`run_ablation_study`),
`src/evaluation/run_eval.py:646-801` (`run_comparison_evaluation`).

**Analogy:** Like A/B testing a new SIM activation script against the old one on
the same 100 subscribers, measuring which change actually improved activation
success.

---

# ============================================================
# 7. DEPLOYMENT
# ============================================================

## 7.1 Docker image: slim base, CPU-only torch, non-root user

**Typical web app (X):** `FROM python:latest`, installs GPU torch (multi-GB image),
runs as root, no healthcheck.

**This repo (Y):** `python:3.11-slim` with CPU-only torch (keeps image small and
affordable), creates `appuser` and runs `USER appuser`, copies `.env`-free code
(`.dockerignore` keeps `.env`/`.git`/caches out), and sets `PYTHONUNBUFFERED` and
`PYTHONDONTWRITEBYTECODE`.

**Because (Z):** Smaller images pull faster and cost less to store/non-root limits
the blast radius if the container is compromised. CPU-only torch is sufficient for
MiniLM/cross-encoder at this scale — no GPU needed.

**file:line:** `Dockerfile:1-31` (base, deps, non-root user),
`Dockerfile:20-22` (CPU-only torch install).

**Analogy:** Like deploying a provisioning gateway in a hardened container that runs
as a non-root service account, with only the CPU it needs — no GPU, no root, no
extraneous tools an attacker could use.

---

## 7.2 Healthcheck via the `/health` endpoint

**Typical web app (X):** No container healthcheck; orchestration can't tell if the
app is actually serving.

**This repo (Y):** `HEALTHCHECK` hits `http://localhost:8000/health` (Python
`urllib`, since `curl` isn't in the slim image), with 30s interval, 5s timeout,
30s start period, 3 retries. `docker-compose.yml` mirrors it and the UI
`depends_on: api: condition: service_healthy`.

**Because (Z):** Orchestration (Docker Compose, Kubernetes) needs to know the API
is ready before sending traffic, and to restart it if `/health` fails. The UI
waits for the API to be healthy before starting.

**file:line:** `Dockerfile:37-38`, `docker-compose.yml:20-26`,
`src/api/main.py:477-499` (`/health` endpoint).

**Analogy:** Like a provisioning gateway that exposes a "ready" port the load
balancer polls — traffic only flows once the gateway can answer, and it's pulled
out if it stops responding.

---

## 7.3 docker-compose: api + ui services, persistent volumes, restart policy

**Typical web app (X):** Run `uvicorn` and `streamlit` manually on a laptop; data
lost on restart; no restart policy.

**This repo (Y):** Two services (`api` on 8000, `ui` on 8501) with volumes for
`./data`, `./uploads`, `./logs` (persistent across restarts), `restart:
unless-stopped`, env vars for `LOG_LEVEL` and `LLM_PROVIDER`, and an optional
commented-out NGINX reverse proxy for production.

**Because (Z):** Persistent volumes keep the vector store and embedding cache
across container restarts. `unless-stopped` keeps services running after crashes.
The UI depends on the API being healthy.

**file:line:** `docker-compose.yml:1-41` (both services, volumes, healthcheck,
depends_on).

**Analogy:** Like a field provisioning setup with separate activation-gateway and
operator-console services, shared persistent storage for provisioning records, and
auto-restart after a failure.

---

# ============================================================
# 8. THE LEARNER'S WEAK AREAS — CALLOUTS
# ============================================================

## 8.1 Concurrency / async

**Typical assumption (X):** "Async/await means everything is concurrent and fast."

**This repo (Y):** The NVIDIA provider uses a *blocking* `httpx.Client` and
`time.sleep` for backoff — synchronous I/O inside potentially async callers. No
async client, no circuit breaker, no jitter on backoff (thundering-herd risk on
429). `ToolRegistry` uses a `ThreadPoolExecutor(max_workers=4)` for tool
execution. `Request ID` uses a `ContextVar` (per-task, not truly thread-safe
across threads). The vector store and rate limiter have no locking.

**file:line:** `providers/llm_nvidia.py:42-44` (GAP: blocking sync client),
`providers/llm_nvidia.py:382-387` (`_sleep` blocking `time.sleep`),
`src/agent/tools.py:86` (`ThreadPoolExecutor`),
`src/observability.py:27` (`request_id_ctx` ContextVar),
`src/store/vector_store.py:46-47` (GAP: no locking),
`src/api/main.py:195-250` (RateLimiter: no lock, per-worker state).

**# INTERVIEW:** "This system is mostly synchronous. If you need true concurrency
(many simultaneous queries), you'd add an async HTTP client, locking on shared
state, and a distributed rate limiter."

**Analogy:** Like a single field technician handling activations one at a time —
fine for low volume, but you'd need a team (thread pool) or an automated switch
(async) to handle a surge.

---

## 8.2 Auth / multi-tenancy

**Typical assumption (X):** "There's login, sessions, and per-customer data
isolation."

**This repo (Y):** Single `API_TOKEN` for all mutating endpoints; `X-Access-Level`
header (0-3) for role checks — advisory unless bearer auth is on. One shared corpus;
no per-customer isolation. Department is inferred from filename keywords (brittle).

**file:line:** `src/api/main.py:268-305` (auth + roles),
`src/api/main.py:290-291` (header trust caveat),
`src/ingestion/loaders.py:118-153` (department inference).

**# INTERVIEW:** "Authentication is a single bearer token plus role flags; there's
no multi-tenancy — every user sees the same corpus. Don't claim per-customer
isolation that doesn't exist."

**Analogy:** Like one provisioning office with a badge system and role-based
counters — everyone works from the same subscriber database; there's no separate
database per customer.

---

## 8.3 Observability

(See section 5.) JSON logs, correlation IDs, per-stage traces, `/metrics`,
`/traces`, `/provider`, secret redaction — all present, but in-memory only (no
Prometheus push, no distributed tracing backend).

---

## 8.4 Evals

(See section 6.) Golden set, retrieval metrics, LLM-as-judge, ablation studies —
present, but the LLM judge isn't ground truth and evals run sequentially (no
parallelism for large sets: `src/generation/answer.py:423-429`).

---

## 8.5 Deployment

(See section 7.) Docker, non-root, healthcheck, volumes, restart policy — present.
No CI/CD pipeline file, no orchestration beyond docker-compose, no autoscaling.

---

## 8.6 Cost / latency tradeoffs

| Tradeoff | Where it shows | Production choice | Why |
|----------|---------------|-------------------|-----|
| Reranker latency vs. answer quality | `src/retrieval/reranker.py` | Cross-encoder on top-k only | Better relevance; model is too slow for the whole corpus |
| Hybrid (BM25+dense) vs. dense-only | `src/retrieval/hybrid.py` | Hybrid default (`use_hybrid=True`) | Better recall for acronym-heavy telecom questions |
| Streaming vs. one-shot | `src/generation/answer.py:330-407` | Streaming (`/ask/stream`) | Lower *perceived* latency; same total work |
| Context budget vs. completeness | `src/generation/answer.py:118-186` | 2000-token cap | Fits model window; bounds cost |
| Mock vs. real provider | `providers/llm.py:377-401` | Env-driven (`LLM_PROVIDER`) | Dev/tests free; prod pays for real tokens |

**Analogy:** Like choosing between a fast automated activation (lower quality) and a
technician-led activation (higher quality, slower, costlier) based on the customer's
SLA.

---

## 8.7 Customer-facing communication

**Typical web app (X):** Error messages expose stack traces, internal IDs, and
sometimes keys; no feedback loop from users.

**This repo (Y):** Client-facing errors are generic (HTTPException detail strings
without internals). `/provider` tells the user whether mock or real is active.
Feedback collection (`/feedback`, thumbs up/down) captures user sentiment. The
grounded prompt forces citations and refusal ("I don't have enough information…")
instead of hallucination.

**file:line:** `src/api/main.py:643-645` (generic 500 detail),
`src/api/main.py:467-475` (`/provider`),
`src/api/main.py:1014-1062` (`/feedback`),
`src/generation/prompt.py:148-188` (grounded templates with refusal rule),
`src/generation/answer.py:213-214` (insufficiency fallback).

**# INTERVIEW:** "Never expose internals or keys to the client; always cite sources;
and when the evidence isn't there, say so rather than guessing."

**Analogy:** Like a customer support portal that tells a subscriber "activation
failed" without exposing internal error codes, and offers a feedback option after
every interaction.

---

# ============================================================
# QUICK REFERENCE — PATTERN → FILE:LINE
# ============================================================

| Pattern | Key file:line |
|---------|--------------|
| Fail-fast config validation | `src/config.py:42`, `src/config.py:177-183` |
| Immutable settings | `src/config.py:70-71` |
| Retry + backoff | `src/embedding/embedder.py:73-113` |
| Rate limiting (embedder) | `src/embedding/embedder.py:116-151` |
| Embedding cache | `src/embedding/embedder.py:240-270`, `:300-350` |
| Mock fallback | `src/embedding/embedder.py:346-350`, `src/generation/answer.py:208-215` |
| ChromaDB → in-memory fallback | `src/store/vector_store.py:67-73` |
| Key pool rotation/failover | `src/key_pool.py:54-174`, `:149-157` |
| Key expiry monitoring + SMTP alert | `src/key_monitor.py` |
| JSON logs + redaction | `src/observability.py:42-89` |
| Correlation ID | `src/observability.py:27-37` |
| Per-stage tracing | `src/observability.py:107-162` |
| Metrics | `src/observability.py:166-221` |
| Query cache | `src/api/main.py:126-189` |
| Rate limiter | `src/api/main.py:195-250` |
| Bearer token auth | `src/api/main.py:268-284` |
| Role-based access | `src/api/main.py:287-305` |
| Path traversal guard | `src/api/main.py:309-330` |
| Middleware logging + `X-Request-Id` | `src/api/main.py:411-438` |
| `/metrics` | `src/api/main.py:451-456` |
| `/traces` | `src/api/main.py:459-464` |
| `/provider` | `src/api/main.py:467-475` |
| `/health` | `src/api/main.py:477-499` |
| `/ask` with trace stages | `src/api/main.py:501-645` |
| `/ask/stream` (SSE) | `src/api/main.py:647-709` |
| `/ingest` (admin) | `src/api/main.py:779-861` |
| Document delete (cascade, admin) | `src/api/main.py:863-901` |
| Upload (allowlist, admin) | `src/api/main.py:903-921` |
| Reindex (admin) | `src/api/main.py:956-969` |
| `/agent/ask` (allow-listed tools) | `src/api/main.py:978-1011` |
| Token budget context | `src/generation/answer.py:118-186` |
| Grounded prompt templates | `src/generation/prompt.py:148-188` |
| LLMProvider ABC | `providers/llm.py:46-108` |
| Mock provider | `providers/llm.py:111-219` |
| Provider factory (env-driven) | `providers/llm.py:312-401` |
| NVIDIA provider (retries, rotation, redaction) | `providers/llm_nvidia.py` |
| Tool registry (validate→auth→run→record) | `src/agent/tools.py:58-136` |
| Admin-only tools | `src/agent/tools.py:165-250` |
| Agent step cap + repeat cap | `src/agent/agent.py:29-30`, `:108-151` |
| Upload validate (extensions, size) | `src/ingestion/manager.py:98-110` |
| Upload dedupe + versioning | `src/ingestion/manager.py:114-148` |
| Delete cascade | `src/ingestion/manager.py:191-199` |
| Reindex | `src/ingestion/manager.py:201-223` |
| Golden set (30 questions) | `src/evaluation/golden_set.json` |
| Retrieval metrics | `src/evaluation/metrics.py:16-241` |
| LLM-as-judge generation metrics | `src/evaluation/metrics.py:244-474` |
| Golden-set eval + ablation | `src/evaluation/run_eval.py` |
| Docker (slim, CPU torch, non-root) | `Dockerfile:1-31` |
| Healthcheck | `Dockerfile:37-38` |
| docker-compose (api+ui, volumes, restart) | `docker-compose.yml:1-41` |

---

# Closing note for the learner

Every pattern above answers the same question a production system must answer:
*"If this fails, or scales, or is attacked, what happens?"* The web-app version
usually assumes the happy path. This repo's production version assumes failures,
rate limits, missing keys, untrusted LLM output, and curious operators — and
handles each explicitly. Use the `file:line` references to verify each claim
yourself before repeating it in an interview.
