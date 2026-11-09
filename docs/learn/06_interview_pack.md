# 06 — FDE Interview Pack: rag-academy RAG System

**Branch:** `learn/annotated` | **Repo:** `C:\projects\rag-project` | **Status:** Draft (doc only — no source code changed)

**Audience:** You — senior QA/automation engineer (~10 yrs QA/CI/CD/test frameworks) moving into an AI Solutions Engineer role in telecom/eSIM/IoT. You know program flow and test pipelines, not line-by-line code or production architecture.

**Purpose:** An interview preparation kit for field-development-engineer (FDE) interviews, written specifically for THIS repo. Every answer is grounded in the actual annotated source (`file:line` citations), docs/learn/01-04, and the recon findings in docs/learn/_plan.md. Nothing here is invented — where I could not verify a claim directly, I mark it **[INFERRED]**.

**Constraint honored:** no source code was changed to produce this document. It lives entirely under `docs/learn/`.

**How to read the `# INTERVIEW:` blocks:** these are say-it-out-loud points. Say them aloud in your own words before the interview so they come naturally under pressure.

**Telecom/provisioning analogies used throughout:** think of ingestion as loading a tariff/circuit into an OSS, the vector store as the circuit-inventory database, retrieval as querying that inventory for a fault symptom, answer generation as the NOC engineer writing a trouble-ticket resolution quoting only the inventory record, caching as a MAC/ARP fast-lookup table, rate limiting as a policer on an access port, and auth as RADIUS at the network edge.

---

## Contents

1. [25+ interview questions (first-person model answers)](#1-questions)
2. [2-minute spoken walkthrough](#2-2-minute-walkthrough)
3. [5-minute spoken walkthrough](#3-5-minute-walkthrough)
4. [5 customer scenario prompts](#4-customer-scenarios)
5. [10 break-it drills](#5-break-it-drills)
6. [Answer key (drills)](#6-answer-key)
7. [Honest weaknesses — how to talk about them candidly](#7-honest-weaknesses)
8. [UNVERIFIED items](#8-unverified)

---

<a id="1-questions"></a>
# 1. 25+ interview questions (first-person model answers)

Each answer is in first person ("I chose X because..."), references your QA/automation → AI SE background where relevant, and cites `file:line` for every behavioral claim. The `# INTERVIEW:` blocks are say-it-out-loud points.

---

### Design choices

#### Q1. Why did you build this as RAG instead of fine-tuning a model on the telecom documents?

**Model answer (first person):**

I chose RAG (retrieval-augmented generation) because the system's stated purpose is to answer from verified telecom documents (LTE/5G, Roaming, SIM Security, eSIM) with citations, and never to speculate beyond the source material (`CLAUDE.md` Core Purpose). Fine-tuning would bake knowledge into the model's weights, making it impossible to cite a specific document section and easy to hallucinate. With RAG, the answer generator only sees retrieved chunks, so every claim is traceable back to a source (`src/generation/prompt.py:351` `create_grounded_prompt`, `src/generation/answer.py:217` `generate_answer_with_context`).

The pipeline is load → clean → chunk → embed → store, then retrieve → (rerank) → generate (`docs/learn/_plan.md:21`, `src/api/main.py:396` `initialize_rag_pipeline`). Chunks carry metadata like `source`, `doc_version`, and `is_current`, so the answer can cite exactly which document section supported it (`src/ingestion/loaders.py:25` `DocumentLoader`, `src/store/vector_store.py:236` `list_sources`).

If the retrieved evidence is insufficient, the grounded prompt forces a refusal ("I don't have enough information…") rather than a guess (`src/generation/prompt.py:148-188` templates, `src/generation/answer.py:213-214` insufficiency fallback). That's the "no speculation" rule in practice (`CLAUDE.md`).

**QA/automation angle:** This is like asserting against an actual log line instead of relying on memory of what the system *usually* does — the evidence is right there, cited.

**# INTERVIEW:** Say it out loud — "RAG lets us cite the exact document section for every answer and refuse when evidence is missing, which fine-tuning can't do."

**file:line:** `CLAUDE.md:3-6`, `src/generation/prompt.py:351`, `src/generation/answer.py:217`, `src/api/main.py:396`, `docs/learn/_plan.md:21`.

---

#### Q2. Why two retrieval paths — dense (semantic) and sparse (BM25) — fused with RRF, instead of just one?

**Model answer (first person):**

I kept both because neither signal alone is sufficient for telecom QA. Dense retrieval embeds the query and finds semantically similar chunks (`src/retrieval/retriever.py:51` `Retriever`, `:98` `retrieve`), which is great for conceptual questions like "what is an EPS bearer?". Sparse BM25 finds exact keyword/acronym matches (`src/retrieval/sparse_retriever.py:53` `BM25Retriever`, `:213` `create_bm25_retriever`), which matters for telecom acronyms (TAC, IMEI, EPS, S1-AP) that dense similarity can under-rank. Hybrid search fuses them with Reciprocal Rank Fusion (RRF), which combines ranked lists by position rather than raw score, so the different score scales combine fairly (`src/retrieval/hybrid.py:51` `HybridRetrieverRRF`, `:164` `create_hybrid_retriever_rrf`, `:327` `reciprocal_rank_fusion`).

The default query path uses hybrid (`use_hybrid=True`) (`src/api/main.py:62-67` `QueryRequest` defaults). The gap: RRF's `weight_dense`/`weight_sparse` are stored but not actually applied in the fusion math (`docs/learn/02_architecture.md:136` note), so the weights are decorative today.

**# INTERVIEW:** Say it out loud — "Dense finds meaning, sparse finds exact acronyms, and RRF fuses the two ranked lists by position so the score scales don't fight each other."

**file:line:** `src/retrieval/retriever.py:51,98`, `src/retrieval/sparse_retriever.py:53,213`, `src/retrieval/hybrid.py:51,164,327`, `src/api/main.py:62-67`.

---

#### Q3. Why is the `Settings` object frozen/immutable and loaded only from environment variables?

**Model answer (first person):**

`Settings` is a frozen dataclass (`frozen=True`) (`src/config.py:69`). Once `get_settings()` runs at startup, no module can mutate timeouts, keys, log level, or the API token mid-flight (`src/config.py:127` `get_settings`). Everything comes from `os.environ` — there is no checked-in config file (`.env` is gitignored) — so the same Docker image deploys to dev/test/prod by changing env vars alone, twelve-factor style (`docs/learn/_plan.md:13`, `src/config.py:164-212`).

The fail-fast rule: if `LLM_PROVIDER=nvidia` but no NVIDIA key is set, `get_settings()` raises `MissingConfigError` and the process exits before serving (`src/config.py:40` `MissingConfigError`, `:176-182` key check). A RAG system without an LLM key cannot answer anything, so crashing fast with a clear message beats intermittent 500s per request.

**# INTERVIEW:** Say it out loud — "Settings is frozen so no module can change config mid-flight; all values come from env vars at startup, and a missing required secret crashes fast with a clear message."

**file:line:** `src/config.py:40,69,127,176-182`, `docs/learn/_plan.md:13`.

---

#### Q4. Why does the NVIDIA key pool rotate keys round-robin with failover, instead of using one key?

**Model answer (first person):**

Production LLM traffic must survive a single key being revoked, expired, or rate-limited without a page to on-call. `KeyPool` holds multiple NVIDIA keys from `NVIDIA_API_KEYS` (comma-separated), selects them round-robin under a `threading.Lock`, and on a 401/403 marks the key bad and rotates to the next, bounded by pool size (`src/key_pool.py:136` `KeyPool`, `:165` `_lock`, `:337` `current_key`, `:428` `mark_bad`). Full keys are never logged or stored — only a `first4…last4` fingerprint (`src/key_pool.py:65` `fingerprint`, `:92`). The registry file `data/key_registry.json` (gitignored, local) holds expiry metadata (`src/key_pool.py:487` `load_registry`, `:513` `save_registry`, `:62` `DEFAULT_REGISTRY_PATH`).

This spreads usage so no single key burns its quota first, and failover keeps the generation path available (`providers/llm_nvidia.py:420` `_post` rotates on 401/403 via `key_pool.mark_bad`).

**# INTERVIEW:** Say it out loud — "Round-robin key selection with a lock; on a 401/403 we mark the key bad and try the next, bounded by pool size; full keys are never logged, only fingerprints."

**file:line:** `src/key_pool.py:136,165,337,428,65,92,487,513,62`, `providers/llm_nvidia.py:420`.

---

#### Q5. Why both a mock and a real NVIDIA provider, selected by env var?

**Model answer (first person):**

The `LLMProvider` ABC lets the generation code stay provider-agnostic (`providers/llm.py:46` `LLMProvider`). The mock provider (`providers/llm.py:111` `MockLLMProvider`) parses the prompt for Context/Question markers and returns a canned answer with no network call — used for tests and local dev only, zero cost, no key. The NVIDIA provider (`providers/llm_nvidia.py:226` `NVIDIAProvider`) calls `integrate.api.nvidia.com/v1` with Bearer auth, retries, backoff, key-pool failover, streaming, usage tracking, and redaction (`:68` `DEFAULT_BASE_URL`, `:69` `DEFAULT_MODEL`).

Selection is env-driven: `create_llm_provider_from_env()` reads `LLM_PROVIDER` (`providers/llm.py:377`), defaulting to `mock` (`src/config.py:100`). So CI and local dev never touch the network or spend tokens, while production uses the real model. The `/provider` endpoint exposes which is active, so a mock can't silently substitute in production (`src/api/main.py:639` `/provider`).

**# INTERVIEW:** Say it out loud — "The provider is chosen purely by env var; mock is free and offline for dev/tests, NVIDIA is real and paid for prod, and `/provider` makes a mock impossible to hide."

**file:line:** `providers/llm.py:46,111,377`, `providers/llm_nvidia.py:226,68,69`, `src/config.py:100`, `src/api/main.py:639`.

---

#### Q6. Why is the agent's tool use "controlled" — allow-list + validation + auth + timeout — instead of letting the LLM call tools freely?

**Model answer (first person):**

LLM output is untrusted, so the agent never gets raw filesystem/DB/OS access. `ToolRegistry.execute()` is the single choke point: it validates input against a Pydantic schema, authorizes against `is_admin`, runs the handler under a 30-second timeout via a `ThreadPoolExecutor(max_workers=4)`, and records every call in an audit history (`src/agent/tools.py:205` `ToolRegistry`, `:272` `execute`, `:171` `requires_admin`, `:86` `ThreadPoolExecutor`). The LLM may *propose* a tool; the application *validates + authorizes + executes* (`docs/learn/04_gap_bridge.md:350-377`).

The agent plans via deterministic-heuristic first, falling back to an LLM planner only when a provider is present (`src/agent/agent.py:50` `RAGAgent`, `:142` `plan`). Execution is bounded: `MAX_STEPS = 5` and `MAX_REPEAT_SAME_CALL = 2` (`src/agent/agent.py:54-55`), so the agent can't loop forever. `generate_report` is admin-only (`src/agent/tools.py:468`).

**# INTERVIEW:** Say it out loud — "The LLM proposes, the application disposes — every tool call is schema-validated, authorized, timed, and audited before execution, with steps capped at 5 and repeats at 2."

**file:line:** `src/agent/tools.py:205,272,171,86,468`, `src/agent/agent.py:50,142,54-55`.

---

#### Q7. Why does ingestion dedupe by SHA-256 and version documents instead of just appending?

**Model answer (first person):**

`DocumentManager.upload()` dedupes by SHA-256 content hash — re-uploading identical bytes returns `duplicate` status and is a no-op (`src/ingestion/manager.py:105` `DocumentManager`, `:98` `validate` returns the hash, `:114` `upload` dedupe). When the same filename comes back with different content, old chunks are marked `is_current=False` and `doc_version` is bumped (`src/store/vector_store.py:281` `mark_source_current`, `:170` metadata `doc_version` in manager). Delete-then-write, not upsert (`src/store/vector_store.py:519` `delete_by_source`).

This prevents stale or duplicate chunks from accumulating, which would otherwise make retrieval return contradictory answers for the same question — exactly the inconsistency a QA engineer would catch in a regression suite.

**# INTERVIEW:** Say it out loud — "Re-uploading identical bytes is a no-op (dedupe by SHA-256); a changed file supersedes the old version and bumps doc_version — delete-then-write, not upsert."

**file:line:** `src/ingestion/manager.py:105,98,114`, `src/store/vector_store.py:281,170,519`.

---

#### Q8. Why is the answer grounded with citations and a refusal path instead of letting the model answer freely?

**Model answer (first person):**

The project's teaching rule is "No Speculation": answers must be grounded in retrieved chunks, and if information isn't in the source documents, the system must say so (`CLAUDE.md`). The grounded prompt templates (`grounded_v1`/`v2`, `citation_focused`) instruct the LLM to answer only from context, cite `[source_filename]`, and refuse when evidence is insufficient (`src/generation/prompt.py:148-188`). `generate_answer_with_context` builds context within a 2000-token budget and returns a `GenerationResult` with sources (`src/generation/answer.py:47` `AnswerGenerator`, `:217`, `:16` `GenerationResult`).

The golden set even includes "unanswerable" questions (e.g. CEO salary) whose expected answer is "cannot be determined" — testing the refusal behavior, not just retrieval (`docs/learn/04_gap_bridge.md:625-648`, `src/evaluation/golden_set.json`).

**# INTERVIEW:** Say it out loud — "Cite sources for every claim, and when the evidence isn't there, say 'I can't determine that from the available documents' rather than guessing."

**file:line:** `CLAUDE.md`, `src/generation/prompt.py:148-188`, `src/generation/answer.py:47,217,16`, `docs/learn/04_gap_bridge.md:625-648`, `src/evaluation/golden_set.json`.

---

### Tradeoffs

#### Q9. What's the tradeoff with the reranker — latency versus answer quality?

**Model answer (first person):**

The cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is more accurate than first-pass cosine/BM25, but it's too slow to run over the whole corpus. So it runs only on the top-k candidates retrieved by hybrid search (`src/retrieval/reranker.py:53` `Reranker`, `:100` `rerank`, `:227` `create_reranker`), and retrieval fetches `top_k × 2` candidates when reranking is on to give the re-ranker a larger pool (`src/api/main.py:809`).

The tradeoff: better relevance for the chunks the generator reads, at the cost of an extra model pass and added latency. It's optional (`use_reranker` flag, default `True`) so you can trade quality for speed per request (`src/api/main.py:65` `QueryRequest.use_reranker`).

**# INTERVIEW:** Say it out loud — "The re-ranker lifts the most relevant chunks to the top before the generator reads them, but it only runs on the top-k candidates because a cross-encoder over the whole corpus would be too slow."

**file:line:** `src/retrieval/reranker.py:53,100,227`, `src/api/main.py:809,65`.

---

#### Q10. What's the tradeoff with the 2000-token context budget versus completeness?

**Model answer (first person):**

`_build_context_within_budget` bounds context to `max_tokens=2000` (chars/token estimate 4.0), taking highest-ranked chunks first and stopping before the budget (`src/generation/answer.py:118` `_build_context_within_budget`, `:78` `chars_per_token = 4.0`, `:220` `max_context_tokens=2000`). This keeps prompts within the model's context window and bounds per-answer token cost — important because NVIDIA NIM is billed per token.

The tradeoff: character-based token estimation is imprecise vs. a real tokenizer, and `format_context` truncates by character count (`docs/learn/04_gap_bridge.md:484`), so the budget is an approximation. A long answer may lose lower-ranked chunks that could have helped.

**# INTERVIEW:** Say it out loud — "We cap context at ~2000 estimated tokens to fit the model window and bound cost, but character-based estimation is imprecise, so completeness is traded for safety."

**file:line:** `src/generation/answer.py:118,78,220`, `docs/learn/04_gap_bridge.md:484`.

---

#### Q11. What's the tradeoff with streaming (`/ask/stream`) versus one-shot (`/ask`)?

**Model answer (first person):**

Streaming emits SSE token events as they arrive, then a `done` event with sources/usage, then `[DONE]` (`src/api/main.py:893` `/ask/stream`, `:964` `done` event, `:971` `[DONE]`). The non-streaming `/ask` returns the complete answer in one JSON (`src/api/main.py:714`).

The tradeoff: streaming cuts *perceived* latency (the user sees the first token immediately) but the total work is the same — retrieval + rerank + generation still happen before tokens flow. Streaming answers are also *not* cached, so a repeated streamed query pays full pipeline cost each time (`src/api/main.py:921-922` gap). For an FDE interview-prep UI, perceived responsiveness matters, so streaming is the default chat experience.

**# INTERVIEW:** Say it out loud — "Streaming lowers perceived latency by showing tokens as they arrive, but total work is unchanged and streamed answers aren't cached."

**file:line:** `src/api/main.py:893,714,964,971,921-922`.

---

#### Q12. What's the tradeoff with ChromaDB embedded (linear scan) versus a managed vector database?

**Model answer (first person):**

ChromaDB runs embedded (single-process) at `data/chroma_db`, with an in-memory cosine-similarity fallback when Chroma is unavailable (`src/store/vector_store.py:76` `VectorStore`, `:67-73` fallback per docs, `:334` `search`). Search is a linear scan over all vectors — no HNSW index, no sharding, no replication (`docs/learn/02_architecture.md:271-293`).

The tradeoff: at demo/eval scale (tens of documents) linear scan is fast enough and keeps the system runnable offline with zero infrastructure. It does not scale beyond a modest corpus or a single node — you'd migrate to a managed vector DB (HNSW, sharding) before a nationwide rollout, like moving from a field-office card catalog to an HLR-scale database.

**# INTERVIEW:** Say it out loud — "Embedded ChromaDB with linear scan is fine for tens of documents and zero infra, but it's single-node and won't scale beyond a modest corpus."

**file:line:** `src/store/vector_store.py:76,67-73,334`, `docs/learn/02_architecture.md:271-293`.

---

#### Q13. What's the tradeoff with in-memory cache and rate limiter versus Redis-backed?

**Model answer (first person):**

`QueryCache` is an in-memory FIFO dict (max 200 entries, no TTL) (`src/api/main.py:126` `QueryCache`, `:260` instance `max_size=200`). `RateLimiter` is a per-process sliding window (30 req / 60 s) (`src/api/main.py:195` `RateLimiter`, `:261` instance). Both reset on restart and are *not* shared across uvicorn workers — each worker has its own counter, so a client can multiply its limit by worker count (`docs/learn/02_architecture.md:298-302`).

The tradeoff: simplicity and zero infra (no Redis to operate) versus correctness under scale. Behind a load balancer with multiple workers, the cache and rate limit are effectively per-worker, so a determined client can exceed the intended quota and stale cached answers persist across restarts.

**# INTERVIEW:** Say it out loud — "Cache and rate limiter are per-process in-memory, so they reset on restart and don't share state across uvicorn workers — a client can multiply its limit by worker count."

**file:line:** `src/api/main.py:126,260,195,261`, `docs/learn/02_architecture.md:298-302`.

---

#### Q14. What's the tradeoff with the blocking NVIDIA client (`time.sleep` backoff) versus async?

**Model answer (first person):**

The NVIDIA provider uses a blocking `httpx.Client` and `time.sleep` for backoff — no async client, no jitter (`providers/llm_nvidia.py:42-44` blocking sync client gap per docs, `:382-387` `_sleep` `time.sleep`, `:420` `_post` retries). This is simplest to reason about and fine for low concurrency, but it blocks the event loop if called from an async context, and fixed backoff without jitter risks a thundering herd on 429.

The tradeoff: simplicity and predictability versus true concurrency. For many simultaneous queries you'd add an async HTTP client, locking on shared state, jittered backoff, and a circuit breaker.

**# INTERVIEW:** Say it out loud — "The NVIDIA client is blocking with fixed `time.sleep` backoff — simple and predictable, but it blocks the event loop and risks a thundering herd on 429 without jitter."

**file:line:** `providers/llm_nvidia.py:42-44,382-387,420`, `docs/learn/04_gap_bridge.md:807-812`.

---

### Failure scenarios

#### Q15. What happens if an NVIDIA API key expires mid-production?

**Model answer (first person):**

NVIDIA keys don't expose expiry over the API, so expiry is tracked locally in `data/key_registry.json` (`src/key_pool.py:34` design rule). The `key_monitor` CLI registers key expiries, live-probes keys against NVIDIA `/models`, and emails SMTP alerts when keys expire soon (default `key_warn_days=14`) (`src/key_monitor.py:53` `register_key`, `:86` `probe_key`, `:125` `send_alert`, `:166` `main`; `src/config.py:117` `key_warn_days`).

If a key returns 401/403, `mark_bad` flips it to "bad" and `current_key` rotates to the next active key (`src/key_pool.py:428` `mark_bad`, `:337` `current_key`). If *all* keys are bad or expired, `current_key` raises `RuntimeError` ("No active API keys in pool") and generation fails (`src/key_pool.py:340`). There's no automatic key acquisition — an operator must register a new key via `key_monitor`. So expiry is an operational, not automatic, concern.

**# INTERVIEW:** Say it out loud — "Expiry is tracked locally because NVIDIA doesn't expose it over the API; a 401/403 rotates to the next key, but if every key is bad or expired the generation path errors out until an operator registers a new key."

**file:line:** `src/key_pool.py:34,428,337,340`, `src/key_monitor.py:53,86,125,166`, `src/config.py:117`.

---

#### Q16. What happens if ChromaDB is unavailable at startup or runtime?

**Model answer (first person):**

`VectorStore` tries `chromadb.PersistentClient` at `data/chroma_db`; if ChromaDB is absent or fails, it falls back to in-memory lists with cosine similarity (`src/store/vector_store.py:76` `VectorStore`, `:44` `_initialize_client`, `:81` fallback per docs, `:67-73`). The system stays runnable — retrieval still works — but the in-memory fallback is lost on restart (`docs/learn/02_architecture.md:306`).

So a missing ChromaDB doesn't crash the API; it degrades gracefully to reduced persistence. The gap: silent fallback can mask a real problem (you might not realize vectors aren't persisting), which is why `get_stats()` reports `storage_type` so you can tell ChromaDB vs. fallback (`src/store/vector_store.py:620` `get_stats`).

**# INTERVIEW:** Say it out loud — "ChromaDB unavailability falls back to an in-memory vector store — the system keeps serving, but vectors are lost on restart, so check `get_stats().storage_type` to know which backend is active."

**file:line:** `src/store/vector_store.py:76,44,81,67-73,620`, `docs/learn/02_architecture.md:306`.

---

#### Q17. What happens if the LLM returns invalid JSON in the agent planner?

**Model answer (first person):**

The LLM planner (`_plan_with_llm`) asks the model to return one JSON tool call, extracts it with a greedy brace-regex, and validates the tool name is in the allow-list (`src/agent/agent.py:209` `_plan_with_llm`, `:238` regex, `:242` allow-list check). If it returns no JSON, invalid JSON, or a non-allow-listed tool, it raises `ValueError`, which `plan()` catches and falls back to the heuristic planner (`src/agent/agent.py:163-168`).

So a malformed planner response degrades gracefully to keyword-based routing — the agent never crashes or executes an unauthorized tool. The gap: the greedy regex can over-match if the LLM's response contains multiple/nested braces (`src/agent/agent.py:228-229` note), and the heuristic planner's filename regex may return `""`, producing a tool call with an empty filename that fails at execution (`src/agent/agent.py:195-197`).

**# INTERVIEW:** Say it out loud — "If the LLM planner returns bad JSON or an unauthorized tool, we fall back to the deterministic heuristic planner — the agent never executes an unvalidated or unallow-listed tool."

**file:line:** `src/agent/agent.py:209,238,242,163-168,228-229`, `src/agent/agent.py:195-197`.

---

#### Q18. What happens if a client floods `/ask`? What breaks first?

**Model answer (first person):**

`RateLimiter.is_allowed(client_id)` enforces 30 requests per 60 seconds per client and returns 429 when exceeded (`src/api/main.py:195` `RateLimiter`, `:261` instance, checked at `:768` in `/ask`). `client_id` defaults to `X-Forwarded-For` or the request host (`src/api/main.py:767`).

What breaks first: the rate limiter is per-process in-memory, so behind multiple uvicorn workers each worker tracks its own counter — a client can effectively send 30 × worker_count requests per minute (`docs/learn/02_architecture.md:300`). There's no client eviction, so memory grows with distinct clients (`src/api/main.py:206-209` docstring). The LLM token budget and pipeline capacity are protected per-worker, but not cluster-wide.

**# INTERVIEW:** Say it out loud — "Each client gets 30 requests per 60 seconds, then 429 — but it's per-process in-memory, so multiple uvicorn workers multiply the effective limit and there's no cross-worker sharing."

**file:line:** `src/api/main.py:195,261,767,768,206-209`, `docs/learn/02_architecture.md:300`.

---

#### Q19. What happens on a cache hit — could a stale or wrong answer persist?

**Model answer (first person):**

`QueryCache.get()` returns the cached `QueryResponse` immediately, skipping retrieve/rerank/LLM (`src/api/main.py:126` `QueryCache`, `:784` cache lookup in `/ask`, `:788` returns cached result with `latency_ms=1.0`). The cache is keyed by MD5 of the full request (query + all flags), so different `top_k` or reranker flags get distinct entries (`:157` `_get_key`).

The gap: there's no TTL, so a cached wrong answer persists until the FIFO cache (max 200) evicts it (`docs/learn/02_architecture.md:298`, `:123-125` gap comment). Cached responses report `latency_ms=1.0`, which could mislead an operator into thinking the pipeline ran fast. For interview-prep repeat questions this is usually fine, but it's a correctness risk if source documents change without cache invalidation.

**# INTERVIEW:** Say it out loud — "Cache hits skip the whole pipeline and report 1 ms latency, but there's no TTL — a stale wrong answer lives until FIFO eviction at 200 entries."

**file:line:** `src/api/main.py:126,784,788,157`, `docs/learn/02_architecture.md:298,123-125`.

---

#### Q20. What happens if `API_TOKEN` is not set and you expose the API to a network?

**Model answer (first person):**

`verify_api_token` is a no-op when `API_TOKEN` is unset — local dev is open (`src/api/main.py:276` `verify_api_token`, `:304` `if not expected: return`). So anyone who can reach the network is authenticated. Worse, `verify_user_role` reads `X-Access-Level` from a client-controlled header, which is only trustworthy when bearer auth is enabled (`src/api/main.py:313` `verify_user_role`, `:316` note). Without `API_TOKEN`, a caller can assert `X-Access-Level: 3` (admin) and call mutating endpoints (`/ingest`, `/upload`, `/delete`, `/reindex`).

So exposing this API without `API_TOKEN` is a real auth bypass — fine for local dev, not for production exposure.

**# INTERVIEW:** Say it out loud — "Without API_TOKEN, auth is a no-op and the role header is client-controlled, so anyone can claim admin — this is local-dev only, not production-safe."

**file:line:** `src/api/main.py:276,304,313,316`.

---

### Scaling

#### Q21. How does this scale beyond a single node?

**Model answer (first person):**

It doesn't — not without rework. Everything is single-process, single-node: in-memory cache/rate limiter/metrics/traces/feedback store (`src/api/main.py:260-262`), in-memory BM25 index (`src/retrieval/sparse_retriever.py:53` `BM25Retriever`, `:72` index per docs), in-memory conversation memory (`src/retrieval/conversation_memory.py:50` `ConversationMemory`, `:69` turns), embedded ChromaDB (`src/store/vector_store.py:76`), local files for uploads/status/key registry (`src/ingestion/manager.py:41-42`, `src/key_pool.py:62`). There's no message bus, no shared cache, no distributed tracing, no multi-tenancy (`docs/learn/02_architecture.md:281,308`).

To scale past one node you'd need Redis for cache/rate limiting, persistent durable feedback/telemetry, an async NVIDIA client with circuit breaker, and real identity-backed auth. Until then it's a prototype-grade, single-node deployment (`docs/learn/02_architecture.md:308`).

**# INTERVIEW:** Say it out loud — "This is single-process, single-node with in-memory state — it won't survive a load balancer or multi-node rollout without Redis, persistent telemetry, async I/O, and real auth."

**file:line:** `src/api/main.py:260-262`, `src/retrieval/sparse_retriever.py:53,72`, `src/retrieval/conversation_memory.py:50,69`, `src/store/vector_store.py:76`, `docs/learn/02_architecture.md:281,308`.

---

#### Q22. What breaks if you run two uvicorn workers?

**Model answer (first person):**

Each worker has its own `query_cache`, `rate_limiter`, `feedback_store`, `metrics`, and trace ring buffer (`src/api/main.py:260-262`, `src/observability.py:166` `Metrics`, `:230` traces). So:
- Rate limiting: a client can send 30 × 2 = 60 requests/minute (one per worker).
- Caching: a query cached on worker A is a miss on worker B.
- Metrics/traces: each worker reports only its own slice; `/metrics` and `/traces` show per-worker views.
- `rag_pipeline` global: concurrent `/ingest` calls racing on rebinding could clobber each other's index (no lock) (`src/api/main.py:396` `initialize_rag_pipeline`, `:423` gap comment, `:257` `rag_pipeline = None`).

There's no Redis or shared store to synchronize these, so horizontal scaling doesn't work as-is.

**# INTERVIEW:** Say it out loud — "Two workers means two independent caches, rate limiters, and metric slices — a client effectively doubles its quota and `/metrics` shows only a per-worker view."

**file:line:** `src/api/main.py:260-262,257,396`, `src/observability.py:166,230`, `docs/learn/02_architecture.md:298-302`.

---

#### Q23. How would you add per-customer isolation? (And does it exist today?)

**Model answer (first person):**

It doesn't exist today — there's one shared corpus and no per-customer isolation. All users see the same vector-store collection `rag_academy` (`src/store/vector_store.py:695` `create_vector_store` `collection_name="rag_academy"`). Department is inferred from filename keywords, which is brittle and not enforced access control (`src/ingestion/loaders.py:118-153` `_infer_department` per docs; actual `loaders.py:287` `PDFLoader`, `:364` `DocxLoader`).

To add isolation I'd need: per-customer collections or a `tenant_id` metadata filter on every retrieval, real auth (JWT/OIDC) replacing the single bearer token, and authorization that scopes `can_ingest`/`can_delete` per tenant. That's a significant rework — today `USER_ROLES` is a single shared admin/manager/employee/guest map (`src/api/main.py:268` `USER_ROLES`).

**# INTERVIEW:** Say it out loud — "There's no multi-tenancy — one shared corpus, one bearer token, department inferred from filenames. Per-customer isolation would need tenant-scoped collections and real identity-backed auth."

**file:line:** `src/store/vector_store.py:695`, `src/ingestion/loaders.py:287,364`, `src/api/main.py:268`, `docs/learn/04_gap_bridge.md:404-409`.

---

### Cost

#### Q24. Where does cost come from, and how do you control it?

**Model answer (first person):**

The only recurring cash cost is NVIDIA NIM API tokens (prompt + completion) when `LLM_PROVIDER=nvidia` (`providers/llm_nvidia.py:545` `chat`). Embedding runs locally via sentence-transformers (effectively zero API cost, bounded by compute) (`src/embedding/embedder.py:154` `Embedder`), ChromaDB is local open-source (no per-query fee) (`src/store/vector_store.py:76`), and infra is a small VM running two Docker containers (API + UI) (`docker-compose.yml:3-41`, `Dockerfile:1-31`).

Cost controls built in:
- **Mock provider** for CI/dev — zero token cost (`providers/llm.py:111`).
- **Query cache** (200 entries, in-memory) skips the pipeline for repeats (`src/api/main.py:126,260`).
- **Rate limiter** (30/min/client) prevents one client from exhausting the token budget (`src/api/main.py:195,261`).
- **Key pool** rotates keys to avoid burning one quota (`src/key_pool.py:136,337`).
- **Token usage tracked** per call from provider-reported usage (`src/api/main.py:840` `metrics.add_tokens`, `:593-594` per docs).

**# INTERVIEW:** Say it out loud — "The only cash cost is NVIDIA tokens; mock provider, query cache, rate limiter, key pool, and per-call token tracking keep it bounded."

**file:line:** `providers/llm_nvidia.py:545`, `providers/llm.py:111`, `src/embedding/embedder.py:154`, `src/store/vector_store.py:76`, `src/api/main.py:126,260,195,261,840,593-594`, `src/key_pool.py:136,337`, `docker-compose.yml:3-41`.

---

#### Q25. How does the embedding cache save money and latency?

**Model answer (first person):**

`Embedder` caches vectors on disk under `data/embedding_cache/`, keyed by an MD5 content hash of the text (`src/embedding/embedder.py:154` `Embedder`, `:159` `cache_dir`, `:246` cache path). `embed_text` checks the cache first — identical text hits the cache and skips the model forward pass (`:574` `embed_text`). `embed_batch` splits hits from misses and only re-embeds changed text (`:582` `embed_documents` per docs; actual `embed_text`/batch at `:574`/`:582`).

This is the single biggest ingestion cost/latency win: re-uploading unchanged documents never re-embeds. The gap: the cache has no TTL or automated eviction — it grows without bound (`Embedder.clear_cache` exists at `:560` per docs but isn't automated).

**# INTERVIEW:** Say it out loud — "Embeddings are cached on disk by content-hash MD5, so unchanged text skips the model entirely — but there's no TTL, so the cache grows until manually cleared."

**file:line:** `src/embedding/embedder.py:154,159,246,574,582`, `docs/learn/02_architecture.md:289-291`.

---

### Security

#### Q26. How is auth handled, and what are its weaknesses?

**Model answer (first person):**

AuthN: if `API_TOKEN` is set, every endpoint requires `Authorization: Bearer <token>` compared in constant time with `hmac.compare_digest`; the token is never logged (`src/api/main.py:276` `verify_api_token`, `:309` `hmac.compare_digest`). AuthZ: `USER_ROLES` maps `X-Access-Level` to admin/manager/employee/guest with `can_ingest`/`can_delete`; mutating endpoints require admin (`src/api/main.py:268` `USER_ROLES`, `:313` `verify_user_role`, `:804-808` `/ingest` requires `can_ingest`).

Weaknesses: without `API_TOKEN` the role header is client-controlled (anyone can claim admin) (`src/api/main.py:316`); it's a single shared bearer token with no scopes, rotation, or expiry (`src/api/main.py:293-295` gap); `/metrics` and `/traces` are bearer-protected but the token itself is static (`src/api/main.py:451,459`).

**# INTERVIEW:** Say it out loud — "Bearer token with constant-time compare, role-based gating for mutating endpoints — but without API_TOKEN the role header is spoofable and there's no token rotation or scopes."

**file:line:** `src/api/main.py:276,309,268,313,804-808,316,451,459,293-295`.

---

#### Q27. How are secrets kept out of logs and error responses?

**Model answer (first person):**

Defense in depth: `redact()` scrubs `nvapi-`, `sk-`, and `Bearer` patterns from every log record via `RedactFilter`; `JsonFormatter` redacts `getMessage()`; `NVIDIAProvider._redact` scrubs the key from error messages; only key fingerprints (`first4…last4`) are stored in the registry and shown in alert emails (`src/observability.py:126` `redact`, `:156` `RedactFilter`, `:196` `JsonFormatter`, `:76-89` per docs; `providers/llm_nvidia.py:201` `_redact`; `src/key_pool.py:65` `fingerprint`, `:92`; `src/key_monitor.py:146-160` per docs).

Secrets never enter source: `NVIDIA_API_KEY`/`NVIDIA_API_KEYS` come from env only, `.env` is gitignored and excluded from the Docker image (`.dockerignore:1-4`), and `get_settings()` crashes at startup if a required key is missing (`src/config.py:176-182`).

**# INTERVIEW:** Say it out loud — "Redaction at every log/exception call site, fingerprints only in logs/registry, secrets from env only, and `.env` excluded from the image — so a leaked log never reveals a full key."

**file:line:** `src/observability.py:126,156,196,76-89`, `providers/llm_nvidia.py:201`, `src/key_pool.py:65,92`, `src/config.py:176-182`, `.dockerignore:1-4`.

---

#### Q28. How is file upload protected from path traversal and bad file types?

**Model answer (first person):**

`DocumentManager.validate()` allows only `.pdf/.docx/.md/.txt`, requires a plain filename (no path separators), rejects empty files, and caps size at 10 MB (`src/ingestion/manager.py:105` `DocumentManager`, `:44` `ALLOWED_EXTENSIONS`, `:45` `MAX_FILE_SIZE_BYTES`, `:204` extension check). `resolve_safe_ingest_dir` confines ingestion to the project's `data/` or `uploads/` directories, blocking `../` escapes (`src/api/main.py:360` `resolve_safe_ingest_dir`). PDF/DOCX loaders record extraction errors instead of fabricating text (`src/ingestion/loaders.py:287` `PDFLoader`, `:364` `DocxLoader` per docs; actual `loaders.py:157-160,205-207` per docs).

So an attacker can't write outside the intended directory or upload executable/bomb files.

**# INTERVIEW:** Say it out loud — "Only .pdf/.docx/.md/.txt up to 10 MB, plain filenames only, and ingestion is confined to data/ or uploads/ — path traversal and bad file types are blocked at validation."

**file:line:** `src/ingestion/manager.py:105,44,45,204`, `src/api/main.py:360`, `src/ingestion/loaders.py:287,364`, `docs/learn/04_gap_bridge.md:431-434`.

---

### "What would you change"

#### Q29. What's the first thing you'd change to make this production-ready?

**Model answer (first person):**

I'd add a shared, persistent cache and rate-limit store (Redis) so state survives restarts and is shared across uvicorn workers, plus persistent durable telemetry (Prometheus push / distributed tracing) instead of in-memory counters (`src/api/main.py:126,195`, `src/observability.py:166,230`). That's the foundation: without it, horizontal scaling and observability don't work (`docs/learn/02_architecture.md:308`).

Then I'd turn on real auth (JWT/OIDC) instead of the single static bearer token, add an async NVIDIA client with jittered backoff and a circuit breaker, and add per-customer isolation. But cache/rate-limit/telemetry is the first change because it unlocks safe multi-worker operation.

**# INTERVIEW:** Say it out loud — "First I'd add Redis for cache/rate limiting and persistent telemetry — without shared, durable state the system can't run behind multiple workers or be observed in production."

**file:line:** `src/api/main.py:126,195`, `src/observability.py:166,230`, `docs/learn/02_architecture.md:308`.

---

#### Q30. How would you add multi-tenancy / per-customer isolation?

**Model answer (first person):**

Today there's one shared corpus and one bearer token, so I'd start by replacing the single `API_TOKEN` with real identity-backed auth (JWT/OIDC) and adding a `tenant_id` to chunk metadata. Every retrieval would filter `is_current=True AND tenant_id = <caller's tenant>`, and mutating endpoints would authorize per-tenant `can_ingest`/`can_delete`. Separate ChromaDB collections per tenant (or a metadata filter) would prevent cross-tenant leakage (`src/store/vector_store.py:695`, `src/api/main.py:268,313`).

I'd be candid in the interview that this is a significant rework — department is currently inferred from filename keywords, which is brittle (`src/ingestion/loaders.py:287,364` per docs).

**# INTERVIEW:** Say it out loud — "Multi-tenancy doesn't exist today — one corpus, one token, filename-inferred department. I'd add JWT auth, tenant-scoped collections, and per-tenant authorization, and I'd flag that as a real rework, not a small tweak."

**file:line:** `src/store/vector_store.py:695`, `src/api/main.py:268,313`, `src/ingestion/loaders.py:287,364`.

---

#### Q31. What would you change about the eval / golden set?

**Model answer (first person):**

The golden set currently holds 30 questions over *generic Acme Corp corporate documents* (HR leave, IT security, benefits, travel) — not the telecom documents (LTE/5G, Roaming, SIM Security, eSIM) the project's stated purpose describes (`src/evaluation/golden_set.json`, `docs/learn/_plan.md:48` recon finding). So the eval measures mechanical correctness on the wrong content. I'd re-create the golden set with telecom questions and expected telecom sources, and run it regularly to guard against regressions (`src/evaluation/run_eval.py:334` `evaluate_golden_set`, `:469` `run_ablation_study`).

I'd also add parallelism for large golden sets (evals run sequentially today) and treat the LLM judge as a signal, not ground truth, since it has its own biases (`docs/learn/04_gap_bridge.md:690-695` gap).

**# INTERVIEW:** Say it out loud — "The golden set is 30 questions on corporate Acme docs, not the telecom content the system is actually for — so the eval passes on the wrong material. I'd re-create it with telecom questions and treat the LLM judge as a signal, not ground truth."

**file:line:** `src/evaluation/golden_set.json`, `docs/learn/_plan.md:48`, `src/evaluation/run_eval.py:334,469`, `docs/learn/04_gap_bridge.md:690-695`.

---

**Total: 31 interview questions** (exceeds the 25+ requirement).

---

<a id="2-2-minute-walkthrough"></a>
# 2. 2-minute spoken walkthrough

> "This is the rag-academy RAG system — a study aid for FDE interview prep in telecom (LTE/5G, Roaming, SIM Security, eSIM). It ingests verified documents, embeds their chunks into a local vector store (ChromaDB), and answers questions using only retrieved chunks, with citations and a refusal when evidence is missing.
>
> Data flow in sixty seconds: a document lands in `uploads/`, is validated (extension, 10 MB cap), cleaned, chunked (~500 chars / 50 overlap), embedded with a sentence-transformer (content-hash cached), and indexed into ChromaDB (`src/ingestion/manager.py:98,114,159`, `src/embedding/embedder.py:154,574`, `src/store/vector_store.py:76,334`).
>
> When a user asks a question via `/ask`, the API rate-limits the caller (30/min), checks an in-memory cache, embeds the query, retrieves top-k chunks via dense + BM25 hybrid with RRF, optionally re-ranks with a cross-encoder, builds context within a ~2000-token budget, and calls the LLM provider (mock or NVIDIA NIM). The answer comes back with sources, latency, and a feedback ID (`src/api/main.py:714`, `src/retrieval/retriever.py:51,98`, `src/retrieval/hybrid.py:51`, `src/retrieval/reranker.py:53`, `src/generation/answer.py:118,217`, `providers/llm_nvidia.py:226,545`).
>
> Quality is guarded by a golden set (hit-rate, recall, MRR, LLM-as-judge faithfulness/contradiction) (`src/evaluation/run_eval.py:96,334`, `src/evaluation/metrics.py:39,332`). Everything is observable — JSON logs, correlation IDs, per-stage traces, `/metrics`, `/traces`, `/provider` (`src/observability.py:71,302,379`, `src/api/main.py:588,615,639`). Config comes from env only, required secrets crash fast at startup, and full keys are never logged (`src/config.py:127,176`, `src/key_pool.py:65`).
>
> The honest caveat: today the indexed documents are generic Acme Corp policies, not the telecom materials the system's stated purpose requires — that's a known purpose/source mismatch recorded in `docs/learn/_plan.md:48`, not something to fix in code per the mission constraint."

*(~90-110 words spoken at a measured pace; fits in 2 minutes with pauses.)*

**# INTERVIEW:** Keep it under 2 minutes. Lead with purpose and the cite-or-refuse principle — that's what an FDE interviewer cares about most.

---

<a id="3-5-minute-walkthrough"></a>
# 3. 5-minute spoken walkthrough

> **What it solves.** FDE interview prep requires accurate, sourced telecom answers. Without this system a learner hunts documents manually, risks mis-citing, and can't prove an answer came from the source. The system enforces grounding: answer only from retrieved chunks, cite sections, refuse when silent (`CLAUDE.md`).
>
> **Components.** Config (`src/config.py`) is the frozen, env-only foundation. The key pool (`src/key_pool.py`) rotates NVIDIA keys round-robin with failover on 401/403. Observability (`src/observability.py`) gives JSON logs, correlation IDs, per-stage traces, and in-memory metrics. Ingestion (`src/ingestion/`) loads→cleans→chunks→embeds→stores with SHA-256 dedupe and versioning. Retrieval (`src/retrieval/`) fuses dense + BM25 with RRF, optionally re-ranks. Generation (`src/generation/`, `providers/`) builds context within a token budget and calls mock or NVIDIA. The controlled agent (`src/agent/`) proposes tool calls that are validated, authorized, timed, and audited. FastAPI (`src/api/main.py`) is the HTTP entry point with auth, cache, rate limiting, tracing; Streamlit (`ui/streamlit_app.py`) is the UI.
>
> **The ask/answer flow.** `POST /ask` rate-limits, cache-checks, then runs retrieve → (rerank) → generate, traced per stage (retrieve/rerank/llm), caches the result, and returns sources + feedback ID (`src/api/main.py:714-887`).
>
> **The ingest lifecycle.** `POST /documents/upload` validates, dedupes by SHA-256, versions on collision, writes to `uploads/`, runs the load→clean→chunk→embed→store pipeline, records status to `data/ingest_status.json` (`src/ingestion/manager.py:105-223`, `src/api/main.py:1244`). Delete cascade removes all chunks for a source (`src/store/vector_store.py:519`). Reindex rebuilds (`src/ingestion/manager.py:201`).
>
> **Key production patterns.** Fail-fast config validation, retry-with-backoff on embedding, graceful mock fallbacks, agent step/repeat caps, bearer auth + role gating, path-traversal guards, secret redaction, golden-set eval, Docker non-root image with healthcheck (`Dockerfile:1-41`, `docker-compose.yml:3-41`).
>
> **Where I'd dig in first.** As the owner, I'd start with `src/api/main.py` (the request surface), `src/retrieval/retriever.py` (recall is the first thing that breaks), and `src/generation/answer.py` (grounding and token budget). Then `src/observability.py` to see where latency actually lives.
>
> **Honest caveat.** It's single-node, in-memory state — no Redis, no distributed tracing, no multi-tenancy, single static bearer token. And the indexed content is corporate policies, not telecom — a known gap (`docs/learn/_plan.md:48`).

**# INTERVIEW:** 5 minutes means ~650 words. Structure: problem → components → flow → lifecycle → patterns → dig-in-first → honest caveat. End with the caveat — it shows judgment.

---

<a id="4-customer-scenarios"></a>
# 4. 5 customer scenario prompts

Each is ambiguous, has constraints and pushback. For each: discovery (what to ask, what to check in code) and a proposal. Inferences are marked **[INFERRED]**.

---

### Scenario 1 — "Our answers keep contradicting themselves across sessions."

**Context:** A telecom operator's FDE team says the same roaming question returns different answers on different days. They need consistency for interview prep.

**Constraints:**
- Must keep using the existing ingested corpus.
- Can't add new infra (no budget for Redis/managed vector DB).
- Answers must remain citable.

**Likely pushback:** "We can't re-ingest everything every night." / "Caching is already on, why is it still inconsistent?"

**Discovery — what to ask:**
- Are they re-uploading the same documents (creating duplicates) or updating them?
- Is `current_version_only` filtering being used on retrieval?
- Are they hitting a cache hit vs. a fresh pipeline run?

**Discovery — what to check in code:**
- `src/ingestion/manager.py:98` `validate` (SHA-256 dedupe) and `:114` `upload` (versioning) — are duplicates being created?
- `src/store/vector_store.py:281` `mark_source_current` and `retriever.py:126` `current_version_only` — are stale versions shadowing current answers?
- `src/api/main.py:126` `QueryCache` — no TTL, so stale answers persist until FIFO eviction at 200 (`docs/learn/02_architecture.md:298`).

**Proposal:**
- Enforce dedupe-by-hash on upload and always re-ingest with `current_version_only=True` so only the current version is retrievable (`src/ingestion/manager.py:114`, `src/retrieval/retriever.py:126`).
- Add cache invalidation on re-index (currently no TTL/invalidation — **[INFERRED]** there's no hook for this).
- Run the golden set after each re-ingest to confirm consistency (`src/evaluation/run_eval.py:334`).

**[INFERRED]** There's no documented cache-invalidation hook on re-index today; adding one is a proposal, not existing behavior.

---

### Scenario 2 — "One customer is burning our entire NVIDIA token budget."

**Context:** A single FDE learner (or script) is making hundreds of `/ask` calls, exhausting the token budget and slowing the service for everyone.

**Constraints:**
- Must remain available to other users.
- Can't change the LLM provider or pricing.
- Must stay on the current single-node deployment.

**Likely pushback:** "The rate limiter already exists — why isn't it working?" / "We don't want to block legitimate power users."

**Discovery — what to ask:**
- Is the client behind a NAT/proxy (all sharing one `client_id`)?
- Are they using `/ask/stream` (not cached)?
- How many uvicorn workers are running?

**Discovery — what to check in code:**
- `src/api/main.py:195` `RateLimiter` (30 req/60s) and `:261` instance — per-process in-memory, not shared across workers (`docs/learn/02_architecture.md:300`).
- `src/api/main.py:767` — `client_id` is `X-Forwarded-For` or request host; behind a NAT all users share one ID.
- `src/api/main.py:840` `metrics.add_tokens` — token usage is tracked, so you can identify the top consumer (`docs/learn/01_business_context.md:85`).

**Proposal:**
- Confirm worker count; if >1, the effective limit is 30 × workers (`docs/learn/02_architecture.md:300`). Add Redis-backed rate limiting to share state (**[INFERRED]** — not present today).
- Shorten the window or lower `max_requests` for anonymous/guest access levels (`src/api/main.py:268` `USER_ROLES`, `:313` `verify_user_role`).
- Use token-usage metrics (`/metrics`) to identify and contact the top consumer before hard-blocking.

**[INFERRED]** Redis-backed rate limiting and per-tenant token budgets are not implemented; proposing them is new work.

---

### Scenario 3 — "We need to add eSIM provisioning docs, but the team is afraid of breaking existing answers."

**Context:** A new batch of eSIM/LPA/LDS documents must be ingested. The team worries re-ingestion will corrupt or regress existing LTE/5G answers.

**Constraints:**
- No downtime for the QA UI.
- Must be able to roll back if eSIM answers degrade LTE answers.
- Must keep citations correct.

**Likely pushback:** "Just re-run the ingest pipeline — what could go wrong?" / "We don't have a staging environment."

**Discovery — what to ask:**
- Are the new docs new filenames, or re-uploads of existing filenames with different content?
- Is there a baseline golden set to compare against before/after?

**Discovery — what to check in code:**
- `src/ingestion/manager.py:114` `upload` (dedupe + versioning) — re-uploading a changed filename supersedes old versions (`is_current=False`) rather than deleting (`src/store/vector_store.py:281`).
- `src/ingestion/manager.py:201` `reindex` and `:191` `delete` (cascade) — how to remove a source cleanly.
- `src/api/main.py:396` `initialize_rag_pipeline` — `/ingest` rebinds the global `rag_pipeline` with no lock, so concurrent ingests race (`src/api/main.py:423-425` gap).

**Proposal:**
- Ingest eSIM docs as new filenames first (additive, no supersede), verify against the golden set, then cut over.
- Run the golden set before and after (`src/evaluation/run_eval.py:334`, `:469` ablation) to detect regression.
- Serialize ingests (one at a time) because `rag_pipeline` rebinding isn't thread-safe (`src/api/main.py:396`, `:423`).
- Keep old versions retrievable via `doc_version` until validated, then mark old `is_current=False` (`src/ingestion/manager.py:114`).

**[INFERRED]** No staging environment or blue/green ingest exists; the rollback strategy is versioning + manual reindex, which is a proposal.

---

### Scenario 4 — "The mock provider is answering in production."

**Context:** An operator discovers `LLM_PROVIDER=mock` was left set in production, so answers are canned, not from the real NVIDIA model. They want to know how it happened and how to prevent it.

**Constraints:**
- Must detect this automatically, not rely on manual checks.
- Can't disrupt live traffic while fixing.
- Must prove to auditors which model answered each response.

**Likely pushback:** "`/provider` already shows the model — just check it." / "We can't add monitoring; we have no budget."

**Discovery — what to ask:**
- How was the env var set in production (CI, Docker, orchestrator)?
- Is `/provider` being polled or checked manually?

**Discovery — what to check in code:**
- `src/api/main.py:639` `/provider` returns `{provider, model, is_real}` — makes a mock impossible to hide (`docs/learn/04_gap_bridge.md:601-617`).
- `src/config.py:100` `llm_provider` defaults to `mock`; `get_settings()` only fails fast if `nvidia` with no key — it does *not* fail if `mock` is set in prod (`src/config.py:127`).
- `src/generation/answer.py:47` `AnswerGenerator` and `QueryResponse.provider` (`src/api/main.py:70` `provider` field) — each response carries the provider.

**Proposal:**
- Poll `/provider` (bearer-protected) on a schedule and alert if `is_real` is false in production (**[INFERRED]** — no automated guard exists today).
- Add a startup/deploy gate that rejects `LLM_PROVIDER=mock` in non-dev environments (**[INFERRED]** — `get_settings` doesn't distinguish env tiers).
- Include `provider`/`model` in every `QueryResponse` so answers are auditable (`src/api/main.py:70-71`, `:863-873`).

**[INFERRED]** Automated drift detection for `LLM_PROVIDER=mock` in prod and env-tier gating are not implemented.

---

### Scenario 5 — "Sales wants to white-label this per customer with isolated document libraries."

**Context:** A customer wants their own document library and user roles, isolated from other customers, on the same deployment.

**Constraints:**
- Single deployment (can't run separate instances per customer yet).
- Must use existing auth model if possible.
- Must not let Customer A see Customer B's chunks.

**Likely pushback:** "Just add a `customer_id` filter to retrieval." / "We already have roles — isn't that enough?"

**Discovery — what to ask:**
- How many customers, and how large is each corpus?
- Do customers share any documents, or are libraries fully disjoint?

**Discovery — what to check in code:**
- `src/store/vector_store.py:695` `create_vector_store` — one collection `rag_academy`, shared by all (`docs/learn/04_gap_bridge.md:404-409` gap).
- `src/api/main.py:268` `USER_ROLES` — single shared role map, no per-customer scoping.
- `src/ingestion/loaders.py:287,364` — department inferred from filename keywords, brittle, not enforced access control.
- `src/retrieval/retriever.py:126` — `current_version_only` filter exists, but no `tenant_id`/`customer_id` filter.

**Proposal:**
- Be candid: true isolation needs per-customer ChromaDB collections or a `customer_id` metadata filter on every retrieval, plus real auth (JWT/OIDC) replacing the single bearer token (`src/store/vector_store.py:695`, `src/api/main.py:276,313`).
- Until then, the honest answer is: this deployment is single-tenant; per-customer isolation is a rework, not a config toggle (`docs/learn/04_gap_bridge.md:824-843` gap).
- **[INFERRED]** No `customer_id`/`tenant_id` field exists in chunk metadata today; adding one is new schema work across ingestion, storage, and retrieval.

---

<a id="5-break-it-drills"></a>
# 5. 10 break-it drills

Each drill is a change request or bug you must *locate in the code* — do not apply any fix. For each: the scenario, the `file:line` to inspect, and what the fix would look like. The answer key (root cause + location) is at the bottom — try each drill before peeking.

---

### Drill 1 — Stale cached answer after document update

**Scenario:** An FDE learner re-uploads an updated LTE policy (same filename, changed content). The old answer still shows in `/ask` because a cached response is returned.

**Where to inspect:** `src/api/main.py:126` (`QueryCache`) and `:260` (instance `max_size=200`); `:784-788` (cache get/set in `/ask`). Also `src/ingestion/manager.py:114` (upload versioning).

**What the fix would look like:** Invalidate (delete) the cache entries affected by a re-upload/reindex, or add a TTL + version-aware cache key. Currently there's no TTL and no invalidation on re-ingest — a stale wrong answer persists until FIFO eviction.

---

### Drill 2 — Rate limit multiplied by uvicorn workers

**Scenario:** Behind a load balancer with 4 uvicorn workers, a client sends 120 requests/minute and isn't blocked, even though the limit is 30/min.

**Where to inspect:** `src/api/main.py:195` (`RateLimiter`) and `:261` (instance `max_requests=30, window_seconds=60`). `:206-209` docstring notes per-process state.

**What the fix would look like:** Back the rate limiter with Redis (shared `INCR`/`EXPIRE` per client) so the counter is cluster-wide instead of per-process.

---

### Drill 3 — Missing NVIDIA key only surfaces as per-request 500s

**Scenario:** `LLM_PROVIDER=nvidia` is set but `NVIDIA_API_KEYS` is empty. The API starts, but every `/ask` returns 500 when generation is attempted.

**Where to inspect:** `src/config.py:127` (`get_settings`) and `:176-182` (nvidia key required check). `:40` (`MissingConfigError`).

**What the fix would look like:** The fail-fast check already raises `MissingConfigError` at startup when `nvidia` with no key — verify it's actually reached (e.g., `get_settings()` is called at startup before serving, `src/api/main.py:558` `startup_event`). If the 500s occur anyway, the check may be bypassed somewhere.

---

### Drill 4 — Agent executes an empty-filename tool call

**Scenario:** The agent is asked "what's the current version of the roaming policy" but no filename is mentioned. The heuristic planner returns `compare_document_versions` with `filename=""`, which fails.

**Where to inspect:** `src/agent/agent.py:170` (`_plan_heuristic`), `:194-197` (version branch filename regex, `fn = m.group(1) if m else ""`), `:108` (`run`).

**What the fix would look like:** Validate that a filename was extracted before planning a version tool; if none, fall back to `search_documents` or ask the user to specify the document. Currently an empty filename reaches `ToolRegistry.execute` and fails at handler time.

---

### Drill 5 — Blocking NVIDIA call stalls the event loop

**Scenario:** A slow NVIDIA response blocks other concurrent requests because the provider uses a blocking `httpx.Client` + `time.sleep` backoff.

**Where to inspect:** `providers/llm_nvidia.py:42-44` (blocking sync client gap) and `:382-387` (`_sleep` `time.sleep`), `:420` (`_post` retries).

**What the fix would look like:** Replace `httpx.Client` with `httpx.AsyncClient`, replace `time.sleep` with `asyncio.sleep`, and add jittered backoff. Also add a circuit breaker so repeated 429s don't keep retrying.

---

### Drill 6 — Spoofed admin role without bearer token

**Scenario:** `API_TOKEN` is unset, and a caller sends `X-Access-Level: 3` to `/documents/upload` and successfully ingests a document.

**Where to inspect:** `src/api/main.py:276` (`verify_api_token`, no-op when token unset) and `:313` (`verify_user_role`, `:316` note that the header is client-controlled without bearer auth).

**What the fix would look-like:** Require `API_TOKEN` in production (fail fast at startup if mutating endpoints are exposed without it), or derive roles from a verified JWT instead of trusting the client header.

---

### Drill 7 — Path traversal via ingest directory

**Scenario:** A caller sends `/ingest` with `data_directory=../../etc` and attempts to read files outside `data/`/`uploads/`.

**Where to inspect:** `src/api/main.py:360` (`resolve_safe_ingest_dir`) and `:811` (used by `/ingest`). `src/ingestion/manager.py:44-45` (`ALLOWED_EXTENSIONS`, `MAX_FILE_SIZE_BYTES`).

**What the fix would look like:** Confirm `resolve_safe_ingest_dir` rejects paths outside the allowed roots (realpath + prefix check), and that `validate` rejects filenames with path separators. Test with `../../etc/passwd`.

---

### Drill 8 — All NVIDIA keys marked bad, generation hard-fails

**Scenario:** Every NVIDIA key returns 401. `current_key` raises `RuntimeError` and `/ask` returns 500 instead of a graceful message.

**Where to inspect:** `src/key_pool.py:337` (`current_key`, raises when no active keys) and `:428` (`mark_bad`, rotates or returns `None`). `:340` (`RuntimeError` message).

**What the fix would look like:** Catch the `RuntimeError` at the API layer and return a clear 503/"all keys exhausted" message; surface pool health via an endpoint (`key_pool.status()` at `:446` returns fingerprint-only status, safe to expose).

---

### Drill 9 — In-memory feedback lost on restart

**Scenario:** The API restarts and all thumbs-up/down feedback is gone; there's no history of what users liked.

**Where to inspect:** `src/api/main.py:262` (`feedback_store = []`) and `:1536` (`log_feedback_request`). `:106-109` (`FeedbackRequest`).

**What the fix would look like:** Persist feedback to a database or append-only file instead of a per-process list; add a `/feedback` listing endpoint with auth.

---

### Drill 10 — Golden set tests the wrong content

**Scenario:** The eval passes, but the questions are about Acme Corp HR/IT policies, not LTE/5G/Roaming/SIM Security/eSIM — so "passing" is meaningless for the stated purpose.

**Where to inspect:** `src/evaluation/golden_set.json` (30 questions, corporate Acme sources like `hr_leave_policy.md`, `it_security_policy.md`) and `docs/learn/_plan.md:48` (recon finding: purpose/source mismatch). `src/evaluation/run_eval.py:334` (`evaluate_golden_set`).

**What the fix would look like:** Re-create the golden set with telecom-domain questions and expected telecom sources; add telecom documents to `data/raw/` (content change, out of scope for this mission) and re-run the eval. Treat current "pass" as mechanical only.

---

<a id="6-answer-key"></a>
# 6. Answer key (drills) — root cause and location

*Cover this section until you've attempted each drill.*

| Drill | Root cause | Location |
|-------|-----------|----------|
| 1 | `QueryCache` has no TTL and no invalidation on re-upload/reindex; stale entries persist until FIFO eviction at 200 entries. | `src/api/main.py:126` (`QueryCache`), `:260` (instance), `:784-788` (get/set in `/ask`); `src/ingestion/manager.py:114` (upload has no cache invalidation) |
| 2 | `RateLimiter` is per-process in-memory; each uvicorn worker tracks its own counter, so effective limit = 30 × workers. | `src/api/main.py:195` (`RateLimiter`), `:261` (instance), `:206-209` (docstring) |
| 3 | Fail-fast check exists but may not be enforced before serving; if `get_settings()` isn't called at startup or the check is bypassed, 500s occur per request. | `src/config.py:127` (`get_settings`), `:176-182` (nvidia key check), `:40` (`MissingConfigError`); `src/api/main.py:558` (`startup_event`) |
| 4 | Heuristic planner returns `compare_document_versions` with `filename=""` when no filename regex matches; the empty filename reaches the handler and fails. | `src/agent/agent.py:170` (`_plan_heuristic`), `:194-197` (filename extraction) |
| 5 | Blocking `httpx.Client` + `time.sleep` backoff; no async client, no jitter, no circuit breaker. | `providers/llm_nvidia.py:42-44`, `:382-387`, `:420` |
| 6 | `verify_api_token` is a no-op when `API_TOKEN` unset, and `X-Access-Level` header is client-controlled without bearer auth. | `src/api/main.py:276,304`; `:313,316` |
| 7 | Relies on `resolve_safe_ingest_dir` confinement and `validate` extension/filename checks; must confirm realpath+prefix and separator rejection. | `src/api/main.py:360` (`resolve_safe_ingest_dir`), `:811`; `src/ingestion/manager.py:44-45`, `:204` |
| 8 | `current_key` raises `RuntimeError` when every key is bad/expired; API layer doesn't convert to a graceful 503. | `src/key_pool.py:337,340`; `:428` (`mark_bad`) |
| 9 | `feedback_store` is a per-process Python list, lost on restart, not queryable. | `src/api/main.py:262` (`feedback_store`), `:1536` (`log_feedback_request`) |
| 10 | Golden set covers corporate Acme docs, not the telecom content the system's stated purpose requires — a purpose/source mismatch. | `src/evaluation/golden_set.json`; `docs/learn/_plan.md:48`; `src/evaluation/run_eval.py:334` |

---

<a id="7-honest-weaknesses"></a>
# 7. Honest weaknesses — how to talk about them candidly

Interviewers value candor about limitations more than a perfect pitch. Here are the repo's honest weaknesses, each grounded in `file:line`, and how to discuss them.

---

### Weakness 1 — Purpose/source mismatch: indexed content isn't telecom

**The issue:** `data/raw/` holds generic Acme Corp corporate policies (HR leave, IT security, benefits, travel), not the LTE/5G, Roaming, SIM Security, eSIM documents the system's stated purpose describes (`CLAUDE.md:6`, `docs/learn/_plan.md:48` recon finding). The system works mechanically but isn't indexed with the content it's meant to serve.

**How to say it:** "The pipeline is real, but the indexed documents are corporate policies, not the telecom materials the project's stated purpose requires. That's a known purpose/source mismatch I'd flag, not something I'd quietly ship as 'telecom-ready.'"

**file:line:** `docs/learn/_plan.md:48`, `CLAUDE.md:3-6`, `src/evaluation/golden_set.json` (corporate sources).

---

### Weakness 2 — In-memory-only cache, rate limiter, metrics, traces, feedback

**The issue:** `QueryCache` (`src/api/main.py:126`), `RateLimiter` (`:195`), `Metrics` (`src/observability.py:166`), traces (`:230` ring buffer of 100), and `feedback_store` (`src/api/main.py:262`) are all per-process in-memory. They reset on restart and aren't shared across uvicorn workers — so horizontal scaling and durable observability don't work (`docs/learn/02_architecture.md:298-302,308`).

**How to say it:** "Cache, rate limiting, metrics, traces, and feedback are all in-memory per process. That's fine for a single-node demo, but behind a load balancer each worker has independent state, and everything resets on restart. I'd add Redis for cache/rate limiting and persistent telemetry first."

**file:line:** `src/api/main.py:126,195,262`, `src/observability.py:166,230`, `docs/learn/02_architecture.md:298-302,308`.

---

### Weakness 3 — Blocking NVIDIA client, no async, no jitter, no circuit breaker

**The issue:** `NVIDIAProvider` uses a blocking `httpx.Client` and `time.sleep` backoff (`providers/llm_nvidia.py:42-44`, `:382-387`). No async client, no jittered backoff (thundering-herd risk on 429), no circuit breaker (`docs/learn/04_gap_bridge.md:807-812`).

**How to say it:** "The NVIDIA client is synchronous with fixed `time.sleep` backoff. It's simple and predictable for low concurrency, but it blocks the event loop and risks a thundering herd on 429. I'd move to an async client with jitter and a circuit breaker."

**file:line:** `providers/llm_nvidia.py:42-44,382-387,420`, `docs/learn/04_gap_bridge.md:807-812`.

---

### Weakness 4 — No CI beyond docker-compose; no CI/CD pipeline file

**The issue:** Deployment is Docker + docker-compose (`Dockerfile:1-41`, `docker-compose.yml:3-41`), but there's no CI pipeline file (GitHub Actions, etc.) and no automated test run on deploy. Eval runs locally via `python -m src.evaluation.run_eval` (`src/evaluation/run_eval.py:1057` `main`), not on a schedule.

**How to say it:** "There's Docker and compose, but no CI/CD pipeline file — tests and the golden-set eval run locally, not on deploy. I'd add a CI job that runs the test suite and golden set on every change."

**file:line:** `Dockerfile:1-41`, `docker-compose.yml:3-41`, `src/evaluation/run_eval.py:1057`.

---

### Weakness 5 — Single static bearer token; no scopes, rotation, or per-customer isolation

**The issue:** Auth is one `API_TOKEN` (`src/config.py:119` `api_token`, `src/api/main.py:276` `verify_api_token`). Without it, the `X-Access-Level` header is client-controlled and spoofable (`src/api/main.py:316`). There's no multi-tenancy — one shared corpus, no per-customer isolation (`src/store/vector_store.py:695` single collection `rag_academy`, `src/api/main.py:268` `USER_ROLES`).

**How to say it:** "Authentication is a single static bearer token plus role flags; without the token the role header is spoofable. There's no multi-tenancy — every user sees the same corpus. I wouldn't claim per-customer isolation that doesn't exist."

**file:line:** `src/api/main.py:276,313,316,268`, `src/store/vector_store.py:695`, `src/config.py:119`.

---

### Weakness 6 — RRF weights stored but not applied; heuristic planner is fragile

**The issue:** Hybrid RRF stores `weight_dense`/`weight_sparse` but doesn't apply them in the fusion math (`docs/learn/02_architecture.md:136`). The agent's heuristic planner uses keyword matching that can return an empty filename (`src/agent/agent.py:194-197`).

**How to say it:** "The hybrid RRF weights are declared but not actually used in the fusion — they're decorative today. And the agent's heuristic planner is keyword-based, so it can generate an empty-filename tool call that fails at execution."

**file:line:** `docs/learn/02_architecture.md:136`, `src/agent/agent.py:194-197`.

---

### Weakness 7 — Golden-set eval runs sequentially, no parallelism; LLM judge isn't ground truth

**The issue:** Evals run sequentially (`src/evaluation/run_eval.py`); the LLM-as-judge is another model with its own biases, not ground truth (`docs/learn/04_gap_bridge.md:690-695`).

**How to say it:** "Evals run sequentially, so large golden sets are slow. And the LLM judge is a signal, not ground truth — I'd treat its scores as directional and verify with human review on critical questions."

**file:line:** `src/evaluation/run_eval.py`, `docs/learn/04_gap_bridge.md:690-695`.

---

**Interview stance:** Name the weakness, cite the code, and say what you'd change. That demonstrates senior judgment — exactly what an FDE/SE interviewer wants from someone transitioning from QA.

---

<a id="8-unverified"></a>
# 8. UNVERIFIED items

The following claims were inferred from reading the code and docs but not confirmed by running the system:

1. The system starts and serves end-to-end (`uvicorn src.api.main:app` + Streamlit) — no run was performed as part of this task (`docs/learn/02_architecture.md:320`).
2. ChromaDB is installed and `data/chroma_db` is active at runtime (the in-memory fallback may be used if `chromadb` is missing) (`docs/learn/02_architecture.md:321`).
3. The `sentence-transformers` model `all-MiniLM-L6-v2` downloads/loads successfully; if unavailable, the mock embedder is used silently (`src/embedding/embedder.py:233` per docs; actual `Embedder` at `:154`).
4. Line numbers are accurate for the current `learn/annotated` working tree; some source files were modified after docs/learn/02-04 were written, so a small number of `file:line` references may be off by a line or two. I re-verified the route decorators and key class/def lines via `grep` before writing this doc.
5. The NVIDIA provider path (`LLM_PROVIDER=nvidia`) was not exercised; behavior depends on a real API key and network access to `integrate.api.nvidia.com` (`docs/learn/02_architecture.md:324`).

---

## Report

- **File written:** `C:\projects\rag-project\docs\learn\06_interview_pack.md`
- **Question count:** 31 interview questions (Q1–Q31), exceeding the 25+ requirement. Grouped into design choices (Q1–8), tradeoffs (Q9–14), failure scenarios (Q15–20), scaling (Q21–23), cost (Q24–25), security (Q26–28), and "what would you change" (Q29–31).
- **Other sections included:** 2-minute spoken walkthrough, 5-minute spoken walkthrough, 5 customer scenarios (each with discovery + proposal, [INFERRED] marks), 10 break-it drills, separated answer key, 7 honest weaknesses with interview talk tracks, UNVERIFIED items.
- **Grounding:** Every factual claim about behavior carries a `file:line` citation to the actual source (re-verified via `grep` where line numbers had shifted since docs/learn/02-04), plus docs/learn/01-04 and docs/learn/_plan.md.
- **Learner background references:** Senior QA/automation → AI SE (telecom/eSIM/IoT) woven into answers via analogies (provisioning circuits, HLR/HSS, test pipelines, regression suites, MAC/ARP cache, policers, RADIUS) and explicit "QA/automation angle" notes.
- **Convention:** All model answers are first person ("I chose X because..."); `# INTERVIEW:` blocks are say-it-out-loud points; inferences marked `[INFERRED]`.
- **No source code changed:** This document lives entirely under `docs/learn/`.

**UNVERIFIED:** See section 8 — the five items above, primarily that the system hasn't been run end-to-end and the NVIDIA path hasn't been exercised.
