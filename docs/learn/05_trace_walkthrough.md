# 05 — Trace Walkthrough: One Request End-to-End

**Branch:** `learn/annotated` | **Repo:** `C:\projects\rag-project` | **Status:** Draft (doc only, no code changed)

> **Mission constraint honored:** no source code was changed. Only this new file was added under `docs/learn/`. Every `file:line` below refers to the actual annotated source at `learn/annotated`.

---

## Who this is for

**Learner profile:** senior QA / automation engineer (~10 years QA/CI/CD/test frameworks) moving toward an AI Solutions Engineer role in telecom/eSIM/IoT. You know program **flow** but are weak on production architecture, concurrency/async, auth/multi-tenancy, observability, and cost/latency tradeoffs.

**Analogy frame (keep this in your head):** think of the RAG system like a telecom provisioning / fault-ticket operation.

| System piece | Telecom / provisioning analogy |
|---|---|
| Ingestion (load → clean → chunk → embed → store) | Loading a new tariff / circuit record into the OSS inventory |
| Vector store | The circuit-inventory database |
| Retrieval | Querying the inventory for circuits matching a fault symptom |
| Reranker | A second, more careful judge re-sorting the shortlist |
| Answer generation | The NOC engineer writing the trouble-ticket resolution, quoting only the inventory record |
| Query cache | A fast lookup table (MAC / ARP cache) |
| Rate limiter | A policer on an access port |
| Bearer-token auth | RADIUS / TACACS+ at the network edge |
| Request ID / correlation ID | A tracking number on a fault ticket |
| Per-stage trace | Timestamps on each substage of a provisioning workflow |
| Mock LLM provider | A flight simulator — realistic enough to practice, never leaves the ground |

---

## How to read each hop

Every step in the traces below follows this template:

```
### N. Step name — file:line
- **Input:** concrete values (query, filename, chunk size, score, etc.)
- **Where:** `file:line` — the exact code that does the work
- **What it does:** plain English
- **Output / return:** what this hop produces
- **# FAILURE:** what happens here if X goes wrong (symptom the user/operator sees)
```

Where runtime behavior could not be confirmed by running the system (no live server, deps not installed), the hop is marked **[UNVERIFIED]** and the limitation is stated plainly — no behavior is invented.

---

## Prerequisites assumed for both traces

- `LLM_PROVIDER=mock` (default, `src/config.py:164`) — no NVIDIA key, no network, zero cost.
- `API_TOKEN` is **unset** (dev mode), so `verify_api_token` is a no-op (`src/api/main.py:276-305`) and the `X-Access-Level` header is client-controlled/advisory (`src/api/main.py:313-350`).
- The FastAPI app has finished `startup_event` (`src/api/main.py:557-585`), which called `initialize_rag_pipeline()` (`src/api/main.py:396-493`), so the global `rag_pipeline` is populated and not `None` (`src/api/main.py:257`).
- The vector store is populated (either ChromaDB at `data/chroma_db` or the in-memory fallback, `src/store/vector_store.py:67-73,116-114`).
- **[UNVERIFIED]** The system has not been started end-to-end in this session; line numbers are grounded in the source but runtime behavior (e.g. whether ChromaDB vs fallback is active) is not confirmed by a live run.

---

## Trace A — Ask a question → get an answer (the FastAPI path)

**Concrete request:** `POST /ask` with body

```json
{
  "query": "What is an EPS bearer in LTE?",
  "top_k": 5,
  "use_reranker": true,
  "use_hybrid": true,
  "use_rrf": false,
  "current_version_only": true
}
```

Headers: `X-Access-Level: 1` (employee), no `Authorization` header (dev mode, token unset).

This is the "fault ticket" path: the subscriber (user) reports a symptom (a question), the system queries the inventory (retrieve), re-sorts candidates (rerank), writes a resolution quoting only inventory records (generate), and returns it.

---

### A1. Middleware: request ID + trace + metrics — `src/api/main.py:507-555`

- **Input:** incoming HTTP request `POST /ask` (method, path, headers).
- **Where:** `src/api/main.py:507` (`@app.middleware("http") async def log_requests`).
- **What it does:** Calls `new_request_id()` (`src/observability.py:71`) to mint a 12-hex-char correlation ID (e.g. `a3f7c2e9b1d4`), stores it in a `ContextVar` (`src/observability.py:68`), starts a `Trace` (`src/observability.py:301`), increments `requests_total` (`src/observability.py:416`), logs "request start", then delegates to the downstream handler (`call_next`). On the way out it records `request_ms` latency (`src/observability.py:421`), sets the `X-Request-Id` response header (`src/api/main.py:551`), and records the trace (`src/observability.py:481`).
- **Output / return:** the downstream `Response` with `X-Request-Id: a3f7c2e9b1d4` added; a trace record exists in the ring buffer.
- **# FAILURE:** If `call_next` raises (e.g. a 500 from a later hop), the middleware still sets `X-Request-Id` on the error response (`src/api/main.py:541-545`), records the error on the trace (`trace.error = str(e)`), increments the error counter (`metrics.inc_error`), and re-raises. The client sees the error *with* a request ID it can quote to support.

---

### A2. AuthN / AuthZ: verify_api_token + verify_user_role — `src/api/main.py:719-720`

- **Input:** `Authorization` header (absent), `X-Access-Level: 1` header.
- **Where:** `src/api/main.py:719` (`user_info: dict = Depends(verify_user_role)`) and `:720` (`_token: None = Depends(verify_api_token)`), with implementations at `src/api/main.py:276` (`verify_api_token`) and `src/api/main.py:313` (`verify_user_role`).
- **What it does:** `verify_api_token` checks `API_TOKEN` from settings (`src/config.py:211`); since it's unset, it returns immediately (dev mode, no auth). `verify_user_role` maps `X-Access-Level: 1` → `{"role": "employee", "access_level": 1, "can_ingest": false, "can_delete": false}` by looking up `USER_ROLES` (`src/api/main.py:268-273`).
- **Output / return:** `user_info = {"role": "employee", "access_level": 1, "can_ingest": false, "can_delete": false}`; auth dependency returns `None`.
- **# FAILURE:** If `API_TOKEN` *were* set and the caller sent no bearer token, `verify_api_token` returns 401 "Missing bearer token" (`src/api/main.py:307`). If an unknown access level is sent, it falls back to "guest" with no privileges (`src/api/main.py:328`). **[GAP]** Without `API_TOKEN`, `X-Access-Level` is client-controlled — anyone can self-assert `access_level: 3` (admin) (`src/api/main.py:332`).

---

### A3. Rate limit: RateLimiter.is_allowed — `src/api/main.py:767-769`

- **Input:** `client_id` derived from `X-Forwarded-For` or the client host (e.g. `"192.168.1.42"`), limit 30 requests per 60 s.
- **Where:** `src/api/main.py:767` (`client_id = x_forwarded_for or ...`) and `:768-769` (`if not rate_limiter.is_allowed(client_id): raise HTTPException(429, ...)`), limiter defined at `:261` (`RateLimiter(max_requests=30, window_seconds=60)`), `is_allowed` at `:224`.
- **What it does:** Prunes timestamps older than 60 s for that `client_id`, counts remaining requests in the window, and returns `False` (reject) if the count is already >= 30; otherwise appends `time.time()` and returns `True`. This is the policer on the access port — one subscriber can't flood the NOC.
- **Output / return:** `True` (allowed) — request proceeds; or 429 "Rate limit exceeded. Please try again later."
- **# FAILURE:** If the client has already made 30 requests in the last 60 s, hop returns 429 and no retrieval/LLM work happens — the user sees "Rate limit exceeded" and must wait. **[GAP]** The limiter is in-memory per process (`src/api/main.py:219-250`), resets on restart, and is not shared across uvicorn workers, so a client behind a load balancer can effectively multiply its limit by worker count.

---

### A4. Cache lookup: QueryCache.get — `src/api/main.py:774-788`

- **Input:** `cache_request = {"query": "What is an EPS bearer in LTE?", "top_k": 5, "use_reranker": true, "use_hybrid": true, "use_rrf": false, "current_version_only": true, "user_access_level": 1}`.
- **Where:** `src/api/main.py:774-782` (builds the key dict), `:784` (`query_cache.get(cache_request)`), cache instance at `:260` (`QueryCache(max_size=200)`), `get` at `:163`, key hash at `:157-161` (MD5 of JSON-serialized request).
- **What it does:** Serializes the full request (query + *all* flags, not just the query string) to JSON, MD5-hashes it, and looks it up in the in-memory dict. Hashing all flags means two requests with the same query but different `top_k` or `use_reranker` get distinct cache entries — they must not return each other's answers. On a hit, `hits` counter increments; on a miss, `misses` increments (`src/api/main.py:164-169`).
- **Output / return:** Cache **miss** (first time this exact request is asked) → `None`, proceed to pipeline. On a hit, returns `cached_result.model_copy(update={"latency_ms": 1.0})` (`src/api/main.py:788`) — a near-instant cached answer with minimal reported latency.
- **# FAILURE:** Cache **hit** on a stale/wrong answer: because there is **no TTL** (`src/api/main.py:123-124`), a previously cached incorrect answer persists until the FIFO of 200 entries rotates it out. The user gets a fast but wrong answer with `latency_ms: 1.0`, which looks suspiciously good. Cache **miss** is the happy path here — proceed to A5.

---

### A5. Pipeline guard + trace setup — `src/api/main.py:790-799`

- **Input:** none beyond the request already in flight.
- **Where:** `src/api/main.py:790` (`start_time = time.time()`), `:792-793` (`if rag_pipeline is None: raise 503`), `:795-796` (`trace = Trace(request_id=get_request_id())`), `:799` (`feedback_id = str(uuid.uuid4())`).
- **What it does:** Records the wall-clock start time for end-to-end latency. Guards that `rag_pipeline` (the global wiring dict built at startup) is not `None` — if startup failed, every endpoint returns 503 "RAG pipeline not initialized" (`src/api/main.py:792-793`, also `:695-696` in `/health`). Creates an observability `Trace` carrying the request ID from the `ContextVar` (`src/observacy:301,99`), and mints a `feedback_id` (UUID) so the user can later thumbs-up/down this answer via `/feedback` (`src/api/main.py:1014`).
- **Output / return:** `start_time` float; trace object; `feedback_id` e.g. `"550e8400-e29b-41d4-a716-446655440000"`.
- **# FAILURE:** If `rag_pipeline is None` (startup failed — e.g. `initialize_rag_pipeline` raised at boot, `src/api/main.py:557-585`), returns 503 and the user sees "RAG pipeline not initialized". The request never reaches retrieval.

---

### A6. Retrieve: dense/hybrid vector search — `src/api/main.py:806-815`

- **Input:** `query = "What is an EPS bearer in LTE?"`, `k = top_k * 2 = 10` (because `use_reranker` is true — fetch extra candidates for the reranker to choose from), `current_version_only = True`.
- **Where:** `src/api/main.py:806` (`with trace.stage("retrieve", k=10, hybrid=True):`), `:807-811` (`rag_pipeline["retriever"].retrieve(request.query, k=..., current_version_only=...)`), `:813-814` (splits `(doc, score)` pairs), `:815` (`metrics.inc("retrieved_chunks", len(retrieved_docs))`). Retriever implementation: `src/retrieval/retriever.py:98` (`Retriever.retrieve`), `:122` (`query_embedding = self.embedder.embed_text(query)`), `:126-127` (injects `{"is_current": True}` filter), `:131-135` (`vector_store.search(query_vector=..., k=30 if filtering else 10, filter_metadata=...)`), `:142` (cap at k).
- **What it does:** Embeds the query into a 384-d vector (`src/embedding/embedder.py:299`, mock or sentence-transformers), asks the vector store for nearest neighbours (`src/store/vector_store.py:334`), converts ChromaDB distance to similarity (`1 - min(distance, 1.0)`, `src/store/vector_store.py:399`), applies the `is_current=True` filter so stale document versions don't shadow the current answer, thresholds by similarity (default 0.0 = no threshold here), and caps at `k=10`. Because `current_version_only` is set and no other filters, it fetches `k*3 = 30` from the store to survive filtering (`retriever.py:133`).
- **Output / return:** `retrieved_docs_with_scores` — e.g. 8 `(doc, score)` tuples like `[({"content": "EPS bearers ...", "metadata": {"source": "lte.md", "chunk_index": 2, "is_current": true}}, 0.87), ...]`; `retrieved_docs` (list of dicts) and `scores` (list of floats); `metrics.counters["retrieved_chunks"] += 8`.
- **# FAILURE:** Vector store empty (nothing ingested yet) → `retrieved_docs_with_scores == []`, downstream gets no chunks, and the generator produces an insufficiency answer ("I don't have enough information…"). Embedding failure → `embedder.embed_text` degrades to a deterministic mock vector (`src/embedding/embedder.py:346-350`, `:_mock_embed` at `:508`) so retrieval still runs but with poor semantic quality. `vector_store.search` raises `ValueError` if the query vector dimension != 384 (`src/store/vector_store.py:374-375`) — caught by Chroma fallback or propagates as 500.

---

### A7. (Optional) Rerank: cross-encoder re-scoring — `src/api/main.py:819-831`

- **Input:** `request.query`, `list(zip(retrieved_docs, scores))` (8 candidates), `k = request.top_k = 5`.
- **Where:** `src/api/main.py:819-827` (`if request.use_reranker and rag_pipeline["reranker"] is not None: with trace.stage("rerank", k=5): ... reranker.rerank(...)`), reranker implementation `src/retrieval/reranker.py:100` (`Reranker.rerank`), `:127-130` (passthrough when `self.model is None`), `:134-137` (builds `[query, doc["content"]]` pairs, `self.model.predict(pairs)`), `:145` (sort descending), `:148-149` (top k).
- **What it does:** The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly encodes the query + each candidate document and produces a more accurate relevance score than first-pass cosine/BM25. It re-sorts descending and returns the top 5. If the model isn't installed, it degrades to passthrough (original scores, `reranker.py:127-130`). This is the second, more careful judge re-sorting the shortlist after the first quick scan.
- **Output / return:** `final_docs` = top 5 docs, `final_scores` = their rerank scores. If reranker is disabled/None, falls back to `retrieved_docs[:5]` (`src/api/main.py:830-831`).
- **# FAILURE:** Reranker model fails mid-predict → caught, logged, falls back to original scores (`reranker.py:154-157`) — no exception to the caller, answer proceeds with retrieval order. Reranker returns empty → `final_docs == []`, generator gets no context and answers from insufficiency.

---

### A8. Generate: grounded answer from retrieved chunks — `src/api/main.py:834-843`

- **Input:** `request.query`, `final_docs` (top 5 chunks), provider = mock (`LLM_PROVIDER=mock`).
- **Where:** `src/api/main.py:834` (`with trace.stage("llm", provider="mock"):`), `:835-836` (`rag_pipeline["answer_generator"].generate_answer_with_context(request.query, final_docs)`), answer generator `src/generation/answer.py:217` (`generate_answer_with_context`), `:118` (`_build_context_within_budget`, max 2000 tokens), `:256` (`create_grounded_prompt(context, query, "v2")`), `:188` (`generate_answer` → mock provider `providers/llm.py:136`), mock provider `providers/llm.py:111` (`MockLLMProvider`), `:198` (`get_model_info` → `{"provider": "mock", ...}`).
- **What it does:** Builds the context string within a 2000-token budget (chars/4 estimate, highest-ranked chunks first, `src/generation/answer.py:118-186`), renders a grounded prompt v2 that instructs the LLM to answer only from context, cite `[source_filename]`, and refuse when evidence is insufficient (`src/generation/prompt.py:351`), calls the mock provider (no network — parses Context/Question markers and returns a canned answer with mock citations, `providers/llm.py:136-172`), records prompt/completion tokens and latency, and assembles a `GenerationResult` (`src/generation/answer.py:15`, `:304-321`).
- **Output / return:** `answer_result.answer` e.g. `"Based on the provided context, ... [source_document_1.md] ..."` (mock-canned), `usage` (None for mock; provider-reported for NVIDIA), `answer_result.metadata = {"provider": "mock", "model": "mock", "num_context_docs": 5, "context_length": ..., "answer_generation_latency_ms": ..., ...}`. The mock never calls NVIDIA — zero cost, deterministic.
- **# FAILURE:** LLM provider raises → `generate_answer` degrades to an insufficiency fallback message instead of crashing (`src/generation/answer.py:208-214`); the user gets "I don't have enough information… (LLM provider error)". Empty `final_docs` → empty context → mock returns an insufficiency message (`providers/llm.py:169-170` or `answer.py:441-491`). **[UNVERIFIED]** NVIDIA path (`LLM_PROVIDER=nvidia`) not exercised — no key, no network; would route through `providers/llm_nvidia.py:545` with retries/backoff/key-failover.

---

### A9. Sources + context + latency assembly — `src/api/main.py:846-873`

- **Input:** `final_docs` (5 chunks), `answer_text`, `usage`, `latency_ms`.
- **Where:** `src/api/main.py:846-854` (builds `sources` list with 200-char snippets), `:857` (`context_used = format_context(final_docs)`, `src/generation/prompt.py:282`), `:860` (`latency_ms = (time.time() - start_time) * 1000`), `:863-873` (constructs `QueryResponse`).
- **What it does:** For each final doc, extracts `source` (filename), a 200-character `content_snippet`, and full `metadata`. Joins the chunks into `context_used` via `format_context` (dedupes and truncates by char count, `src/generation/prompt.py:282-307`). Computes end-to-end latency from `start_time` (A5). Builds the `QueryResponse` Pydantic model with answer, sources, context_used, query, latency_ms, feedback_id, provider, model, usage.
- **Output / return:** `QueryResponse` object e.g. `{answer: "...", sources: [{source: "lte.md", content_snippet: "...", metadata: {...}}], context_used: "...", query: "What is an EPS bearer in LTE?", latency_ms: 1240.5, feedback_id: "...", provider: "mock", model: "mock", usage: None}`.
- **# FAILURE:** A doc missing `metadata` → `metadata.get("source", "Unknown")` defaults gracefully (`src/api/main.py:848-850`), no crash. `format_context` truncation by character count (not tokens) can under-fill the budget relative to the model's actual window (`src/generation/prompt.py:306-307`, noted `# GAP` in `answer.py:141-143`).

---

### A10. Cache write: QueryCache.set — `src/api/main.py:876`

- **Input:** `cache_request` (same dict as A4), `response` (the QueryResponse).
- **Where:** `src/api/main.py:876` (`query_cache.set(cache_request, response)`), `set` at `:171-178`.
- **What it does:** MD5-hashes the request dict, stores the QueryResponse in the in-memory dict. If the cache is full (>= 200 entries), evicts the oldest (FIFO) before inserting (`src/api/main.py:174-177`). No TTL — entries persist until evicted by FIFO.
- **Output / return:** None (mutates `self.cache`); a repeat identical query will now hit at A4.
- **# FAILURE:** Cache is unbounded in entry count until FIFO evicts; no TTL means a wrong cached answer lingers (`src/api/main.py:123-124`). Concurrent reads/writes are not locked — `# GAP` at `src/api/main.py:755-757` — though the GIL makes dict ops mostly atomic, not transactional.

---

### A11. Feedback logging (background) + return response — `src/api/main.py:879-887`

- **Input:** `feedback_id`, `request.query`, `user_info`, `client_id`.
- **Where:** `src/api/main.py:879-885` (`background_tasks.add_task(log_feedback_request, ...)`), `:887` (`return response`).
- **What it does:** Schedules `log_feedback_request` (a background task, `src/api/main.py:1065`) so the response isn't blocked by feedback logging. Returns the `QueryResponse` to the client as JSON. The middleware (A1) then sets `X-Request-Id` and records the trace.
- **Output / return:** HTTP 200 with `QueryResponse` JSON body and `X-Request-Id: a3f7c2e9b1d4`.
- **# FAILURE:** If `log_feedback_request` raises, it's a background task — the response is already sent, so the user is unaffected but the feedback record is lost (`src/api/main.py:1065`, in-memory `feedback_store` at `:262`, lost on restart).

---

### A12. (Variant) /ask/stream — SSE streaming — `src/api/main.py:893-980`

- **Input:** same `QueryRequest` as A1.
- **Where:** `src/api/main.py:893` (`@app.post("/ask/stream")`), `:934` (rate limit), `:943-947` (retrieve), `:950-956` (rerank), `:958-965` (`stream_answer_with_context` yields token events + done event), `:971` (`yield "data: [DONE]\n\n"`), `:972-977` (error path emits error event then [DONE]).
- **What it does:** Same retrieve/rerank as A6-A7, then streams answer tokens as SSE `data:` events (`text/event-stream`), ending with a `done` event carrying sources/usage/latency/provider/model, then `[DONE]`. On exception, emits an `error` event then `[DONE]` so the client is never left hanging (`src/api/main.py:972-977`).
- **Output / return:** `StreamingResponse` with SSE frames; `X-Request-Id` header (`src/api/main.py:980`).
- **# FAILURE:** Mid-stream LLM failure → error event + `[DONE]`; client must handle partial answers (`src/api/main.py:919-920`). Streaming answers are **not cached** (`src/api/main.py:921-922`) — repeated streamed queries pay full pipeline cost each time.

---

## Trace B — Ingest a document (upload → validate → dedupe → version → persist → load/clean/chunk/embed/store → status log)

**Concrete request:** `POST /documents/upload` with file `lte_bearers.md` (a telecom document describing EPS bearers), content ~3 KB. Content = markdown text about EPS bearers in LTE. Called by an admin (`X-Access-Level: 3`, `can_ingest: true`).

This is the "provision a new circuit into the OSS inventory" path: validate the service order, check for duplicates, write the record, index it, and log the result.

> Note: the actual `data/raw/` directory currently holds generic Acme Corp corporate policies (HR/IT/travel), not telecom docs — a purpose/source mismatch noted in `docs/learn/_plan.md:48` and `docs/learn/01_business_context.md:181-182`. The trace below uses `lte_bearers.md` as a concrete example of what *would* be ingested for the system's stated telecom purpose.

---

### B1. Upload endpoint + authZ — `src/api/main.py` (upload route)

- **Input:** `filename = "lte_bearers.md"`, `content` (bytes, ~3000 bytes), `user_info` with `can_ingest: true` (admin).
- **Where:** upload route `src/api/main.py` — the `/documents/upload` endpoint (admin-only, `can_ingest` required at `:911-912`), delegating to `doc_manager.upload(filename, content, vector_store)` (`src/ingestion/manager.py:214`). `doc_manager` is a module-level `DocumentManager()` at `src/api/main.py:259`.
- **What it does:** Confirms the caller has `can_ingest` (only admin, level 3), then calls `DocumentManager.upload` which orchestrates validate → dedupe → version → persist → index → record.
- **Output / return:** `IngestRecord` (status `ok`/`duplicate`/`superseded`/`failed`), returned to the caller as JSON.
- **# FAILURE:** Non-admin caller → 403 "Insufficient permissions" (`src/api/main.py:911-912`, `:804-808` for `/ingest`). Upload not implemented as a separate route in the version read — single-file upload goes through `/documents/upload`; directory ingest goes through `/ingest` (`src/api/main.py:1071`).

---

### B2. Validate: filename + extension + size + hash — `src/ingestion/manager.py:175-210`

- **Input:** `filename = "lte_bearers.md"`, `content` (bytes).
- **Where:** `src/ingestion/manager.py:175` (`DocumentManager.validate`), `:200-210` (checks), `:44-45` (`ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}`, `MAX_FILE_SIZE_BYTES = 10 MB`).
- **What it does:** Strips path components (`Path(filename).name`), rejects if the name contained path separators (traversal guard), rejects disallowed extensions, rejects empty content, rejects files over 10 MB, then returns the SHA-256 hex digest of the content (`hashlib.sha256(content).hexdigest()`, `:210`).
- **Output / return:** content_hash e.g. `"a3f5c9e1...d4b2"` (64 hex chars); raises `UploadError` on any validation failure (`src/ingestion/manager.py:48`).
- **# FAILURE:** `filename = "../../etc/passwd"` → rejected ("filename must be a plain name, no path separators", `:201-202`). `.exe` → rejected ("extension not allowed", `:204-205`). Empty file → rejected (`:206-207`). >10 MB → rejected (`:208-209`). The caller gets a 400 with the `UploadError` message.

---

### B3. Dedupe: SHA-256 content hash vs. existing sources — `src/ingestion/manager.py:248-254`

- **Input:** `content_hash` (from B2), `name = "lte_bearers.md"`, `vector_store.list_sources()`.
- **Where:** `src/ingestion/manager.py:248` (`existing = vector_store.list_sources()`), `:249-254` (loop comparing `row.get("filename") == name and row.get("content_hash") == content_hash`).
- **What it does:** Lists all sources already in the vector store; if a source with the same filename *and* same content hash already exists, this is a duplicate — no re-index, status `"duplicate"`.
- **Output / return:** If duplicate, `IngestRecord(name, content_hash, status="duplicate", chunks=<existing chunk_count>, doc_version=...)` (`:251-252`), recorded (`:253`) and returned (`:254`) — indexing is skipped entirely.
- **# FAILURE:** Identical bytes re-uploaded → `duplicate`, no-op, no new chunks (protects against duplicate storage, `src/ingestion/manager.py:247-254`). If `list_sources()` fails (store error), the exception propagates and the upload fails.

---

### B4. Version: bump doc_version + supersede old — `src/ingestion/manager.py:256-260`

- **Input:** `name = "lte_bearers.md"`, `content_hash` (different from previous), `prev` (prior version record from `:257`), `new_version = self._next_version(...)`.
- **Where:** `src/ingestion/manager.py:256-260` (`prev = next(...)`, `:258` `_next_version`, `:259-260` `mark_source_current(name, current=False)`), `_next_version` at `:278-304`.
- **What it does:** If the same filename exists with a *different* hash, the old version is superseded: `vector_store.mark_source_current(name, current=False)` (`src/store/vector_store.py:281`) flips old chunks' `is_current=False`, and `doc_version` is bumped (e.g. `"1.0"` → `"1.1"`, `:300-302`). If no previous version, `new_version = "1.0"`.
- **Output / return:** `new_version` string e.g. `"1.1"`; old chunks marked `is_current=False` in the store.
- **# FAILURE:** Unparseable version string → `_next_version` returns `"1.0"` (never raises, `:303-304`). If `mark_source_current` fails, the exception propagates to `upload`'s try/except (`:271-274`) and the record is marked `"failed"`.

---

### B5. Persist: write bytes to uploads/ — `src/ingestion/manager.py:263-264`

- **Input:** `dest = uploads/lte_bearers.md`, `content` (bytes).
- **Where:** `src/ingestion/manager.py:263` (`dest = self.upload_dir / name`), `:264` (`dest.write_bytes(content)`), `UPLOAD_DIR` at `:41` (`project_root / "uploads"`), created at `:136`.
- **What it does:** Writes the raw file bytes to disk under `uploads/`. This is the staging copy; the vector store holds the indexed chunks.
- **Output / return:** file written to `uploads/lte_bearers.md`.
- **# FAILURE:** Disk full / permission error → `write_bytes` raises, caught by `upload`'s try/except (`:271-274`), record marked `"failed"`. **[GAP]** Not atomic: the file is written *before* indexing; if indexing fails, an orphaned file remains in `uploads/` (`src/ingestion/manager.py:235-237`).

---

### B6. Index pipeline: load → clean → chunk → embed → store — `src/ingestion/manager.py:306-348`

- **Input:** `path` (uploads/lte_bearers.md), `vector_store`, `content_hash`, `doc_version` ("1.1").
- **Where:** `src/ingestion/manager.py:306` (`_index_file`), `:336` (`load_document(str(path))`, `src/ingestion/loaders.py:12`), `:340` (`clean_documents([doc])`, `src/ingestion/cleaner.py:25`), `:341` (`chunk_documents(..., strategy="recursive", chunk_size=500, chunk_overlap=50)`, `src/ingestion/chunker.py:43`), `:346` (`embed_documents(chunked)`, `src/embedding/embedder.py:461`), `:347` (`add_documents_to_store(vector_store, embedded)`, `src/store/vector_store.py:704`).
- **What it does:** Runs the full per-file pipeline:
  1. **Load:** `load_document` reads the file into `{"content": ..., "metadata": {...}}` by extension (PDF/DOCX/TXT/MD). PDF/DOCX never fabricate text — record extraction errors instead (`src/ingestion/loaders.py:157-160,205-207`).
  2. **Metadata:** stamps `content_hash`, `doc_version`, `is_current=True` on the doc (`:337-339`).
  3. **Clean:** HTML unescape, tag stripping, whitespace/newline normalization (`src/ingestion/cleaner.py:25`).
  4. **Chunk:** recursive strategy (paragraph → sentence → fixed), 500 chars / 50 overlap — each chunk gets `content_hash`, `doc_version`, `is_current=True` (`:342-345`). A 3 KB doc → ~6-7 chunks of ~500 chars.
  5. **Embed:** `embed_documents` batch-embeds chunks via `embed_batch` → `embed_text` (content-hash cache check first, `src/embedding/embedder.py:299-334`, then `_embed_text_internal` with retry/backoff `:287`, or mock `_mock_embed` producing a deterministic 384-d vector `:508`). Each embedded doc gets `embedding` + metadata flags (`embedded`, `embedding_model`, `embedding_dimension`, `embedding_cached`).
  6. **Store:** `add_documents_to_store` pulls the `"embedding"` vector from each doc and calls `store.add_vectors` (`src/store/vector_store.py:162`) → ChromaDB `PersistentClient` collection `rag_academy` at `data/chroma_db` (`src/store/vector_store.py:79-114`), or in-memory fallback (`:67-73,116-114`).
- **Output / return:** `len(embedded)` = number of chunks embedded and stored (e.g. 6). Sets `rec.chunks = 6`, `rec.status = "ok"` (or `"superseded"` if a prev version existed, `:270`).
- **# FAILURE:** Any stage (loader/cleaner/chunker/embedder) raises → caught by `upload`'s try/except (`src/ingestion/manager.py:271-274`), `rec.status = "failed"`, `rec.error` truncated to 300 chars (`:274`). Embedding failure after retries degrades to mock vectors (no raise, `embedder.py:346-350`) — indexing proceeds but with poor vectors. Large doc → all chunks held in memory before embedding (OOM risk, `# GAP` at `:323-324`).

---

### B7. Record status + return — `src/ingestion/manager.py:275-276`

- **Input:** `rec` (IngestRecord with status/chunks/version/error).
- **Where:** `src/ingestion/manager.py:275` (`self._record(rec)`), `:276` (`return rec`), `_record` at `:162-167`, `_save_status` at `:154-160`.
- **What it does:** Appends the record to the in-memory list (capped at 500, dropping oldest beyond that, `:165-166`) and persists all records to `data/ingest_status.json` (`:154-160`). Returns the `IngestRecord` to the API caller.
- **Output / return:** `IngestRecord` e.g. `{filename: "lte_bearers.md", content_hash: "a3f5...c9e1" (truncated to 12 chars in to_dict, `:99`), status: "ok", chunks: 6, doc_version: "1.1", error: "", ts: <epoch>}`.
- **# FAILURE:** Status file corrupt/unreadable → `_load_status` returns `[]` (start fresh, `:151-152`); `_save_status` failure would surface as an exception. Beyond 500 records, oldest are dropped (`:165-166`) — no concurrency control on the JSON file (`# GAP`, `:124-125`).

---

### B8. (Related) Delete cascade — `src/ingestion/manager.py:362-391`

- **Input:** `filename = "lte_bearers.md"`, `vector_store`.
- **Where:** `src/ingestion/manager.py:362` (`delete`), `:386` (`vector_store.delete_by_source(name)`, `src/store/vector_store.py:519`), `:387-390` (unlink file from `uploads/`).
- **What it does:** Deletes all chunks for the source from the vector store first, then deletes the file from `uploads/` if present. Delete-then-write, not upsert — prevents stale/duplicate chunks on re-ingest.
- **Output / return:** `{"filename": "lte_bearers.md", "chunks_deleted": 6, "file_deleted": true}`.
- **# FAILURE:** Vector-store deletion succeeds but disk deletion fails (or vice versa) → inconsistent state, no rollback (`# GAP`, `:377-379`).

---

### B9. (Related) Reindex — `src/ingestion/manager.py:393-437`

- **Input:** `vector_store`, optional `data_dir`.
- **Where:** `src/ingestion/manager.py:393` (`reindex`), `:423` (`vector_store.reset()` wipes the store), `:429-436` (loads/cleans/chunks/embeds/stores every doc from `uploads/` + optional data dir).
- **What it does:** Full rebuild — wipes the vector store, then re-indexes all documents. Disaster recovery for the index after config changes (e.g. new chunk size).
- **Output / return:** `{"chunks_indexed": <total>, "sources_dirs": [...]}`.
- **# FAILURE:** `reset()` wipes everything *before* reindexing — if reindex fails midway, the store is empty and old data is gone (`# GAP`, `:409-411`). Individual doc load failures are skipped (error printed), indexing continues (`:406-407`).

---

## Failure mode summary table

| # | Hop (Trace A / B) | Top 2 failure modes | Symptom a user/operator would see |
|---|---|---|---|
| A1 | Middleware (request ID + trace + metrics) — `src/api/main.py:507` | Downstream handler raises; trace recording fails | Client gets error response *with* `X-Request-Id` (good); if trace ring buffer full, oldest traces dropped (`src/observability.py:485-486`) |
| A2 | AuthN/AuthZ — `src/api/main.py:276,313` | Missing/invalid bearer token (when `API_TOKEN` set); client spoofs `X-Access-Level` (dev mode) | 401 "Missing/Invalid token"; or any caller self-asserts admin without a token |
| A3 | Rate limit — `src/api/main.py:768` | Client exceeds 30 req/60s; per-process limiter not shared across workers | 429 "Rate limit exceeded"; client can multiply limit by worker count behind a LB |
| A4 | Cache lookup — `src/api/main.py:784` | Stale wrong answer cached (no TTL); hash collision (astronomically unlikely) | Fast but wrong answer, `latency_ms: 1.0`; wrong answer persists until FIFO eviction |
| A5 | Pipeline guard — `src/api/main.py:792` | `rag_pipeline is None` (startup failed) | 503 "RAG pipeline not initialized" |
| A6 | Retrieve — `src/api/main.py:807` | Vector store empty; embedding/model failure; dimension mismatch | No chunks → insufficiency answer; poor semantic vectors; 500 on dimension mismatch |
| A7 | Rerank — `src/api/main.py:821` | Cross-encoder missing → passthrough; predict exception | Answers ranked by weaker first-pass scores; no exception to caller |
| A8 | Generate — `src/api/main.py:835` | LLM provider raises; empty context | "I don't have enough information…" fallback (mock) or provider error message |
| A9 | Sources/latency assembly — `src/api/main.py:846` | Missing metadata fields; char-count truncation under-fills token budget | "Unknown" source; context shorter than budget allows |
| A10 | Cache write — `src/api/main.py:876` | FIFO eviction of useful entries; no TTL on stale answers | Cache miss on repeat (evicted); wrong answer lingers until rotation |
| A11 | Feedback background — `src/api/main.py:879` | Background task raises; in-memory store lost on restart | User unaffected; feedback record lost |
| A12 | Stream variant — `src/api/main.py:893` | Mid-stream LLM failure; no cache on streaming | Error event + `[DONE]`; client gets partial answer, pays full cost each stream |
| B1 | Upload authZ — `src/api/main.py:911` | Non-admin caller | 403 "Insufficient permissions" |
| B2 | Validate — `src/ingestion/manager.py:175` | Path traversal; bad extension; empty; >10 MB | 400 `UploadError` with reason |
| B3 | Dedupe — `src/ingestion/manager.py:248` | Identical re-upload | `duplicate` status, no re-index (correct behavior, not a failure) |
| B4 | Version — `src/ingestion/manager.py:258` | Unparseable version string | Falls back to `"1.0"` (never raises) |
| B5 | Persist — `src/ingestion/manager.py:264` | Disk full / permission | Record marked `"failed"`; orphaned file may remain in `uploads/` |
| B6 | Index pipeline — `src/ingestion/manager.py:306` | Loader/cleaner/chunker/embedder raises; OOM on large doc | Record marked `"failed"`, error truncated to 300 chars; degrades to mock vectors |
| B7 | Status record — `src/ingestion/manager.py:275` | Corrupt status file; >500 records | Start fresh (lose history); oldest records dropped |
| B8 | Delete cascade — `src/ingestion/manager.py:362` | Vector-store delete succeeds, disk delete fails (or vice versa) | Inconsistent state, no rollback |
| B9 | Reindex — `src/ingestion/manager.py:393` | `reset()` wipes store then reindex fails | Store empty, old data gone |

---

## UNVERIFIED items

The following could not be confirmed by running the system in this session (no live server, dependencies may not be installed). They are grounded in source reading, not runtime observation:

1. **End-to-end startup:** `uvicorn src.api.main:app` + Streamlit actually serving requests was not run (`docs/learn/02_architecture.md:320-321`).
2. **ChromaDB vs. fallback:** whether `chromadb` is installed and `data/chroma_db` is active at runtime, or the in-memory fallback is used (`src/store/vector_store.py:67-73`).
3. **sentence-transformers model:** whether `all-MiniLM-L6-v2` downloads/loads; if unavailable the mock embedder is used silently (`src/embedding/embedder.py:229-238`).
4. **Line-number drift:** annotations may have shifted lines by one or two; each `file:line` should be re-checked against the working tree (`docs/learn/02_architecture.md:323`).
5. **NVIDIA provider path:** `LLM_PROVIDER=nvidia` not exercised (no key, no network); behavior depends on real key + `integrate.api.nvidia.com` (`docs/learn/02_architecture.md:324`).
6. **`/documents/upload` route:** the single-file upload endpoint was referenced (`src/api/main.py:903` per architecture doc) but the version read here has `/documents/upload` at a different location; exact line not confirmed in this read.

---

## Related docs in this series

- `00_START_HERE.md` — how to read the learn docs
- `01_business_context.md` — customer, pain, cost, success
- `02_architecture.md` — components, data flow, trust boundaries, scaling
- `03_glossary.md` — term definitions with `file:line`
- `04_gap_bridge.md` — "web app does X; production system does Y because Z"
- `06_interview_pack.md` — interview Q&A and break-it drills
- `07_issues_found.md` — bugs, smells, UNVERIFIED items

No source code was changed to produce this document.
