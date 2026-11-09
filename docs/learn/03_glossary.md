# Glossary — rag-academy RAG System

Plain-English definitions of every domain term, library, pattern, acronym, and config key used in the repo.
Each entry lists where it appears (`file:line`), why it matters, and a one-line analogy.

Entries are grouped alphabetically within sections: RAG pipeline, retrieval, generation, providers,
ingestion & document lifecycle, key management & auth, observability & operations, libraries &
frameworks, evaluation, and telecom domain.

---

## RAG Pipeline

### RAG (Retrieval-Augmented Generation)
**Definition:** An architecture that retrieves relevant document chunks from a vector store and feeds them
as context to an LLM, so the generated answer is grounded in source material rather than the model's
parametric memory.
**file:line:** `src/generation/answer.py:30` (AnswerGenerator), `src/agent/agent.py:50` (RAGAgent),
`src/config.py:1` (module docstring: "configuration for the RAG system")
**Why it matters:** It is the core pattern of the entire repo — without retrieval, the LLM would have no
access to the ingested telecom documents and would hallucinate or refuse every answer.
**Analogy:** An open-book exam: the model looks up the relevant pages before writing the answer.

### chunk
**Definition:** A contiguous piece of text produced by splitting a document into smaller units, each with its
own metadata (chunk_index, total_chunks, source, doc_version, is_current).
**file:line:** `src/ingestion/chunker.py:21` (TextChunker), `:43` (chunk_text), `:217` (chunk_document),
`src/ingestion/manager.py:172` (_index_file calls chunk_documents)
**Why it matters:** Embedding and retrieval operate on chunks, not whole documents; chunk size and overlap
directly control recall and context-window fit.
**Analogy:** Tearing a textbook into indexed flashcards so you can study only the relevant pages.

### embedding
**Definition:** A fixed-size numeric vector (384 dimensions for all-MiniLM-L6-v2) that represents the
semantic meaning of a piece of text, produced by a sentence-transformers model or a deterministic mock.
**file:line:** `src/embedding/embedder.py:154` (Embedder class), `:299` (embed_text), `:424` (embed_document),
`:508` (_mock_embed fallback)
**Why it matters:** Embeddings are what make semantic (dense) search possible — chunks with similar meaning
land close together in vector space.
**Analogy:** A fingerprint for a paragraph: two paragraphs about the same topic get similar fingerprints.

---

## Retrieval

### vector store
**Definition:** A database that stores embedding vectors plus document metadata and answers nearest-neighbor
queries ("which chunks are most like this query?").
**file:line:** `src/store/vector_store.py:76` (VectorStore class), `:334` (search), `:519` (delete_by_source),
`:236` (list_sources)
**Why it matters:** It is the long-term memory of the RAG system — ingested chunks live here and retrieval
queries it at question time.
**Analogy:** A library's card catalog that also knows which books are about the same subject.

### ChromaDB
**Definition:** The embedded, on-disk vector database used by default (persisted under `data/chroma_db`),
with an in-memory cosine-similarity fallback when ChromaDB is unavailable.
**file:line:** `src/store/vector_store.py:67` (`import chromadb`), `:128` (PersistentClient),
`:154` (_init_fallback), `:143` (create_collection); `src/config.py:26` (per plan: ChromaDB at data/chroma_db)
**Why it matters:** It provides persistent, scalable similarity search; the fallback keeps the system runnable
offline and in tests without a database server.
**Analogy:** A filing cabinet with a built-in "find similar" feature, plus a notebook backup if the cabinet
is missing.

### dense retrieval
**Definition:** Retrieval that embeds the query into a vector and uses the vector store to find the most
semantically similar chunks (cosine similarity).
**file:line:** `src/retrieval/retriever.py:51` (Retriever class), `:98` (retrieve), `:122`
(query_embedding = self.embedder.embed_text(query))
**Why it matters:** It surfaces conceptually related chunks even when they share no keywords with the query.
**Analogy:** Asking a librarian "books about X" and getting thematically similar titles.

### sparse retrieval
**Definition:** Retrieval that scores chunks by exact term overlap using BM25 (term frequency /
inverse document frequency), without embeddings.
**file:line:** `src/retrieval/sparse_retriever.py:53` (BM25Retriever class), `:79` (_tokenize),
`:139` (get_score), `:185` (retrieve)
**Why it matters:** It catches exact acronym/keyword matches (e.g. "TAC", "IMEI") that dense similarity can
under-rank.
**Analogy:** Ctrl-F in a document: finds the exact words you typed.

### hybrid search
**Definition:** Retrieval that combines dense (semantic) and sparse (BM25) result lists and fuses them into
one ranked list.
**file:line:** `src/retrieval/hybrid.py:51` (HybridRetrieverRRF class), `:129` (_reciprocal_rank_fusion),
`:82` (retrieve); `src/retrieval/retriever.py:193` (HybridRetriever weighted blending)
**Why it matters:** Neither signal alone is sufficient — dense finds meaning, sparse finds exact terms; together
they improve recall.
**Analogy:** Hiring two researchers, one who understands the topic and one who is great at keyword search,
then merging their shortlists.

### BM25
**Definition:** A classical ranking function that scores a query against a document using term frequency and
inverse document frequency, with length normalization (k1=1.5, b=0.75 defaults).
**file:line:** `src/retrieval/sparse_retriever.py:53` (BM25Retriever), `:175` (idf formula),
`:178` (numerator/denominator scoring)
**Why it matters:** It is the sparse half of hybrid retrieval; rare matching terms boost a document's score.
**Analogy:** A word-count contest where uncommon shared words count more than common ones.

### RRF (reciprocal rank fusion)
**Definition:** A fusion method that combines several ranked lists by position rather than raw score:
each document earns 1/(k + rank + 1) per list, and the totals determine the final order.
**file:line:** `src/retrieval/retriever.py:327` (reciprocal_rank_fusion function), `:352`
(formula 1.0 / (k + rank + 1)); `src/retrieval/hybrid.py:51` (HybridRetrieverRRF), `:129`
(_reciprocal_rank_fusion), `:155` (formula)
**Why it matters:** Dense cosine scores and BM25 scores live on different scales; RRF fuses them fairly
without re-normalization.
**Analogy:** Two judges rank contestants; a contestant ranked #1 by either judge rises to the top, even if
their raw scores differ.

### reranker
**Definition:** A model that re-scores an initial set of retrieved (query, document) pairs more accurately
than first-pass retrieval. This repo has a cross-encoder Reranker and a model-free MetadataReranker.
**file:line:** `src/retrieval/reranker.py:53` (Reranker class), `:100` (rerank), `:160` (MetadataReranker),
`:179` (MetadataReranker.rerank)
**Why it matters:** First-pass retrieval (cosine/BM25) is approximate; reranking lifts the most relevant
chunks to the top before the generator reads them.
**Analogy:** A second, more careful judge re-sorts the shortlist after the first quick scan.

---

## Generation

### LLM provider
**Definition:** The abstract interface (LLMProvider) plus concrete backends (mock, NVIDIA NIM, placeholder
OpenAI/Anthropic) that the RAG system calls to generate text, keeping generation code provider-agnostic.
**file:line:** `providers/llm.py:46` (LLMProvider ABC), `:111` (MockLLMProvider), `:312`
(create_llm_provider factory), `:377` (create_llm_provider_from_env)
**Why it matters:** It is the seam that lets the same code run offline with mock and in production with NVIDIA
NIM, selected purely by env var.
**Analogy:** A universal power adapter: the device (generation code) doesn't care which wall socket
(backend) it's plugged into.

### NVIDIA NIM
**Definition:** The production LLM backend — an OpenAI-compatible chat-completions API at
`https://integrate.api.nvidia.com/v1` (default model `meta/llama-3.1-70b-instruct`) with retries,
backoff, multi-key failover, streaming, structured JSON output, usage tracking, and secret redaction.
**file:line:** `providers/llm_nvidia.py:1` (module docstring), `:68` (DEFAULT_BASE_URL), `:69` (DEFAULT_MODEL),
`:226` (NVIDIAProvider class), `:420` (_post with retry/failover)
**Why it matters:** It is the real production generator; the file is hardened for outages, rate limits, and
key rotation so answers keep flowing.
**Analogy:** A licensed contractor you call for the real job, with backup subcontractors if one is unavailable.

### mock provider
**Definition:** A deterministic fake LLM (MockLLMProvider) that parses the prompt for Context/Question
markers and returns a canned answer with no network call — used for tests and local development only.
**file:line:** `providers/llm.py:111` (MockLLMProvider class), `:136` (generate), `:174` (stream_chat);
`src/config.py:26` (llm_provider defaults to "mock")
**Why it matters:** It lets tests and local dev run without an API key, network, or cost, and never reaches
production traffic.
**Analogy:** A flight simulator: realistic enough to practice, never leaves the ground.

### token budget
**Definition:** The maximum number of estimated tokens allowed for retrieved context before truncation,
so the prompt stays within the model's context window (default max_tokens=2000, chars_per_token=4.0).
**file:line:** `src/generation/answer.py:122` (_build_context_within_budget docstring), `:78`
(chars_per_token = 4.0), `:168` (budget check `used_tokens + doc_tokens > max_tokens`),
`:220` (generate_answer_with_context max_context_tokens=2000)
**Why it matters:** It bounds cost and latency and prevents the context from overflowing the model's context
window, which would truncate or break generation.
**Analogy:** A suitcase with a weight limit — you pack the most important items first and stop when full.

### context window
**Definition:** The maximum amount of text (tokens) an LLM can process in a single prompt; the answer
generator budgets context against it to avoid overflow.
**file:line:** `src/generation/answer.py:102` ("used to keep prompts within the model's context window"),
`:126` ("Ensures the context fed to the LLM fits the model's context window"), `:243` (GAP note)
**Why it matters:** Exceeding it cuts off the prompt; staying within it keeps the full question and context
visible to the model.
**Analogy:** A whiteboard of fixed size — write only what fits, or the top gets erased.

---

## Ingestion & Document Lifecycle

### dedupe by SHA-256
**Definition:** Content-addressed deduplication: re-uploading a file whose SHA-256 hash matches an already
ingested file is a no-op (status "duplicate"), so identical bytes are never indexed twice.
**file:line:** `src/ingestion/manager.py:98` (validate returns hashlib.sha256(content).hexdigest()),
`:119-126` (dedupe check against list_sources), module docstring `:8`
**Why it matters:** It prevents duplicate chunks from polluting the store and contradicting retrieval results
on re-upload.
**Analogy:** Returning a library book you already own because its barcode matches one on your shelf.

### doc_version
**Definition:** A version string (e.g. "1.0", "1.1") recorded in each chunk's metadata, bumped when the same
filename is re-uploaded with different content; old versions keep is_current=False.
**file:line:** `src/ingestion/loaders.py:36` ("doc_version" extracted from filename), `:56` (_extract_version),
`src/ingestion/manager.py:45` (IngestRecord.doc_version field), `:130` (_next_version bumps minor),
`:170` (metadata["doc_version"] = doc_version)
**Why it matters:** It lets the system keep multiple versions of a document while surfacing only the current
one, so answers don't go stale.
**Analogy:** Edition numbers on a textbook — you cite the latest printing.

### is_current
**Definition:** A boolean metadata flag on each chunk marking whether it belongs to the current version of
its source file; retrieval filters by it (current_version_only) to exclude stale versions.
**file:line:** `src/ingestion/loaders.py:37` ("is_current": True), `:170` (manager sets is_current=True),
`src/store/vector_store.py:259` (list_sources default is_current=False), `:281` (mark_source_current flips it),
`src/retrieval/retriever.py:126` (current_version_only injects {"is_current": True})
**Why it matters:** It prevents old, superseded document versions from shadowing the current answer.
**Analogy:** A "latest edition" sticker on the current book; older editions are archived.

### idempotency
**Definition:** A property where repeating an operation has the same effect as doing it once — re-uploading
identical content (dedupe) or re-ingesting (delete-then-write via delete_by_source) produces no duplicate
chunks.
**file:line:** `src/ingestion/manager.py:8` (module docstring: "Dedupe by SHA-256... re-uploading identical bytes
is a no-op"), `:119-126` (dedupe), `:545` (delete_by_source docstring: delete-then-write, not upsert)
**Why it matters:** It makes uploads and re-ingests safe to retry without creating duplicates or stale data.
**Analogy:** Pressing an elevator button twice still calls the elevator once.

### allow-list
**Definition:** A registry of permitted tool names (ToolRegistry) that the agent may call; the LLM can only
propose, and the application validates + authorizes before execution — tools not on the list are rejected.
**file:line:** `src/agent/tools.py:81` (ToolRegistry docstring: "Allow-list of tools"), `:111`
("tool {name!r} is not in the allow-list"), `:244-249` (build_default_tools registers the allow-listed tools),
`src/agent/agent.py:102` (planner choice validated against self.tools.names())
**Why it matters:** It is the primary safety rail — the LLM never gets raw filesystem/DB/OS access, only
pre-approved, schema-validated tool calls.
**Analogy:** A restaurant menu: the customer (LLM) may order only what's listed, nothing off-menu.

---

## Key Management & Auth

### key pool
**Definition:** A thread-safe collection of NVIDIA API keys with round-robin rotation and failover; keys come
from NVIDIA_API_KEYS (comma-separated) or the single NVIDIA_API_KEY for back-compat.
**file:line:** `src/key_pool.py:54` (KeyPool class), `:70` (from_env), `:118` (current_key advances round-robin),
`:149` (mark_bad rotates on auth failure)
**Why it matters:** It spreads usage across keys and automatically rotates away a bad/expired key, keeping the
production LLM path available.
**Analogy:** A keyring where you try each key in turn until one opens the door.

### key fingerprint
**Definition:** A non-reversible identifier for an API key (first 4 + last 4 characters, e.g. "abcd...1234")
used in logs, registry, and alerts instead of the full secret.
**file:line:** `src/key_pool.py:28` (fingerprint function), `:43` (KeyEntry.fingerprint), `:151` (mark_bad looks
up by fingerprint); `src/key_monitor.py:57` (fingerprint(key)); `src/observability.py:42` (_SECRET_PATTERNS
redacts full keys)
**Why it matters:** It lets operators correlate logs to a key without ever exposing the secret itself.
**Analogy:** A luggage tag showing only the first and last letters of your name — enough to identify, not to
steal.

### round-robin
**Definition:** A selection strategy that cycles through active pool entries in order, advancing after each
use so load is spread evenly across keys.
**file:line:** `src/key_pool.py:102` (_next_active: "Raw-index round-robin over entries"), `:118`
(current_key advances), `:55` (class docstring: "round-robin rotation"), `:141` (rotate)
**Why it matters:** It prevents any single NVIDIA key from being overused and rate-limited while others sit
idle.
**Analogy:** Dealing cards one at a time to each player in turn.

### failover
**Definition:** On an NVIDIA auth failure (401/403), the current key is marked bad and the next key in the
pool is tried, bounded by pool size; if all keys are bad, the request errors out.
**file:line:** `src/key_pool.py:1` (module docstring: "rotation, failover"), `:149` (mark_bad returns next usable
key), `:159` (mark_expired); `providers/llm_nvidia.py:451` (_post rotates on 401/403 via key_pool.mark_bad)
**Why it matters:** A revoked or rate-limited key doesn't take down the whole service — the system self-heals
by switching keys.
**Analogy:** If one cashier's register is down, you move to the next open line.

---

## Observability & Operations

### rate limiter
**Definition:** A sliding-window request limiter keyed by client_id that rejects clients exceeding
max_requests within window_seconds, protecting the LLM token budget and pipeline capacity.
**file:line:** `src/api/main.py:195` (RateLimiter class), `:224` (is_allowed), `:257` (rate_limiter instance,
30 requests per 60s); `src/embedding/embedder.py:116` (rate_limit decorator for embedding calls)
**Why it matters:** It stops one client from flooding the RAG pipeline and exhausting token budget or
vector-store capacity.
**Analogy:** A bouncer letting only so many people in per minute.

### query cache
**Definition:** An in-memory FIFO cache keyed by an MD5 hash of the full request (query + all flags) that
returns cached QueryResponse objects for repeated identical queries, skipping retrieve/rerank/LLM.
**file:line:** `src/api/main.py:126` (QueryCache class), `:151` (__init__, max_size=100), `:256`
(query_cache = QueryCache(max_size=200)), `:541` (get), `:630` (set)
**Why it matters:** Repeating the same telecom question doesn't re-run the whole pipeline, cutting latency and
LLM token cost for hot queries.
**Analogy:** A "frequently asked questions" board — you read the answer instead of recomputing it.

### correlation ID / request_id
**Definition:** A per-request unique identifier (12-hex-char uuid) attached to every log line, trace, and
response header (X-Request-Id) so a request's journey can be followed through pipeline stages.
**file:line:** `src/observability.py:27` (request_id_ctx ContextVar), `:30` (new_request_id), `:36`
(get_request_id), `:84` (JsonFormatter includes request_id); `src/api/main.py:416` (middleware assigns request_id),
`:434` (response.headers["X-Request-Id"])
**Why it matters:** It ties together logs, metrics, and traces for a single request across distributed stages.
**Analogy:** A tracking number on a package that follows it through every handoff.

### secret redaction
**Definition:** Scrubbing secret-looking tokens (NVIDIA API keys, sk- keys, Bearer tokens) from logs and
exception messages so secrets never leak to stdout or error responses.
**file:line:** `src/observability.py:49` (redact function), `:57` (RedactFilter logging filter),
`src/observability.py:42` (_SECRET_PATTERNS regexes); `providers/llm_nvidia.py:201` (_redact scrubs api_key),
`:406` (_classify_error redacts before wrapping)
**Why it matters:** It is a compliance requirement — credentials must never appear in logs, traces, or HTTP
error bodies.
**Analogy:** A bank teller blacking out your account number on a receipt before handing it to you.

---

## Libraries & Frameworks

### FastAPI
**Definition:** The Python web framework that exposes the RAG pipeline as an HTTP API (ask, search, ingest,
upload, delete, agent, health, metrics, traces) with Pydantic request/response models and dependency-based
auth.
**file:line:** `src/api/main.py:404` (app = FastAPI(...)), `:32` (`from fastapi import FastAPI`),
`:501` (@app.post("/ask")), `:779` (@app.delete("/documents/{source}"))
**Why it matters:** It is the top entry point — the UI, curl clients, and external services reach the RAG
system only through this HTTP surface.
**Analogy:** The reception desk and phone system of an office building.

### Streamlit
**Definition:** The browser-based UI that calls the FastAPI API, with tabs for Ask (streaming SSE), Search,
Agent, Traces & Metrics, and Documents, plus file upload.
**file:line:** `ui/streamlit_app.py:14` (`import streamlit as st`), `:57` (st.set_page_config), `:80` (tabs),
`:143` (agent tab), `:187` (file uploader)
**Why it matters:** It gives non-technical users (FDE interview candidates) a point-and-click way to query the
RAG system without calling the API directly.
**Analogy:** The storefront window of a shop whose back office is the FastAPI API.

### pydantic
**Definition:** The data-validation library used for request/response models (QueryRequest, SearchRequest,
IngestRequest, FeedbackRequest), tool input schemas (SearchInput, FilenameInput, etc.), and structured LLM
output (GroundedAnswer).
**file:line:** `src/api/main.py:34` (`from pydantic import BaseModel`), `:62` (QueryRequest),
`:106` (FeedbackRequest); `src/agent/tools.py:20` (`from pydantic import BaseModel, ValidationError`),
`:141` (SearchInput); `src/generation/answer.py:11` (`from pydantic import BaseModel, Field`),
`:363` (GroundedAnswer); `providers/llm_nvidia.py:62` (`from pydantic import BaseModel, ValidationError`)
**Why it matters:** It enforces schemas at the edge — untrusted LLM/tool input is validated before execution,
and structured output is guaranteed to match a schema.
**Analogy:** A form with required fields and dropdowns that rejects garbage before it's submitted.

### sentence-transformers
**Definition:** The Python library providing SentenceTransformer (dense bi-encoder embeddings) and CrossEncoder
(reranking) models; optional at import time with graceful mock/passthrough fallback.
**file:line:** `src/embedding/embedder.py:230` (`from sentence_transformers import SentenceTransformer`),
`:291` (self.model.encode([text])); `src/retrieval/reranker.py:48`
(`from sentence_transformers import CrossEncoder`), `:94` (model load)
**Why it matters:** It supplies the embedding and reranking models that power semantic retrieval and re-scoring;
the fallback keeps the system importable without it.
**Analogy:** A reference dictionary the system consults to translate text into numbers.

---

## Evaluation

### golden set
**Definition:** A benchmark dataset of questions with expected answer substrings, expected sources, question
type, and hop count — used to run repeatable end-to-end evaluation.
**file:line:** `src/evaluation/golden_set.json` (the file itself, 30 questions), `src/evaluation/run_eval.py:334`
(evaluate_golden_set), `:808` (--golden-set CLI argument)
**Why it matters:** It is the ground truth that lets the team measure whether changes improve or regress
answer quality over time.
**Analogy:** A practice exam with an answer key used to check if studying helped.

### eval
**Definition:** The process of assessing RAG quality with retrieval metrics (hit-rate@k, recall@k, precision,
MRR, nDCG, average precision) and generation metrics (answer relevance, faithfulness, contradiction, completeness,
some via LLM-as-judge).
**file:line:** `src/evaluation/metrics.py:16` (RetrievalMetrics class), `:244` (GenerationMetrics class),
`:476` (RAGEvaluator); `src/evaluation/run_eval.py` (whole file: RAGEvaluationPipeline, ablation studies)
**Why it matters:** It quantifies whether the pipeline actually finds and answers correctly, replacing vibes
with numbers.
**Analogy:** Grading a practice test with a rubric to know what to improve.

---

## Telecom Domain

### FDE
**Definition:** Field Development Engineer — the target audience for this RAG system, using it as study aid
for interview preparation.
**file:line:** `CLAUDE.md:3` ("study aid for FDE (Field Development Engineer) interview preparation"),
`:42` (prepares users effectively for FDE interviews)
**Why it matters:** It defines the user persona, educational tone, and practical, field-relevant focus of the
system's answers.
**Analogy:** The student the whole system was built to tutor.

### LTE/5G
**Definition:** Mobile network generations (Long Term Evolution / 5th Generation) — core subject areas of the
ingested telecom documents the system answers questions about.
**file:line:** `CLAUDE.md:6` ("verified telecom documents (LTE/5G, Roaming, SIM Security, eSIM)");
`src/retrieval/query_rewriter.py:216` ("lte": "Long Term Evolution"), `:217` ("5g": "5th Generation")
**Why it matters:** These are the domain topics FDE candidates are interviewed on; the system must retrieve
and ground answers in them.
**Analogy:** The curriculum chapters the student must master.

### Roaming
**Definition:** The ability of a mobile subscriber to use services outside their home network — one of the
telecom subject areas in scope.
**file:line:** `CLAUDE.md:6` ("Roaming")
**Why it matters:** A standard FDE interview topic; answers about roaming must come from retrieved documents.
**Analogy:** Using your phone when you're outside your carrier's coverage area.

### SIM Security
**Definition:** Protections around the Subscriber Identity Module (SIM) — one of the telecom subject areas in
scope.
**file:line:** `CLAUDE.md:6` ("SIM Security")
**Why it matters:** SIM cloning and swap attacks are practical FDE concerns; the system must ground answers in
source documents rather than speculate.
**Analogy:** The lock and key that prove your phone belongs to your account.

### eSIM
**Definition:** Embedded SIM — a SIM built into a device rather than a removable card — one of the telecom
subject areas in scope.
**file:line:** `CLAUDE.md:6` ("eSIM"); `src/retrieval/query_rewriter.py:225` ("esim": "Embedded SIM")
**Why it matters:** eSIM provisioning and security are modern FDE interview topics the system covers.
**Analogy:** A SIM card soldered onto the phone's motherboard instead of a removable tray.

---

## How to read this glossary

Each entry follows the pattern:

> **Term** — plain definition  
> **file:line** — where the term is defined or primarily used in the repo  
> **Why it matters** — its role in the system  
> **Analogy** — one sentence to build intuition

No source code was changed to produce this document; it lives entirely under `docs/learn/`. The existing
`learn/GLOSSARY.md` remains as a separate, more general reference.
