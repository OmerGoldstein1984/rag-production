# Architecture

## Pipeline (data flow)

```
data/raw + uploads/  ──▶  ingestion        ──▶  vector store (ChromaDB, persistent)
   .md/.txt/.pdf/.docx     loaders→cleaner→chunker→embedder (cache+retry)   │
                                                                            ▼
   user question ──▶ FastAPI (/ask, /ask/stream, /agent/ask, /search)     retriever
                        │              │                                  dense + BM25
                        │              │                                  + RRF + reranker
                        │              ▼                                  │
                        │        context builder (budget, dedupe, [Source:] tags)
                        │              │
                        ▼              ▼
                 providers/llm_nvidia.py (NVIDIA NIM, OpenAI-compatible) ◀─ key pool (rotation/failover)
                        │              │
                        ▼              ▼
                 grounded answer + citations + usage/latency      /metrics /traces /provider
```

## Layers
- `src/ingestion/` — loaders (real pypdf / stdlib-docx; never fabricate), cleaner, chunker (fixed/recursive/structure-aware), `manager.py` (upload/dedupe/version/delete/reindex + status log)
- `src/embedding/embedder.py` — sentence-transformers with content-hash cache, retry/backoff, rate limit; **mock only as explicit fallback**
- `src/store/vector_store.py` — ChromaDB persistent client (+ in-memory fallback); `list_sources`, `mark_source_current`, `delete_by_source`
- `src/retrieval/` — dense `Retriever`, `BM25Retriever`, `HybridRetrieverRRF` (stable doc keys), cross-encoder `Reranker` (lazy import), rule/LLM `QueryRewriter`, `ConversationMemory`
- `src/generation/` — `prompt.py` (grounded v1/v2, citation, untrusted-data delimiters), `answer.py` (`AnswerGenerator` with injected provider, streaming, usage capture)
- `providers/` — `LLMProvider` ABC; `llm_nvidia.py` (auth, streaming, structured JSON w/ Pydantic + retry, usage, latency, typed errors, redaction); `llm.py` factory; `llm.py` Mock (dev/tests only)
- `src/agent/` — allow-listed `ToolRegistry` (Pydantic-validated inputs, admin gating, timeout, audit history) + deterministic-first `RAGAgent` (step cap, repeat cap)
- `src/key_pool.py` / `src/key_monitor.py` — multi-key rotation/failover; expiry registry + SMTP alert (fingerprints only)
- `src/observability.py` — JSON logs w/ redaction, request IDs, per-stage traces, in-memory metrics
- `src/evaluation/` — retrieval metrics from scratch, LLM-as-judge framework, golden set (30 Q), experiment runner
- `src/api/main.py` — FastAPI: /health /provider /metrics /traces /ask /ask/stream /search /agent/ask /documents* /feedback; bearer-token auth (API_TOKEN), role checks, rate limit, query cache, ingest path guard

## Key decisions (ADRs)
- **ADR-001 NVIDIA NIM as primary provider** — OpenAI-compatible endpoint; single adapter behind `LLMProvider` so business logic is provider-agnostic. Mock is explicit opt-in, never a silent fallback.
- **ADR-002 Env-only secrets** — keys/SMTP/API token only from env; `.env` gitignored + `.dockerignore`d; only fingerprints logged/stored/emailed.
- **ADR-003 Multi-key pool + failover** — 401/403 marks a key bad and rotates; rotation doesn't consume retry budget; all-keys-bad → clear error.
- **ADR-004 Untrusted data** — retrieved chunks and model output are data: delimited in prompts, never eval/exec'd, Pydantic-validated tool inputs, safe errors.
- **ADR-005 Tool allow-list** — LLM proposes, app validates/authorizes/executes with timeout + audit; max steps + repeat cap; admin-only tools gated.
- **ADR-006 Observability built-in** — JSON logs + redaction, request IDs, stage traces, metrics; `/provider` exposes active model so mock can't hide.
- **ADR-007 Local ChromaDB** — persistent local store is enough at this scale; no managed vector DB cost. Re-evaluate at scale.
- **ADR-008 CPU-only torch in Docker** — keeps image small/affordable; GPU not needed for MiniLM/cross-encoder at this scale.
