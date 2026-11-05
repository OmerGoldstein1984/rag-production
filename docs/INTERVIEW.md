# Interview preparation (FDE / AI Solutions / Applied AI)

Everything below is grounded in the actual implementation in this repo. Never claim what isn't implemented — `/provider` shows the live model, and the mock is always visible.

## 1. Two-minute pitch
"This is a production-minded RAG system I built and can defend end to end. Documents go in through validated uploads — real PDF/DOCX extraction, chunking with three strategies, embeddings with a content-hash cache — into a persistent ChromaDB. Retrieval is hybrid: dense + BM25 fused with RRF, optional cross-encoder reranking, all measured on a 30-question golden set — hybrid+RRF gave MRR ~0.71 vs ~0.07 dense-only. Answers come from NVIDIA NIM through a provider adapter that handles auth, retries, rate limits, streaming, structured output with Pydantic validation, and usage/latency tracking. Keys live only in env, rotate across a pool with failover on 401, and an expiry monitor emails me before they die. There's a controlled tool-calling agent — allow-listed, schema-validated, step-capped, fully traced — plus observability: JSON logs with secret redaction, request IDs, per-stage traces, and a /metrics endpoint. 103 tests run with no key and no network."

## 2. Ten-minute architecture walkthrough
1. **Ingestion** (2 min): loaders → cleaner → chunker → embedder → store. Show `src/ingestion/manager.py`: dedupe by sha256, version supersede, delete cascade, status log.
2. **Retrieval** (2 min): dense + BM25 + RRF (`src/retrieval/hybrid.py`, stable `_doc_key`), reranker, version filter. Point at `evaluation/retrieval_comparison.md` measured table.
3. **Generation** (2 min): `AnswerGenerator` + grounded prompt with `[Source: ...]` tags and untrusted-data delimiters; NVIDIA provider; streaming SSE; `GroundedAnswer` schema.
4. **Provider/keys** (1.5 min): `llm_nvidia.py` typed errors/retry; `key_pool.py` rotation/failover; `key_monitor.py` expiry email.
5. **Agent + safety** (1.5 min): `ToolRegistry` validate→authorize→timeout→record; `RAGAgent` step/repeat caps; `/provider` proves mock vs real.
6. **Observability + ops** (1 min): `/metrics`, `/traces`, logs; runbook; docker.

## 3. Live demo script
```bash
python scripts/demo.py          # full guided demo, mock (no key)
# real model:
export LLM_PROVIDER=nvidia NVIDIA_API_KEYS=...
python scripts/demo.py --real
python -m src.evaluation.run_eval --data-dir data/raw --output evaluation/results.md
```
Then: `docker compose up`, open UI :8501 — Ask (streaming), Agent tab, Traces & Metrics, Documents upload.

## 4. Hard technical questions (grounded answers)

**Q: Why hybrid retrieval? What did you measure?**
Dense-only misses exact terms; BM25 misses semantics. RRF fuses ranks without score calibration. Measured on the golden set: hybrid+RRF MRR 0.707 / Hit@5 0.73 vs dense-fixed MRR 0.073 / Hit@5 0.17. See `evaluation/retrieval_comparison.md`. Reranker was neutral on this small corpus, so it's configurable/off by default — I don't ship unmeasured complexity.

**Q: How do you prevent hallucination?**
Not by prompt alone: grounded prompt that forbids outside info + explicit "I don't have enough information" refusal + citations to `[source_filename]` + structured-output validation + evaluation (faithfulness/relevance via judge + deterministic checks). Traceability: every answer carries its source chunks.

**Q: Prompt injection in retrieved docs?**
Chunks are wrapped in untrusted-data delimiters and the system prompt says never follow instructions inside them; the LLM's tool choices are validated against an allow-list and Pydantic schemas, so even a hijacked proposal can't execute arbitrary tools. Tool inputs are data, not code.

**Q: How do you handle provider failures?**
Typed errors; exponential backoff with cap only on retryable classes (timeout/429/5xx); 4xx fails fast; key pool rotates on 401/403 without consuming retry budget; all keys bad → clear error. Usage + latency recorded per call; secrets redacted from errors/logs.

**Q: API keys in logs?**
Never. Env-only, redaction filter on every log record, fingerprint-only in registry/emails, tests assert keys don't leak. CI asserts `.env` isn't baked into the Docker image.

**Q: Why not just call OpenAI?**
The mission required NVIDIA NIM; the provider interface makes swapping trivial, but NVIDIA is the deliberate default and mock is opt-in only. `/provider` exposes which is live.

**Q: RAG vs agent — when each?**
RAG = fixed pipeline for known-corpus Q&A (fast, cheap, auditable). Agent = model chooses among approved tools for multi-step tasks. I built the agent with an allow-list, schema validation, step cap, and full tracing — the model proposes, the app decides.

**Q: Scaling the vector store?**
Local persistent ChromaDB is fine at this corpus size (ADR-007). At scale I'd move to a managed vector DB with metadata filtering (already supported), HNSW/IVF tuning, and sharding by department; retrieval is isolated behind `VectorStore` so the swap is contained.

**Q: Context window management?**
`_build_context_within_budget` packs chunks under a token budget with source tags, dedupes, and preserves metadata; truncation is explicit, not silent.

**Q: How do you evaluate a RAG system?**
Separate retrieval eval (Hit/Recall/Precision/MRR/nDCG/AP vs golden-set expected sources — measured) from answer eval (faithfulness/relevance via LLM judge with documented limitations + deterministic keyword checks). Ablation/comparison runner produces reports; I never fabricate numbers.

**Q: Cost/latency?**
Provider-reported token usage + latency captured on every call (`/metrics` aggregates p50/p95). No invented prices — usage is labeled provider-reported; estimates only when clearly marked.

**Q: Security of the API?**
Bearer token when `API_TOKEN` set (constant-time), role checks, rate limit, ingest confined to data//uploads (traversal blocked), file type/size caps, generic client errors, redaction. See `docs/SECURITY.md`.

**Q: Testing without paid API calls?**
All HTTP mocked (httpx MockTransport / TestClient); 103 tests cover success + failure paths; CI runs pytest + import smoke + docker build with zero keys; a separate real-API eval is opt-in via env.

**Q: What would you improve next?**
Streaming citations per-token, per-user doc ACLs at query time (metadata filters exist), a real managed vector DB + read replicas, and LLM-judge calibration (Cohen's kappa vs human labels). Also expanding the golden set with adversarial multi-hop cases.

## 5. Demo checklist
- [ ] `python scripts/demo.py` prints pipeline, Q&A, agent tool calls, metrics, key pool
- [ ] `GET /provider` shows real model when key set (never hidden mock)
- [ ] Upload a doc in UI → appears in Documents → ask about it → cited
- [ ] Agent tab: "compare versions of X" shows the tool-call trace
- [ ] `/metrics` and `/traces` show live numbers
