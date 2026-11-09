# 07 — Issues Found (bugs, smells, UNVERIFIED, security, production needs)

Consolidated from all PHASE 1 annotator groups (G1–G6) plus recon. Per the mission constraint, **none of these were fixed in code** — they are logged here only. Each entry carries `file:line` where the annotator could verify it.

## A. Recon findings (project-level)

### A1. Purpose / source mismatch — LOGGED, not fixed
`data/raw/` contains generic Acme Corp corporate documents (HR/benefits/IT/travel policies: `benefits_summary.md`, `hr_leave_policy.md`, `travel_policy.md`, `it_security_policy.md`, …), **not** the telecom documents (LTE/5G, Roaming, SIM Security, eSIM) that `CLAUDE.md` states the system is for. This is a purpose/source mismatch.
- Where: `CLAUDE.md` (stated purpose) vs. `data/raw/` (actual corpus)
- Why it matters: answers about telecom will be grounded in HR/benefits content, defeating the FDE interview-prep goal.
- Talk track: "The ingest pipeline works, but the current `data/raw/` corpus doesn't match the stated telecom domain; swapping in LTE/5G/Roaming/SIM Security/eSIM docs is the first thing I'd do."

### A2. Root-level `learn/` directory overlaps `docs/learn/`
A pre-existing root-level `learn/` directory (`00-foundations.md` … `INTERVIEW-PREP.md`, `GLOSSARY.md`) overlaps the new `docs/learn/` structure produced by this mission.
- Where: `learn/` (root) vs. `docs/learn/`
- Action left to PHASE 2: reconcile (don't duplicate; decide which is canonical). Not fixed in code.

## B. Bugs / smells by file

### B1. G1 Foundational — no bugs found
- `src/config.py` — none
- `src/key_pool.py` — none
- `src/key_monitor.py` — none
- `src/observability.py` — none
- Note: `tests/test_key_monitor.py` does not exist; `key_monitor.py` docstring hedges a reference as "(if present)".
- Verification: `py_compile` clean; `tests/test_key_pool.py` + `tests/test_observability.py` pass 26/26.

### B2. G2 Embedding + Store
| # | File:line | Issue | Why it matters |
|---|-----------|-------|----------------|
| B2-1 | `src/embedding/embedder.py:328` | Empty input returns hardcoded `[0.0] * 384`, tied to `all-MiniLM-L6-v2` dimension. A different `model_name` yields a dimension-mismatched zero vector. | Silent wrong-dimension vector pollutes the index. |
| B2-2 | `src/embedding/embedder.py:409,410,420` | `uncached_texts[uncached_indices.index(idx)]` inside a loop: `list.index()` is O(n) per lookup and returns the *first* match, so duplicate texts in one batch get the wrong hash/cache entry; quadratic overall. | Wrong embeddings cached/returned for duplicate chunks; slow on large batches. |
| B2-3 | `src/embedding/embedder.py` | Rate-limit decorator uses `last_call_time = [0.0]` closure — shared mutable state, not thread-safe. | Race on concurrent embed calls. |
| B2-4 | `src/embedding/embedder.py` | `retry_with_backoff` retries on *any* `Exception`, including non-transient ones like `ValueError`. | Over-retries bad inputs instead of failing fast. |
| B2-5 | `src/store/vector_store.py:205` | Auto-generated IDs `doc_{count + i}` depend on current count; concurrent `add_vectors` calls can collide (no lock / no idempotency guard). | Lost or overwritten vectors under concurrent ingest. |
| B2-6 | `src/store/vector_store.py:661,666` | Duplicate key `"total_vectors"` in `_fallback_get_stats` (last wins). Harmless but sloppy, carried over from original. | Minor correctness sloppiness in stats. |

### B3. G3 Ingestion
#### `src/ingestion/loaders.py`
| # | File:line | Issue |
|---|-----------|-------|
| B3-1 | `:35` (orig) | No retry/backoff on `Path.stat()` failure; a transient filesystem hiccup fails the whole load. |
| B3-2 | `:199` | No encoding detection; assumes UTF-8. `UnicodeDecodeError` on non-UTF-8 files. |
| B3-3 | `:20` | `mimetypes` imported but never used (dead import). |
| B3-4 | `:53` | `_infer_department` is brittle keyword matching (e.g. "travel_report.md" about IT travel tagged "Finance"); no confidence score or override. |
| B3-5 | `:78-80` | Regex-based version parsing can misfire (e.g. "section_1.2.txt" yields "1.2"); no semantic version validation. |

#### `src/ingestion/cleaner.py`
| # | File:line | Issue |
|---|-----------|-------|
| B3-6 | `:69` | Whitespace collapse removes intentional indentation in code blocks or poetry. |
| B3-7 | `:101` | `cleaning_ratio` divides by `max(len(content),1)`, masking truly empty documents. |
| B3-8 | `:106` | Module `clean_documents` creates a new `TextCleaner()` instead of using `self`, recompiling all regex patterns wastefully; duplicates the same-named method (two code paths can diverge). |

#### `src/ingestion/chunker.py`
| # | File:line | Issue |
|---|-----------|-------|
| B3-9 | `:42` | `chunk_size` is characters, not tokens; a 500-char chunk may still exceed the embedding model's token limit. |
| B3-10 | `:77` | Empty text returns `[]` silently; callers can't distinguish empty input from no chunks. |
| B3-11 | `:105` | Sentence-boundary search only looks within the last 100 chars before chunk end. |
| B3-12 | `:176` | A single paragraph longer than `chunk_size` gets sub-chunked by fixed-size, losing paragraph-boundary benefit. |
| B3-13 | `:202` | Section reconstruction can mishandle content before the first heading. |
| B3-14 | `:252` | `total_chunks` computed after `chunk_text`; if zero chunks produced, the document is lost from the index entirely. |
| B3-15 | `:285` | Flattening loses document boundaries; callers can't tell which chunks came from which document without reading metadata. |
| B3-16 | `:503` | Invalid strategy strings silently default to `FIXED_SIZE` instead of raising; typos are hidden. |

#### `src/ingestion/manager.py`
| # | File:line | Issue |
|---|-----------|-------|
| B3-17 | `:49` | `UploadError` has no error code or structured detail; callers must parse the message string. |
| B3-18 | `:56` | `IngestRecord.status` is a free-form string (no enum); a typo like "succeeded" goes unnoticed. |
| B3-19 | `:126` | No locking; concurrent uploads of the same filename could race on versioning/dedup. |
| B3-20 | `:161` | No rate limiting; a client could loop-upload thousands of files and fill disk. |
| B3-21 | `:195` | File is written to disk before indexing; if indexing fails, an orphaned file remains in `uploads/`. |
| B3-22 | `:212-214` | Only minor-version bump; can't signal a major version change for breaking content changes. |
| B3-23 | `:233` | All chunks held in memory before embedding; OOM risk for very large documents. |
| B3-24 | `:254-256` | Vector-store deletion and disk deletion are not atomic; inconsistent state if one succeeds and the other fails. |
| B3-25 | `:269` | `vector_store.reset()` wipes everything before reindexing; if reindex fails midway, the store is empty and old data is gone. |

#### `src/ingestion/run.py`
| # | File:line | Issue |
|---|-----------|-------|
| B3-26 | `:19` | `os` imported but never used (dead import). |
| B3-27 | `:73` | No documents found → silent warning and early return; caller can't distinguish "empty dir" from "all files failed". |
| B3-28 | `:82` | No retry/backoff on transient failures; one corrupt document or DB error aborts the whole pipeline. |
| B3-29 | `:126/150` | Bad-split heuristic only checks the first 5 chunks per strategy. |
| B3-30 | `:228` | Invalid `--log-level` causes `AttributeError` on `getattr`; no validation that `--data-dir` exists. |

### B4. G4 Retrieval
| # | File:line | Issue |
|---|-----------|-------|
| B4-1 | `src/retrieval/retriever.py:167` | `HybridRetriever` keys cross-list matches by `id(doc)`, which never matches across dense/sparse lists (weak de-dup; see `_doc_key`). |
| B4-2 | `src/retrieval/sparse_retriever.py:91` | `get_score` returns `0.0` for out-of-range `doc_index` (hides caller bug instead of surfacing it). |
| B4-3 | `src/retrieval/hybrid.py:7` | `import numpy as np` unused. |
| B4-4 | `src/retrieval/hybrid.py:10` | `from .reranker import Reranker` unused. |
| B4-5 | `src/retrieval/hybrid.py:31-35` | `weight_dense`/`weight_sparse` stored but never applied in `_reciprocal_rank_fusion` (dead config). |
| B4-6 | `src/retrieval/hybrid.py:73-75` | `similarity_threshold` compared against RRF position-based scores (small values; a non-zero threshold may filter out everything — semantics trap). |
| B4-7 | `src/retrieval/query_rewriter.py` | `llm_provider` accepted but never actually invoked; `_llm_based_rewrite`/`_hyde_rewrite`/`_multi_query_expansion` fall back to rule-based logic. |
| B4-8 | `src/retrieval/conversation_memory.py:42` | `metadata.get("timestamp")` assumes `metadata` is a dict when non-`None` (AttributeError risk). |
| B4-9 | `src/retrieval/conversation_memory.py:106` | `_condense_with_llm` is a placeholder that always falls back to `_condense_with_rules`. |

### B5. G5 Providers + Generation + Agent
| # | File:line | Issue |
|---|-----------|-------|
| B5-1 | `providers/llm_nvidia.py:420` | Key-pool rotation does not consume the retry budget, so total attempts can exceed `max_retries` (no hard attempt ceiling). |
| B5-2 | `providers/llm_nvidia.py:368` | Blocking `time.sleep` backoff, no jitter, no async client — thundering-herd risk on 429; blocks event loop if used async. |
| B5-3 | `providers/llm_nvidia.py:654` | `stream_chat` silently `continue`s on malformed SSE lines — a corrupt chunk can truncate the answer without signaling an error. |
| B5-4 | `providers/llm_nvidia.py:749,768` | No context manager; `__del__` cleanup is non-deterministic — httpx client leak risk in long-running processes. |
| B5-5 | `src/generation/prompt.py:344` | `format_context` truncates by **character count**, not token count (poor proxy for the model's context window). |
| B5-6 | `src/generation/answer.py:118` | `_build_context_within_budget` bounds only the context, not the full prompt (instructions + context + question can still exceed the window). |
| B5-7 | `src/generation/answer.py:398` | `stream_answer_with_context` `usage` falls back to estimated `token_stats` when provider-reported usage is unavailable — mixes estimated and actual metrics. |
| B5-8 | `src/generation/answer.py:409` | `generate_multiple_answers` strictly sequential — no batching/parallelism, so N queries take N× single-call latency. |
| B5-9 | `src/generation/chat.py:150` | `reranker = create_reranker()` is a local var never stored on `self`, so `self.retriever.reranker` is likely None → the reranker branch is effectively dead code (the `hasattr` fallback always wins). |
| B5-10 | `src/generation/chat.py` `__init__` | Re-ingests and re-indexes on every `RAGChat` construction — no persistence/reuse of an existing index. |
| B5-11 | `src/agent/tools.py:213` | `ThreadPoolExecutor` is never shut down — resource leak in long-running processes. |
| B5-12 | `src/agent/tools.py:215` | `register` silently overwrites a duplicate tool name (no duplicate-name guard). |
| B5-13 | `src/agent/agent.py:248` | `run` returns after the **first** tool call in both branches, so `MAX_STEPS` (5) and `MAX_REPEAT_SAME_CALL` (2) never actually come into play — multi-step reasoning is not implemented. |
| B5-14 | `src/agent/agent.py:170` | `_plan_heuristic` filename regex may return `""` when no doc name matches → tool call with empty filename fails at execution. |

### B6. G6 API + Eval + UI
#### `src/api/main.py`
| # | File:line | Issue |
|---|-----------|-------|
| B6-1 | `:1151` | `rag_pipeline = rag_pipeline` is a no-op reassignment in `/ingest` (harmless dead code). |
| B6-2 | `RateLimiter`/`QueryCache` | In-memory and per-process: not shared across uvicorn workers, no TTL, no Redis backing. |
| B6-3 | `verify_user_role` | `X-Access-Level` header is client-controlled unless `API_TOKEN` auth is enabled — a caller can self-assert admin in dev mode. |
| B6-4 | `initialize_rag_pipeline` | Rebinding the global `rag_pipeline` with no lock; concurrent `/ingest` calls can race. |
| B6-5 | `startup_event` | No retry/backoff if pipeline init fails; app stays in 503 until restart. |

#### `src/evaluation/metrics.py`
| # | File:line | Issue |
|---|-----------|-------|
| B6-6 | Retrieval metrics | Fall back to `id(doc)` when `metadata.doc_id` is missing, which can undercount hits. |
| B6-7 | `_fallback_faithfulness` | Splits sentences on `.!?` (crude, breaks on abbreviations) and ignores negation. |
| B6-8 | `contradiction_detection` | Returns `0.0` on judge exception (assumes no contradiction — may under-detect) and `0.5` on non-YES/NO output. |

#### `src/evaluation/run_eval.py`
| # | File:line | Issue |
|---|-----------|-------|
| B6-9 | `:324` | `question = question_data["question"]` uses `[]` (KeyError if missing) while sibling fields use `.get()` (inconsistent). |
| B6-10 | `run_ablation_study` / `run_comparison_evaluation` | No per-config try/except; one bad config aborts the whole study. |
| B6-11 | `hybrid_alpha` | Not range-checked to [0,1]. |
| B6-12 | `make_llm_judge` | Degrades to `"0"` silently with no retry/circuit breaker. |

#### `ui/streamlit_app.py`
| # | File:line | Issue |
|---|-----------|-------|
| B6-13 | Response accesses | Several assume schema (`res["results"]`, `res["steps"]`); a changed API shape raises `KeyError`. |
| B6-14 | `call()` | Does not catch non-`RequestException` JSON-decode errors. |
| B6-15 | General | No retry/offline fallback. |

## C. Security concerns
| # | File:line | Concern |
|---|-----------|---------|
| C-1 | `src/api/main.py` | Single bearer token (`API_TOKEN`) for all mutating endpoints; no per-customer/tenant isolation — shared corpus, single auth boundary. |
| C-2 | `src/api/main.py` | `X-Access-Level` client-controlled unless `API_TOKEN` set (see B6-3). |
| C-3 | `src/observability.py` | Secret redaction patterns (NVIDIA/`sk-`/Bearer) applied to logs — good, but not exhaustively confirmed on every log/exception path (UNVERIFIED). |
| C-4 | `src/ingestion/manager.py:98-110` | Upload allowlist + path-traversal guard present (`Path(filename).name`), extension + size cap — a strength, not a bug. |
| C-5 | `src/agent/tools.py` | Tool allow-list + Pydantic validation + admin gating + timeout — a strength; full keys never logged (key_pool fingerprint only). |

## D. Missing tests / test gaps
- `tests/test_key_monitor.py` does not exist (referenced hedged as "(if present)").
- `tests/` covers key_pool, observability, agent, chunker, document_lifecycle, end_to_end, generation_real, key_pool, loaders, metrics, nvidia_provider, rrf, security_api, streaming — but no test for `vector_store` concurrency (B2-5), `cleaner`/`chunker` edge cases (B3-6…B3-16), or `manager` race/orphan cases (B3-19…B3-25).
- Golden set exists at `src/evaluation/golden_set.json` (30 Qs incl. unanswerable).

## E. What a production version would need (from gap_bridge + annotators)
1. **Multi-tenant isolation** — per-customer corpus/key, not one shared token + corpus (C-1).
2. **Distributed cache/rate-limiter/metrics** — Redis-backed, shared across workers, with TTL (B6-2).
3. **Async NVIDIA client** with jittered backoff and hard attempt ceiling (B5-1, B5-2).
4. **Idempotent, locked vector-store writes** — UUID-based IDs + lock to prevent concurrent collisions (B2-5).
5. **Atomic delete + indexing** — transaction-like delete cascade; reindex that doesn't wipe-before-rebuild (B3-24, B3-25).
6. **CI/CD pipeline** beyond docker-compose — tests, lint, AST-diff guard on every PR.
7. **Retry/backoff with jitter** on ingest pipeline and startup init (B3-28, B6-5).
8. **Token-count-based context budget** instead of char-count (B5-5).
9. **Exhaustive secret-redaction verification** across all log/exception paths (C-3).
10. **Telecom corpus** in `data/raw/` matching the stated purpose (A1).

## F. UNVERIFIED items
| # | Item |
|---|------|
| F-1 | End-to-end app startup not actually run (`uvicorn` + Streamlit not exercised). |
| F-2 | ChromaDB active vs. in-memory fallback at runtime not confirmed. |
| F-3 | `sentence-transformers` model (`all-MiniLM-L6-v2`) download/load not confirmed. |
| F-4 | NVIDIA provider path (`LLM_PROVIDER=nvidia`) not exercised — needs a real key + network. |
| F-5 | Some `file:line` references may have shifted by a line or two after comment/docstring insertions (annotators verified logic via token-stream compare to `main`, but line numbers are approximate for learner reference). |
| F-6 | Secret redaction not exhaustively confirmed on every log/exception path (C-3). |

## G. Verification status (PHASE 3)
- Code-unchanged proof: pending AST-dump compare of each annotated file vs `main`.
- Test suite: pending full run on both branches.
- Lint: pending.
- Fact-check of `# GAP:` / `# INTERVIEW:` tags: pending (≥30% sample + every tag).
- `file:line` link resolution: pending.
