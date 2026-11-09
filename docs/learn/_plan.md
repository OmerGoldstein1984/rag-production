# PHASE 0 — Recon & Plan (docs/learn/_plan.md)

**Learner:** senior QA/automation engineer → AI Solutions Engineer (telecom/eSIM, IoT). 10 yrs QA/CI/CD/test frameworks. Knows program FLOW, not line-by-line CODE. Weak on production architecture, concurrency/async, auth/multi-tenancy, observability, evals, deployment, cost/latency tradeoffs, customer-facing communication.

**Mission constraint (HARD):** CODE MUST NOT CHANGE. Only add comments/docstrings in source files, and new files under `docs/learn/`. No renames, reformat, import reorder, or "small fixes". Bugs go in `docs/learn/07_issues_found.md` (never fix inline).

**Target output tree:** `docs/learn/_plan.md`, `00_START_HERE.md`, `01_business_context.md`, `02_architecture.md`, `03_glossary.md`, `04_gap_bridge.md`, `05_trace_walkthrough.md`, `06_interview_pack.md`, `07_issues_found.md`, plus per-file source annotations (comments/docstrings only).

---

## 1. Repo map (recon summary)

Entrypoint: `src/api/main:app` (FastAPI), served by `uvicorn src.api.main:app --port 8000` (`Dockerfile:41`). CLI: `python -m src.ingestion.run` (`src/ingestion/run.py:172`). UI: `streamlit run ui/streamlit_app.py` (`docker-compose.yml:33`). Tests: `tests/` (pytest). Config: `src/config.py` (`get_settings()` reads env only; `.env` gitignored).

### Source files to annotate (disjoint, by module group)

| Group | Files | Role |
|-------|-------|------|
| G1 Foundational | `src/config.py`, `src/key_pool.py`, `src/key_monitor.py`, `src/observability.py` | settings, secret key pool, key expiry alerting, structured logging/tracing |
| G2 Embedding + Store | `src/embedding/embedder.py`, `src/store/vector_store.py` | sentence-transformer embeddings, ChromaDB/in-memory vector store |
| G3 Ingestion | `src/ingestion/loaders.py`, `src/ingestion/cleaner.py`, `src/ingestion/chunker.py`, `src/ingestion/manager.py`, `src/ingestion/run.py` | doc load → clean → chunk → embed → store; upload/dedupe/version/delete/reindex lifecycle |
| G4 Retrieval | `src/retrieval/retriever.py`, `src/retrieval/sparse_retriever.py`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py`, `src/retrieval/query_rewriter.py`, `src/retrieval/conversation_memory.py` | dense/hybrid/RRF retrieval, re-ranking, query rewrite, chat memory |
| G5 Providers + Generation + Agent | `providers/llm.py`, `providers/llm_nvidia.py`, `src/generation/prompt.py`, `src/generation/chat.py`, `src/generation/answer.py`, `src/agent/agent.py`, `src/agent/tools.py` | LLM providers (mock/NVIDIA), prompt building, answer gen, controlled tool-using RAG agent with allow-list |
| G6 API + Eval + UI | `src/api/main.py`, `src/evaluation/metrics.py`, `src/evaluation/run_eval.py`, `ui/streamlit_app.py` | FastAPI routes + cache + rate limit, golden-set eval, Streamlit UI |

### Dependency order (leaves first → entrypoint last)

1. `src/config.py` → `src/key_pool.py`, `src/key_monitor.py`, `src/observability.py`
2. `src/embedding/embedder.py`
3. `src/store/vector_store.py`
4. `src/ingestion/{loaders,cleaner,chunker}.py` → `src/ingestion/manager.py` → `src/ingestion/run.py`
5. `src/retrieval/{retriever,sparse_retriever,hybrid,reranker,query_rewriter,conversation_memory}.py`
6. `providers/{llm,llm_nvidia}.py` → `src/generation/{prompt,chat,answer}.py`
7. `src/agent/{tools,agent}.py`
8. `src/api/main.py`
9. `src/evaluation/{metrics,run_eval}.py`
10. `ui/streamlit_app.py`

### Data stores / external services
- Vector store: ChromaDB at `data/chroma_db` (falls back to in-memory if absent) — `src/config.py:26`, `Dockerfile:21`.
- Embedding cache: `data/embedding_cache/`; key registry: `data/key_registry.json` (gitignored, local only) — `src/key_pool.py:25`.
- Uploads: `uploads/` (write-only staging for ingested files) — `src/ingestion/manager.py:28`.
- Status log: `data/ingest_status.json` — `src/ingestion/manager.py:29`.
- External: NVIDIA NIM API (`integrate.api.nvidia.com/v1`) when `LLM_PROVIDER=nvidia` — `src/config.py:14`.
- Golden eval set: `src/evaluation/golden_set.json`.

### Recon findings (to record in 07_issues_found.md)
- `data/raw/` holds generic corporate docs (HR/benefits/IT/travel policies), NOT the telecom (LTE/5G, Roaming, SIM Security, eSIM) documents the project's stated purpose (`CLAUDE.md`) describes — a purpose/source mismatch.
- Existing root-level `learn/` directory already contains 00-foundations … INTERVIEW-PREP.md that overlaps this mission's `docs/learn/` structure — avoid duplicating; reconcile in PHASE 2.

---

## 2. PHASE 1 — parallel annotator + specialist agents (disjoint file ownership)

Spawn all at once. Each annotator owns a disjoint group (above) and writes in-file comments/docstrings only. Specialists write their doc under `docs/learn/`.

| Agent | Owns | Produces |
|-------|------|----------|
| Annotator G1 | `src/config.py`, `src/key_pool.py`, `src/key_monitor.py`, `src/observability.py` | comments/docstrings in those files |
| Annotator G2 | `src/embedding/embedder.py`, `src/store/vector_store.py` | comments/docstrings in those files |
| Annotator G3 | `src/ingestion/loaders.py`, `src/ingestion/cleaner.py`, `src/ingestion/chunker.py`, `src/ingestion/manager.py`, `src/ingestion/run.py` | comments/docstrings in those files |
| Annotator G4 | `src/retrieval/retriever.py`, `src/retrieval/sparse_retriever.py`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py`, `src/retrieval/query_rewriter.py`, `src/retrieval/conversation_memory.py` | comments/docstrings in those files |
| Annotator G5 | `providers/llm.py`, `providers/llm_nvidia.py`, `src/generation/prompt.py`, `src/generation/chat.py`, `src/generation/answer.py`, `src/agent/tools.py`, `src/agent/agent.py` | comments/docstrings in those files |
| Annotator G6 | `src/api/main.py`, `src/evaluation/metrics.py`, `src/evaluation/run_eval.py`, `ui/streamlit_app.py` | comments/docstrings in those files |
| Architecture | code + config + docker-compose + Dockerfile | `docs/learn/02_architecture.md` (Mermaid diagram, data flow, sequence, trust boundaries, state, scaling) |
| Business-context | README/CLAUDE.md + code + commits | `docs/learn/01_business_context.md` (customer, pain, cost, success, client questions; mark inferences) |
| Glossary | all source + config keys | `docs/learn/03_glossary.md` (term, plain def, file:line, why it matters, analogy) |
| Gap-bridge | all source + patterns | `docs/learn/04_gap_bridge.md` ("web app does X; production system does Y because Z") |
| Run-and-trace | one real request end-to-end | `docs/learn/05_trace_walkthrough.md` (real values, file:line at each hop, failure at each hop) |
| Interview | all of the above | `docs/learn/06_interview_pack.md` (25+ Q&A first-person, 2-min & 5-min walkthroughs, 5 customer scenarios, 10 break-it drills + answer key, honest weaknesses) |

### Annotation standard (every annotator, every file)
- FILE HEADER: business problem solved, place in flow, what calls it, what it calls, what breaks if removed.
- Every function/class/method docstring: WHAT / WHY / INPUTS+OUTPUTS (concrete example) / SIDE EFFECTS / FAILURE MODES / ALTERNATIVE.
- Every non-trivial line: inline intent comment (plain language, not syntax). Group obvious lines under one comment.
- Tags: `# GAP:` (concept learner likely lacks) 2-3 sentences; `# BUSINESS:` (what it protects); `# INTERVIEW:` (say-it-out-loud point).
- Match language conventions (Python docstrings). Quote `file:line` for claims.

---

## 3. PHASE 2 — reading guide (after Phase 1)
Write `docs/learn/00_START_HERE.md` (5-sentence summary, file-by-file reading order with question + time per file, 2-hr/day day-by-day plan, one-page mental map, links to all docs/learn/ files) and `docs/learn/07_issues_found.md` (bugs, smells, UNVERIFIED items, missing tests, security concerns, production needs).

---

## 4. PHASE 3 — verification (independent reviewer)
1. Prove code unchanged: strip comments/docstrings, compare to `main` (Python AST dump). Run test suite + lint on both branches. Revert any file differing beyond comments.
2. Fact-check: sample ≥30% of annotations and every `# GAP:` / `# INTERVIEW:` against code + library docs; fix wrong ones.
3. Readability: smart non-expert can follow each file header + docstring. Cut syntax-only comments.
4. Check every doc link and `file:line` reference resolves.

---

## 5. Output
Commit in logical chunks on `learn/annotated` (never touch `main`). Final message: tree of `docs/learn/`, files annotated, verification results, 10 concepts to master first, anything UNVERIFIED.
