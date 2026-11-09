# 00 — Start Here (reading guide)

Welcome to the rag-academy RAG teaching copy. This guide tells you exactly what this project is, what order to read everything in, and how to study it over ~2 hours/day so you can explain, defend, and extend every part of it in an FDE (Field Development Engineer) interview — as if you built it yourself.

## What this project is, in 5 sentences

The rag-academy RAG system is a study aid for FDE interview preparation focused on telecom domain knowledge (LTE/5G, Roaming, SIM Security, eSIM). It ingests source documents, chunks and embeds them, stores them in a vector database, and answers user questions by retrieving relevant chunks and generating grounded answers with citations — never speculating beyond the source material. It's built in Python with FastAPI (API), Streamlit (UI), sentence-transformers (embeddings), and ChromaDB (vector store, with an in-memory fallback), plus a controlled tool-using agent with an allow-list, a multi-key NVIDIA provider pool with failover, observability (structured logs, correlation IDs, per-stage tracing), and a golden-set evaluation harness. The learner profile is a senior QA/automation engineer (10 years) moving into AI Solutions Engineering: strong on program flow and test pipelines, weak on production architecture, concurrency, auth/multi-tenancy, observability, evals, deployment, and cost/latency tradeoffs. Every source file is annotated with teaching comments/docstrings, and every design decision is explained in `docs/learn/` with plain-English telecom/provisioning/test-pipeline analogies.

## The docs/learn/ tree (what exists, in order)

| File | What it is | When to read it |
|------|-----------|-----------------|
| `_plan.md` | PHASE 0 plan: repo map, dependency order, agent assignments, verification standard | First — read this to understand the layout |
| `00_START_HERE.md` | This reading guide + study plan | Second |
| `01_business_context.md` | Customer, pain, cost, success metrics, client questions, limitations | After foundations |
| `02_architecture.md` | Mermaid system diagram, components, data flow, sequence, trust boundaries, state, scaling | After you've read the source modules |
| `03_glossary.md` | 40 terms: plain definition, `file:line`, why it matters, analogy | Keep open as a reference while reading source |
| `04_gap_bridge.md` | "Typical web app does X; production system does Y, because Z" for 7 pattern areas | After source + architecture |
| `05_trace_walkthrough.md` | Step-by-step trace of one ask→answer request and one ingest, with failure modes at each hop | After retrieval + generation + API |
| `06_interview_pack.md` | 31 interview Q&As (first-person), 2-min & 5-min walkthroughs, 5 customer scenarios, 10 break-it drills + answer key, honest weaknesses | Last — interview prep |
| `07_issues_found.md` | Consolidated bug/smell/UNVERIFIED/security log (nothing fixed in code) | Read alongside source; know what's broken |

**Source files annotated** (comments/docstrings only, zero logic change — verified vs `main`):

| Group | Files | Read order |
|-------|-------|------------|
| G1 Foundational | `src/config.py`, `src/key_pool.py`, `src/key_monitor.py`, `src/observability.py` | 1st |
| G2 Embedding + Store | `src/embedding/embedder.py`, `src/store/vector_store.py` | 2nd |
| G3 Ingestion | `src/ingestion/loaders.py`, `src/ingestion/cleaner.py`, `src/ingestion/chunker.py`, `src/ingestion/manager.py`, `src/ingestion/run.py` | 3rd |
| G4 Retrieval | `src/retrieval/retriever.py`, `src/retrieval/sparse_retriever.py`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py`, `src/retrieval/query_rewriter.py`, `src/retrieval/conversation_memory.py` | 4th |
| G5 Providers + Generation + Agent | `providers/llm.py`, `providers/llm_nvidia.py`, `src/generation/prompt.py`, `src/generation/chat.py`, `src/generation/answer.py`, `src/agent/tools.py`, `src/agent/agent.py` | 5th |
| G6 API + Eval + UI | `src/api/main.py`, `src/evaluation/metrics.py`, `src/evaluation/run_eval.py`, `ui/streamlit_app.py` | 6th |

## How to read each source file

For every file, answer these questions before moving on:
1. What business problem does this file solve? (file header)
2. What calls it, and what does it call? (file header)
3. Pick one function: what are its inputs/outputs with a concrete example? What are its side effects and failure modes? (docstring)
4. Where are the `# GAP:`, `# BUSINESS:`, `# INTERVIEW:` tags? Read the `# INTERVIEW:` ones out loud.
5. What breaks if this file is removed? (file header)

If a file header or docstring only restates syntax, skip it — it's not doing its job.

## Day-by-day study plan (assume 2 hours/day)

**Day 1 — Foundations + business context** (~2h)
- Read `src/config.py` (settings, fail-fast, env-only secrets) — Q: "Why does provider=nvidia without a key raise instead of defaulting?" (~30 min)
- Read `src/key_pool.py` (round-robin, failover, fingerprinting) — Q: "Why are full keys never logged?" (~30 min)
- Read `src/observability.py` (correlation IDs, secret redaction, per-stage tracing) — Q: "What would appear in `/traces` for one request?" (~30 min)
- Read `docs/learn/01_business_context.md` — Q: "What happens without this system?" (~30 min)

**Day 2 — Embedding + store + ingestion parsing** (~2h)
- Read `src/embedding/embedder.py` — Q: "What happens on an empty input? On a duplicate chunk?" (~35 min)
- Read `src/store/vector_store.py` — Q: "How are IDs generated, and what breaks under concurrent writes?" (~35 min)
- Read `src/ingestion/loaders.py`, `cleaner.py`, `chunker.py` — Q: "Where can a document get silently lost?" (~50 min)

**Day 3 — Ingestion lifecycle + retrieval** (~2h)
- Read `src/ingestion/manager.py`, `run.py` (upload → dedupe → version → persist → index → status log; delete cascade; reindex) — Q: "What's not atomic, and what's the risk?" (~45 min)
- Read `src/retrieval/retriever.py`, `sparse_retriever.py`, `hybrid.py` — Q: "How does RRF combine dense+sparse? Where does de-dup break?" (~45 min)
- Skim `reranker.py`, `query_rewriter.py`, `conversation_memory.py` (~30 min)

**Day 4 — Providers + generation + agent** (~2h)
- Read `providers/llm.py`, `llm_nvidia.py` — Q: "How does key failover work, and what's not thread-safe?" (~35 min)
- Read `src/generation/prompt.py`, `chat.py`, `answer.py` — Q: "Why is the context budget char-based, not token-based?" (~40 min)
- Read `src/agent/tools.py`, `agent.py` — Q: "Why does `run` return after the first tool call?" (~45 min)

**Day 5 — API + eval + UI + architecture** (~2h)
- Read `src/api/main.py` — Q: "Where are auth, rate-limit, cache, retrieve, rerank, generate?" (~40 min)
- Read `src/evaluation/metrics.py`, `run_eval.py` — Q: "How is faithfulness measured, and what's the flaw?" (~40 min)
- Read `ui/streamlit_app.py` (~20 min)
- Read `docs/learn/02_architecture.md` (Mermaid diagrams, trust boundaries, state, scaling) (~20 min)

**Day 6 — Glossary + gap bridge + trace** (~2h)
- Read `docs/learn/03_glossary.md` (use as reference while re-skimming any unclear term) (~30 min)
- Read `docs/learn/04_gap_bridge.md` — Q: "What does a typical web app do that this production system doesn't, and why?" (~40 min)
- Read `docs/learn/05_trace_walkthrough.md` — Q: "At each hop, what's the top failure mode and the user-visible symptom?" (~50 min)

**Day 7 — Interview pack + issues + review** (~2h)
- Read `docs/learn/06_interview_pack.md` — Q: "Can I answer Q1–Q8 in first person without notes?" (~50 min)
- Read `docs/learn/07_issues_found.md` — Q: "Which of these would I fix first, and why?" (~40 min)
- Re-run `pytest` and trace one request end-to-end if you can get the environment up (~30 min)

## One-page mental map

```
data/raw/ (docs) ──► loaders/cleaner/chunker ──► embedder ──► vector_store (ChromaDB)
                                              ▲          ▲
                                              │          │
                                    manager/run.py   retrieval/*.py
                                              │          │
                                    uploads/ + status   providers/llm_nvidia.py
                                    log (JSON)          generation/*.py
                                                          agent/*.py
                                                             │
                                    FastAPI src/api/main.py ◄┘
                                                             │
                                                        Streamlit ui/streamlit_app.py
                                                             │
                                                    evaluation/*.py (golden set)
```

- **State lives in:** `data/chroma_db` (vectors), `data/key_registry.json` (key fingerprints/expiry), `data/ingest_status.json` (ingest log), `uploads/` (staged files), `data/embedding_cache/` (embedding cache), plus in-memory cache/rate-limiter/metrics/traces (per-process, no TTL — does not survive restart).
- **Entrypoint:** `uvicorn src.api.main:app` (port 8000); CLI `python -m src.ingestion.run`; UI `streamlit run ui/streamlit_app.py` (port 8501).
- **Trust boundary:** FastAPI app — bearer token + role header, allow-listed tools, admin-only tools, secret redaction in logs.
- **External:** NVIDIA NIM API (`integrate.api.nvidia.com/v1`) when `LLM_PROVIDER=nvidia`; mock otherwise.

## Links to every doc in docs/learn/

- [_plan.md](_plan.md) — PHASE 0 plan
- [00_START_HERE.md](00_START_HERE.md) — this guide
- [01_business_context.md](01_business_context.md)
- [02_architecture.md](02_architecture.md)
- [03_glossary.md](03_glossary.md)
- [04_gap_bridge.md](04_gap_bridge.md)
- [05_trace_walkthrough.md](05_trace_walkthrough.md)
- [06_interview_pack.md](06_interview_pack.md)
- [07_issues_found.md](07_issues_found.md)

## Suggested first action

Open `src/config.py` and read its file header + `get_settings()` docstring. Then answer out loud: "Why does the system fail-fast when `LLM_PROVIDER=nvidia` but no key is set?" If you can answer that in plain English with a concrete example, you're ready for Day 1.
