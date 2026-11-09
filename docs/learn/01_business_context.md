# 01 — Business Context: rag-academy RAG System

**Branch:** `learn/annotated` | **Repo:** `C:\projects\rag-project` | **Status:** Draft (doc only, no code changed)

---

## Purpose of this document

This document describes the **business context** of the rag-academy RAG system: who it is for, what pain it solves, what happens without it, what it costs, how success is measured, what a client would ask an FDE about it, and its known limitations.

It is written for a learner (senior QA/automation engineer transitioning to AI Solutions Engineer, telecom/eSIM/IoT) who knows program flow but not production architecture. Everything here is grounded in `CLAUDE.md`, the commit history, and the source code. Anything I could not verify directly from those sources is marked **[INFERRED]**.

---

## 1. Who the user/customer is

**[STATED]** The primary user is an **FDE (Field Development Engineer) interview preparation learner** working in the **telecom domain** — specifically LTE/5G, Roaming, SIM Security, and eSIM. This is stated explicitly in `CLAUDE.md`:

> "This RAG system is designed as a study aid for FDE (Field Development Engineer) interview preparation, specifically focusing on telecom domain knowledge."

**[STATED]** The learner profile (from `docs/learn/_plan.md`) is a senior QA/automation engineer (~10 years QA/CI/CD/test frameworks) moving toward an AI Solutions Engineer role in telecom/eSIM/IoT. They understand program flow but are weak on production architecture, concurrency/async, auth/multi-tenancy, observability, evals, deployment, cost/latency tradeoffs, and customer-facing communication.

**[STATED]** The system is **not** for the general public or for generating new knowledge. `CLAUDE.md` says:

> "It is NOT designed to generate new information or speculate beyond the provided source materials."

**[INFERRED]** Because the learner's background is QA/automation rather than software engineering, the system is optimized for **answer correctness and citability** over code-level sophistication. The UI is a Streamlit app (`ui/streamlit_app.py`) and the API is FastAPI (`src/api/main.py`), both chosen for accessibility rather than raw performance.

**[INFERRED]** The secondary "customer" is an interviewer or hiring manager evaluating an FDE candidate's telecom knowledge. The system exists to help the learner give accurate, sourced answers in an interview setting.

---

## 2. The pain this system solves

**[STATED]** FDE interview prep requires accurate, domain-specific telecom answers drawn from verified documents (LTE/5G, Roaming, SIM Security, eSIM), with citations to specific document sections. Without a structured system, a learner must:

- Hunt through documents manually to find the right chunk.
- Trust their own memory or paraphrasing, risking inaccuracy.
- Produce answers without source citations, which an FDE interviewer would not accept.
- Risk hallucinating details that sound plausible but are wrong.

**[STATED]** `CLAUDE.md` requires **citation** and **no speculation**:

> "Include references to specific document sections when answering questions."
> "If information is not in the source documents, state clearly that the answer cannot be determined from available materials."

**[STATED]** The system addresses this with a full RAG pipeline: load → clean → chunk → embed → store (ChromaDB), then retrieve (dense + BM25 hybrid, optional RRF, optional reranker), then generate a grounded answer citing sources (`src/ingestion/`, `src/retrieval/`, `src/generation/`).

**[INFERRED]** A secondary pain is **inconsistency**. Different study sessions could surface different chunks for the same question, giving the learner contradictory answers. The system mitigates this with deterministic retrieval, a query cache (`src/api/main.py:126`), and a golden test set (`src/evaluation/golden_set.json`) used to check consistency over time.

---

## 3. What happens without this system

**[STATED]** Without the system, the learner relies on manual document review. `CLAUDE.md`'s Limitations Awareness section describes the failure modes that the system is designed to contain but cannot eliminate:

- Knowledge is limited to the ingested documents.
- Chunking can separate related information, so answers may seem incomplete.
- Retrieval relevance scoring may miss the most pertinent chunks.
- The system cannot reason beyond combining and extracting information — it is an extraction/combination tool, not a reasoning engine.

**[INFERRED]** Without the pipeline, a learner answering "What is EPS bearers in LTE?" would have to:
1. Remember which document discusses bearers.
2. Locate the right section by eye.
3. Synthesize the answer themselves, with no guarantee they captured all relevant chunks or cited the right section.

With the system, the same question goes through `/ask`, which retrieves top-k chunks, optionally reranks them, generates a grounded answer, and returns sources with snippets — all traceable via request ID and per-stage timings (`src/observability.py`).

**[INFERRED]** Without the document lifecycle (upload, dedupe, versioning, delete cascade, reindex — commit `27b1cf8`), there is no controlled way to add, replace, or retire source documents, so outdated or duplicate telecom material could linger and mislead the learner.

---

## 4. What it costs

This section breaks cost into **(a) LLM/API keys**, **(b) embedding**, **(c) vector store**, and **(d) infrastructure**. Where the code does not state a price, I mark it **[INFERRED]**.

### 4.1 LLM / API keys — the largest variable cost

**[STATED]** The system supports two LLM providers (`src/config.py`):
- **mock** — default (`LLM_PROVIDER=mock`). No network, no API key, zero cost. Used for dev and tests.
- **nvidia** — NVIDIA NIM at `https://integrate.api.nvidia.com/v1`, default model `meta/llama-3.1-70b-instruct`. Requires `NVIDIA_API_KEY` or `NVIDIA_API_KEYS` (comma-separated). Fail-fast if `nvidia` is selected without a key.

**[STATED]** There is a **multi-key pool with round-robin rotation, failover on 401/403, and expiry tracking** (`src/key_pool.py`), plus an **expiry email monitor** with SMTP config (`src/config.py` smtp fields, `key_warn_days` default 14). Commit `6dcbb44` added "multi-key NVIDIA pool with failover + expiry email monitor."

**[INFERRED]** NVIDIA NIM is billed per token (prompt + completion). The exact price is not in the repo; I infer it from the presence of token-usage tracking (`metrics.add_tokens` in `src/api/main.py:593-594`, `usage` in `QueryResponse`) and the multi-key pool (suggesting the operator expects to manage multiple paid keys and rotate before expiry). Cost scales with:
- Number of questions asked (`/ask`, `/ask/stream`).
- Chunk count sent to the LLM (top_k, chunk size 500 default).
- Whether streaming is used (tokens still counted).

**[INFERRED]** There is a **key registry file** (`data/key_registry.json`, gitignored, local only) keyed by key fingerprint (first4…last4, never the full key — `src/key_pool.py:25-33`). Expiry is tracked locally because NVIDIA keys do not expose expiry over the API (`src/key_pool.py` docstring).

### 4.2 Embedding

**[STATED]** Embeddings use a sentence-transformer model (`src/embedding/embedder.py`). There is an **embedding cache** at `data/embedding_cache/` (`_plan.md` recon section 1.9).

**[INFERRED]** Sentence-transformers run locally, so embedding cost is effectively **zero** (no per-call API fee), bounded only by compute/time. The cache avoids re-embedding unchanged chunks on reindex.

### 4.3 Vector store

**[STATED]** ChromaDB at `data/chroma_db`, with an **in-memory fallback** if Chroma is absent (`src/config.py:26`, `Dockerfile:21`, `_plan.md` section 1.9).

**[INFERRED]** ChromaDB is local and open-source, so there is no per-query vector-store fee. Cost is disk space and RAM on the host.

### 4.4 Infrastructure

**[STATED]** Deployment is via **Docker** (`Dockerfile`, `docker-compose.yml`):
- `python:3.11-slim` base image, CPU-only torch pre-installed to keep the image small.
- FastAPI backend on port 8000 (`uvicorn src.api.main:app`).
- Streamlit UI on port 8501.
- Persistent volumes for `data/`, `uploads/`, `logs/`.
- Optional NGINX reverse proxy (commented out in `docker-compose.yml`).

**[INFERRED]** Infrastructure cost is a small VM or container host running two containers (API + UI). Because the vector store, embedding cache, and logs are all local, there are no managed-service fees (no Pinecone, no Redis, no managed OpenAI). The only recurring cash cost is the NVIDIA NIM API tokens (when not using mock). An NGINX reverse proxy is noted as optional for production but not deployed.

### 4.5 Cost controls built in

- **Query cache** (in-memory FIFO, 200 entries, `src/api/main.py:126`) — repeated identical queries skip retrieval+LLM, saving tokens and latency.
- **Rate limiter** (30 requests/minute per client, in-memory, `src/api/main.py:195`) — prevents one client from exhausting the token budget.
- **Mock provider** for CI and local dev — zero token cost during development and testing.
- **Bearer token auth** (`API_TOKEN`, `src/api/main.py:268`) and role-based access (admin/manager/employee/guest) — mutating endpoints (ingest, delete, upload, reindex) require admin; read endpoints are open in dev mode.

---

## 5. How success is measured

**[STATED]** Success is measured through a **golden-set evaluation** (`src/evaluation/golden_set.json`, `src/evaluation/run_eval.py`, `src/evaluation/metrics.py`). Commit `b2741a4` "repair eval runner, wire NVIDIA judge"; commit `bd12d6c` "recreate golden dataset with questions for all telecom documents." CLAUDE.md's Quality Control section says:

> "Regularly test with known questions from the golden dataset to ensure consistency."

The golden set contains questions with `expected_answer_contains`, `expected_sources`, `question_type`, and `hop_count`. Each question is evaluated for:

**[STATED] Retrieval metrics:** hit_rate_at_5, hit_rate_at_10, recall_at_5, recall_at_10, precision_at_5, MRR, NDCG@5, average_precision (`src/evaluation/run_eval.py:415-424`).

**[STATED] Generation metrics:** answer_relevance, faithfulness, contradiction, answer_completeness (`src/evaluation/run_eval.py:426-431`), using an **LLM-as-judge** (NVIDIA provider when real, mock when not — `src/evaluation/run_eval.py:32-50`).

**[STATED] Operational metrics:** request count, latency percentiles (retrieve_ms, llm_ms, request_ms), token usage (prompt/completion), error rates, cache hit rate (`src/observability.py`, `src/api/main.py`). Exposed at `/metrics` and `/traces` (auth-gated).

**[STATED] User feedback:** thumbs up/down (rating 1/0) plus optional comment, stored per `feedback_id` (`src/api/main.py:106-109`, `/feedback` endpoint).

**[INFERRED]** Success for the learner specifically means: the golden set passes consistently (no regression when chunking, hybrid, or reranking settings change), answers are faithful to source chunks (low contradiction score), and the learner can explain the answer with citations. For an operator, success means the service stays up (health endpoint `/health`), stays within token budget (rate limit + cache), and keys do not expire unexpectedly (expiry monitor + email alert).

---

## 6. What a client would ask an FDE about it

**[STATED]** `CLAUDE.md` defines the interaction patterns the system supports — these map directly to what an interviewer/client would ask an FDE:

- **Conceptual:** definition, purpose, key components, practical relevance. (e.g. "What is an EPS bearer in LTE, and why does it matter?")
- **Procedural:** steps, prerequisites, expected outcomes. (e.g. "Walk me through eSIM provisioning.")
- **Troubleshooting:** symptom → resolution. (e.g. "A roaming subscriber can't attach — where do you look first?")
- **Comparison:** similarities and differences. (e.g. "LTE vs 5G NR architecture — what changed and why?")

**[STATED]** The golden set shows the question style: factual, single-hop, with expected answer terms and expected sources (`src/evaluation/golden_set.json`). Examples in the current (corporate) golden set include policy/leave/security/FAQ questions; the telecom set would parallel this with LTE/5G/Roaming/SIM Security/eSIM questions.

**[INFERRED]** A client interviewing an FDE would ask **practical, field-relevant** questions, not pure theory — matching `CLAUDE.md` Context Awareness:

> "focus on practical, field-relevant knowledge rather than purely theoretical concepts."

Example telecom questions an FDE might field (inferred, not in repo):
- "Walk me through the LTE attach procedure end to end."
- "What is the difference between a home subscriber and a roaming subscriber in the HSS/HSS diameter flow?"
- "How does SIM security (Ki, OTA) protect against cloning?"
- "What is eSIM profile download and how does LPA/LDS work?"

**[INFERRED]** The system's answer-with-citations design means an FDE can answer with a source reference ("per section 4.2 of the LTE spec...") — exactly what an interviewer wants to see. If the answer is not in the ingested documents, the system (per CLAUDE.md) should say so rather than speculate — which is itself a signal of honesty an interviewer would value.

---

## 7. Limitations awareness

All four limitations below are stated explicitly in `CLAUDE.md` (Limitations Awareness section). I list them here verbatim-paraphrased and add what each means in practice.

### 7.1 Knowledge is limited to the ingested documents

**[STATED]** CLAUDE.md:

> "Recognize that the system's knowledge is limited to the ingested documents."

**Practical meaning:** If a telecom topic is not in the ingested corpus, the system cannot answer it — even if it is common knowledge to an FDE. The `/provider` endpoint reveals whether the live (NVIDIA) or mock provider is active, but neither knows anything outside the retrieved chunks.

**[INFERRED — important gap]** The current `data/raw/` directory holds **generic corporate documents** (HR/benefits/IT/travel policies for "Acme Corp": `hr_leave_policy.md`, `it_security_policy.md`, `product_faq.md`, `travel_policy.md`, etc.), **not** the telecom documents (LTE/5G, Roaming, SIM Security, eSIM) that CLAUDE.md says the system is for. This is a **purpose/source mismatch** — noted in the recon findings of `docs/learn/_plan.md` (section 1.11). The system works mechanically, but it is not yet indexed with the telecom content its stated purpose requires. This is recorded as a known issue (to go in `07_issues_found.md`), not something to fix in code per the mission constraint (code must not change).

### 7.2 Chunking may separate related information

**[STATED]** CLAUDE.md:

> "Understand that chunking may separate related information - consider broader context when answers seem incomplete."

**Practical meaning:** Documents are split into ~500-character chunks with 50-character overlap (`src/api/main.py:95-98`, `src/ingestion/run.py:28-32`). A sentence that spans a chunk boundary, or two facts that belong together but live in different chunks, may be retrieved separately and appear disconnected. Overlap mitigates but does not eliminate this.

### 7.3 Retrieval relevance scoring may miss pertinent chunks

**[STATED]** CLAUDE.md:

> "Be aware that retrieval relevance scoring may not always surface the most pertinent chunks."

**Practical meaning:** Dense (semantic) and BM25 (keyword) retrieval are fused with RRF (`src/retrieval/hybrid.py`), optionally reranked (`src/retrieval/reranker.py`), but relevance scoring is statistical, not semantic understanding. The top-k may omit the one chunk that actually answers the question. `CLAUDE.md`'s quality control advises retrieving additional chunks when uncertain.

### 7.4 No reasoning beyond combination and extraction

**[STATED]** CLAUDE.md:

> "Acknowledge that the system cannot perform reasoning beyond information combination and extraction."

**Practical meaning:** The LLM answer generator combines retrieved chunks into a response; it does not perform multi-hop inference, arithmetic, or domain reasoning beyond what the chunks state. `CLAUDE.md` Core Purpose:

> "All answers must be grounded in the retrieved document chunks to prevent hallucination."
> "It is NOT designed to generate new information or speculate beyond the provided source materials."

**[INFERRED]** The controlled tool-calling agent (`src/agent/agent.py`, commit `131bbb5`) adds a small amount of orchestration (decide whether to retrieve, which allow-listed tool to call, validate/authorize/execute, trace), but it still answers from retrieved document content — it does not reason independently.

---

## Appendix: stated vs. inferred summary

| Topic | Stated (in CLAUDE.md / code / commits) | Inferred (my deduction) |
|---|---|---|
| User/customer | FDE interview prep learner; telecom (LTE/5G, Roaming, SIM Security, eSIM) | Learner profile is QA→AI Solutions Engineer; accessibility over performance |
| Pain | Manual doc hunting, no citations, hallucination risk | Inconsistent answers across sessions without caching/golden tests |
| Without system | (described via Limitations Awareness) | Manual section-hunting; no controlled doc lifecycle |
| Cost | mock vs nvidia provider; NVIDIA NIM URL+model; multi-key pool; SMTP expiry monitor; ChromaDB local; sentence-transformers local; Docker/uvicorn/Streamlit | NVIDIA per-token pricing (not in repo); small VM host; zero embedding/vector-store fee |
| Success measured | golden set (hit rate, recall, MRR, NDCG, answer relevance, faithfulness, contradiction, completeness); LLM-as-judge; /metrics, /traces; feedback thumbs up/down; latency, cache hit rate, token usage | Learner success = consistent, citable, faithful answers |
| Client asks FDE | conceptual/procedural/troubleshooting/comparison patterns; factual single-hop golden-set style | practical field-relevant telecom questions; cite-by-section answers |
| Limitations | all four (knowledge bounded by docs; chunking separates; relevance may miss; no reasoning beyond combine/extract) | current data/raw/ is corporate, not telecom — purpose/source mismatch |

---

**Bottom line:** The rag-academy RAG system is a low-cost, locally-hosted study tool for FDE telecom interview prep. Its biggest cost is NVIDIA NIM API tokens (avoided entirely with the mock provider). Success is measured by golden-set consistency and answer faithfulness. Its critical gap right now is that the indexed documents are generic corporate policies, not the telecom materials its stated purpose requires — a mismatch to record in `07_issues_found.md`, not to fix in code (per the mission constraint that source code must not change).
