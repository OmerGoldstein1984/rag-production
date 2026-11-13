# RAG Production — Senior AI Builder / Forward Deployed Engineer Cheat Sheet

> **Interview rule:** Describe what the code does today, then separate that from what you would build next. This repo is a useful RAG prototype with real components, but it is not evidence of production-scale quality or reliability.

## 1. The 30-Second Elevator Pitch

This is a telecom/company knowledge-base RAG app. It ingests local PDF, DOCX, Markdown, and text documents, cleans and chunks them, embeds the chunks, and indexes them. A user asks a question through a FastAPI API (or Streamlit UI); the service retrieves relevant chunks, may rerank them, sends their context to a configured LLM provider, and returns an answer with source snippets and usage metadata. It also has document-management routes, an optional controlled tool-calling agent, streaming, and an offline evaluation harness.

**Be honest:** ChromaDB and NVIDIA NIM are supported, but local/mock fallbacks exist. Some evaluation labels are inferred from retrieved results, and the agent currently returns after one tool call. So call it a working prototype, not a proven production RAG service.

```text
Documents ──> load / clean / chunk / embed ──> ChromaDB (or in-memory fallback)
                                                     │
User ──> Streamlit ──HTTP──> FastAPI ──> retrieve ──> rerank ──> grounded prompt + LLM
  └──────────────────────────────────────────────────────────────────> answer + sources
```

## 2. The Happy Path (Step-by-Step Flow)

Standard, non-streaming question path:

1. At server startup, `src/api/main.py` — `startup_event()` calls `initialize_rag_pipeline()` to prepare the index and components. This happens before the question, not once per request.
2. The UI’s `ui/streamlit_app.py` — `main()` collects the question; `call_ask()` sends it to `POST /ask` (or the user can call the API directly).
3. `src/api/main.py` — `ask_question()` applies rate limiting and input sanitization, builds a cache key, returns a cache hit if present, and checks that the pipeline is ready.
4. `src/api/main.py` — `ask_question()` calls the configured retriever’s `retrieve()`. With reranking enabled it requests `top_k * 2` candidates; the retriever in `src/retrieval/retriever.py` embeds the query and searches the store.
5. `src/api/main.py` — `ask_question()` calls `Reranker.rerank()` from `src/retrieval/reranker.py` when requested and available, then keeps up to `top_k` documents.
6. `src/generation/answer.py` — `AnswerGenerator.generate_answer_with_context()` formats a context within an estimated token budget, builds the grounded prompt, calls the provider, checks the output, and packages answer/source/usage metadata.
7. `src/api/main.py` — `ask_question()` redacts/sanitizes the answer, builds `QueryResponse`, caches it, schedules feedback-request logging, and returns JSON. `ui/streamlit_app.py` — `main()` renders the answer and sources.

**Separate streaming route:** `ui/streamlit_app.py` — `call_ask()` uses `/ask/stream`; `src/api/main.py` — `ask_stream()` streams events from `AnswerGenerator.stream_answer_with_context()`.

## 3. Actual Tech Stack & Design Choices

- **API and UI:** FastAPI and Pydantic request/response models (`src/api/main.py`); Streamlit and `requests` (`ui/streamlit_app.py`).
- **Document pipeline:** loaders, cleaning, and custom chunking in `src/ingestion/`; `TextChunker` supports `fixed_size`, `recursive`, and `structure_aware` (`src/ingestion/chunker.py`). Chunk sizes are character-based, not token-based.
- **Embeddings:** `sentence-transformers` `all-MiniLM-L6-v2`, 384 dimensions, plus disk JSON cache keyed by MD5 text hash (`src/embedding/embedder.py`). If model loading/computation fails, hash-based mock vectors are used; that fallback is deterministic, not semantic.
- **Vector storage:** ChromaDB `PersistentClient`, collection `rag_academy`, under `./data/chroma_db`; on missing/unusable ChromaDB, an in-memory cosine-similarity fallback is used (`src/store/vector_store.py`). The fallback loses data on restart.
- **Retrieval:** dense vector search (`src/retrieval/retriever.py`) and a from-scratch in-memory BM25 implementation (`src/retrieval/sparse_retriever.py`). `src/retrieval/hybrid.py` contains RRF; its actual formula adds `1 / (k + rank + 1)` per list. The stored `weight_dense` and `weight_sparse` are not applied by this RRF implementation. A separate weighted hybrid implementation also exists in `src/retrieval/retriever.py`.
- **Reranking:** optional `sentence_transformers.CrossEncoder`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`; if unavailable it passes through the original order/scores (`src/retrieval/reranker.py`).
- **Generation:** `LLMProvider` abstraction and env-based factory (`providers/llm.py`); NVIDIA NIM via synchronous `httpx` (`providers/llm_nvidia.py`), or mock provider. `AnswerGenerator` uses `GenerationResult` dataclass, retry/timeout/circuit-breaker logic, and heuristic token estimate of about four characters per token (`src/generation/answer.py`). OpenAI and Anthropic classes in `providers/llm.py` are placeholders, not real integrations.
- **State and agent:** Pydantic models for HTTP/tool input; dataclasses such as `ToolContext`, `ToolSpec`, `ToolCall`, and `AgentResult` (`src/agent/tools.py`, `src/agent/agent.py`). The agent uses a keyword heuristic by default; optional LLM planning is allow-listed, schema-validated, authorized, timed, and recorded. Its current run returns after one tool call.
- **State persistence:** the vector store may persist via ChromaDB, but query cache, feedback list, `rag_pipeline`, and BM25 corpus are process-local Python state (`src/api/main.py`, `src/retrieval/sparse_retriever.py`).

## 4. Testing & Evaluation Reality Check

- `tests/` contains unit and API-oriented tests for chunking, loaders, lifecycle, agent/tool guardrails, security, RRF, metrics, providers, observability, streaming, and an end-to-end flow. `tests/conftest.py` resets API globals before and after each test.
- **What is mocked:** `tests/test_nvidia_provider.py` uses `httpx.MockTransport` for HTTP success/errors/retries/streaming—no network or real API key. `tests/test_generation_real.py` injects a `FakeProvider` for generation and usage assertions. `providers/llm.py` and `src/generation/answer.py` also provide deterministic mock generation paths. `tests/test_end_to_end.py` creates temporary documents and runs the component pipeline, but it mainly checks that results exist and include expected terms; it is not a broad quality benchmark.
- **Metrics implemented:** retrieval `hit_rate@5/10`, `recall@5/10`, `precision@5`, `MRR`, binary `nDCG@5`, and `average_precision`; generation `answer_relevance`, `faithfulness`, `contradiction`, and optional keyword-based `answer_completeness` (`src/evaluation/metrics.py`). Without an LLM judge, relevance and faithfulness use simple lexical heuristics; contradiction is set to `0.0`.
- **Critical evaluation caveat:** the golden set has 30 questions (`src/evaluation/golden_set.json`), but `src/evaluation/run_eval.py` derives “relevant” document IDs by checking expected answer terms inside the already retrieved final documents. If none match, it labels the top three retrieved docs as pseudo-relevant. `expected_sources` is recorded but not used to establish relevance. Therefore these retrieval scores are not an independent ground-truth measurement and can look good without proving retrieval quality. The LLM judge is also a judge, not ground truth.

## 5. 5 "Gotcha" Interview Questions (With Code-Grounded Answers)

1. **Question: “How do you know your retrieval evaluation is measuring real relevance?”**  
   **Answer:** “Today, I do not claim it proves that. In `src/evaluation/run_eval.py`, relevance is inferred from expected-term matches in returned documents, and if there are no matches, the top three are labeled relevant. That creates evaluation leakage. I would create an independently reviewed query-to-document/chunk relevance set, score all retrieved candidates against it, separate answer evaluation from retrieval labels, and report per-query results and confidence intervals.”

2. **Question: “What happens if the embedding model is missing or breaks—will search still be meaningful?”**  
   **Answer:** “It will still run, but not necessarily search meaningfully. `src/embedding/embedder.py` falls back to deterministic hash-based vectors. That helps offline tests, but these vectors do not encode meaning. In production I would fail readiness or mark the service degraded, expose the active embedding model, and prevent indexing/querying with mismatched or mock embeddings.”

3. **Question: “Can a request turn hybrid retrieval or RRF off, and do the RRF weights tune the result?”**  
   **Answer:** “The API request model has these flags, but the normal path does not use them to choose a retriever: `initialize_rag_pipeline()` chooses the retriever when the pipeline is built, while `ask_question()` calls that stored retriever. In `src/retrieval/hybrid.py`, `weight_dense` and `weight_sparse` are stored but the RRF formula does not use them. I would make retrieval configuration explicit per pipeline, reject unsupported combinations, and test that each flag changes the actual retrieval path.”

4. **Question: “Your agent has a five-step limit. Show me a question that makes it use multiple tools.”**  
   **Answer:** “I can’t show one today. `src/agent/agent.py` — `RAGAgent.run()` plans and executes one tool, then returns from either the search or non-search branch. `MAX_STEPS` exists, but multi-step reasoning is not implemented. I would add an observation/replan loop, validate every next action, cap steps/time/cost, and test multi-tool tasks plus failures.”

5. **Question: “What breaks if you run multiple API workers or restart the service?”**  
   **Answer:** “The process-local query cache and `feedback_store` are lost on restart and not shared across workers; `rag_pipeline` and BM25 state are also local (`src/api/main.py`, `src/retrieval/sparse_retriever.py`). ChromaDB can persist the vector index, but that does not make the other state distributed. I would move shared state to appropriate managed stores, define safe index updates/locking, and add multi-worker and restart recovery tests.”
