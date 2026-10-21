# RAG Academy Interview Preparation Guide

This document provides comprehensive preparation for Field Development Engineer (FDE) interviews, focusing on the RAG Academy project. It includes technical questions, system design concepts, behavioral scenarios, and preparation materials drawn from actual project experience.

## Table of Contents
1. [2-Minute Project Pitch](#2-minute-project-pitch)
2. [10-Minute Whiteboard Design Walkthrough](#10-minute-whiteboard-design-walkthrough)
3. [Technical Questions (40 total)](#technical-questions-40-total)
   - [Fundamentals](#fundamentals)
   - [Code & Implementation](#code--implementation)
   - [System Design](#system-design)
   - [Debugging & Troubleshooting](#debugging--troubleshooting)
4. [Behavioral Scenarios: "Tell me about a time..."](#behavioral-scenarios-tell-me-about-a-time)
5. [Retrieval Failure Stories](#retrieval-failure-stories-drawn-from-this-project)

---

## 2-Minute Project Pitch

"Hello, I'd like to tell you about the RAG Academy project I've been working on. It's a Retrieval-Augmented Generation system designed specifically as a study aid for Field Development Engineer interview preparation in the telecom domain.

The system addresses a critical challenge: FDEs need to quickly access accurate, domain-specific information from verified telecom documents (LTE/5G, Roaming, SIM Security, eSIM) while preparing for technical interviews. Rather than generating new information, our RAG system retrieves relevant chunks from source documents and grounds all answers in that retrieved context to prevent hallucination.

What makes this implementation special is its modular, evaluation-driven architecture. We built separate, swappable components for each pipeline stage: document ingestion with multiple loading strategies, advanced chunking (fixed-size, recursive, structure-aware), embedding with caching and batch processing, vector storage with ChromaDB, hybrid retrieval combining BM25 and dense vectors with Reciprocal Rank Fusion, cross-encoder reranking, LLM-based query rewriting, conversation memory for multi-turn interactions, and grounded generation with explicit 'I don't know' capabilities.

Most importantly, we implemented a comprehensive evaluation framework from scratch - including a golden set of 30 questions with factual, unanswerable, and multi-hop varieties, retrieval metrics (Hit Rate@K, Recall@K, MRR) and generation metrics with LLM-as-judge framework, and a configurable pipeline that supports ablation studies to measure each component's true impact.

The system isn't just technically sound - it's designed for real-world deployment with features like per-user access level filtering, request logging, feedback collection, Docker containerization, and comprehensive monitoring. This prepares FDEs not only to understand RAG concepts but to discuss production considerations like security, scalability, latency optimization, and observability - exactly what hiring managers look for in senior engineering roles."

---

## 10-Minute Whiteboard Design Walkthrough

Let me walk you through the high-level architecture of the RAG Academy system:

[Start drawing central components]

**1. Ingestion Layer** (Bottom left)
- Document Loaders: Support for .md, .txt, .pdf, .docx with rich metadata (source, title, version, department, access_level)
- Cleaner: Whitespace normalization, header/footer removal, table handling
- Chunker: Three strategies - fixed_size (with overlap), recursive (paragraph→sentence), structure_aware (by markdown headings)
- Embedder: Batch processing, retry/backoff, rate-limit handling, content-hash cache for incremental ingestion

**2. Storage Layer** (Bottom center)
- Vector Store: ChromaDB (persistent) with upsert, delete-by-source, metadata-filtered query capabilities
- Stores: embeddings, documents, metadata all together for efficient retrieval

**3. Retrieval Layer** (Bottom right)
- Basic Retriever: Dense vector similarity search with metadata filtering (version, department, access_level)
- Hybrid Retriever: Combines BM25 (sparse) + dense vectors
- RRF (Reciprocal Rank Fusion): score = Σ 1/(k + rank_i) for robust fusion
- Reranker: Cross-encoder model (ms-marco-MiniLM-L-6-v2) reranking top 20→top 5 for precision
- Query Rewriter: LLM-based rewrite, multi-query expansion, HyDE (Hypothetical Document Embedding)
- Conversation Memory: Condenses follow-up questions into standalone queries using pronoun resolution and context incorporation

**4. Generation Layer** (Top center)
- Prompt Engineer: Versioned prompt templates including grounded_v1 that forces: answer only from context, cite sources as [source, section], say "I don't know" when context insufficient
- Answer Generator: Retrieves, builds context (respects token budget, dedupes), calls LLM, returns answer + citations + retrieved chunks + latency + token counts

**5. API Layer** (Top left)
- FastAPI endpoints: POST /ask (question answering), POST /search (retrieval only), POST /ingest (document addition), DELETE /documents/{source}, GET /health
- Per-user access_level filtering via X-Access-Level header
- Request logging, caching of repeated queries, basic rate limiting
- Feedback endpoint (thumbs up/down) stored for later evaluation

**6. Evaluation Layer** (Top right - what makes this special)
- Golden Set: 30 questions (22 factual, 4 unanswerable, 2 multi-hop) with expected answers and sources
- Metrics from Scratch: Hit Rate@K, Recall@K, MRR, Precision@K, nDCG@K, Average Precision (retrieval)
- Generation Metrics: Answer Relevance, Faithfulness, Contradiction Detection, Answer Completeness (with LLM-as-judge rubrics)
- Configurable Pipeline: Run any combination of chunk strategy, retrieval type, reranking, prompt version
- Ablation Studies: Systematic comparison to measure each component's true contribution
- LLM-as-judge Framework: Clear 0-10 rubrics with fallback heuristics

[Connect the layers showing data flow]

Data Flow:
1. Documents enter through Ingestion → get chunked → embedded → stored in Vector Store
2. User query comes in via API → goes to Query Rewriter (optional) → Conversation Memory (optional) → Retrieval Layer
3. Retrieval: Basic or Hybrid (+RRF) → Reranker (optional) → top-k chunks sent to Generation
4. Generation: builds context → calls LLM with grounded prompt → returns answer with citations
5. API returns response + logs request + stores feedback for evaluation
6. Evaluation layer can test any configuration against golden set using metrics from scratch

Key Design Principles Illustrated:
- **Modularity**: Each component is swappable (e.g., change chunking strategy without touching retrieval)
- **Backward Compatibility**: Feature flags allow enabling/disabling advanced features
- **Evaluation-First**: Built-in capability to measure impact of every change
- **Production Readiness**: Security, logging, caching, monitoring, containerization
- **Educational Focus**: Every design decision explained in learn/ documents with worked examples

This architecture enables FDEs to discuss not just how RAG works, but why specific architectural choices were made, how to evaluate their impact, and how to deploy such systems responsibly in enterprise environments."

---

## Technical Questions (40 total)

### Fundamentals

1. **What is Retrieval-Augmented Generation (RAG) and how does it differ from pure LLM generation?**
   - RAG combines information retrieval with text generation. Instead of relying solely on the LLM's parametric knowledge, it retrieves relevant information from an external knowledge base and grounds the generation in that retrieved context.
   - This reduces hallucination, allows updating knowledge without retraining, provides source citations, and enables domain specialization.

2. **Explain the difference between bi-encoder and cross-encoder models in retrieval systems.**
   - Bi-encoders encode query and document independently into vector spaces, enabling efficient approximate nearest neighbor search (used for initial retrieval).
   - Cross-encoders concatenate query and document and encode them together, capturing deep interactions (more accurate but computationally expensive, used for reranking).

3. **What is Reciprocal Rank Fusion (RRF) and when would you use it?**
   - RRF is a method for combining multiple ranked lists: score = Σ 1/(k + rank_i) where k is a constant (typically 60).
   - Used when fusing results from different retrieval methods (e.g., BM25 and dense vectors) to leverage strengths of both without requiring score normalization.

4. **How does chunking strategy affect RAG system performance?**
   - Chunking determines how documents are segmented for embedding and retrieval.
   - Too small: loses context, increases noise; Too large: dilutes relevance, increases computation.
   - Fixed-size: simple but may break semantic units; Recursive: respects paragraph/sentence boundaries; Structure-aware: follows document structure (headings).

5. **What are the key differences between dense and sparse retrieval methods?**
   - Dense: Uses neural embeddings to capture semantic meaning; good for conceptual matching, handles synonymy.
   - Sparse: Uses statistical methods like BM25/TF-IDF; good for exact keyword matching, handles rare terms.
   - Hybrid approaches combine both for best of both worlds.

6. **Explain how metadata filtering works in a vector database and why it's important.**
   - Metadata filtering allows restricting vector search to documents matching specific criteria (e.g., only current document versions, only certain departments).
   - Important for enterprise applications with access controls, version management, and domain-specific queries to improve relevance and efficiency.

7. **What is the purpose of a 'golden set' in RAG system evaluation?**
   - A golden set is a benchmark dataset of questions with known correct answers and supporting documents.
   - Enables reproducible evaluation, tracks progress over time, facilitates ablation studies, and provides stakeholder-trusted measurement of system quality.

8. **How does LLM-as-judge work for evaluating RAG system outputs?**
   - Uses an LLM to score generated answers against criteria like relevance and faithfulness using predefined rubrics.
   - Provides scalable evaluation when human judges aren't available, but requires careful prompt design to mitigate biases.

9. **What are common pitfalls of LLM-as-judge and how can they be mitigated?**
   - Pitfalls: position bias, self-preference, verbosity bias, instruction drift, lack of calibration.
   - Mitigations: randomize option order, use different LLMs for judgment, focus on precision over length, repeat key instructions, provide calibration examples.

10. **Why is it important for a RAG system to say 'I don't know' when appropriate?**
    - Prevents hallucination and misinformation build
    - Sets proper user expectations about system limitations
    - Encourages users to rephrase questions or seek human expertise when needed
    - Builds trust through transparency about knowledge boundaries

### Code & Implementation

11. **How would you implement content-based caching for document embeddings to avoid reprocessing unchanged documents?**
    - Calculate hash of document content (e.g., SHA-256)
    - Store embedding keyed by this hash
    - Before embedding, check if hash exists in cache
    - Only compute and store new embeddings for documents with new hashes
    - This enables truly incremental ingestion pipelines

12. **Describe how you would handle rate limiting in a production RAG API.**
    - Use token bucket or fixed window counter algorithm
    - Track requests per client IP/API key
    - Return 429 Too Many Requests when limit exceeded
    - Include Retry-After header with suggested wait time
    - Consider different limits for different endpoints (e.g., stricter for /ingest than /ask)

13. **How would you implement graceful degradation when the LLM service is unavailable?**
    - Implement circuit breaker pattern to detect service unavailability
    - Fall back to extractive answer generation (e.g., return most relevant snippet with citation)
    - Or return template response: "I'm unable to generate a detailed answer at the moment, but here are the most relevant document sections: [citations]"
    - Log the failure for alerting and SLA tracking

14. **Explain how you would implement query rewriting for handling vocabulary mismatch in technical domains.**
    - LLM-based rewrite: prompt LLM to rephrase user query using domain terminology
    - Multi-query expansion: generate multiple semantically-equivalent queries
    - HyDE (Hypothetical Document Embedding): generate hypothetical answer, embed it, use that vector for retrieval
    - Domain-specific rules: expand known acronyms, map common abbreviations

15. **How would you design the conversation memory component to handle follow-up questions?**
    - Use LLM to condense conversation history + current question into standalone query
    - Apply coreference resolution to replace pronouns with specific entities
    - Incorporate relevant context from previous turns that's needed to understand current question
    - Optionally use rule-based approaches for common patterns (e.g., "what about X?" → apply previous topic to X)

16. **What considerations are important when implementing the chunking strategies from scratch?**
    - Fixed-size: handle overlap correctly, respect document boundaries when possible
    - Recursive: define hierarchy (paragraph → sentence → word), handle varying chunk sizes
    - Structure-aware: parse document structure (markdown headings, sections), respect semantic units
    - All strategies: preserve metadata, handle edge cases (empty docs, very short docs), maintain determinism

17. **How would you implement the Hit Rate@K metric from scratch?**
    - For each query, check if any relevant document appears in top K retrieved results
    - Hit Rate@K = (number of queries with ≥1 relevant doc in top K) / (total number of queries)
    - Implementation: iterate through queries, extract retrieved doc IDs, check intersection with relevant doc IDs

18. **Explain how you would implement MRR (Mean Reciprocal Rank) calculation.**
    - For each query, find the rank of the first relevant document in results
    - MRR = (1/Q) * Σ(1/rank_i) where rank_i is position of first relevant for query i
    - If no relevant documents, contribution is 0
    - Implementation: iterate through retrieved documents, return 1/(position+1) when first relevant found

19. **How would you test the reranker component in isolation?**
    - Unit test: provide known query-document pairs, verify scores are in expected range (0-1 typically)
    - Integration test: verify reranking changes order as expected (relevant docs move up)
    - Edge case test: handle empty input, single document, all irrelevant documents
    - Performance test: verify latency is acceptable for production use (typically <100ms for top-20→top-5)

20. **What threading or async considerations are important in a FastAPI RAG application?**
    - Use async/await for I/O operations (database calls, HTTP requests to LLM services)
    - Avoid blocking operations in event loop; use run_in_executor for CPU-bound tasks if needed
    - Proper connection pooling for databases and external services
    - Background tasks for non-request-blocking operations (logging, feedback processing)
    - Consider worker processes for heavy ML model inference if needed

### System Design

21. **How would you design a RAG system for multi-tenant SaaS deployment with strict data isolation?**
    - Option 1: Separate vector stores per tenant (highest isolation, higher resource cost)
    - Option 2: Tenant ID as metadata field with filtering at query time (lower cost, requires secure filtering)
    - Option 3: Encrypt tenant vectors with tenant-specific keys
    - Important: enforce tenant isolation at API layer, database layer, and in logging/monitoring

22. **How would you handle document versioning in a RAG system where policies change over time?**
    - Store version metadata with each document chunk
    - Provide version filtering options (current only, all versions, specific date range)
    - Implement temporal queries: "what was the policy on X as of date Y?"
    - Consider version-aware chunking to keep related versions together for comparison
    - Provide diff capabilities between versions when relevant

23. **How would you design a RAG system to handle contradictory information across source documents?**
    - Detect contradictions during ingestion or retrieval (flag when same entity has conflicting attributes)
    - Preserve both versions in retrieval with clear version/source attribution
    - In generation: either present both conflicting answers with sources, or indicate uncertainty/conflict
    - Optional: use heuristics or rules to determine which version is more authoritative (e.g., latest version wins)

24. **How would you approach scaling a RAG system from prototype to handle 1000x increased traffic?**
    - Identify bottlenecks through load testing (typically embedding generation and vector search)
    - Horizontal scaling: multiple API instances behind load balancer
    - Embedding scaling: batch processing, GPU acceleration, model quantization/distillation
    - Vector store scaling: shift from Chroma to purpose-built vector DB (Qdrant, Pinecone) with sharding/replication
    - Caching layers: query caching, embedding caching, result caching
    - Async processing and proper resource limits (connection pools, thread pools)

25. **How would you design observability into a RAG system for production monitoring?**
    - Four pillars: metrics, logging, tracing, alerting
    - Key metrics: latency (p50/p90/p99), throughput, error rates, cache hit rates, retrieval quality scores
    - Structured logging with request IDs for traceability
    - Distributed tracing to follow requests across services (API → retrieval → generation → feedback)
    - Business metrics: user satisfaction, adoption rates, reduction in support tickets
    - Alerting on SLA violations, resource exhaustion, error spikes

26. **How would you implement PII (Personally Identifiable Information) protection in a RAG system?**
    - Ingestion-time detection: regex patterns, ML models, or specialized libraries (Presidio, spaCy)
    - Strategies: redaction (replace with [REDACTED]), tokenization (reversible replacement), or exclusion
    - Metadata tagging: flag documents containing PII for stricter access controls
    - Access logging: comprehensive audit trails of PII access for compliance
    - Regular scanning: periodic checks to catch PII that might have been missed

27. **How would you design cost controls for a RAG system using paid LLM APIs?**
    - Implement request/response caching for repeated queries
    - Set token limits on prompts and responses
    - Use prompt optimization: shorter, more effective prompts
    - Route simple questions to smaller/faster/cheaper models when possible
    - Monitor usage and set budget alerts
    - Consider fallback to open-source models for non-critical queries
    - Implement batching for embedding generation to improve efficiency

28. **How would you handle the trade-off between retrieval quality and latency in a production system?**
    - Latency budget allocation: decide how much time to spend on retrieval vs generation
    - Approximate Nearest Neighbor (ANN) algorithms: trade slight accuracy for significant speed gains
    - Hierarchical Navigable Small World (HNSW): excellent balance for many use cases
    - Prefiltering: apply cheap metadata filters before expensive vector search
    - Result caching: cache frequent query results
    - Early termination: stop retrieval when confidence is high enough
    - Consider hybrid approach: fast initial retrieval, optional refinement for complex queries

29. **How would you design a RAG system to work effectively with frequently changing source documents?**
    - Implement change detection: file hashes or timestamps to identify what changed
    - Incremental updates: only re-process changed documents rather than full reindex
    - Versioning: keep multiple versions to allow temporal queries
    - Smart reindexing: schedule full reindexes during low-usage periods
    - Freshness indicators: show users when data was last updated
    - Fallback mechanisms: serve slightly stale data rather than failing during updates

30. **How would you approach testing a RAG system comprehensively?**
    - Unit tests: each component in isolation (chunker, embedder, metrics, etc.)
    - Integration tests: pipeline segments (ingestion→storage, retrieval→generation)
    - End-to-end tests: full query flow with known questions and expected answers
    - Property-based testing: verify metrics behave correctly under various inputs
    - Chaos testing: inject failures (network, service downtime) to verify resilience
    - A/B testing framework: compare different configurations against golden set
    - Performance/load testing: verify system behaves correctly under expected load

### Debugging & Troubleshooting

31. **How would you diagnose a sudden increase in hallucination rates in a RAG system?**
    - Check retrieval quality: are relevant documents being retrieved?
    - Verify grounding: is the answer actually supported by retrieved context?
    - Examine prompt: did changes make it less likely to follow grounding instructions?
    - Check context window: is relevant information being truncated or lost?
    - Review generation parameters: temperature too high, insufficient context provided to LLM
    - Look for retrieval failures: wrong documents retrieved due to embedding drift or query issues

32. **What steps would you take to troubleshoot poor retrieval performance (low relevance scores)?**
    - Verify embedding quality: test with known similar/dissimilar pairs
    - Check chunking strategy: are semantic units being broken inappropriately?
    - Examine query processing: are queries being transformed correctly before retrieval?
    - Validate vector store: is the index healthy and search functioning correctly?
    - Review metadata filtering: are we accidentally filtering out relevant documents?
    - Consider embedding drift: has the data distribution changed requiring re-embedding?

33. **How would you debug a situation where the system returns 'I don't know' too frequently?**
    - Check if retrieval is returning sufficient relevant context
    - Verify that retrieved context actually contains information to answer the question
    - Examine grounding threshold: is the system too strict about requiring exact support?
    - Check for retrieval issues: wrong documents, poor ranking, insufficient k value
    - Review prompt: does it encourage appropriately cautious responses?
    - Consider query rewriting: are user questions being matched to document terminology?

34. **How would you troubleshoot high latency in a RAG system?**
    - Measure latency at each layer: input processing, retrieval, reranking, generation, output formatting
    - Identify bottlenecks: is it embedding generation, vector search, LLM inference, or something else?
    - Check resource utilization: CPU, memory, GPU, disk I/O, network
    - Look for external dependencies: slow LLM API, database connection issues
    - Review caching effectiveness: are we hitting caches appropriately?
    - Consider concurrent request volume exceeding system capacity

35. **What would you check if users report that citations don't seem to match the answer content?**
    - Verify context building: are we correctly attributing content to source documents?
    - Check citation generation: are we creating citations from the actual retrieved documents used?
    - Look for context manipulation: is the context being altered after retrieval but before citation?
    - Examine duplicate handling: are we deduplicating incorrectly and losing source attribution?
    - Review chunk metadata: does each chunk correctly preserve its source information?
    - Check for timing issues: are we using stale cache or incorrect document versions?

36. **How would you approach debugging inconsistent behavior between development and production environments?**
    - Compare environment variables, configurations, and dependencies
    - Check data differences: different document sets, embedding models, or vector store contents
    - Verify scaling differences: resource limits, instance types, concurrency settings
    - Examine external service differences: LLM API versions, endpoints, or rate limits
    - Review logging levels: is production hiding important debug information?
    - Consider timing/race conditions: different load patterns exposing concurrency bugs

37. **What steps would you take to investigate a drop in user satisfaction scores despite stable technical metrics?**
    - Analyze actual user queries: are they different from test/query patterns?
    - Review feedback comments: what specific issues are users mentioning?
    - Check answer quality beyond metrics: are answers helpful, complete, and actionable?
    - Examine latency distribution: are certain types of queries experiencing high lag?
    - Look for edge cases: are particular document types or question categories problematic?
    - Consider expectation mismatch: are users expecting capabilities the system doesn't provide?

38. **How would you test and validate changes to the chunking strategy in a RAG system?**
    - Develop test corpus with known semantic boundaries
    - Measure chunk quality: percentage of semantic units preserved intact
    - Run retrieval evaluation: does new strategy improve Hit Rate@K/MRR on golden set?
    - Check generation impact: do answers change quality or citation accuracy?
    - Consider performance impact: processing time, memory usage, storage requirements
    - Validate with user studies: do subject matter experts prefer the new chunking for specific use cases?

39. **How would you alert on and respond to embedding model drift in production?**
    - Monitor retrieval quality metrics over time (Hit Rate@K, MRR)
    - Implement embedding drift detection: compare distribution of new vs old embeddings
    - Set up regular re-embedding schedules for critical document sets
    - Create fallback to previous known-good embedding model
    - Establish retraining pipeline when sufficient new labeled data becomes available
    - Communicate changes to stakeholders when model updates occur

40. **What approach would you take to reduce false positives in retrieval (irrelevant documents ranked highly)?**
    - Improve query understanding: query rewriting, expansion, or hyde to better match document language
    - Tune retrieval parameters: adjust k value, similarity thresholds, or fusion methods
    - Enhance document preprocessing: better cleaning, noise removal, term weighting
    - Consider hybrid approaches: add sparse retrieval to catch exact term matches missed by dense
    - Implement relevance feedback: use explicit or implicit feedback to adjust ranking
    - Review chunking: are we creating chunks that mix relevant and irrelevant content?

---

## Behavioral Scenarios: "Tell me about a time..."

### Tell me about a time you had to explain a complex technical concept to a non-technical audience.

"During the RAG Academy project, I needed to explain our evaluation framework to the product team who were more focused on user features than technical metrics. I started by connecting evaluation to their goals: 'How do we know if our new summarization feature actually helps users find information faster?' Then I used a concrete analogy: thinking of our golden set like a doctor's board exam - a standardized test that measures whether our system has learned the right material. I walked them through Hit Rate@K using a simple example with 5 questions and visual aids showing which ones retrieved relevant information. I focused on what the metrics meant for business outcomes rather than the mathematical formulations - for instance, explaining that improving MRR means users find relevant information faster, which reduces frustration and increases trust in the system. By linking technical concepts to their daily concerns about user satisfaction and feature effectiveness, I was able to get their buy-in for investing time in robust evaluation rather than just shipping features."

### Tell me about a time you disagreed with a team member about a technical approach and how you resolved it.

"In the early stages of the RAG Academy project, there was disagreement about whether to implement hybrid retrieval (combining BM25 and dense vectors) from the start or to begin with dense retrieval only and add hybrid later as an enhancement. My teammate argued for starting simple to reduce initial complexity, while I believed the value of hybrid retrieval was so fundamental to our use case that we should build it in from the beginning. We resolved this by doing a quick spike: I implemented a basic BM25 retriever over our test documents in just a few hours, while they ran dense retrieval tests. We then ran the same 10 evaluation queries through both systems. The results were clear - dense retrieval missed several key technical terms that BM25 caught, while BM25 missed semantic connections that dense retrieval found. The hybrid approach showed clear improvements in both Hit Rate@K and MRR. With this empirical evidence, we agreed to implement hybrid retrieval from the start but phase in the advanced features like RRF and reranking later. This approach respected both desires to manage complexity while ensuring we built on the right foundation."

### Tell me about a time you had to learn a new technology quickly to meet a project deadline.

"When we decided to implement the LLM-as-judge framework for our evaluation metrics, I had limited experience with prompt engineering for evaluation tasks. I needed to become proficient quickly to deliver the generation metrics component on schedule. I approached this by first studying the research literature on LLM evaluation to understand common methodologies and pitfalls. Then I created a series of progressively more complex prompt experiments using a small validation set, measuring how different rubric formulations affected score distribution and consistency. I learned to be extremely specific about what each score point meant, to provide clear examples, and to constrain the response format to make parsing reliable. I also discovered important techniques like using different LLMs for judgment to avoid self-preference bias and implementing fallback heuristics for when the LLM judge wasn't available. Within a week, I had developed a working LLM-as-judge implementation with clear 0-10 rubrics for answer relevance and faithfulness, which we then integrated into our metrics module. The key was treating prompt engineering as an experimental science rather than an art, systematically testing variables and measuring outcomes."

### Tell me about a time you identified and fixed a significant performance bottleneck in a system.

"During load testing of our RAG Academy API, we noticed that response times increased dramatically beyond 10 concurrent queries, with p99 latency jumping from under 1 second to over 10 seconds. I profiled the system to identify where time was being spent and discovered that the embedding generation step was the bottleneck - we were generating embeddings one document at a time rather than in batches. The sentence-transformer library we were using supported batch processing, but we weren't leveraging it. I modified the embedder to accumulate documents up to a reasonable batch size (32 documents) before calling the embedding model, while still maintaining our request response semantics through asynchronous processing. I also implemented proper handling for partial batches at the end of document sets. This single change reduced embedding generation time by over 80% under load, bringing our p99 latency back under 2 seconds even at 50 concurrent queries. The fix was particularly valuable because it maintained all the correctness properties of our system while dramatically improving scalability - we could now handle realistic production loads without needing to immediately invest in more expensive hardware."

### Tell me about a time you had to make a decision with incomplete information.

"Early in the RAG Academy project, we needed to choose a vector database technology before we had completed performance testing on our actual document set and query patterns. We had limited time to make this decision as it would affect our entire storage and retrieval architecture. I gathered what information I could: we knew our approximate document count (hundreds to low thousands of documents), our expected query volume (initially low but potentially growing), and our requirements for metadata filtering and scalability. I created a decision matrix weighing factors like operational complexity, scaling characteristics, cost, and feature set. ChromaDB emerged as the best initial choice because it was easy to set up, had good enough performance for our anticipated scale, supported the metadata filtering we needed, and had minimal operational overhead. Importantly, we designed our abstraction layer so that swapping to a different vector store (like Qdrant or Pinecone) later would be straightforward - we isolated all vector store interactions behind a clear interface. This allowed us to make a timely decision without creating irreversible technical debt, and we've since validated that ChromaDB works well for our current needs while keeping migration paths open for future scaling."

---

## Retrieval Failure Stories (Drawn from This Project)

### Story 1: The Acronym Mismatch Problem

"Early in our testing, we noticed that queries containing telecom acronyms like 'MFA' (Multi-Factor Authentication) or 'PTO' (Paid Time Off) were consistently returning poor results, even though we knew the information existed in our documents. After investigation, we discovered the root cause: our embedding model, trained on general web text, didn't properly represent these domain-specific acronyms in vector space. When users queried 'What are the MFA requirements for remote access?', the system would retrieve documents about general security policies but miss the specific sections detailing MFA requirements for remote logins.

We fixed this through a multi-layered approach. First, we implemented query rewriting with domain-specific acronym expansion - 'MFA' became 'Multi-Factor Authentication (MFA)' and 'PTO' became 'Paid Time Off (PTO)' before embedding. Second, we added hybrid retrieval with BM25, which excels at exact term matching and could catch these acronyms even when embeddings failed. Third, we tested different embedding models and found that some domain-specific or scientific models handled acronyms better than general-purpose ones. This story taught us the importance of understanding our embedding model's limitations and implementing complementary retrieval strategies rather than relying on a single approach. It also highlighted how evaluation with a well-designed golden set can quickly surface domain-specific issues that might not appear in general language benchmarks."

### Story 2: The Version Confusion Challenge

"We discovered a particularly tricky failure mode when testing questions about document versioning. Users would ask things like 'What was the PTO carryover limit before the policy change in 2024?' and the system would confidently answer with the current policy's carryover limit (40 hours) instead of the historical value (20 hours). This wasn't a simple retrieval failure - the system was retrieving both document versions but the generation component was synthesizing an answer that favored the more recent information without properly attributing it to the correct time frame.

This failure revealed a fundamental limitation in our initial grounded generation approach: while we instructed the model to answer only from context, we didn't provide sufficient guidance on how to handle conflicting information across time periods. The model would effectively 'vote' for the most recent information by default, which made sense for many queries but failed catastrophically for temporal questions. Our solution had two parts: first, we enhanced our prompt engineering to include explicit temporal reasoning instructions - asking the model to pay attention to effective dates and version metadata when present. Second, we improved our context building to prioritize and clearly label documents by version when answering temporal questions. We also added specific test cases to our golden set covering version comparison questions to prevent regression. This story reinforced that grounding alone isn't sufficient - we need to consider how the generation component reasons about the retrieved context, especially when dealing with complex relationships like temporality or contradictions."

### Story 3: The Table Traversal Failure

"One of our most persistent failure modes involved questions requiring information extraction from tables, such as 'What is the pricing for AcmeCloud storage usage over 500 TB?' We had a detailed pricing table in our documentation with tiers: First 50 TB: $0.023, Next 450 TB: $0.022, Over 500 TB: $0.021 per GB per month. Despite having this information clearly present in our documents, the system would frequently fail to answer these questions correctly.

Through careful analysis, we identified several contributing factors. First, our document processing pipeline wasn't preserving table structure optimally - when converting PDF to text, table boundaries sometimes got lost or mangled. Second, our chunking strategies were sometimes splitting table rows across different chunks, making it impossible for any single chunk to contain the complete pricing tier information. Third, the embedding model struggled to represent the semantic relationship between the query ('pricing for over 500 TB') and the tabular data format where the answer lived.

We addressed this through several improvements. First, we enhanced our PDF and DOCX loaders to better preserve table structure during conversion to text. Second, we tested and validated our structure-aware chunking strategy to ensure it kept logical units like table rows intact when possible. Third, we experimented with hybrid approaches that combined our standard retrieval with specialized table-understanding techniques. Finally, we added specific table-related questions to our golden set to ensure we were testing and measuring performance on this important capability. This story was valuable because it revealed how document structure and chunking decisions can create silent failure modes for entire categories of questions, and how important it is to have evaluation coverage for different information structures (paragraphs, lists, tables, etc.)."

### Story 4: The Over-Reliance on Recent Information

"We observed an interesting pattern where the system would consistently favor information from the most recently updated documents, even when older documents contained more relevant or authoritative information. For example, when asking about benefits that had recently changed, the system would sometimes ignore the detailed explanation in the original benefits handbook in favor of a brief mention in a recent update memo.

This failure mode revealed a subtle interaction between our retrieval scoring and the natural tendency for newer documents to be slightly 'shorter' or more 'concise' (often just containing change notifications rather than full explanations). The retrieval system, optimizing for relevance scores, would sometimes prefer these shorter documents because they had higher term density for the query terms, even though they contained less complete information. We fixed this through a combination of approaches: first, we adjusted our retrieval scoring to include length normalization factors that prevented overly short documents from having an unfair advantage. Second, we enhanced our context building to prefer longer, more comprehensive chunks when relevance scores were close. Third, we added specific test cases to our golden set that pitted comprehensive older documentation against brief newer updates to ensure we were measuring and optimizing for information completeness, not just superficial relevance scoring. This highlighted how retrieval metrics alone don't capture the full picture of what makes a good answer - we need to evaluate both relevance and information completeness."

---

## Additional Preparation Tips

### For the Technical Interview:
- Be ready to discuss trade-offs in your design decisions (not just what you did, but why you chose it over alternatives)
- Prepare to explain how you measured success and what metrics mattered most
- Have concrete examples of how you tested and validated your assumptions
- Be able to discuss scalability considerations and when you would re-architect
- Know the limitations of your approach and how you would address them

### For the System Design Discussion:
- Think about how your architecture would handle 10x, 100x, 1000x scale increases
- Consider failure modes and how your system would degrade gracefully
- Think about security, privacy, and compliance requirements from the start
- Consider observability, monitoring, and operational concerns
- Be ready to discuss how you would evolve the system over time based on feedback

### For the Behavioral Interview:
- Use the STAR method (Situation, Task, Action, Result) for all stories
- Focus on your specific contributions and what you learned
- Highlight collaboration, communication, and conflict resolution skills
- Show how you balanced technical excellence with practical constraints
- Demonstrate customer focus and business impact thinking

### Final Preparation:
- Review the learn/ documents in this repository - they contain the educational foundation behind all design decisions
- Be able to walk through the complete data flow from document ingestion to answer generation
- Know the specific telecom domain concepts covered in your source materials (refer to CLAUDE.md)
- Prepare questions to ask the interviewer about their real-world RAG or search system challenges
- Remember that as an FDE candidate, you're expected to think about practical deployment considerations, not just theoretical concepts

Good luck with your interview preparation! The RAG Academy project provides an excellent foundation for discussing modern information retrieval systems in enterprise contexts.