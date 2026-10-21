# Foundations of RAG Systems

Welcome to the foundations of Retrieval-Augmented Generation (RAG) systems. This document covers the essential concepts you need to understand to build, evaluate, and deploy effective RAG systems for company knowledge bases.

## Table of Contents
1. [What is an LLM, token, context window, hallucination, temperature](#what-is-an-llm-token-context-window-hallucination-temperature)
2. [Why LLMs alone fail on private company data](#why-llms-alone-fail-on-private-company-data)
3. [Fine-tuning vs prompt stuffing vs RAG (when to use which)](#fine-tuning-vs-prompt-stuffing-vs-rag-when-to-use-which)
4. [Embeddings, vectors, vector space, cosine similarity vs dot product vs euclidean](#embeddings-vectors-vector-space-cosine-similarity-vs-dot-product-vs-euclidean)
5. [The full RAG pipeline end to end](#the-full-rag-pipeline-end-to-end)
6. [Chunk size/overlap, top-k, metadata filtering, grounding, citations](#chunk-sizeoverlap-top-k-metadata-filtering-grounding-citations)
7. [Sparse (BM25) vs dense retrieval, hybrid search, reranking, query rewriting](#sparse-bm25-vs-dense-retrieval-hybrid-search-reranking-query-rewriting)
8. [Evaluation: recall@k, MRR, faithfulness, answer relevance](#evaluation-recallk-mrr-faithfulness-answer-relevance)
9. [Common failure modes and how to diagnose each](#common-failure-modes-and-how-to-diagnose-each)
10. [FDE view: how I'd explain RAG to a non-technical customer, discovery questions to ask a customer before building](#fde-view-how-id-explain-rag-to-a-non-technical-customer-discovery-questions-to-ask-a-customer-before-building)

---

## What is an LLM, token, context window, hallucination, temperature

### Large Language Model (LLM)
An LLM is a deep learning model trained on vast amounts of text data to understand and generate human-like language. Examples include GPT-4, Claude 3, Llama 2.

**Analogy**: Think of an LLM like a student who has read every book in a massive library and can answer questions based on what they've read, but they might misremember details or make things up when uncertain.

### Token
Tokens are the basic units of text that LLMs process. They can be words, parts of words, or even individual characters depending on the tokenization scheme.

**Example**: 
- "Hello world!" might be tokenized as ["Hello", "world", "!"] (3 tokens)
- "unhappiness" might be tokenized as ["un", "happ", "iness"] (3 tokens with subword tokenization)

**Why it matters in production**: Token limits affect how much text you can process at once. Going over the limit causes truncation or errors. Understanding token usage helps optimize costs and performance.

### Context Window
The context window is the maximum amount of text (measured in tokens) that an LLM can process at one time.

**Example**: 
- GPT-3.5-turbo: 4,096 tokens
- Claude 3: 200,000 tokens
- GPT-4-turbo: 128,000 tokens

**Why it matters in production**: The context window determines how much retrieved information you can pass to the LLM. Larger windows allow more context but increase cost and latency.

### Hallucination
Hallucination occurs when an LLM generates information that sounds plausible but is factually incorrect or not present in the input context.

**Example**: 
Asked "What is Acme Corp's vacation policy?" when the documents don't contain this information, the LLM might invent a policy like "Employees receive 25 days of vacation per year" when the actual policy is 20 days.

**Why it matters in production**: In enterprise settings, hallucinations can lead to misinformation, poor decisions, and loss of trust. RAG systems aim to reduce hallucinations by grounding responses in verified source documents.

### Temperature
Temperature is a parameter that controls the randomness of LLM output. Lower values make output more deterministic and focused; higher values increase creativity and variability.

**Range**: Typically 0.0 to 2.0
- 0.0: Most deterministic (always chooses highest probability token)
- 0.7: Balanced creativity and coherence (common default)
- 1.5+: More creative/random output

**Example with prompt "The future of AI is":**
- Temp 0.0: "The future of AI is bright."
- Temp 0.7: "The future of AI is incredibly promising, with advancements in machine learning"
- Temp 1.5: "The future of AI is dancing unicorns wearing top hats while reciting Shakespeare"

**Why it matters in production**: For factual QA systems like RAG, lower temperatures (0.1-0.3) are preferred to ensure consistent, accurate responses. Higher temperatures might be used for creative tasks but increase hallucination risk.

---

## Why LLMs alone fail on private company data

### The Knowledge Cutoff Problem
LLMs are trained on data up to a certain point in time. They don't know about events, documents, or changes that occurred after their training cutoff.

**Example**: An LLM trained in 2023 won't know about Acme Corp's 2024 policy changes, new product launches, or organizational restructuring.

### Limited Context Capacity
Even with large context windows, LLMs can't absorb an entire company's knowledge base at once.

**Reality**: A typical company knowledge base might contain:
- 10,000+ documents
- 50+ MB of text
- Equivalent to hundreds of books

This exceeds even the largest context windows (like Claude 3's 200k tokens ≈ 150k words).

### Static Knowledge
LLMs cannot be easily updated with new information without expensive retraining.

**Problem**: When Acme Corp updates its HR policy, you would need to retrain the entire LLM to incorporate the change - impractical and costly.

### Lack of Source Attribution
LLMs generate responses without indicating which specific documents or facts they used.

**Issue**: In regulated industries or audit scenarios, you need to verify that information came from approved sources, not the LLM's internal knowledge.

### Domain-Specific Terminology
Company-specific acronyms, project names, and internal terminology may be poorly represented in general training data.

**Example**: Acme Corp's internal project "Project Phoenix" might be confused with the mythological bird or other companies' projects.

**Why RAG solves this**: RAG separates knowledge storage (in a retrievable database) from reasoning (in the LLM), allowing:
- Up-to-date knowledge without retraining
- Scalable knowledge bases of any size
- Source attribution and verification
- Better handling of domain-specific terminology

---

## Fine-tuning vs prompt stuffing vs RAG (when to use which)

### Fine-tuning
**What it is**: Continuing to train a pre-trained LLM on your specific dataset to adapt it to your domain.

**Analogy**: Like sending a general practitioner to specialized medical school to become a cardiologist.

**When to use**:
- You have a large, stable dataset (10k+ examples)
- You need the model to internalize specific patterns or styles
- You have computational resources for training
- The knowledge doesn't change frequently

**Pros**:
- Can learn complex patterns and styles
- More efficient at inference time (no retrieval needed)
- Better for tasks requiring deep domain adaptation

**Cons**:
- Expensive and time-consuming to train
- Requires machine learning expertise
- Knowledge becomes "frozen" - hard to update
- Risk of catastrophic forgetting
- Still prone to hallucinations

**Example use case**: Training a model to generate Acme Corp's specific writing style for marketing materials.

### Prompt Stuffing (Context Stuffing)
**What it is**: Putting as much relevant information as possible directly into the LLM's prompt.

**Analogy**: Like trying to cram all your study notes onto a cheat sheet for an exam.

**When to use**:
- Very small knowledge bases (< 50 documents)
- Extremely low latency requirements
- Simple query patterns
- Prototyping or proof-of-concept

**Pros**:
- Simple to implement
- No additional infrastructure needed
- Zero retrieval latency

**Cons**:
- Severely limited by context window size
- Poor scalability
- Higher costs (more tokens processed)
- Difficult to manage and update
- Still prone to hallucinations when overloaded

**Example use case**: A chatbot for a small restaurant menu with 10 items.

### RAG (Retrieval-Augmented Generation)
**What it is**: Combining a retrieval system (to find relevant information) with a generation system (to formulate responses).

**Analogy**: Like giving an open-book exam where you can look up information in a textbook before answering.

**When to use**:
- Medium to large knowledge bases (hundreds to millions of documents)
- Need for up-to-date information
- Requirement for source attribution
- Varying query patterns
- Production enterprise applications

**Pros**:
- Scalable to any knowledge base size
- Up-to-date without retraining
- Source attribution and verification
- Reduced hallucinations
- Cost-effective (only process relevant tokens)
- Flexible retrieval strategies

**Cons**:
- More complex to implement
- Retrieval latency adds to response time
- Requires tuning of retrieval parameters
- Dependencies on both retrieval and generation quality

**Example use case**: Acme Corp's internal employee knowledge base for HR, IT, and product information - exactly our scenario!

**Decision Framework**:
1. **Is your knowledge base small and static?** → Consider prompt stuffing
2. **Do you need the LLM to internalize specific styles/patterns and have training resources?** → Consider fine-tuning  
3. **Otherwise (most enterprise cases)** → Use RAG

---

## Embeddings, vectors, vector space, cosine similarity vs dot product vs euclidean

### Embeddings
Embeddings are numerical representations of text that capture semantic meaning. Similar meanings have similar numerical representations.

**Analogy**: Like mapping words to points on a map where similar concepts are located near each other.

**Example**: 
- "king" and "queen" might have similar embeddings (both royalty)
- "car" and "automobile" would be very close
- "car" and "banana" would be far apart

### Vectors and Vector Space
Embeddings are stored as vectors (lists of numbers) in a high-dimensional vector space.

**Example**: A 384-dimensional embedding is a point in 384-dimensional space:
[0.2, -0.5, 0.8, 0.1, ..., -0.3] (384 numbers total)

### Similarity Metrics
To find similar texts, we measure the distance between their vectors.

#### Cosine Similarity
Measures the cosine of the angle between two vectors. Range: -1 to 1 (often 0 to 1 for positive space).

**Formula**: cos(θ) = (A·B) / (‖A‖ × ‖B‖)
Where A·B is dot product, ‖A‖ is magnitude of A

**Properties**:
- Independent of vector magnitude
- Focuses on direction/orientation
- Bounded: -1 to 1
- Commonly used for text embeddings

**Example calculation by hand**:
Vector A: [1, 1, 0]
Vector B: [1, 0, 1]

Dot product: (1×1) + (1×0) + (0×1) = 1
‖A‖ = √(1² + 1² + 0²) = √2 ≈ 1.414
‖B‖ = √(1² + 0² + 1²) = √2 ≈ 1.414
Cosine similarity = 1 / (1.414 × 1.414) = 1 / 2 = 0.5

#### Dot Product
Simple sum of products of corresponding elements.

**Formula**: A·B = Σ(Ai × Bi)

**Properties**:
- Not normalized (scores can be very large)
- Sensitive to vector magnitude
- Computationally fastest
- Can be negative or positive

**Same example**:
Dot product = 1 (as calculated above)

#### Euclidean Distance
Straight-line distance between two points in space.

**Formula**: √[Σ(Ai - Bi)²]

**Properties**:
- Always non-negative
- Smaller distance = more similar
- Sensitive to both direction and magnitude
- Dissimilarity measure (need to convert to similarity)

**Same example**:
√[(1-1)² + (1-0)² + (0-1)²] = √[0 + 1 + 1] = √2 ≈ 1.414
Similarity = 1 / (1 + distance) = 1 / (1 + 1.414) ≈ 0.414

### When to Use Which

| Metric | Best For | Characteristics |
|--------|----------|-----------------|
| **Cosine Similarity** | Text embeddings, semantic search | Magnitude-independent, standard for NLP |
| **Dot Product** | When magnitude matters, ultra-fast needs | Fastest, but scores not bounded |
| **Euclidean** | Geometric interpretations, clustering | Natural distance, but need conversion |

**Why cosine similarity dominates in RAG**: 
Text embeddings from models like sentence-transformers are typically normalized to unit length, making cosine similarity equivalent to dot product but more interpretable. It focuses purely on semantic direction rather than being influenced by arbitrary magnitude differences.

**Production implication**: Most vector stores and embedding models are optimized for cosine similarity search. Changing metrics usually requires significant rework.

---

## The full RAG pipeline end to end

Let's walk through each step of the RAG pipeline with our Acme Corp example.

### 1. Ingest
**What happens**: Collect and prepare source documents for processing.

**Example**: 
- Gather HR policies (hr_policies.md), IT security (it_security_policy.md), product FAQ (product_faq.md)
- Supported formats: PDF, DOCX, TXT, MD
- Each document gets basic metadata: filename, path, size, timestamp

**Technical details**: 
- Loaders extract text from various formats
- Initial validation ensures files are readable
- Basic metadata collection

### 2. Clean
**What happens**: Normalize and sanitize text content.

**Example**: 
- Remove extra whitespace, HTML artifacts
- Standardize line endings
- Fix common OCR errors (if from scanned documents)
- Remove sensitive information if needed

**Technical details**:
- Whitespace normalization: "Hello    world" → "Hello world"
- HTML entity decoding: "&amp;" → "&"
- Special character handling
- Language detection and filtering

### 3. Chunk
**What happens**: Split cleaned text into smaller pieces for embedding.

**Example**: 
A 10-page HR policy might become 20 chunks of ~300 tokens each
- Chunk 1: Introduction and policy scope
- Chunk 2: Leave accrual rates
- Chunk 3: Approval procedures
- etc.

**Technical details**:
- Fixed-size chunking with overlap (e.g., 500 tokens with 50 overlap)
- Sentence-boundary aware chunking (prefers to break at periods)
- Metadata propagation: each chunk inherits source document metadata
- Chunk indexing for reconstruction

### 4. Embed
**What happens**: Convert text chunks into numerical vectors.

**Example**: 
Each text chunk gets converted to a 384-dimensional vector:
["Leave accrual starts immediately..."] → [0.2, -0.5, 0.8, ..., 0.3]

**Technical details**:
- embedding_model.encode(chunk_text) → vector
- Batch processing for efficiency
- Error handling for problematic text
- Dimension validation

### 5. Store
**What happens**: Save vectors and associated metadata for efficient retrieval.

**Example**: 
Store the 384-dimensional vector along with:
- Original text content
- Source document: hr_policies.md
- Chunk index: 5
- Total chunks in document: 20
- Timestamp: 2024-01-15

**Technical details**:
- Vector storage (numpy arrays, FAISS, or database)
- Metadata index for filtering
- ID mapping for retrieval
- Persistence to disk

### 6. Retrieve
**What happens**: Given a query, find the most relevant stored vectors.

**Example**: 
Query: "How many days of leave do I get?"
System finds chunks containing leave policy information

**Technical details**:
- Query embedding: same model used for documents
- Similarity search (usually cosine similarity)
- Top-k selection (e.g., k=10)
- Initial retrieval set

### 7. Rerank (Optional but Recommended)
**What happens**: Re-score retrieved results for better relevance.

**Example**: 
Initial retrieval might return chunks that mention "leave" in different contexts
Reranking pushes chunks that actually discuss leave policies higher

**Technical details**:
- Cross-encoder models (more accurate but slower)
- Metadata boosting (prefer certain document types)
- Diversity ensures different aspects are covered

### 8. Augment
**What happens**: Combine retrieved context with the original query.

**Example**: 
Original query: "How many days of leave do I get?"
Augmented prompt: 
"Based on the following context, answer the question:
Context: [HR policy chunks about leave]
Question: How many days of leave do I get?
Answer:"

**Technical details**:
- Context formatting (with source citations)
- Prompt template application
- Token count management (truncate if too long)
- Instruction addition (be concise, cite sources, etc.)

### 9. Generate
**What happens**: The LLM produces the final answer based on the augmented prompt.

**Example**: 
LLM processes the augmented prompt and outputs:
"Employees receive 20 days of paid leave per year, accrued monthly at a rate of 1.67 days per month."

**Technical details**:
- LLM inference with appropriate parameters (temperature, max_tokens)
- Output parsing and cleaning
- Post-processing (remove artifacts, ensure format)

### 10. Evaluate
**What happens**: Assess the quality of the entire pipeline.

**Example**: 
Check if the answer:
- Is factually correct (20 days matches policy)
- Comes from the right source (hr_policies.md)
- Is well-formed and complete
- Doesn't contain hallucinations

**Technical details**:
- Retrieval metrics (recall@k, MRR)
- Generation metrics (faithfulness, relevance)
- End-to-end timing
- Human evaluation for nuanced qualities

### The Data Flow
Documents → [Ingest] → Clean → [Chunk] → Embed → [Store] 
                          ← [Retrieve] ← 
Query → [Embed] → [Retrieve] → [Rerank] → [Augment] → [Generate] → Answer

**Why it matters in production**: Understanding each step helps you:
- Identify bottlenecks (is embedding or retrieval slow?)
- Diagnose failures (did retrieval miss relevant info?)
- Optimize performance (batch embedding, approximate search)
- Improve quality (better chunking, reranking, prompt engineering)
- Troubleshoot issues (is the problem in retrieval or generation?)

---

## Chunk size/overlap, top-k, metadata filtering, grounding, citations

### Chunk Size and Overlap

#### Chunk Size
The target size of each text chunk (usually in tokens or characters).

**Trade-offs**:
- **Small chunks** (100-200 tokens):
  - Pros: More precise relevance, less noise
  - Cons: May split related concepts, higher retrieval overhead
  
- **Large chunks** (500-1000 tokens):
  - Pros: Preserves context, fewer retrievals needed
  - Cons: May contain irrelevant information, lower precision

**Recommendation**: 300-500 tokens for most text-based RAG systems

#### Chunk Overlap
Amount of text shared between adjacent chunks.

**Purpose**: Prevents loss of information at chunk boundaries.

**Example**: 
With 100-token chunks and 20-token overlap:
- Chunk 1: tokens 0-100
- Chunk 2: tokens 80-180  
- Chunk 3: tokens 160-260
- etc.

**Recommendation**: 10-20% overlap (e.g., 50 tokens for 500-token chunks)

**Why it matters in production**: 
- Too little overlap: Information loss at boundaries
- Too much overlap: Redundant storage and computation
- Optimal overlap balances completeness and efficiency

### Top-K
Number of retrieved chunks to pass to the generation stage.

**Trade-offs**:
- **Small k** (1-3):
  - Pros: Less context to process, lower cost, faster
  - Cons: May miss relevant information, higher variance
  
- **Large k** (10-20):
  - Pros: Higher recall, more comprehensive context
  - Cons: More tokens to process, higher cost, potential for contradiction/noise

**Recommendation**: Start with k=5, adjust based on evaluation

**Why it matters in production**: 
- Directly affects cost (more tokens = higher LLM API costs)
- impacts latency (more context = longer generation time)
- influences answer quality (balance of completeness vs focus)

### Metadata Filtering
Using document metadata to narrow retrieval scope before similarity search.

**Examples**:
- Filter by document type: only search HR policies for leave questions
- Filter by date: only use documents from last 6 months
- Filter by department: only search IT documents for technical questions
- Filter by version: exclude outdated policy versions

**Implementation**:
- Pre-filter candidate vectors by metadata before similarity search
- Or post-filter results (less efficient but simpler)

**Why it matters in production**: 
- Dramatically improves retrieval precision
- Reduces computation and latency
- Enables specialized handling of different document types
- Critical for compliance (e.g., only use approved policy versions)

### Grounding
Ensuring the LLM's response is based solely on the provided context.

**Techniques**:
- Prompt engineering: Explicitly instruct to use only context
- Post-processing: Verify claims can be traced to source
- Confidence scoring: Measure how much answer derives from context
- Retrieval augmentation: Only allow information from retrieved chunks

**Why it matters in production**: 
- Primary defense against hallucinations
- Enables auditability and compliance
- Builds user trust in the system
- Essential for factual domains like HR, legal, medical

### Citations
Explicitly referencing which source documents support each part of the answer.

**Formats**:
- Inline: "Employees get 20 days leave [hr_policies.md]"
- Bracketed: "Employees get 20 days leave (Source: HR Policies)"
- Footnote style: Numbered references at end
- Block citations: Group sources by answer section

**Implementation**:
- Track which chunks contribute to answer generation
- Post-process to add citations based on retrieval scores
- Or use retrieval-augmented generation architectures that naturally produce citations

**Why it matters in production**: 
- Enables fact-checking and verification
- Reduces hallucination impact (even if wrong, source is clear)
- Supports follow-up questions ("Can you show me where that's stated?")
- Meets transparency requirements in regulated industries

---

## Sparse (BM25) vs dense retrieval, hybrid search, reranking, query rewriting

### Dense Retrieval
**What it is**: Using neural embeddings to find semantically similar vectors.

**How it works**: 
- Embed query with same model as documents
- Find vectors with highest cosine similarity
- Based on semantic meaning, not exact keywords

**Example**: 
Query: "time off for vacation" 
Matches: documents about "annual leave", "paid time off", "vacation days"
Even if none contain the exact words "time off for vacation"

**Pros**:
- Captures semantic similarity
- Handles synonyms and related concepts
- Works well with natural language queries
- Robust to minor wording differences

**Cons**:
- Computationally expensive (embedding + search)
- Requires good embedding model
- May miss exact keyword matches
- Harder to debug and interpret

**Best for**: Semantic search, conceptual questions, natural language queries

### Sparse Retrieval (BM25)
**What it is**: Traditional information retrieval using term frequency-inverse document frequency.

**How it works**: 
- Treat documents as bags of words
- Score based on term frequency in document vs corpus
- Better score for rare but frequent terms in document
- Okapi BM25 is the standard implementation

**Example**: 
Query: "vacation leave policy" 
High score for documents containing both "vacation" and "leave" frequently
Lower score for documents mentioning these terms only once

**Pros**:
- Fast and efficient (no embedding needed)
- Exact keyword matching
- Interpretable scoring
- Works well with technical terminology
- No training data needed

**Cons**:
- Poor semantic understanding
- Fails with synonyms ("leave" vs "time off")
- Sensitive to exact wording
- Doesn't handle morphology well ("leave" vs "leaves")

**Best for**: Keyword search, technical documentation, exact phrase matching

### Hybrid Retrieval
**What it is**: Combining dense and sparse retrieval to get benefits of both.

**Approaches**:
1. **Parallel**: Run both separately, merge and re-score results
2. **Weighted**: Combine scores: final = α×dense + (1-α)×sparse
3. **Cascaded**: Use one to filter candidates for the other
4. **Reciprocal Rank Fusion**: Combine rankings from multiple methods

**Example implementation**:
- Get top 20 from dense retrieval
- Get top 20 from sparse retrieval  
- Combine scores: 0.7×dense_score + 0.3×sparse_score
- Take top 5 from combined results

**Why it matters in production**: 
- Best of both worlds: semantics + exact matching
- More robust across query types
- Better handling of vocabulary mismatch
- Industry best practice for production RAG systems

### Reranking
**What it is**: Re-scoring retrieved results with a more accurate (but slower) model.

**How it works**: 
- Initial retrieval with fast method (dense or sparse)
- Take top N (e.g., 20) candidates
- Apply more accurate scorer to query-document pairs
- Re-rank based on new scores

**Common approaches**:
- **Cross-encoders**: Process query-document pair together (more accurate than bi-encoders)
- **Metadata boosting**: Increase score based on document properties
- **Diversity reranking**: Promote variety in results
- **Learning-to-rank**: ML model trained on relevance judgments

**Example**: 
Initial dense retrieval returns chunks mentioning "policy" 
Cross-encoder reranker identifies which actually discuss "leave policy" specifically

**Why it matters in production**: 
- Large quality improvement for small latency cost
- Compensates for weaknesses in initial retrieval
- Essential for high-precision applications
- Often provides 10-20% relative improvement in metrics

### Query Rewriting
**What it is**: Transforming the user's query to improve retrieval performance.

**Techniques**:
- **Expansion**: Add synonyms or related terms
  - Original: "leave policy" 
  - Expanded: "leave policy OR vacation time OR paid time off"
- **Decomposition**: Break complex questions into simpler ones
- **Hypothetical Document Embedding (HyDE)**: Generate a fake answer, embed that
- **Query classification**: Route to specialized retrievers based on type

**Example**: 
User: "How many days can I take off for sickness?"
Rewritten: "sick leave policy days permitted absence illness"

**Why it matters in production**: 
- Compensates for user's poor query formulation
- Bridges vocabulary gap between users and documents
- Can significantly improve retrieval recall
- Particularly valuable in enterprise settings with domain-specific language

**Recommendation**: Start with simple expansion based on known synonyms, then consider more advanced techniques based on error analysis.

---

## Evaluation: recall@k, MRR, faithfulness, answer relevance

### Retrieval Metrics

#### Recall@K
**What it measures**: Of all relevant documents, what fraction did we retrieve in the top K?

**Formula**: Recall@K = (Number of relevant docs in top K) / (Total number of relevant docs)

**Example**: 
- 5 relevant documents exist for a query
- We retrieved 3 of them in our top 5
- Recall@5 = 3/5 = 0.6

**Why it matters**: Measures whether your retrieval system finds enough of the right information. Low recall means you're missing important context.

**Target**: > 0.8 for good performance

#### Mean Reciprocal Rank (MRR)
**What it measures**: How high ranked is the first relevant document?

**Formula**: For each query, find rank of first relevant doc, compute 1/rank, then average

**Example**: 
- Query 1: First relevant at position 2 → 1/2 = 0.5
- Query 2: First relevant at position 1 → 1/1 = 1.0  
- Query 3: First relevant at position 3 → 1/3 ≈ 0.33
- MRR = (0.5 + 1.0 + 0.33) / 3 = 0.61

**Why it matters**: Measures whether the most important information comes first. Critical when users expect quick answers.

**Target**: > 0.5 for good performance

#### nDCG@K (Normalized Discounted Cumulative Gain)
**What it measures**: Quality of ranking, giving higher weight to top positions.

**Why it matters**: Rewards getting relevant documents early in the list, which is crucial for user experience.

### Generation Metrics

#### Faithfulness
**What it measures**: Whether the answer is supported by the retrieved context.

**Approaches**:
- **Token overlap**: What percentage of answer tokens appear in context?
- **Entailment checking**: Does context entail the answer? (NLI models)
- **Question answering**: Can you answer the question using only the context?
- **Hallucination detection**: Specific detectors for fabricated content

**Simple implementation**: 
Check if key facts in answer can be found in source documents

**Why it matters**: Primary defense against harmful hallucinations. In enterprise RAG, this is often the most important metric.

**Target**: > 0.9 for production systems

#### Answer Relevance
**What it measures**: How well the answer addresses the original question.

**Approaches**:
- **Semantic similarity**: Embed answer and question, compute similarity
- **Keyword overlap**: Does answer contain question's key concepts?
- **Human judgment**: Does it actually answer what was asked?
- **Specificity**: Is it appropriately detailed for the question type?

**Why it matters**: You can have a faithful answer that doesn't answer the question (e.g., "I don't know" is faithful but not helpful).

**Target**: > 0.7 for useful responses

### Putting It Together: Evaluation Workflow

1. **Create golden set**: Manual curation of questions with known good answers
2. **Run pipeline**: Process each question through your RAG system
3. **Measure retrieval**: 
   - Recall@K: Did we get enough relevant chunks?
   - MRR: Did the most important chunks come first?
4. **Measure generation**:
   - Faithfulness: Is answer supported by context?
   - Relevance: Does answer actually address the question?
5. **Analyze failures**: 
   - Low recall? → Improve retrieval (better embeddings, chunking, hybrid search)
   - Low faithfulness? → Improve grounding (better prompts, post-processing)
   - Low relevance? → Improve query understanding or answer generation

**Production practice**: 
- Weekly automatic evaluation against golden set
- Alert when metrics drop below thresholds
- Root cause analysis for failures
- Continuous improvement loop

---

## Common failure modes and how to diagnose each

### 1. Missing Information (Low Recall)
**Symptoms**: 
- Answers are incomplete or vague
- System says "I don't have enough information" when it should
- Specific facts known to exist are not mentioned

**Diagnosis**:
- Check retrieval results: Are relevant chunks missing?
- Verify with manual search: Can you find the information in source?
- Measure recall@K on golden set

**Common causes**:
- Chunking too aggressive (split related info)
- Poor embedding quality for domain
- Inadequate top-K setting
- Metadata filtering too restrictive
- Query preprocessing issues

**Solutions**:
- Increase chunk size or decrease overlap
- Try different embedding models
- Increase top-K
- Review metadata filtering rules
- Add query expansion or rewriting
- Consider hybrid or sparse retrieval supplement

### 2. Hallucinations (Low Faithfulness)
**Symptoms**: 
- Answers contain facts not in source documents
- Confident statements that are incorrect
- Information contradicts known sources
- Vague or overly broad statements that could be true anywhere

**Diagnosis**:
- Fact-check answers against source documents
- Identify which specific claims are hallucinated
- Check if hallucinated concepts appear in retrieval results
- Measure faithfulness metric

**Common causes**:
- Overly generous generation parameters (high temperature)
- Insufficient grounding in prompts
- Retrieval retrieved irrelevant but plausible chunks
- LLM defaulting to training data when uncertain
- Prompt doesn't enforce context-only usage

**Solutions**:
- Lower temperature (0.1-0.3)
- Strengthen grounding instructions in prompts
- Improve retrieval precision (reranking, hybrid search)
- Add post-processing verification step
- Consider constrained generation techniques
- Improve chunking to reduce irrelevant context

### 3. Irrelevant Answers (Low Relevance)
**Symptoms**: 
- Answers don't address the actual question asked
- Go off on tangents or discuss related but wrong topics
- Provide information that's technically correct but not responsive
- Answer a different question than was asked

**Diagnosis**:
- Manually review question-answer pairs
- Check if answer addresses question's intent
- Measure relevance metrics
- Look for patterns in wrong answers

**Common causes**:
- Poor query understanding or preprocessing
- Retrieval focused on wrong aspects of question
- Generation not following instructions
- Context contains relevant but misprioritized information
- Prompt doesn't sufficiently constrain to question

**Solutions**:
- Improve query preprocessing (expansion, rewrite)
- Adjust retrieval weighting (focus on question intent)
- Strengthen answer generation prompts
- Improve context selection and prioritization
- Consider query classification and routing

### 4. Inconsistent Answers
**Symptoms**: 
- Same question yields different answers at different times
- Similar questions get wildly different answers
- Answers contradict each other for related questions
- Non-deterministic behavior

**Diagnosis**:
- Run same question multiple times
- Check for sources of randomness
- Look for temperature or sampling effects
- Check if retrieval order varies

**Common causes**:
- Non-deterministic retrieval (approximate nearest neighbors)
- Random sampling in generation (high temperature)
- Retrieval instability (similar scores)
- Race conditions or caching issues
- Non-deterministic chunking or embedding

**Solutions**:
- Set deterministic seeds where possible
- Lower temperature to reduce randomness
- Use exact nearest neighbor search for retrieval
- Ensure consistent preprocessing
- Consider caching strategies carefully

### 5. Slow Performance
**Symptoms**: 
- Response times exceed acceptable thresholds
- Users notice delay
- System fails to meet SLAs
- Throughput too low for expected load

**Diagnosis**:
- Profile each stage: ingest, embed, store, retrieve, rerank, generate
- Identify bottlenecks
- Check resource utilization (CPU, memory, GPU)
- Measure with realistic load patterns

**Common causes**:
- Embedding computation (especially without GPU)
- Inefficient vector search (brute force vs approximate)
- Overly large top-K or context window
- Inefficient reranking (cross-encoders are slow)
- Generation model too large or slow

**Solutions**:
- Add GPU acceleration for embedding
- Use approximate nearest neighbor (FAISS, HNSW)
- Optimize top-K and context window sizes
- Consider distillation or smaller reranker models
- Optimize generation (caching, batching, smaller models)
- Consider hybrid approach: fast first pass, rerank only if needed

### 6. Stale Information
**Symptoms**: 
- Answers reflect outdated information
- Recent policy changes not reflected
- System contradicts current official documents
- Users report getting old information

**Diagnosis**:
- Check document ingestion timestamps
- Verify when source documents were last updated
- Trace specific outdated facts to source documents
- Measure freshness of knowledge base

**Common causes**:
- Ingestion pipeline not running regularly
- Document update detection not working
- Versioning issues (old vs new documents)
- Manual processes not automated
- Storage not reflecting latest state

**Solutions**:
- Automate ingestion pipeline (scheduled or trigger-based)
- Implement document change detection
- Use proper versioning and retirement policies
- Consider real-time updates for critical documents
- Add freshness metrics to monitoring

### 7. Bias or Fairness Issues
**Symptoms**: 
- Systematic differences in quality for different user groups
- Certain types of questions consistently worse
- Answers reflect unintended biases from training data
- Disparate performance across document types or topics

**Diagnosis**: 
- Stratify evaluation by user demographics or question types
- Check for patterns in errors
- Compare performance across different document categories
- Review source materials for bias

**Common causes**:
- Biased training data in LLMs or embedding models
- Skewed representation in knowledge base
- Retrieval favoring certain document types
- Generation reflecting societal biases
- Lack of diversity in test set

**Solutions**:
- Audit and curate knowledge base for balance
- Use debiasing techniques where available
- Stratified sampling in evaluation and testing
- Post-processing to detect and mitigate bias
- Diversity-aware retrieval and reranking
- Regular fairness audits

**General diagnostic approach**:
1. **Reproduce**: Can you consistently observe the issue?
2. **Isolate**: Which stage is responsible? (Retrieval vs Generation)
3. **Measure**: What metrics show the problem?
4. **Hypothesize**: What's the root cause?
5. **Test**: Does your proposed fix resolve it?
6. **Monitor**: Does it stay fixed over time?

**Production monitoring**: 
- Track key metrics over time
- Set up alerts for degradation
- Sample answers for manual review
- Maintain error database for pattern analysis
- Regular golden set re-evaluation

---

## FDE view: how I'd explain RAG to a non-technical customer, discovery questions to ask a customer before building

### How I'd Explain RAG to a Non-Technical Customer

**The Librarian Analogy**:
"Think of our RAG system like having a brilliant research assistant who's also an expert librarian. 

When you ask a question, here's what happens:
1. **The librarian** (our retrieval system) quickly searches through all company documents to find the most relevant information - like finding the right books in a massive library.
2. **The research assistant** (our language model) then reads those specific documents and formulates a clear, accurate answer based only on what they found.
3. **Importantly**: The assistant doesn't make things up or rely on general knowledge - they strictly use what the librarian found in the actual company documents.

This means you get answers that are:
- ✅ **Accurate**: Based on your actual company documents, not guesses
- ✅ **Up-to-date**: Reflects the latest information as soon as documents are updated
- ✅ **Transparent**: We can show you exactly which documents the answer came from
- ✅ **Secure**: No company data leaves your environment - everything stays internal
- ✅ **Trustworthy**: You can verify the information against source documents"

**The Open-Book Exam Analogy**:
"Remember taking open-book exams in school? You weren't expected to memorize everything - you just needed to know how to find and use the right information.

Our RAG system works the same way:
- Instead of memorizing the entire company knowledge base (which is impossible), 
- It learns how to quickly find the right information when needed
- Then it uses that information to answer your question accurately
- Just like you would in an open-book exam, but automated and instant"

### Discovery Questions to Ask a Customer Before Building

#### About the Knowledge Base
1. **What types of documents make up your knowledge base?**
   - Formats: PDF, Word, HTML, plain text, scanned images?
   - Sources: Policies, procedures, FAQs, technical docs, meeting notes?

2. **How large is your knowledge base approximately?**
   - Number of documents?
   - Total size in MB/GB?
   - Growth rate (new documents per month)?

3. **How frequently do documents change?**
   - How often are policies/procedures updated?
   - Are there version control requirements?
   - Do you need to track document history?

4. **What languages are used in the documents?**
   - Primarily English? Other languages?
   - Need for multilingual support?

5. **Are there sensitive or restricted documents?**
   - Need for access controls or filtering?
   - PII or confidential information handling requirements?

#### About the Use Cases
1. **What types of questions will users ask?**
   - Factual: "What is the vacation policy?"
   - Procedural: "How do I request time off?"
   - Troubleshooting: "Why won't my laptop connect to WiFi?"
   - Comparative: "What's the difference between Plan A and Plan B?"
   - Analytical: "What are the trends in Q3 sales?"

2. **Who will be using the system?**
   - Employees? Customers? Partners? Regulators?
   - Technical vs non-technical users?
   - Frequency of use (daily, occasional)?

3. **What are the most critical questions that must be answered correctly?**
   - Compliance-related questions
   - Safety procedures
   - Customer commitments
   - Financial information

4. **What would constitute a bad or harmful answer?**
   - Wrong medical advice in healthcare context
   - Incorrect legal interpretation
   - Missing safety procedure steps
   - Outdated financial information

#### About Technical Requirements
1. **Where should the system be deployed?**
   - On-premises only? Cloud acceptable? Hybrid?
   - Specific security or compliance requirements (HIPAA, SOC2, GDPR)?

2. **What are the performance requirements?**
   - Maximum acceptable response time?
   - Expected concurrent users?
   - Throughput requirements (queries per hour/minute)?

3. **What integration is needed?**
   - Standalone application?
   - Integration with existing intranet/portal?
   - API for other systems to use?
   - Widget for internal tools?

4. **What are the success metrics?**
   - What makes this project successful in your view?
   - How will you measure ROI or effectiveness?
   - What specific problems should this solve?

#### About Resources and Constraints
1. **What technical resources are available?**
   - In-house expertise for maintenance?
   - Budget for licensing or cloud services?
   - Hardware constraints or preferences?

2. **What is the timeline for implementation?**
   - Hard deadlines or milestones?
   - Phased approach preferred?
   - Pilot group or full rollout?

3. **What existing systems should we consider?**
   - Current search or knowledge base solutions?
   - Document management systems?
   - Authentication systems (SSO, LDAP, etc.)?

4. **What are the deal-breakers or constraints?**
   - Must work offline?
   - Cannot use certain technologies?
   - Specific vendors to avoid or prefer?
   - Regulatory constraints on data handling?

**Why these questions matter**: 
Understanding these aspects ensures we build a system that:
- Actually solves your problems (not just technically impressive)
- Fits within your organizational constraints
- Meets your users' real needs
- Complies with your policies and regulations
- Delivers measurable value
- Can be maintained and evolved over time

**Final thought**: The best RAG systems aren't just about the technology - they're about understanding how people work with information and making that process better, faster, and more reliable.