# Advanced Retrieval Techniques in RAG Systems

This document covers advanced retrieval techniques that improve the quality and relevance of retrieved information in RAG systems, moving beyond basic dense retrieval to production-grade solutions.

## Table of Contents
1. [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
2. [Bi-encoder vs Cross-encoder Models](#bi-encoder-vs-cross-encoder-models)
3. [Query Rewriting Techniques](#query-rewriting-techniques)
4. [Conversation Memory](#conversation-memory)
5. [Decision Guide for Troubleshooting Search Results](#decision-guide-for-troubleshooting-search-results)
6. [When Each Technique Helps and Hurts](#when-each-technique-helps-and-hurts)

---

## Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion is a technique for combining multiple ranked lists of documents into a single ranked list. It's particularly useful for hybrid retrieval systems that combine different retrieval methods (like BM25 and dense vector retrieval).

### The RRF Formula

For each document appearing in any of the ranked lists, its RRF score is calculated as:

```
RRF_score = Σ [1 / (k + rank_i)]
```

Where:
- `k` is a ranking constant (typically 60)
- `rank_i` is the position of the document in the i-th ranked list
- The sum is over all ranked lists where the document appears

### Worked Example

Let's say we have two retrieval methods (BM25 and Dense) that return these ranked lists for a query:

**BM25 Rankings:**
1. Document A (score: 0.8)
2. Document C (score: 0.6)
3. Document B (score: 0.5)
4. Document D (score: 0.4)

**Dense Rankings:**
1. Document B (score: 0.9)
2. Document A (score: 0.7)
3. Document E (score: 0.5)
4. Document F (score: 0.3)

Using RRF with k=60:

**Document A:**
- BM25 rank: 1 → 1/(60+1) = 0.0164
- Dense rank: 2 → 1/(60+2) = 0.0159
- **RRF Score: 0.0323**

**Document B:**
- BM25 rank: 3 → 1/(60+3) = 0.0156
- Dense rank: 1 → 1/(60+1) = 0.0164
- **RRF Score: 0.0320**

**Document C:**
- BM25 rank: 2 → 1/(60+2) = 0.0159
- Dense rank: Not in top 4 → 0
- **RRF Score: 0.0159**

**Document D:**
- BM25 rank: 4 → 1/(60+4) = 0.0152
- Dense rank: Not in top 4 → 0
- **RRF Score: 0.0152**

**Document E:**
- BM25 rank: Not in top 4 → 0
- Dense rank: 3 → 1/(60+3) = 0.0156
- **RRF Score: 0.0156**

**Document F:**
- BM25 rank: Not in top 4 → 0
- Dense rank: 4 → 1/(60+4) = 0.0152
- **RRF Score: 0.0152**

**Final RRF Rankings:**
1. Document A (0.0323)
2. Document B (0.0320)
3. Document C (0.0159)
4. Document E (0.0156)
5. Document D (0.0152)
6. Document F (0.0152)

### Benefits of RRF
- **Rank-normalization**: Different scoring scales between methods don't matter
- **Robustness**: Performs well even when individual methods have weaknesses
- **Simplicity**: No need to learn weighting parameters
- **Effectiveness**: Often outperforms simple linear combination of scores

---

## Bi-encoder vs Cross-encoder Models

### Bi-encoders (for Initial Retrieval)

Bi-encoders encode queries and documents independently using the same encoder model, then compute similarity between the resulting vectors.

**How it works:**
1. Query → Encoder → Query Vector
2. Document → Encoder → Document Vector  
3. Similarity = cos_sim(Query Vector, Document Vector)

**Advantages:**
- **Extremely fast** at query time (document vectors pre-computed)
- **Scales to millions** of documents (only vector comparison needed)
- **Efficient storage** (store only document vectors)

**Disadvantages:**
- **Lower accuracy** (no interaction between query and document during encoding)
- **Limited expressivity** (independent encoding loses fine-grained signals)

### Cross-encoders (for Reranking)

Cross-encoders concatenate the query and document and process them together through the encoder layers.

**How it works:**
1. [Query] [Document] → Encoder → Relevance Score

**Advantages:**
- **Higher accuracy** (full interaction between query and document)
- **Better at handling** complex linguistic phenomena
- **State-of-the-art** performance on relevance tasks

**Disadvantages:**
- **Computationally expensive** (must encode every query-document pair)
- **Slow** at query time (can't pre-compute document representations)
- **Doesn't scale** to large corpora (typically used to rerank top 10-100 results)

### When to Use Each

**Use Bi-encoders for:**
- Initial retrieval from large document collections
- Applications requiring low latency (<100ms)
- Systems with limited computational resources
- First-stage retrieval in a pipeline

**Use Cross-encoders for:**
- Reranking top results from bi-encoder retrieval
- Applications where accuracy is more important than speed
- Reranking stages with small candidate sets (top 20-100)
- Final ranking before presentation to user

---

## Query Rewriting Techniques

Query rewriting improves retrieval by transforming user queries into forms that better match document content.

### LLM-based Query Rewriting

Uses a language model to rewrite ambiguous or vague queries into more specific, retrieval-friendly forms.

**Example:**
- Original: "What about the new antenna specs?"
- Rewritten: "What are the specifications for the new 5G NR antenna parameters including frequency range, gain, and beamforming capabilities?"

**Benefits:**
- Resolves ambiguity and pronoun references
- Adds domain-specific terminology
- Expands abbreviations and acronyms
- Converts questions to declarative statement format

### Multi-query Expansion

Generates multiple variations of the original query to capture different aspects of the information need.

**Example:**
- Original: "5G handover procedure"
- Variations:
  1. "5G handover procedure steps and signaling"
  2. "What is the handover process in 5G networks?"
  3. "5G NR handover types and triggers"
  4. "Handover failure causes in 5G systems"
  5. "Measurement and optimization of 5G handover"

**Benefits:**
- Captures different interpretations of the query
- Increases recall by covering related concepts
- Reduces sensitivity to exact query phrasing
- Particularly good for exploratory search

### HyDE (Hypothetical Document Embedding)

Generates a hypothetical document that answers the query, then uses that document's embedding for retrieval.

**Process:**
1. Generate hypothetical answer to query using LLM
2. Embed the hypothetical answer
3. Search for real documents similar to the hypothetical answer

**Benefits:**
- Bridges the gap between query vocabulary and document vocabulary
- Particularly effective for technical domains with specialized terminology
- Helps when queries use different terms than documents
- Can capture implicit knowledge needed to formulate the query

**Drawbacks:**
- Requires an LLM for hypothetical document generation
- Quality depends on LLM's ability to generate accurate hypothetical answers
- Adds latency to the retrieval process

---

## Conversation Memory

Conversation memory enables multi-turn interactions by condensing follow-up questions into standalone queries that can be processed independently.

### The Problem with Follow-up Questions

In a conversation, users often ask follow-up questions that rely on context:
- Turn 1: "What are the 5G NR timing requirements?"
- Turn 2: "How about for FR2?" (implies: "What are the 5G NR timing requirements for FR2?")
- Turn 3: "And what are the penalties for violation?" (implies: "What are the penalties for violating 5G NR timing requirements for FR2?")

Without context, these follow-up questions are often too vague for effective retrieval.

### Solution: Question Condensation

Process follow-up questions to resolve pronouns, ellipsis, and contextual references using conversation history.

**Example Condensation:**
- Current: "How about for FR2?"
- With context from Turn 1: "What are the 5G NR timing requirements for FR2?"
- Current: "And what are the penalties for violation?"
- With context: "What are the penalties for violating 5G NR timing requirements for FR2?"

### Implementation Approaches

1. **Rule-based**: Use linguistic patterns to identify and resolve references
2. **LLM-based**: Use language model to rewrite question with full context
3. **Hybrid**: Combine rule-based preprocessing with LLM refinement

### Benefits
- Enables natural, conversational interaction
- Reduces user effort in repeating context
- Improves retrieval accuracy for follow-up questions
- Maintains conversational flow

### Considerations
- Privacy: Conversation history may contain sensitive information
- Drift: Long conversations may accumulate irrelevant context
- Complexity: Requires careful handling of coreference resolution
- Latency: Adds processing time to each turn

---

## Decision Guide for Troubleshooting Search Results

When a customer says "search results are bad," follow this systematic approach to identify and fix the issue:

### Step 1: Verify Basic Functionality
- [ ] Is the system returning any results at all?
- [ ] Are results clearly unrelated to the query topic?
- [ ] Are there obvious errors in the retrieval pipeline?

### Step 2: Check Query Processing
- [ ] Is the query being preprocessed correctly? (lowercasing, tokenization)
- [ ] Are domain-specific terms being handled properly? (acronyms expanded)
- [ ] Is the query too vague or ambiguous?
- [ ] Would query rewriting improve this query?

### Step 3: Examine Retrieval Method
- [ ] For dense retrieval: Are embedding models appropriate for the domain?
- [ ] For sparse retrieval: Is BM25 properly tuned? (k1, b parameters)
- [ ] For hybrid retrieval: Are weights balanced correctly?
- [ ] Is RRF being applied correctly if used?
- [ ] Are sufficient candidates being retrieved for reranking?

### Step 4: Evaluate Document Representation
- [ ] Are documents chunked appropriately? (not too small/large)
- [ ] Does chunking preserve important context?
- [ ] Are metadata fields being indexed/searchable?
- [ ] Is document preprocessing removing important information?

### Step 5: Check Reranking (if applicable)
- [ ] Is the reranker model appropriate for the task?
- [ ] Is the reranker being applied to enough candidates?
- [ ] Are query-document pairs formatted correctly for the cross-encoder?
- [ ] Is the reranker returning scores in the expected range?

### Step 6: Assess Result Presentation
- [ ] Are enough results being returned to the user?
- [ ] Are results properly diversified (not all similar)?
- [ ] Is the ranking aligns with user expectations?
- [ ] Are sources being cited correctly?

### Step 7: Consider Domain-specific Factors
- [ ] Are technical documents using specialized notation?
- [ ] Are tables and figures properly processed?
- [ ] Are version differences handled correctly?
- [ ] Are language-specific characteristics accounted for?

### Quick Wins to Try First
1. **Increase top-k**: Retrieve more candidates before reranking
2. **Adjust hybrid weights**: Favor the method that usually works better for your domain
3. **Enable query rewriting**: Especially LLM-based or HyDE for technical domains
4. **Check similarity threshold**: Ensure it's not filtering out relevant results
5. **Verify metadata filtering**: Ensure it's not too restrictive

### Advanced Diagnostics
- **Error analysis**: Log queries where top result is irrelevant and examine why
- **Ablation testing**: Disable components one-by-one to identify impact
- **Embedding visualization**: Check if query and relevant documents are close in vector space
- **BM25 term analysis**: Check if query terms match document terms appropriately

---

## When Each Technique Helps and Hurts

### Reciprocal Rank Fusion (RRF)

**Helps when:**
- Combining heterogeneous retrieval methods (BM25 + dense)
- Individual methods have complementary strengths
- Score distributions vary significantly between methods
- Need robustness against failures in individual methods

**Hurts when:**
- One method is clearly superior to others
- Computational resources are extremely limited (minimal overhead)
- Methods produce very similar rankings (little to gain from fusion)
- Real-time constraints are extremely tight (<10ms latency required)

### Bi-encoder vs Cross-encoder

**Bi-encoder helps:**
- Large-scale retrieval (millions of documents)
- Low-latency requirements (<50ms)
- Resource-constrained environments
- First-stage retrieval in pipeline

**Bi-encoder hurts:**
- When nuanced query-document understanding is needed
- For entities or technical terms requiring precise matching
- When query contains complex linguistic constructs
- High-precision requirements where recall is already good

**Cross-encoder helps:**
- Reranking top candidates (top 20-100)
- When accuracy is paramount
- Complex semantic matching requirements
- Domains with specialized terminology

**Cross-encoder hurts:**
- Large candidate sets (computational cost)
- Real-time applications with tight latency budgets
- When resources are limited (GPU memory for large models)
- First-stage retrieval from large collections

### Query Rewriting

**Helps when:**
- Users pose vague or ambiguous questions
- Domain has significant terminology gaps between queries and documents
- Conversational context needs to be resolved
- Users frequently use acronyms or abbreviations
- Exploratory search where intent is not well-formed

**Hurts when:**
- Queries are already specific and well-formed
- Computational resources are very limited
- Rewriting introduces noise or incorrect concepts
- Latency is critical and LLM is slow
- Domain has stable, consistent terminology

### Conversation Memory

**Helps when:**
- Users ask natural follow-up questions
- Domain knowledge builds across turns
- Users employ pronouns and ellipsis
- Exploratory or learning-oriented conversations
- Need to reduce user effort in repeating context

**Hurts when:**
- Each query is independent (no conversational context)
- Privacy concerns prohibit storing history
- Conversations are very short (single-turn predominant)
- Memory constraints are severe
- Risk of context contamination or drift
- Real-time systems where added latency is unacceptable

### Overall Recommendation for Telecom RAG

For a telecom-focused RAG system serving Field Development Engineers:

1. **Start with**: Dense retrieval (bi-encoder) + light query rewriting
2. **Add**: RRF hybrid retrieval if BM25 shows value for keyword matching
3. **Include**: Cross-encoder reranking for top 20→5
4. **Consider**: Conversation memory for interactive use cases
5. **Monitor**: Latency and accuracy trade-offs continuously
6. **Adjust**: Based on actual query patterns and user feedback

The key is to measure the impact of each technique on your specific workload and user satisfaction rather than assuming universal benefits.