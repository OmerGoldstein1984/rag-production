# Evaluation in RAG Systems

This document covers the evaluation of RAG systems, focusing on metrics, methodologies, and best practices for assessing both retrieval and generation components. Proper evaluation is essential for understanding system performance and guiding improvements.

## Table of Contents
1. [Retrieval Metrics](#retrieval-metrics)
2. [Generation Metrics](#generation-metrics)
3. [LLM-as-Judge: Principles and Pitfalls](#llm-as-judge-principles-and-pitfalls)
4. [Building a Golden Set](#building-a-golden-set)
5. [Presenting Results to Stakeholders](#presenting-results-to-stakeholders)
6. [Ablation Studies](#ablation-studies)

---

## Retrieval Metrics

Retrieval evaluation assesses how well the system finds relevant documents for a given query. Unlike traditional IR evaluation, RAG retrieval evaluation must consider that the end goal is answer generation, not just document retrieval.

### Hit Rate@K

**Definition**: The proportion of queries for which at least one relevant document appears in the top K retrieved results.

**Formula**:
```
Hit Rate@K = (Number of queries with at least one relevant doc in top K) / (Total number of queries)
```

**Worked Example**:
Suppose we have 5 queries with the following retrieval results (where ✓ indicates a relevant document):

- Query 1: [✓, ✗, ✗, ✗, ✗] → Hit in top 5: YES
- Query 2: [✗, ✗, ✓, ✗, ✗] → Hit in top 5: YES  
- Query 3: [✗, ✗, ✗, ✗, ✗] → Hit in top 5: NO
- Query 4: [✗, ✓, ✗, ✗, ✗] → Hit in top 5: YES
- Query 5: [✗, ✗, ✗, ✗, ✓] → Hit in top 5: YES

Hit Rate@5 = 4/5 = 0.80

**When to Use**: 
- When you care about whether the system can find *any* relevant information
- Binary success metric (either you got something relevant or you didn't)
- Particularly important for RAG because missing all relevant information guarantees a poor answer

### Recall@K

**Definition**: The proportion of relevant documents that are retrieved in the top K results.

**Formula**:
```
Recall@K = (Number of relevant documents retrieved in top K) / (Total number of relevant documents)
```

**Worked Example**:
Suppose for a single query there are 3 relevant documents in the corpus, and our retrieval results are:

- Retrieved documents: [DocA(✓), DocB(✗), DocC(✓), DocD(✗), DocE(✗)]
- Where ✓ indicates relevance

Recall@5 = 2/3 = 0.67

**When to Use**:
- When you want to measure how comprehensively the system finds relevant information
- Important for complex questions requiring multiple pieces of information
- Higher recall gives the generation component more context to work with

### Mean Reciprocal Rank (MRR)

**Definition**: The average of the reciprocal ranks of the first relevant document for each query.

**Formula**:
```
MRR = (1/Q) * Σ(1/rank_i)
```
Where rank_i is the position of the first relevant document for query i.

**Worked Example**:
Suppose we have 3 queries with the following retrieval results:

- Query 1: [✗, ✗, ✓, ✗, ✓] → First relevant at rank 3 → 1/3 = 0.33
- Query 2: [✓, ✗, ✗, ✗, ✗] → First relevant at rank 1 → 1/1 = 1.00
- Query 3: [✗, ✗, ✗, ✗, ✗] → No relevant documents → 0/1 = 0.00

MRR = (0.33 + 1.00 + 0.00) / 3 = 0.44

**When to Use**:
- When you care about how quickly the system finds relevant information
- Emphasizes getting relevant results earlier in the ranking
- Particularly important when users only look at top results

### Normalized Discounted Cumulative Gain (nDCG@K)

**Definition**: Measures the usefulness of a document based on its position in the result list, with gains discounted logarithmically.

**Formula**:
```
DCG@K = Σ(rel_i / log₂(i+1)) for i=1 to K
nDCG@K = DCG@K / IDCG@K
```
Where IDCG@K is the ideal DCG (perfect ranking).

**When to Use**:
- When relevance is graded (not just binary relevant/irrelevant)
- When you want to reward highly relevant documents appearing at the top
- More sophisticated than simple precision/recall for graded relevance

---

## Generation Metrics

Generation evaluation assesses how well the system produces answers based on retrieved context. This is where RAG systems differ most from traditional IR systems.

### Answer Relevance

**Definition**: Measures how well the generated answer addresses the original question.

**LLM-as-Judge Approach**:
Instead of simple keyword matching, we use a language model to judge relevance with a clear rubric:

```
Score the relevance of the answer to the question on a scale of 0-10:
- 9-10: The answer directly and completely addresses the question
- 7-8: The answer addresses most aspects of the question with minor gaps
- 5-6: The answer addresses some aspects but misses important elements
- 3-4: The answer has minimal relevance to the question
- 0-2: The answer is not relevant to the question

Provide only the numerical score (0-10) as your response.
```

**Worked Example**:
Question: "What is Acme Corp's policy on remote work?"
Answer: "Acme Corp allows eligible employees to work remotely up to 3 days per week with manager approval."

LLM Judge Score: 9/10 → Normalized relevance = 0.90

**Heuristic Fallback**:
When LLM-as-judge is not available, we can use keyword overlap:
- Extract keywords from question and answer (removing stop words)
- Relevance = (Number of overlapping keywords) / (Number of question keywords)

### Faithfulness

**Definition**: Measures whether the generated answer is factually consistent with the retrieved context.

**LLM-as-Judge Approach**:
We use a language model to judge faithfulness with a clear rubric:

```
Score the faithfulness of the answer to the context on a scale of 0-10:
- 9-10: All facts in the answer are directly supported by the context
- 7-8: Most facts in the answer are supported by the context with minor inferences
- 5-6: About half of the facts in the answer are supported by the context
- 3-4: Few facts in the answer are supported by the context
- 0-2: No facts in the answer are supported by the context or answer contradicts context

Provide only the numerical score (0-10) as your response.
```

**Worked Example**:
Context: "Acme Corp allows eligible employees to work remotely up to 3 days per week."
Answer: "Acme Corp allows remote work up to 3 days per week for eligible employees."

LLM Judge Score: 10/10 → Normalized faithfulness = 1.00

**Heuristic Fallback**:
When LLM-as-judge is not available, we check term overlap:
- For each sentence in the answer, check if sufficient terms appear in the context
- Faithfulness = (Number of faithful sentences) / (Total sentences)

### Answer Completeness

**Definition**: Measures whether the answer contains all expected information elements.

**Approach**:
1. Define expected keywords/concepts for each question in the golden set
2. Check what percentage of these appear in the generated answer
3. Completeness = (Number of expected keywords found) / (Total expected keywords)

**Worked Example**:
Question: "What are the password requirements for Acme Corp's IT security policy?"
Expected keywords: ["12 characters", "uppercase", "lowercase", "numbers", "special characters", "90 days"]
Answer: "Passwords must be at least 12 characters long with uppercase, lowercase, numbers, and special characters, changed every 90 days."

Completeness = 6/6 = 1.00

---

## LLM-as-Judge: Principles and Pitfalls

LLM-as-judge has become a popular approach for evaluating generation quality, but it comes with important considerations.

### Principles of Effective LLM-as-Judge

1. **Clear Rubrics**: As shown above, provide explicit scoring guidelines
2. **Consistent Formatting**: Always ask for the same output format (e.g., "Provide only the numerical score")
3. **Context Window Awareness**: Ensure the prompt + context fits within the LLM's context window
4. **Bias Mitigation**: Use balanced examples and consider position bias
5. **Uncertainty Handling**: Allow the judge to express uncertainty when appropriate

### Common Pitfalls and How to Avoid Them

#### 1. Position Bias
**Problem**: LLMs often favor text that appears at the beginning or end of their input.
**Solution**: 
- Randomize the order of options when comparing multiple answers
- Use symmetrical prompt constructions
- Consider using pairwise comparisons instead of absolute scoring

#### 2. Self-Preference Bias
**Problem**: LLMs tend to prefer their own outputs over others.
**Solution**:
- When evaluating different systems, ensure the judge LLM is different from the candidate LLMs
- Use blind evaluation where possible
- Calibrate judges with known-good and known-bad examples

#### 3. Verbosity Bias
**Problem**: LLMs may favor longer, more verbose answers regardless of quality.
**Solution**:
- Normalize scores by expected answer length
- Focus on precision and relevance rather than mere word count
- Instruct judges to penalize irrelevant verbosity

#### 4. Instruction Drift
**Problem**: LLMs may forget or misinterpret complex instructions over long interactions.
**Solution**:
- Keep evaluation prompts concise and focused
- Repeat key instructions in the prompt
- Validate judge outputs for format compliance

#### 5. Lack of Calibration
**Problem**: Judges may use different parts of the scale inconsistently.
**Solution**:
- Provide calibration examples with known scores
- Use rating scales with clear semantic labels
- Consider using comparative judgments instead of absolute scores

### Best Practices for LLM-as-Judge in RAG Evaluation

1. **Use Multiple Judges**: Aggregate scores from multiple LLM calls or different LLMs
2. **Include Uncertainty Options**: Allow judges to express when they're unsure
3. **Spot-Check with Humans**: Periodically validate LLM judgments with human evaluators
4. **Track Judge Agreement**: Measure inter-judge reliability to detect drift
5. **Document Judge Settings**: Record temperature, model version, and prompt exactness

---

## Building a Golden Set

A golden set (or benchmark dataset) is crucial for reliable evaluation. Here's how to build one that stakeholders will trust.

### Principles of a Good Golden Set

1. **Representative Questions**: Cover the full spectrum of question types users will ask
2. **Clear Expected Answers**: Each question should have unambiguous correct answers
3. **Document Ground Truth**: Specify exactly which source documents support each answer
4. **Include Edge Cases**: Add unanswerable questions, multi-hop questions, and ambiguous queries
5. **Size Matters**: Aim for statistical significance (typically 30-100+ questions)

### Question Types to Include

| Type | Description | Example | Purpose |
|------|-------------|---------|---------|
| **Factual** | Direct lookup questions | "What is the PTO accrual rate?" | Test basic retrieval accuracy |
| **Multi-hop** | Requires combining information | "How does remote work policy affect IT security requirements?" | Test reasoning and synthesis |
| **Comparative** | Ask for differences/similarities | "How do current and previous leave policies differ?" | Test ability to handle contrasting information |
| **Unanswerable** | Not in source documents | "What is the CEO's salary?" | Test hallucination resistance and "I don't know" behavior |
| **Temporal** | Version-dependent questions | "What was the PTO carryover limit before 2024?" | Test version awareness and temporal reasoning |
| **Procedural** | Ask about processes | "How do employees request PTO?" | Test understanding of workflows |

### Steps to Build a Golden Set with Stakeholders

1. **Collaborate with Domain Experts**: Work with FDEs or subject matter experts to identify real questions
2. **Extract from Documentation**: Pull questions naturally arising from the source materials
3. **Add Edge Cases**: Intentionally include questions that test system boundaries
4. **Validate Answers**: Have experts verify that expected answers are correct and complete
5. **Document Sources**: Clearly specify which document chunks support each answer
6. **Review and Iterate**: Have multiple stakeholders review the set for quality and completeness

### Example Golden Set Entry

```json
{
  "id": "q17",
  "question": "What security measures are required for remote logins to Acme Corp networks?",
  "expected_answer_contains": ["Multi-factor authentication (MFA)", "company VPN"],
  "expected_sources": ["hr_remote_work_policy.md", "it_security_policy.md"],
  "question_type": "factual",
  "hop_count": 2,
  "validation_notes": "Requires combining information from both policies: HR policy mentions VPN requirement, IT policy specifies MFA for remote access"
}
```

---

## Presenting Results to Stakeholders

Technical metrics alone don't build trust—you need to present results in a way that resonates with your audience.

### For Technical Audiences (Engineers, Data Scientists)

**Focus On**:
- Detailed metric breakdowns
- Error analysis and failure modes
- Statistical significance and confidence intervals
- Ablation study results showing component contributions
- Comparison to baselines and state-of-the-art

**Presentation Format**:
- Detailed tables with precision metrics
- Error categorization (retrieval failures, generation failures, etc.)
- Learning curves showing improvement over iterations
- Resource trade-off analyses (latency vs. accuracy)

### For Business Audiences (Product Managers, Executives)

**Focus On**:
- Business impact metrics
- User experience implications
- Risk assessment (hallucination rates, error types)
- Cost-benefit analysis of different configurations
- Clear recommendations

**Presentation Format**:
- Executive summary with key findings
- Before/after examples showing concrete improvements
- Simple visualizations (bar charts, not complex statistical plots)
- Clear action items and recommendations

### Key Metrics to Highlight by Audience

| Audience | Key Metrics | Why They Matter |
|----------|-------------|-----------------|
| **Engineers** | Recall@K, MRR, nDCG, per-component ablation scores | Shows where to focus improvement efforts |
| **Product Managers** | Answer relevance, faithfulness, hallucination rate | Directly impacts user satisfaction and trust |
| **Executives** | % of queries answered correctly, reduction in escalation needs | Ties to business outcomes and ROI |
| **FDEs (Interview Prep)** | Accuracy on technical questions, version awareness, "I don't know" behavior | Measures effectiveness as a study aid |

### Building Trust Through Transparency

1. **Show Both Successes and Failures**: Don't hide challenging cases
2. **Provide Examples**: For each metric category, show concrete examples
3. **Explain Limitations**: Be clear about what the metrics don't capture
4. **Track Progress Over Time**: Show how the system improves with iterations
5. **Benchmark Against Meaningful Baselines**: Compare to reasonable alternatives, not straw men

### Sample Stakeholder Presentation Outline

1. **Executive Summary** (2-3 minutes)
   - Overall performance score
   - Key strengths and weaknesses
   - Primary recommendation

2. **Methodology** (2 minutes)
   - Golden set size and composition
   - Evaluation configurations tested
   - Key metrics used

3. **Results Deep Dive** (5-7 minutes)
   - Retrieval performance trends
   - Generation quality analysis
   - Failure mode breakdown
   - Ablation study insights

4. **Examples** (3-4 minutes)
   - Successful case walkthrough
   - Failure case analysis and root cause
   - Before/after comparison showing improvement

5. **Recommendations and Next Steps** (2-3 minutes)
   - Prioritized improvement backlog
   - Resource estimates
   - Success metrics for next iteration

---

## Ablation Studies

Ablation studies systematically remove or modify components to understand their individual contributions to overall performance.

### Why Run Ablation Studies?

1. **Component Attribution**: Understand which improvements actually move metrics
2. **Resource Allocation**: Focus effort on high-impact components
3. **Interaction Detection**: Identify synergistic or interfering effects
4. **Baseline Understanding**: Establish what the system would look like without each piece
5. **Stakeholder Communication**: Show concrete ROI for each investment

### How to Design an Effective Ablation Study

1. **Define a Meaningful Baseline**: 
   - Should represent a reasonable alternative, not a broken system
   - Example: Basic dense retrieval with naive chunking and simple prompts

2. **Test One Variable at a Time**:
   - Change only one aspect while holding others constant
   - This isolates the effect of each component

3. **Test Combinations**:
   - After individual tests, test logical combinations
   - Reveals synergies (where A+B > A + B)

4. **Use Consistent Evaluation**:
   - Same golden set, same metrics, same conditions
   - Only the component under test should vary

5. **Statistical Significance**:
   - Ensure differences are meaningful, not noise
   - Consider confidence intervals or bootstrap testing

### Common Ablation Dimensions in RAG Systems

| Dimension | What to Vary | What It Tests |
|-----------|--------------|---------------|
| **Retrieval** | Dense vs. sparse vs. hybrid | Value of different retrieval paradigms |
| **Fusion Method** | Weighted sum vs. RRF vs. rank merging | effectiveness of combination strategies |
| **Reranking** | None vs. cross-encoder vs. LLM-based | value of relevance refinement |
| **Context Window** | Different k values | optimal context size for generation |
| **Chunking** | Fixed size vs. recursive vs. structure-aware | impact of document segmentation |
| **Prompting** | Basic vs. grounded vs. chain-of-thought | effectiveness of prompt engineering |
| **Query Processing** | None vs. rewriting vs. expansion | value of query understanding |

### Interpreting Ablation Results

Look for:
- **Large Individual Effects**: Components that significantly move metrics on their own
- **Synergies**: Cases where A+B performs much better than expected from A and B individually
- **Diminishing Returns**: When adding more of a component gives progressively smaller gains
- **Negative Interactions**: Cases where A+B performs worse than expected (rare but informative)

### Example Ablation Study Results Table

| Configuration | Description | Hit Rate@5 | Recall@5 | MRR | Answer Relevance | Faithfulness | Key Insight |
|---------------|-------------|------------|----------|-----|------------------|--------------|-------------|
| Baseline | Fixed size chunks, dense retrieval, basic prompt | 0.65 | 0.45 | 0.32 | 0.58 | 0.62 | Starting point |
| + Improved Chunking | Recursive chunks, dense retrieval, basic prompt | 0.72 (+0.07) | 0.52 (+0.07) | 0.38 (+0.06) | 0.63 (+0.05) | 0.68 (+0.06) | Better chunks help retrieval |
| + Hybrid Retrieval | Recursive chunks, BM25+dense, basic prompt | 0.78 (+0.06) | 0.60 (+0.08) | 0.45 (+0.07) | 0.68 (+0.05) | 0.72 (+0.04) | Keyword + semantic synergy |
| + RRF | Recursive chunks, BM25+dense+RRF, basic prompt | 0.81 (+0.03) | 0.64 (+0.04) | 0.50 (+0.05) | 0.70 (+0.02) | 0.75 (+0.03) | Rank fusion improves robustness |
| + Reranking | Recursive chunks, BM25+dense+RRF + reranking, basic prompt | 0.82 (+0.01) | 0.65 (+0.01) | 0.58 (+0.08) | 0.75 (+0.05) | 0.79 (+0.04) | Reranking boosts precision |
| + Advanced Prompting | Recursive chunks, BM25+dense+RRF + reranking + advanced prompt | 0.82 (+0.00) | 0.65 (+0.00) | 0.58 (+0.00) | 0.83 (+0.08) | 0.84 (+0.05) | Prompting crucial for generation quality |
| + Query Rewriting | + LLM-based query rewriting | 0.85 (+0.03) | 0.69 (+0.04) | 0.62 (+0.04) | 0.86 (+0.03) | 0.86 (+0.02) | Query understanding helps retrieval |

**Interpretation**:
1. Improved chunking provides solid baseline gains (~10% relative improvement)
2. Hybrid retrieval is a major leap (~18% relative improvement in MRR)
3. RRF provides meaningful but smaller gains (~11% relative improvement in MRR over hybrid alone)
4. Reranking provides the largest single jump in MRR (~16% relative improvement)
5. Advanced prompting is critical for generation quality (~10% absolute improvement in relevance/faithfulness)
6. Query rewriting provides additional retrieval benefits, especially for vocabulary mismatch cases

### Recommendations for RAG System Development

Based on ablation study findings:
1. **Invest First in Hybrid Retrieval with RRF**: Largest retrieval gains for reasonable complexity
2. **Always Include Reranking**: Critical for precision of final results
3. **Optimize Chunking Strategy**: Recursive or structure-aware beats fixed size
4. **Invest in Prompt Engineering**: As important as retrieval improvements for final answer quality
5. **Consider Query Rewriting**: Especially valuable for technical domains with terminology gaps
6. **Monitor Diminishing Returns**: Each successive improvement typically yields smaller gains

---

## Conclusion

Effective RAG evaluation goes beyond measuring accuracy—it requires understanding why the system performs as it does and how to make it better. By combining:
1. **Robust Metrics**: Both retrieval (hit-rate@K, recall@K, MRR) and generation (LLM-as-judge relevance/f faithfulness)
2. **Thoughtful Golden Sets**: Representative questions with clear ground truth
3. **Systematic Ablation Studies**: To understand component contributions
4. **Stakeholder-Centered Presentation**: Tailoring insights to different audiences

...you can build RAG systems that not only perform well technically but also earn the trust of the people who depend on them.

For Field Development Engineers using this system as interview preparation aid, reliable evaluation ensures that:
- Technical answers are factually correct
- The system knows its limits and says "I don't know" appropriately
- Complex telecom concepts are explained clearly and accurately
- Users can trust the system as a reliable study companion

The metrics and methodologies described in this document provide a foundation for continuous improvement and validated performance in production RAG systems.