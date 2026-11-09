"""
src/evaluation/metrics.py - RAG evaluation metrics implemented from scratch.

BUSINESS PROBLEM: Provides the offline scoring functions that measure how well
the RAG pipeline retrieves relevant telecom document chunks (hit-rate@k,
recall@k, precision@k, MRR, nDCG@k, average precision) and how well generated
answers are faithful and relevant (LLM-as-judge with rubric, plus heuristic
fallbacks). These metrics power the golden-dataset evaluation used to validate
the telecom QA system for FDE interview prep.

WHERE IT SITS IN THE DATA FLOW: Downstream of retrieval and generation. It
consumes retrieved documents, generated answers, questions, and expected
document IDs/keywords, and produces numeric scores. It does not modify the
pipeline; it only observes and scores.

WHAT CALLS IT: src/evaluation/run_eval.py (RAGEvaluationPipeline), tests, and
any external evaluation harness importing RetrievalMetrics / GenerationMetrics /
RAGEvaluator or the module-level convenience functions.

WHAT IT CALLS: numpy (for nDCG log discounts), re (for score parsing and
keyword tokenization), collections.Counter (imported but unused), logging.
Calls src.generation.prompt.format_context (inside RAGEvaluator.evaluate_rag_pipeline)
via a lazy import.

WHAT BREAKS IF REMOVED: There is no way to quantitatively score retrieval or
generation quality; the evaluation script and golden-dataset checks cannot
produce metrics, so regression or improvement claims become unverifiable.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import numpy as np
import re
from collections import Counter
import json

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Offline retrieval-quality metrics computed from scratch (no ML lib).

    WHAT: Static-methods collection scoring retrieval: hit-rate@k, recall@k,
    precision@k, MRR, nDCG@k, and average precision over retrieved documents
    vs a set of relevant document IDs.
    WHY: Quantifies whether the retriever surfaces the right telecom document
    chunks for a question, which is the foundation of answer quality.
    INPUTS (per method): retrieved_docs list of dicts (with metadata.doc_id),
        relevant_doc_ids set of strings, k int.
    OUTPUTS: float scores in [0, 1] (MRR can be 0..1 too).
    SIDE EFFECTS: none (pure functions).
    FAILURE MODES: all methods return 0.0 when relevant_doc_ids is empty (or
        retrieved_docs is empty for nDCG); missing doc_id falls back to id(doc)
        which may not match expected ids, silently under-scoring.
    ALTERNATIVE: use a library like rank_eval / pytrec_eval / trec_eval for
        standardized, battle-tested retrieval metrics.
    # GAP: validation - no check that retrieved_docs is sorted by relevance
        score; the metrics assume rank order is already correct.
    # GAP: async / retries - none needed (pure in-memory computation).
    # BUSINESS: gives an objective number for "did we retrieve the right chunk?"
        that FDE learners can track across chunking/retrieval changes.
    # INTERVIEW: "Why fall back to id(doc) when doc_id is missing?" - because
        some retrieved chunks lack metadata; using object identity keeps the
        metric from crashing but may undercount hits if the same chunk object
        isn't reused.
    """

    @staticmethod
    def hit_rate_at_k(retrieved_docs: List[Dict[str, Any]],
                     relevant_doc_ids: Set[str],
                     k: int = 5) -> float:
        """Binary: is at least one relevant doc in the top k?

        WHAT: Returns 1.0 if any relevant document appears in the top-k
        retrieved results, else 0.0.
        WHY: A simple "did we find anything relevant?" signal for a question.
        INPUTS: retrieved_docs e.g. [{"metadata": {"doc_id": "lte_ho.md"}},
            ...], relevant_doc_ids e.g. {"lte_ho.md", "lte_pdcch.md"}, k=5.
        OUTPUTS: 1.0 on hit, 0.0 on miss; 0.0 if relevant_doc_ids is empty.
        SIDE EFFECTS: none.
        FAILURE MODES: chunks without doc_id use id(doc) which won't match
            expected ids, potentially yielding a false 0.0.
        ALTERNATIVE: precision@k or recall@k give a graded rather than binary
            signal.
        # BUSINESS: quick yes/no gate for whether retrieval works at all.
        # INTERVIEW: "Hit rate is all-or-nothing - why still use it?" - because
            it is easy to explain and a strong baseline before finer metrics.
        """
        if not relevant_doc_ids:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            # Try to get document ID from metadata (doc_id, then filename, then source)
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source")
            if doc_id:
                retrieved_ids.add(doc_id)
            else:
                # Fallback to using index as ID
                retrieved_ids.add(id(doc))

        # Check if at least one relevant document is in top k
        has_hit = len(retrieved_ids.intersection(relevant_doc_ids)) > 0
        return 1.0 if has_hit else 0.0

    @staticmethod
    def recall_at_k(retrieved_docs: List[Dict[str, Any]],
                   relevant_doc_ids: Set[str],
                   k: int = 5) -> float:
        """Fraction of relevant documents found in the top k.

        WHAT: Returns (number of relevant docs retrieved in top k) / (total
        relevant docs).
        WHY: Measures coverage - did the retriever find all the relevant chunks
        or only some of them.
        INPUTS: retrieved_docs list, relevant_doc_ids set e.g. {"a.md","b.md"},
        k=5.
        OUTPUTS: float in [0, 1]; 0.0 if relevant_doc_ids is empty.
        SIDE EFFECTS: none.
        FAILURE MODES: same doc_id fallback as hit_rate_at_k may undercount.
        ALTERNATIVE: precision@k measures correctness of retrieved set rather
            than coverage of relevant set.
        # BUSINESS: tells whether important telecom chunks are being missed.
        # INTERVIEW: "Recall can be gamed by retrieving everything - why not
            just always return all docs?" - because precision would collapse;
            the two metrics must be read together.
        """
        if not relevant_doc_ids:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            # Try to get document ID from metadata (doc_id, then filename, then source)
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source")
            if doc_id:
                retrieved_ids.add(doc_id)
            else:
                # Fallback to using index as ID
                retrieved_ids.add(id(doc))

        # Calculate recall
        relevant_retrieved = len(retrieved_ids.intersection(relevant_doc_ids))
        recall = relevant_retrieved / len(relevant_doc_ids)

        return recall

    @staticmethod
    def precision_at_k(retrieved_docs: List[Dict[str, Any]],
                      relevant_doc_ids: Set[str],
                      k: int = 5) -> float:
        """Fraction of the top k retrieved documents that are relevant.

        WHAT: Returns (number of relevant docs in top k) / k.
        WHY: Measures correctness - how much of the retrieved set is noise.
        INPUTS: retrieved_docs list, relevant_doc_ids set, k int (k=0 guarded).
        OUTPUTS: float in [0, 1]; 0.0 if k==0 or no retrieved docs.
        SIDE EFFECTS: none.
        FAILURE MODES: doc_id fallback may undercount relevant retrieved docs.
        ALTERNATIVE: recall@k measures coverage instead of correctness.
        # BUSINESS: keeps noisy/irrelevant chunks from cluttering the context
            sent to the LLM.
        # INTERVIEW: "If precision is 1.0 but recall is 0.1, is retrieval
            good?" - no: it retrieved only one relevant chunk and missed most;
            precision alone is misleading without recall.
        """
        if k == 0:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source")
            if doc_id:
                retrieved_ids.add(doc_id)
            else:
                retrieved_ids.add(id(doc))

        # Calculate precision
        relevant_retrieved = len(retrieved_ids.intersection(relevant_doc_ids))
        precision = relevant_retrieved / len(top_k_docs) if top_k_docs else 0.0

        return precision

    @staticmethod
    def mrr(retrieved_docs: List[Dict[str, Any]],
           relevant_doc_ids: Set[str]) -> float:
        """Mean Reciprocal Rank: 1 / rank of the first relevant document.

        WHAT: Returns 1/(rank+1) of the first relevant document in the ranked
        list, or 0.0 if none is relevant.
        WHY: Rewards retrievers that surface relevant docs early in the ranking.
        INPUTS: retrieved_docs list (ranked), relevant_doc_ids set.
        OUTPUTS: float in (0, 1]; 0.0 if no relevant doc found.
        SIDE EFFECTS: none.
        FAILURE MODES: doc_id fallback may miss relevant docs, yielding 0.0.
        ALTERNATIVE: average_precision considers all relevant docs, not just
            the first.
        # BUSINESS: a user scanning results sees the relevant chunk sooner when
            MRR is high.
        # INTERVIEW: "MRR only looks at the first relevant doc - what if the
            second is more important?" - MRR assumes any relevant doc satisfies;
            use nDCG or average_precision when relevance is graded.
        """
        if not relevant_doc_ids:
            return 0.0

        # Find the rank of the first relevant document
        for rank, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id") or metadata.get("filename") or metadata.get("source")
            if doc_id and doc_id in relevant_doc_ids:
                return 1.0 / (rank + 1)
            elif not doc_id and id(doc) in relevant_doc_ids:
                return 1.0 / (rank + 1)

        # No relevant document found
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[Dict[str, Any]],
                 relevant_doc_ids: Set[str],
                 k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain at k (binary relevance).

        WHAT: Computes DCG with log-position discounting, normalized by the
        ideal DCG (IDCG) where all relevant docs rank first.
        WHY: Rewards ranking relevant docs higher while penalizing relevant
        docs buried deep in the results.
        INPUTS: retrieved_docs list, relevant_doc_ids set, k int.
        OUTPUTS: float in [0, 1]; 0.0 if no retrieved docs or no relevant ids.
        SIDE EFFECTS: none.
        FAILURE MODES: doc_id fallback may misjudge relevance; idcg==0 guard
            returns 0.0.
        ALTERNATIVE: precision@k/recall@k are simpler but do not model rank
            position.
        # BUSINESS: measures ranking quality, not just membership - important
            when the top chunks go into the LLM context.
        # INTERVIEW: "Why divide by IDCG instead of raw DCG?" - normalization
            makes scores comparable across queries with different numbers of
            relevant docs.
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(top_k_docs):
            doc_id = doc.get("metadata", {}).get("doc_id")
            relevant = 1 if (doc_id and doc_id in relevant_doc_ids) or \
                           (not doc_id and id(doc) in relevant_doc_ids) else 0

            if relevant:
                dcg += relevant / np.log2(i + 2)  # i+2 because we start at position 1

        # Calculate IDCG (ideal DCG)
        # Ideal case: all relevant documents in top positions
        num_relevant = min(len(relevant_doc_ids), k)
        idcg = 0.0
        for i in range(num_relevant):
            idcg += 1 / np.log2(i + 2)

        # Calculate nDCG
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    @staticmethod
    def average_precision(retrieved_docs: List[Dict[str, Any]],
                         relevant_doc_ids: Set[str]) -> float:
        """Average Precision: mean of precision@i at each relevant doc rank.

        WHAT: Averages the precision computed at each position where a
        relevant document appears (deduped by doc_id).
        WHY: A single summary score that balances precision and recall across
        the whole ranked list.
        INPUTS: retrieved_docs list, relevant_doc_ids set.
        OUTPUTS: float in [0, 1]; 0.0 if no relevant docs or no hits.
        SIDE EFFECTS: none.
        FAILURE MODES: doc_id dedupe uses id(doc) fallback which may not
            deduplicate correctly across different chunk objects.
        ALTERNATIVE: ndcg_at_k when relevance is graded rather than binary.
        # BUSINESS: the standard single-number retrieval score used in IR.
        # INTERVIEW: "Why divide by len(relevant_doc_ids) at the end rather
            than len(precisions)?" - that is the standard AP definition: sum of
            precision-at-each-relevant divided by total relevant, which
            penalizes missed relevant docs.
        """
        if not relevant_doc_ids:
            return 0.0

        # Calculate precision at each relevant document (dedupe by doc_id:
        # the same chunk must not be counted twice if retrieved twice)
        precisions = []
        relevant_count = 0
        seen = set()

        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.get("metadata", {}).get("doc_id")
            if not doc_id:
                doc_id = id(doc)
            if doc_id in seen:
                continue
            seen.add(doc_id)

            if doc_id in relevant_doc_ids:
                relevant_count += 1
                # Calculate precision at this point
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)

        # Calculate average precision
        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant_doc_ids)


class GenerationMetrics:
    """Answer-quality metrics using LLM-as-judge with heuristic fallbacks.

    WHAT: Static-methods collection scoring generated answers for relevance,
    faithfulness, completeness, and contradiction against context/question,
    using an LLM judge with a rubric when available and keyword-overlap
    fallbacks otherwise.
    WHY: Retrieval metrics only tell us whether the right chunks were found;
    generation metrics tell us whether the answer is grounded in and answers
    the question - critical for trustworthy telecom QA.
    INPUTS (per method): answer string, question/context strings,
        llm_judge_fn callable(prompt)->str.
    OUTPUTS: float scores in [0, 1].
    SIDE EFFECTS: calls llm_judge_fn (may hit network / cost tokens) in the
        LLM-as-judge methods; logs warnings/errors on parse failures.
    FAILURE MODES: all methods return 0.0 when answer/context/question is
        empty; on LLM failure or unparseable score, fall back to heuristics
        (or 0.0/0.5 for contradiction).
    ALTERNATIVE: use deterministic metrics (BLEU/ROUGE/bert-score) against a
        reference answer, though these correlate poorly with factuality.
    # GAP: observability - LLM judge calls are not traced or retried; a judge
        outage silently falls back to heuristics without surfacing the failure.
    # GAP: validation - the rubric expects a numeric 0-10 response; a
        verbose judge breaks the regex and triggers the fallback.
    # BUSINESS: protects against hallucinated telecom facts by scoring whether
        the answer is supported by retrieved context.
    # INTERVIEW: "Why LLM-as-judge instead of exact match?" - because telecom
        answers can be phrased many ways; exact match would mark correct
        paraphrases wrong, while a judge evaluates meaning.
    """

    @staticmethod
    def answer_relevance_llm_as_judge(answer: str, question: str, llm_judge_fn) -> float:
        """Score answer relevance to the question via LLM judge (0-10 rubric).

        WHAT: Builds a rubric prompt, calls llm_judge_fn, parses a 0-10 score,
        and normalizes to 0-1.
        WHY: Lets an LLM judge semantic relevance rather than keyword overlap.
        INPUTS: answer string, question string e.g. "What is an EPS bearer?",
            llm_judge_fn callable returning a string.
        OUTPUTS: float in [0, 1]; falls back to keyword-overlap on parse/LLM error.
        SIDE EFFECTS: invokes llm_judge_fn (network/cost); logs warnings on
            parse failure, errors on exception.
        FAILURE MODES: empty answer/question returns 0.0; unparseable judge
            output uses _fallback_answer_relevance.
        ALTERNATIVE: the _fallback_answer_relevance keyword-overlap heuristic
            when no judge is available.
        # GAP: validation - judge output format is not enforced (no schema),
            so a chatty judge breaks the regex.
        # BUSINESS: ensures answers actually address the asked telecom question.
        # INTERVIEW: "Why normalize the 0-10 judge score by dividing by 10?"
            - to put relevance on the same 0-1 scale as the other metrics for
            aggregation.
        """
        if not answer or not question:
            return 0.0

        # Define the rubric for relevance judgment
        rubric = """
        Score the relevance of the answer to the question on a scale of 0-10:
        - 9-10: The answer directly and completely addresses the question
        - 7-8: The answer addresses most aspects of the question with minor gaps
        - 5-6: The answer addresses some aspects but misses important elements
        - 3-4: The answer has minimal relevance to the question
        - 0-2: The answer is not relevant to the question

        Provide only the numerical score (0-10) as your response.
        """

        prompt = f"""
        {rubric}

        Question: {question}

        Answer: {answer}

        Relevance Score (0-10):"""

        try:
            # Get judgment from LLM
            score_raw = llm_judge_fn(prompt)
            # Extract numerical score
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_raw)
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 scale
                return min(max(score / 10.0, 0.0), 1.0)
            else:
                logger.warning(f"Could not extract score from LLM response: {score_raw}")
                return GenerationMetrics._fallback_answer_relevance(answer, question)
        except Exception as e:
            logger.error(f"Error in LLM-as-judge relevance scoring: {e}")
            return GenerationMetrics._fallback_answer_relevance(answer, question)

    @staticmethod
    def _fallback_answer_relevance(answer: str, question: str) -> float:
        """Heuristic answer-relevance via keyword overlap (stop-word filtered).

        WHAT: Computes overlap between question keywords and answer keywords
        after removing stop words, capped at 1.0.
        WHY: A no-LLM fallback so relevance can still be scored offline or in
        CI without a judge provider.
        INPUTS: answer string, question string.
        OUTPUTS: float in [0, 1]; 0.0 if no question keywords.
        SIDE EFFECTS: none.
        FAILURE MODES: keyword overlap rewards term stuffing and misses
            paraphrases/synonyms; overestimates relevance for keyword-rich but
            off-topic answers.
        ALTERNATIVE: the LLM-as-judge variant captures semantics better.
        # GAP: validation - stop-word list is English-only and crude; telecom
            acronyms (e.g. "EPS", "MME") are kept as keywords, which is good,
            but no stemming is applied.
        # BUSINESS: provides a deterministic relevance floor when the LLM judge
            is unavailable (CI, no API key).
        # INTERVIEW: "Why remove stop words before overlap?" - because words
            like "the"/"is" appear in everything and would inflate overlap
            without signaling relevance.
        """
        if not answer or not question:
            return 0.0

        # Extract keywords (simple approach)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who', 'which'}

        question_keywords = question_words - stop_words
        answer_keywords = answer_words - stop_words

        if not question_keywords:
            return 0.0

        # Calculate overlap
        overlap = len(question_keywords.intersection(answer_keywords))
        relevance = overlap / len(question_keywords)

        return min(relevance, 1.0)  # Cap at 1.0

    @staticmethod
    def faithfulness_llm_as_judge(answer: str, context: str, llm_judge_fn) -> float:
        """Score answer faithfulness to the retrieved context via LLM judge.

        WHAT: Builds a rubric prompt asking whether answer facts are supported
        by the context, calls llm_judge_fn, parses 0-10, normalizes to 0-1.
        WHY: Detects hallucination - answers that state facts not in context.
        INPUTS: answer string, context string, llm_judge_fn callable.
        OUTPUTS: float in [0, 1]; falls back to _fallback_faithfulness on error.
        SIDE EFFECTS: invokes llm_judge_fn (network/cost); logs warnings/errors.
        FAILURE MODES: empty answer/context returns 0.0; unparseable judge
            output uses the heuristic fallback.
        ALTERNATIVE: NLI-based entailment (answer entailed by context) instead
            of an LLM rubric.
        # GAP: validation - judge output format unenforced; non-numeric output
            triggers fallback silently.
        # BUSINESS: is the key guardrail that telecom answers do not invent
            facts outside retrieved documents.
        # INTERVIEW: "Why is faithfulness more important than relevance for a
            RAG system?" - because a relevant but hallucinated answer is worse
            than a partially relevant but factually grounded one.
        """
        if not answer or not context:
            return 0.0

        # Define the rubric for faithfulness judgment
        rubric = """
        Score the faithfulness of the answer to the context on a scale of 0-10:
        - 9-10: All facts in the answer are directly supported by the context
        - 7-8: Most facts in the answer are supported by the context with minor inferences
        - 5-6: About half of the facts in the answer are supported by the context
        - 3-4: Few facts in the answer are supported by the context
        - 0-2: No facts in the answer are supported by the context or answer contradicts context

        Provide only the numerical score (0-10) as your response.
        """

        prompt = f"""
        {rubric}

        Context: {context}

        Answer: {answer}

        Faithfulness Score (0-10):"""

        try:
            # Get judgment from LLM
            score_raw = llm_judge_fn(prompt)
            # Extract numerical score
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_raw)
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 scale
                return min(max(score / 10.0, 0.0), 1.0)
            else:
                logger.warning(f"Could not extract score from LLM response: {score_raw}")
                return GenerationMetrics._fallback_faithfulness(answer, context)
        except Exception as e:
            logger.error(f"Error in LLM-as-judge faithfulness scoring: {e}")
            return GenerationMetrics._fallback_faithfulness(answer, context)

    @staticmethod
    def _fallback_faithfulness(answer: str, context: str) -> float:
        """Heuristic faithfulness via term overlap of answer sentences in context.

        WHAT: Splits the answer into sentences, checks whether >=50% of each
        sentence's content words appear in the context, returns the fraction of
        sentences judged faithful.
        WHY: A cheap no-LLM approximation of whether answer claims appear in
        retrieved context.
        INPUTS: answer string, context string.
        OUTPUTS: float in [0, 1]; 0.0 if no answer sentences.
        SIDE EFFECTS: none.
        FAILURE MODES: sentence splitting on .!? is crude (abbreviations break
            it); term-overlap can mark a sentence faithful even if the meaning
            is inverted (negation not handled).
        ALTERNATIVE: the LLM-as-judge faithfulness handles semantics/negation.
        # GAP: validation - no NLI/entailment model; "not supported" and
            "contradicted" are conflated.
        # BUSINESS: gives a deterministic groundedness floor without an LLM
            judge (CI/no-key environments).
        # INTERVIEW: "Why the 50% content-word threshold per sentence?" - it is
            an arbitrary heuristic chosen to be lenient; there is no
            telecom-domain calibration behind it.
        """
        if not answer or not context:
            return 0.0

        # Extract factual statements (simplified)
        answer_sentences = re.split(r'[.!?]+', answer)
        context_lower = context.lower()

        faithful_sentences = 0
        total_sentences = 0

        for sentence in answer_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            total_sentences += 1
            # Check if key phrases from sentence appear in context
            # This is a simplified check - in practice would use NLI
            words = re.findall(r'\b\w+\b', sentence.lower())
            if len(words) > 3:  # Only check sentences with substantial content
                # Check if at least 50% of content words appear in context
                matches = sum(1 for word in words if word in context_lower)
                if len(words) > 0 and matches / len(words) >= 0.5:
                    faithful_sentences += 1

        if total_sentences == 0:
            return 0.0

        return faithful_sentences / total_sentences

    @staticmethod
    def answer_completeness(answer: str, expected_keywords: List[str]) -> float:
        """Fraction of expected keywords present in the answer.

        WHAT: Counts how many expected_keywords appear (case-insensitive) in
        the answer, divided by total expected keywords.
        WHY: Verifies the answer mentions the key telecom terms it should.
        INPUTS: answer string, expected_keywords list e.g. ["EPS bearer", "MME"].
        OUTPUTS: float in [0, 1]; 1.0 if no expected keywords (vacuous), 0.0
        if answer empty but keywords expected.
        SIDE EFFECTS: none.
        FAILURE MODES: substring matching can match partial words; no stemming
        or synonym handling.
        ALTERNATIVE: LLM-as-judge completeness that understands meaning rather
        than substrings.
        # BUSINESS: confirms required telecom terminology appears in answers.
        # INTERVIEW: "Why return 1.0 when expected_keywords is empty?" - a
            vacuous truth: if nothing is required, completeness is trivially
            satisfied.
        """
        if not answer or not expected_keywords:
            return 0.0 if expected_keywords else 1.0

        answer_lower = answer.lower()
        found_count = 0

        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found_count += 1

        return found_count / len(expected_keywords)

    @staticmethod
    def contradiction_detection(answer: str, context: str, llm_judge_fn) -> float:
        """Detect answer-vs-context contradiction via LLM judge (0=no, 1=yes).

        WHAT: Asks the LLM judge YES/NO whether the answer contradicts the
        context; returns 1.0 for YES, 0.0 for NO, 0.5 for unparseable.
        WHY: Surfaces direct contradictions between the answer and retrieved
        context, a strong hallucination signal.
        INPUTS: answer string, context string, llm_judge_fn callable.
        OUTPUTS: float in {0.0, 0.5, 1.0}; 0.0 if empty answer/context or error.
        SIDE EFFECTS: invokes llm_judge_fn (network/cost); logs warnings/errors.
        FAILURE MODES: judge output not YES/NO returns 0.5 (uncertain);
        exceptions return 0.0 (assumes no contradiction - may under-detect).
        ALTERNATIVE: an NLI contradiction model instead of an LLM yes/no prompt.
        # GAP: validation - judge must answer exactly YES/NO; anything else is
            treated as uncertain (0.5), which may not be the desired default.
        # BUSINESS: catches answers that directly deny retrieved telecom facts.
        # INTERVIEW: "Why return 0.0 (no contradiction) on exception instead of
            0.5?" - failing closed would penalize every judge outage as a
            contradiction; failing open avoids false alarms but may miss real
            contradictions.
        """
        if not answer or not context:
            return 0.0

        # Define the prompt for contradiction detection
        prompt = f"""
        Determine if the answer contradicts the context.
        Answer with only "YES" if there is a contradiction, or "NO" if there is no contradiction.

        Context: {context}

        Answer: {answer}

        Contradiction (YES/NO):"""

        try:
            # Get judgment from LLM
            judgment = llm_judge_fn(prompt).strip().upper()
            if judgment == "YES":
                return 1.0
            elif judgment == "NO":
                return 0.0
            else:
                logger.warning(f"Unexpected judgment for contradiction detection: {judgment}")
                return 0.5  # Uncertain
        except Exception as e:
            logger.error(f"Error in contradiction detection: {e}")
            return 0.0

class RAGEvaluator:
    """End-to-end evaluator for RAG systems (retrieval + generation scores).

    WHAT: Orchestrates RetrievalMetrics and GenerationMetrics to score a
    single RAG query end-to-end, returning a nested dict of metrics.
    WHY: Provides one entry point for evaluating a full RAG turn so the
    evaluation script and tests do not wire metric classes together manually.
    INPUTS: query, retrieved_docs, relevant_doc_ids; for generation: answer,
        question, context, optional expected_keywords and llm_judge_fn.
    OUTPUTS: dict of metric scores (nested retrieval/generation dicts).
    SIDE EFFECTS: evaluate_rag_pipeline lazily imports format_context from
        src.generation.prompt.
    FAILURE MODES: metric methods each return 0.0 on empty inputs; no method
        raises for normal input shapes.
    ALTERNATIVE: call RetrievalMetrics / GenerationMetrics static methods
        directly for finer control.
    # GAP: async / retries - none; purely synchronous in-memory scoring.
    # BUSINESS: gives a single reproducible score per telecom QA question for
        regression tracking across pipeline changes.
    # INTERVIEW: "Why does evaluate_rag_pipeline rebuild context from
        retrieved_docs instead of taking it as input?" - so retrieval and
        generation are scored on the exact same context the generator saw,
        keeping the evaluation faithful to the live pipeline.
    """

    def __init__(self):
        """Initialize the evaluator with retrieval and generation metric instances."""
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()

    def evaluate_retrieval(self,
                          query: str,
                          retrieved_docs: List[Dict[str, Any]],
                          relevant_doc_ids: Set[str]) -> Dict[str, float]:
        """Evaluate retrieval performance for a single query.

        WHAT: Computes hit-rate@5/10, recall@5/10, precision@5, MRR, nDCG@5,
        and average precision for one query.
        WHY: Summarizes retrieval quality across the standard IR metrics in one
        call.
        INPUTS: query string (context only, not used in computation),
            retrieved_docs list, relevant_doc_ids set.
        OUTPUTS: dict {"hit_rate_at_5", "hit_rate_at_10", "recall_at_5",
            "recall_at_10", "precision_at_5", "mrr", "ndcg_at_5",
            "average_precision"}.
        SIDE EFFECTS: none (delegates to pure static methods).
        FAILURE MODES: each sub-metric returns 0.0 on empty inputs.
        ALTERNATIVE: call individual RetrievalMetrics methods for a subset.
        # BUSINESS: the standard retrieval report card for a telecom QA query.
        # INTERVIEW: "Why both hit_rate and recall?" - hit rate is binary
            (found anything), recall is graded (found how much); they answer
            different questions.
        """
        return {
            "hit_rate_at_5": self.retrieval_metrics.hit_rate_at_k(retrieved_docs, relevant_doc_ids, 5),
            "hit_rate_at_10": self.retrieval_metrics.hit_rate_at_k(retrieved_docs, relevant_doc_ids, 10),
            "recall_at_5": self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_doc_ids, 5),
            "recall_at_10": self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_doc_ids, 10),
            "precision_at_5": self.retrieval_metrics.precision_at_k(retrieved_docs, relevant_doc_ids, 5),
            "mrr": self.retrieval_metrics.mrr(retrieved_docs, relevant_doc_ids),
            "ndcg_at_5": self.retrieval_metrics.ndcg_at_k(retrieved_docs, relevant_doc_ids, 5),
            "average_precision": self.retrieval_metrics.average_precision(retrieved_docs, relevant_doc_ids)
        }

    def evaluate_generation(self,
                           answer: str,
                           question: str,
                           context: str,
                           expected_keywords: Optional[List[str]] = None,
                           llm_judge_fn: Optional[callable] = None) -> Dict[str, float]:
        """Evaluate generation quality for a single question.

        WHAT: Computes answer relevance, faithfulness, contradiction (via LLM
        judge or heuristics), and optionally answer completeness.
        WHY: Summarizes whether the generated answer is relevant, grounded, and
        complete for one question.
        INPUTS: answer string, question string, context string,
            expected_keywords optional list, llm_judge_fn optional callable.
        OUTPUTS: dict with "answer_relevance", "faithfulness",
            "contradiction", and optionally "answer_completeness".
        SIDE EFFECTS: may invoke llm_judge_fn (network/cost) if provided.
        FAILURE MODES: falls back to heuristics when llm_judge_fn is absent or
            raises; contradiction defaults to 0.0 without a judge.
        ALTERNATIVE: call GenerationMetrics methods directly.
        # GAP: validation - llm_judge_fn signature is not checked; a
            mismatched callable raises at call time inside the judge methods.
        # BUSINESS: the generation report card for a telecom answer.
        # INTERVIEW: "Why is contradiction 0.0 without an LLM judge?" -
            contradiction detection requires semantic understanding that the
            heuristic fallback cannot provide, so it is left unset rather than
            guessed.
        """
        metrics = {}

        # Use LLM-as-judge if provided, otherwise fall back to heuristic methods
        if llm_judge_fn:
            metrics["answer_relevance"] = self.generation_metrics.answer_relevance_llm_as_judge(
                answer, question, llm_judge_fn)
            metrics["faithfulness"] = self.generation_metrics.faithfulness_llm_as_judge(
                answer, context, llm_judge_fn)
            metrics["contradiction"] = self.generation_metrics.contradiction_detection(
                answer, context, llm_judge_fn)
        else:
            # Fall back to heuristic methods
            metrics["answer_relevance"] = self.generation_metrics._fallback_answer_relevance(
                answer, question)
            metrics["faithfulness"] = self.generation_metrics._fallback_faithfulness(
                answer, context)
            metrics["contradiction"] = 0.0  # Can't detect contradictions without LLM

        if expected_keywords:
            metrics["answer_completeness"] = self.generation_metrics.answer_completeness(
                answer, expected_keywords)

        return metrics

    def evaluate_rag_pipeline(self,
                             query: str,
                             retrieved_docs: List[Dict[str, Any]],
                             generated_answer: str,
                             relevant_doc_ids: Set[str],
                             expected_keywords: Optional[List[str]] = None,
                             llm_judge_fn: Optional[callable] = None) -> Dict[str, Any]:
        """Evaluate the complete RAG pipeline for a single question.

        WHAT: Formats context from retrieved docs, then runs retrieval and
        generation evaluation, combining everything into one result dict.
        WHY: Mirrors the live pipeline (retrieve -> format context -> generate)
        so scores reflect the actual end-to-end flow.
        INPUTS: query string, retrieved_docs list, generated_answer string,
            relevant_doc_ids set, expected_keywords optional list,
            llm_judge_fn optional callable.
        OUTPUTS: dict {"query", "retrieval": {...}, "generation": {...},
            "num_retrieved_docs": int, "context_length": int}.
        SIDE EFFECTS: imports format_context lazily; may invoke llm_judge_fn.
        FAILURE MODES: propagates errors from format_context or metric methods
            (each sub-metric guards empty input).
        ALTERNATIVE: call evaluate_retrieval and evaluate_generation
            separately for finer granularity.
        # GAP: observability - no timing is recorded around the scoring itself,
            so slow metric computation is invisible.
        # BUSINESS: one number-set per telecom QA question for comparing
            pipeline configurations.
        # INTERVIEW: "Why include num_retrieved_docs and context_length?" - to
            correlate score changes with how much context was actually used
            (more context can raise faithfulness but lower relevance).
        """
        # Format context for evaluation
        from ..generation.prompt import format_context
        context = format_context(retrieved_docs)

        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Evaluate generation
        generation_metrics = self.evaluate_generation(
            generated_answer, query, context, expected_keywords, llm_judge_fn)

        # Combine metrics
        return {
            "query": query,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "num_retrieved_docs": len(retrieved_docs),
            "context_length": len(context)
        }


# Convenience functions
def evaluate_retrieval(retrieved_docs: List[Dict[str, Any]],
                      relevant_doc_ids: Set[str],
                      k: int = 5) -> float:
    """Calculate Recall@K for retrieval.

    WHAT: Thin wrapper returning RetrievalMetrics.recall_at_k with default k=5.
    WHY: Provides a simple module-level function for the common recall@5 case.
    INPUTS: retrieved_docs list, relevant_doc_ids set, k int (default 5).
    OUTPUTS: float recall score in [0, 1].
    SIDE EFFECTS: none.
    FAILURE MODES: returns 0.0 on empty relevant_doc_ids.
    ALTERNATIVE: RetrievalMetrics.recall_at_k directly for other k values.
    # BUSINESS: the single most-cited retrieval metric for quick checks.
    # INTERVIEW: "Why does this convenience function only expose recall, not
        the other metrics?" - recall@5 was the historically primary metric for
        this project; others are available via the class.
    """
    return RetrievalMetrics.recall_at_k(retrieved_docs, relevant_doc_ids, k)


def evaluate_hit_rate(retrieved_docs: List[Dict[str, Any]],
                     relevant_doc_ids: Set[str],
                     k: int = 5) -> float:
    """Calculate Hit Rate@K for retrieval.

    WHAT: Thin wrapper returning RetrievalMetrics.hit_rate_at_k.
    WHY: Simple module-level entry point for the binary hit-rate metric.
    INPUTS: retrieved_docs list, relevant_doc_ids set, k int (default 5).
    OUTPUTS: 1.0 or 0.0.
    SIDE EFFECTS: none.
    FAILURE MODES: 0.0 on empty relevant_doc_ids.
    ALTERNATIVE: RetrievalMetrics.hit_rate_at_k directly.
    # BUSINESS: quick yes/no retrieval gate.
    # INTERVIEW: (same as hit_rate_at_k - see class docstring.)
    """
    return RetrievalMetrics.hit_rate_at_k(retrieved_docs, relevant_doc_ids, k)


def evaluate_mrr(retrieved_docs: List[Dict[str, Any]],
                relevant_doc_ids: Set[str]) -> float:
    """Calculate MRR for retrieval.

    WHAT: Thin wrapper returning RetrievalMetrics.mrr.
    WHY: Simple module-level entry point for mean reciprocal rank.
    INPUTS: retrieved_docs list (ranked), relevant_doc_ids set.
    OUTPUTS: float in (0, 1] or 0.0.
    SIDE EFFECTS: none.
    FAILURE MODES: 0.0 on empty relevant_doc_ids or no relevant doc found.
    ALTERNATIVE: RetrievalMetrics.mrr directly.
    # BUSINESS: rewards surfacing the relevant chunk early.
    # INTERVIEW: (see mrr in RetrievalMetrics.)
    """
    return RetrievalMetrics.mrr(retrieved_docs, relevant_doc_ids)


def evaluate_answer_relevance(answer: str, question: str) -> float:
    """Calculate answer relevance score (heuristic fallback).

    WHAT: Thin wrapper returning GenerationMetrics._fallback_answer_relevance.
    WHY: Simple keyword-overlap relevance without an LLM judge.
    INPUTS: answer string, question string.
    OUTPUTS: float in [0, 1].
    SIDE EFFECTS: none.
    FAILURE MODES: 0.0 on empty inputs.
    ALTERNATIVE: answer_relevance_llm_as_judge when a judge is available.
    # BUSINESS: relevance scoring usable offline (CI, no API key).
    # INTERVIEW: (see _fallback_answer_relevance - keyword overlap limitations.)
    """
    return GenerationMetrics._fallback_answer_relevance(answer, question)


def evaluate_faithfulness(answer: str, context: str) -> float:
    """Calculate faithfulness score (heuristic fallback).

    WHAT: Thin wrapper returning GenerationMetrics._fallback_faithfulness.
    WHY: Simple term-overlap groundedness without an LLM judge.
    INPUTS: answer string, context string.
    OUTPUTS: float in [0, 1].
    SIDE EFFECTS: none.
    FAILURE MODES: 0.0 on empty inputs.
    ALTERNATIVE: faithfulness_llm_as_judge when a judge is available.
    # BUSINESS: groundedness scoring usable offline.
    # INTERVIEW: (see _fallback_faithfulness - term-overlap limitations.)
    """
    return GenerationMetrics._fallback_faithfulness(answer, context)