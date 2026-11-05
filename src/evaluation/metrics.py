"""
Evaluation metrics for RAG systems implemented from scratch.
Includes retrieval metrics (hit-rate@k, recall@k, MRR) and generation metrics
with LLM-as-judge for faithfulness and answer relevance.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import numpy as np
import re
from collections import Counter
import json

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Metrics for evaluating retrieval performance implemented from scratch."""

    @staticmethod
    def hit_rate_at_k(retrieved_docs: List[Dict[str, Any]],
                     relevant_doc_ids: Set[str],
                     k: int = 5) -> float:
        """
        Calculate Hit Rate@K (whether at least one relevant document is in top k).

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Hit Rate@K score (0.0 or 1.0)
        """
        if not relevant_doc_ids:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            # Try to get document ID from metadata
            doc_id = doc.get("metadata", {}).get("doc_id")
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
        """
        Calculate Recall@K (proportion of relevant documents retrieved in top k).

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@K score
        """
        if not relevant_doc_ids:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            # Try to get document ID from metadata
            doc_id = doc.get("metadata", {}).get("doc_id")
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
        """
        Calculate Precision@K (proportion of retrieved documents that are relevant).

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0

        # Get top k retrieved documents
        top_k_docs = retrieved_docs[:k]

        # Extract document IDs from retrieved documents
        retrieved_ids = set()
        for doc in top_k_docs:
            doc_id = doc.get("metadata", {}).get("doc_id")
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
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs

        Returns:
            MRR score
        """
        if not relevant_doc_ids:
            return 0.0

        # Find the rank of the first relevant document
        for rank, doc in enumerate(retrieved_docs):
            doc_id = doc.get("metadata", {}).get("doc_id")
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
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            nDCG@K score
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
        """
        Calculate Average Precision.

        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs

        Returns:
            Average Precision score
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
    """Metrics for evaluating answer generation quality with LLM-as-judge."""

    @staticmethod
    def answer_relevance_llm_as_judge(answer: str, question: str, llm_judge_fn) -> float:
        """
        Calculate answer relevance using LLM-as-judge with a clear rubric.

        Args:
            answer: Generated answer
            question: Original question
            llm_judge_fn: Function that takes a prompt and returns a judgment score

        Returns:
            Relevance score (0-1)
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
        """Fallback relevance calculation based on keyword overlap."""
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
        """
        Calculate faithfulness using LLM-as-judge with a clear rubric.

        Args:
            answer: Generated answer
            context: Retrieved context
            llm_judge_fn: Function that takes a prompt and returns a judgment score

        Returns:
            Faithfulness score (0-1)
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
        """Fallback faithfulness calculation based on term overlap."""
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
        """
        Check if answer contains expected keywords.

        Args:
            answer: Generated answer
            expected_keywords: List of keywords that should appear in answer

        Returns:
            Completeness score (0-1)
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
        """
        Detect contradictions between answer and context using LLM-as-judge.

        Args:
            answer: Generated answer
            context: Retrieved context
            llm_judge_fn: Function that takes a prompt and returns a judgment

        Returns:
            Contradiction score (0-1, where 0 means no contradiction, 1 means clear contradiction)
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
    """End-to-end evaluator for RAG systems."""

    def __init__(self):
        """Initialize the evaluator."""
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()

    def evaluate_retrieval(self,
                          query: str,
                          retrieved_docs: List[Dict[str, Any]],
                          relevant_doc_ids: Set[str]) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a single query.

        Args:
            query: Query string
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of relevant document IDs

        Returns:
            Dictionary of retrieval metrics
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
        """
        Evaluate generation quality for a single question.

        Args:
            answer: Generated answer
            question: Original question
            context: Retrieved context used for generation
            expected_keywords: Optional list of expected keywords
            llm_judge_fn: Optional LLM judge function for faithfulness and relevance

        Returns:
            Dictionary of generation metrics
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
        """
        Evaluate the complete RAG pipeline for a single question.

        Args:
            query: Query string
            retrieved_docs: List of retrieved document dictionaries
            generated_answer: Generated answer string
            relevant_doc_ids: Set of relevant document IDs
            expected_keywords: Optional list of expected keywords
            llm_judge_fn: Optional LLM judge function for faithfulness and relevance

        Returns:
            Dictionary containing all evaluation metrics
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
    """Calculate Recall@K for retrieval."""
    return RetrievalMetrics.recall_at_k(retrieved_docs, relevant_doc_ids, k)

def evaluate_hit_rate(retrieved_docs: List[Dict[str, Any]],
                     relevant_doc_ids: Set[str],
                     k: int = 5) -> float:
    """Calculate Hit Rate@K for retrieval."""
    return RetrievalMetrics.hit_rate_at_k(retrieved_docs, relevant_doc_ids, k)

def evaluate_mrr(retrieved_docs: List[Dict[str, Any]],
                relevant_doc_ids: Set[str]) -> float:
    """Calculate MRR for retrieval."""
    return RetrievalMetrics.mrr(retrieved_docs, relevant_doc_ids)

def evaluate_answer_relevance(answer: str, question: str) -> float:
    """Calculate answer relevance score (heuristic fallback)."""
    return GenerationMetrics._fallback_answer_relevance(answer, question)

def evaluate_faithfulness(answer: str, context: str) -> float:
    """Calculate faithfulness score (heuristic fallback)."""
    return GenerationMetrics._fallback_faithfulness(answer, context)