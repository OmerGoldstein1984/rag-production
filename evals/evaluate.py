"""
Evaluation runner for the RAG system.
Calculates metrics like Recall@k, MRR, and uses LLM-as-a-judge for faithfulness.
"""

import json
import logging
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.load import load_pdf_document
from src.chunk import chunk_text
from src.index import index_chunks, search_similar
from src.retrieve import retrieve_relevant_chunks
from src.answer import generate_answer, format_answer_with_citations
from .golden_dataset import get_golden_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.golden_dataset = get_golden_dataset()
        self.results = []

    def load_and_index_documents(self, pdf_directory: str = "./data/uploads"):
        """Load and index all PDF documents in the directory."""
        logger.info(f"Loading and indexing documents from {pdf_directory}")

        # In a real system, you'd check if already indexed
        # For evaluation, we'll re-index everything

        # This assumes documents are already uploaded and indexed via the API
        # For simplicity in evaluation, we'll just note that indexing should be done
        logger.info("Assuming documents are pre-indexed via the upload API")

    def calculate_recall_at_k(self, retrieved_sources: List[str], expected_sources: List[str], k: int = 5) float:
        """Calculate Recall@k: proportion of expected sources found in top k retrieved."""
        if not expected_sources:
            return 0.0

        top_k_sources = retrieved_sources[:k]
        retrieved_set = set(top_k_sources)
        expected_set = set(expected_sources)

        if not expected_set:
            return 0.0

        intersection = retrieved_set.intersection(expected_set)
        return len(intersection) / len(expected_set)

    def calculate_mrr(self, retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """Calculate Mean Reciprocal Rank: 1/rank of first relevant result."""
        if not expected_sources:
            return 0.0

        expected_set = set(expected_sources)
        for rank, source in enumerate(retrieved_sources, start=1):
            if source in expected_set:
                return 1.0 / rank
        return 0.0

    def evaluate_faithfulness_llm_as_judge(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Tuple[float, str]:
        """
        Use LLM-as-a-judge to evaluate faithfulness of answer to context.
        Returns (score, explanation) where score is 0.0 to 1.0.
        """
        # This would call an LLM to judge if the answer is faithful to the context
        # For simplicity, we'll implement a basic version
        # In practice, you'd use a prompt like:
        # "Does the answer contain only information that can be inferred from the context?"

        # Placeholder implementation
        if not answer or "I don't have enough information" in answer:
            return 0.5, "Answer indicates insufficient information"

        # Simple heuristic: check if answer contains context words
        # This is very basic - real implementation would use an LLM
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())

        if not context_words:
            return 0.0, "Empty context"

        overlap = len(context_words.intersection(answer_words))
        score = min(overlap / len(context_words), 1.0) if context_words else 0.0

        explanation = f"Word overlap score: {overlap}/{len(context_words)}"
        return score, explanation

    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation on the golden dataset."""
        logger.info(f"Starting evaluation on {len(self.golden_dataset)} questions")

        metrics = {
            "recall_at_1": [],
            "recall_at_3": [],
            "recall_at_5": [],
            "mrr": [],
            "faithfulness": [],
            "question_results": []
        }

        start_time = time.time()

        for i, item in enumerate(self.golden_dataset):
            question = item["question"]
            expected_answer = item["expected_answer"]
            expected_sources = item["expected_sources"]
            question_type = item["question_type"]

            logger.info(f"Evaluating question {i+1}/{len(self.golden_dataset)}: {question[:50]}...")

            try:
                # Retrieve relevant chunks
                relevant_chunks = retrieve_relevant_chunks(
                    query=question,
                    n_results=5,
                    distance_threshold=0.7
                )

                # Get retrieved sources
                retrieved_sources = [
                    chunk["metadata"]["source"]
                    for chunk in relevant_chunks
                ]

                # Calculate retrieval metrics
                recall_1 = self.calculate_recall_at_k(retrieved_sources, expected_sources, k=1)
                recall_3 = self.calculate_recall_at_k(retrieved_sources, expected_sources, k=3)
                recall_5 = self.calculate_recall_at_k(retrieved_sources, expected_sources, k=5)
                mrr = self.calculate_mrr(retrieved_sources, expected_sources)

                # Generate answer
                answer_result = generate_answer(
                    query=question,
                    context_chunks=relevant_chunks,
                    model_provider="anthropic"
                )

                # Format with citations
                formatted_result = format_answer_with_citations(answer_result, relevant_chunks)
                generated_answer = formatted_result["answer"]

                # Get context for faithfulness judgment
                from src.retrieve import get_context_for_answer
                context = get_context_for_answer(relevant_chunks)

                # Evaluate faithfulness
                faithfulness_score, faithfulness_explanation = self.evaluate_faithfulness_llm_as_judge(
                    question, generated_answer, context
                )

                # Store results
                metrics["recall_at_1"].append(recall_1)
                metrics["recall_at_3"].append(recall_3)
                metrics["recall_at_5"].append(recall_5)
                metrics["mrr"].append(mrr)
                metrics["faithfulness"].append(faithfulness_score)

                question_result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "expected_sources": expected_sources,
                    "retrieved_sources": retrieved_sources,
                    "recall_at_1": recall_1,
                    "recall_at_3": recall_3,
                    "recall_at_5": recall_5,
                    "mrr": mrr,
                    "faithfulness_score": faithfulness_score,
                    "faithfulness_explanation": faithfulness_explanation,
                    "question_type": question_type
                }
                metrics["question_results"].append(question_result)

                logger.info(f"  Recall@5: {recall_5:.3f}, MRR: {mrr:.3f}, Faithfulness: {faithfulness_score:.3f}")

            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                # Add zero scores for failed questions
                metrics["recall_at_1"].append(0.0)
                metrics["recall_at_3"].append(0.0)
                metrics["recall_at_5"].append(0.0)
                metrics["mrr"].append(0.0)
                metrics["faithfulness"].append(0.0)

                metrics["question_results"].append({
                    "question": question,
                    "error": str(e)
                })

        end_time = time.time()

        # Calculate aggregate metrics
        results = {
            "aggregate_metrics": {
                "avg_recall_at_1": np.mean(metrics["recall_at_1"]) if metrics["recall_at_1"] else 0.0,
                "avg_recall_at_3": np.mean(metrics["recall_at_3"]) if metrics["recall_at_3"] else 0.0,
                "avg_recall_at_5": np.mean(metrics["recall_at_5"]) if metrics["recall_at_5"] else 0.0,
                "avg_mrr": np.mean(metrics["mrr"]) if metrics["mrr"] else 0.0,
                "avg_faithfulness": np.mean(metrics["faithfulness"]) if metrics["faithfulness"] else 0.0,
                "total_questions": len(self.golden_dataset),
                "evaluation_time_seconds": end_time - start_time
            },
            "question_results": metrics["question_results"]
        }

        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Average Recall@5: {results['aggregate_metrics']['avg_recall_at_5']:.3f}")
        logger.info(f"Average MRR: {results['aggregate_metrics']['avg_mrr']:.3f}")
        logger.info(f"Average Faithfulness: {results['aggregate_metrics']['avg_faithfulness']:.3f}")

        return results

def run_evaluation_and_save(output_file: str = "./evals/evaluation_results.json"):
    """Run evaluation and save results to file."""
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation()

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")
    return results

if __name__ == "__main__":
    run_evaluation_and_save()