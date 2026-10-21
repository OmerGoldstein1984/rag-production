"""
Hybrid retrieval using BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np

from .sparse_retriever import BM25Retriever
from .reranker import Reranker

logger = logging.getLogger(__name__)


class HybridRetrieverRRF:
    """Hybrid retriever that combines BM25 and dense retrieval using Reciprocal Rank Fusion."""

    def __init__(self, dense_retriever: Any, sparse_retriever: BM25Retriever,
                 k: int = 60, weight_dense: float = 0.5, weight_sparse: float = 0.5):
        """
        Initialize the hybrid retriever with RRF.

        Args:
            dense_retriever: Dense retrieval method (vector store based)
            sparse_retriever: Sparse retrieval method (BM25 based)
            k: RRF parameter (default 60) - controls influence of lower-ranked items
            weight_dense: Weight for dense retrieval scores in final combination
            weight_sparse: Weight for sparse retrieval scores in final combination
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.k = k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

        logger.info(f"Initialized HybridRetrieverRRF with k={k}, "
                   f"dense_weight={weight_dense}, sparse_weight={weight_sparse}")

    def retrieve(self, query: str, k: int = 5,
                metadata_filters: Optional[Dict[str, Any]] = None,
                similarity_threshold: float = 0.0,
                current_version_only: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents using hybrid search with RRF.

        Args:
            query: Query string
            k: Number of documents to retrieve
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            similarity_threshold: Minimum similarity score threshold (0-1)
            current_version_only: Whether to filter to current document versions only

        Returns:
            List of tuples (document, fused_score) sorted by fused score descending
        """
        # Get dense results (get more for better fusion)
        search_k = max(k * 3, 10)  # Get extra for RRF to work well
        dense_results = self.dense_retriever.retrieve(
            query, k=search_k,
            metadata_filters=metadata_filters,
            similarity_threshold=0.0,  # Apply threshold after fusion
            current_version_only=current_version_only
        )

        # Get sparse results
        sparse_results = self.sparse_retriever.retrieve(query, k=search_k)

        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion([dense_results, sparse_results])

        # Apply similarity threshold if specified
        if similarity_threshold > 0.0:
            fused_results = [(doc, score) for doc, score in fused_results
                           if score >= similarity_threshold]

        # Return top k results
        return fused_results[:k]

    def _reciprocal_rank_fusion(self, rank_lists: List[List[Tuple[Dict[str, Any], float]]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            rank_lists: List of ranked document lists (each element is (doc, score))

        Returns:
            Fused ranked list of documents (doc, score)
        """
        doc_scores = {}
        doc_map = {}  # Map doc_id back to document object

        for rank_list in rank_lists:
            for rank, (doc, _) in enumerate(rank_list):
                # Create a stable ID for the document based on its content
                # In a real system, you'd use a proper document ID
                doc_id = id(doc)  # Using object ID for simplicity
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                # RRF formula: 1 / (k + rank)
                doc_scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Sort by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents in order with their fused scores
        result = []
        for doc_id, score in sorted_docs:
            result.append((doc_map[doc_id], score))

        return result


def create_hybrid_retriever_rrf(dense_retriever: Any, sparse_retriever: BM25Retriever,
                               k: int = 60, weight_dense: float = 0.5, weight_sparse: float = 0.5) -> HybridRetrieverRRF:
    """
    Create a hybrid retriever instance with RRF.

    Args:
        dense_retriever: Dense retrieval method (vector store based)
        sparse_retriever: Sparse retrieval method (BM25 based)
        k: RRF parameter (default 60)
        weight_dense: Weight for dense retrieval scores
        weight_sparse: Weight for sparse retrieval scores

    Returns:
        HybridRetrieverRRF instance
    """
    return HybridRetrieverRRF(dense_retriever, sparse_retriever, k, weight_dense, weight_sparse)