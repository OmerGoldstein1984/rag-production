"""
Document retrieval utilities for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieves relevant documents from a vector store."""

    def __init__(self, vector_store: Any, embedder: Any):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store instance for searching
            embedder: Embedder instance for query embedding
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5,
                metadata_filters: Optional[Dict[str, Any]] = None,
                similarity_threshold: float = 0.0,
                current_version_only: bool = False) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve k most relevant documents for a query with optional filtering.

        Args:
            query: Query string
            k: Number of documents to retrieve
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            similarity_threshold: Minimum similarity score threshold (0-1)
            current_version_only: If True, exclude chunks whose metadata marks them
                as non-current (``is_current == False``)

        Returns:
            List of tuples (document, similarity_score) that meet filters and threshold
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        if current_version_only and metadata_filters is None:
            metadata_filters = {"is_current": True}

        # Search the vector store with metadata filters
        results = self.vector_store.search(
            query_vector=query_embedding,
            k=k*3 if metadata_filters else k,  # Get extra if filtering to account for removed items
            filter_metadata=metadata_filters
        )

        # Apply similarity threshold
        if similarity_threshold > 0.0:
            results = [(doc, score) for doc, score in results if score >= similarity_threshold]

        # Limit to k results
        results = results[:k]

        logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...' "
                   f"(threshold: {similarity_threshold}, filters: {bool(metadata_filters)})")
        return results

    def retrieve_with_version_filter(self, query: str,
                                   k: int = 5,
                                   current_version_only: bool = True,
                                   similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents with version filtering (useful for policy documents).

        Args:
            query: Query string
            k: Number of documents to retrieve
            current_version_only: If True, only return documents with latest version
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (document, similarity_score)
        """
        if current_version_only:
            # Keep only chunks explicitly marked as the current version.
            # (Old versions carry is_current=False in metadata.)
            return self.retrieve(query, k, {"is_current": True}, similarity_threshold)
        else:
            return self.retrieve(query, k, similarity_threshold=similarity_threshold)

    def retrieve_by_department(self, query: str,
                              department: str,
                              k: int = 5,
                              similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents filtered by department.

        Args:
            query: Query string
            department: Department name to filter by (e.g., "Human Resources")
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (document, similarity_score)
        """
        metadata_filters = {"department": department}
        return self.retrieve(query, k, metadata_filters, similarity_threshold)


class HybridRetriever:
    """Combines dense and sparse retrieval methods."""

    def __init__(self, dense_retriever: Retriever, sparse_retriever: Any = None):
        """
        Initialize the hybrid retriever.

        Args:
            dense_retriever: Dense retrieval method (vector store based)
            sparse_retriever: Sparse retrieval method (BM25 based, optional)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(self, query: str,
                k: int = 5,
                dense_weight: float = 0.7,
                sparse_weight: float = 0.3,
                metadata_filters: Optional[Dict[str, Any]] = None,
                similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query string
            k: Number of documents to retrieve
            dense_weight: Weight for dense retrieval scores (0-1)
            sparse_weight: Weight for sparse retrieval scores (0-1)
            metadata_filters: Metadata filters to apply
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (document, combined_score)
        """
        # Get dense results
        search_k = k * 3 if metadata_filters else k * 2
        dense_results = self.dense_retriever.retrieve(
            query, k=search_k,
            metadata_filters=metadata_filters,
            similarity_threshold=0.0  # We'll apply threshold after combining
        )

        # If no sparse retriever, return dense results
        if self.sparse_retriever is None:
            results = dense_results[:k]
            if similarity_threshold > 0.0:
                results = [(doc, score) for doc, score in results if score >= similarity_threshold]
            return results

        # Get sparse results
        sparse_results = self.sparse_retriever.retrieve(query, k=search_k)

        # Combine scores
        doc_scores = {}

        # Add dense scores
        for doc, score in dense_results:
            doc_id = id(doc)  # Use object ID as temporary identifier
            doc_scores[doc_id] = {
                "document": doc,
                "dense_score": score,
                "sparse_score": 0.0
            }

        # Add sparse scores
        for doc, score in sparse_results:
            doc_id = id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]["sparse_score"] = score
            else:
                doc_scores[doc_id] = {
                    "document": doc,
                    "dense_score": 0.0,
                    "sparse_score": score
                }

        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = (dense_weight * scores["dense_score"] +
                            sparse_weight * scores["sparse_score"])
            combined_results.append((scores["document"], combined_score))

        # Sort by combined score descending
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Apply similarity threshold and return top k
        if similarity_threshold > 0.0:
            combined_results = [(doc, score) for doc, score in combined_results if score >= similarity_threshold]

        return combined_results[:k]


def _doc_key(doc: Dict[str, Any]) -> str:
    """Stable identity for a retrieved chunk, independent of object identity.

    Different retrievers (dense vs BM25) return distinct dict objects for the
    same underlying chunk, so id() never matches across lists. Prefer an
    explicit doc_id, then source+chunk_index, then a content hash.
    """
    meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
    if meta.get("doc_id"):
        return str(meta["doc_id"])
    if meta.get("source") and meta.get("chunk_index") is not None:
        return f"{meta['source']}#{meta['chunk_index']}"
    content = doc.get("content", "") if isinstance(doc, dict) else repr(doc)
    import hashlib
    return "h:" + hashlib.md5(content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(rank_lists: List[List[Tuple[Dict[str, Any], float]]],
                          k: int = 60) -> List[Tuple[Dict[str, Any], float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rank_lists: List of ranked document lists (each element is (doc, score))
        k: RRF parameter (default 60)

    Returns:
        Fused ranked list of documents (doc, score)
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank_list in rank_lists:
        for rank, (doc, _) in enumerate(rank_list):
            key = _doc_key(doc)
            if key not in doc_scores:
                doc_scores[key] = 0.0
                doc_map[key] = doc
            doc_scores[key] += 1.0 / (k + rank + 1)

    # Sort by score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[key], score) for key, score in sorted_docs]


# Convenience function
def create_retriever(vector_store: Any, embedder: Any) -> Retriever:
    """Create a retriever instance."""
    return Retriever(vector_store, embedder)


def create_hybrid_retriever(dense_retriever: Retriever,
                           sparse_retriever: Any = None) -> HybridRetriever:
    """Create a hybrid retriever instance."""
    return HybridRetriever(dense_retriever, sparse_retriever)