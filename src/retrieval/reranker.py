"""
Document reranking utilities for improving retrieval quality.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    """Reranker using cross-encoder models for improved relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available for reranking. Using mock reranker.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str,
              documents: List[Tuple[Dict[str, Any], float]],
              k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents based on query-document relevance.

        Args:
            query: Query string
            documents: List of tuples (document, initial_score)
            k: Number of top documents to return (if None, return all)

        Returns:
            List of tuples (document, rerank_score) sorted by rerank score descending
        """
        if not documents:
            return []

        if self.model is None:
            # Return original scores if no model available
            logger.warning("No reranker model available, returning original scores")
            return documents[:k] if k else documents

        try:
            # Prepare query-document pairs
            pairs = [[query, doc["content"]] for doc, _ in documents]

            # Get reranker scores
            rerank_scores = self.model.predict(pairs)

            # Create scored document list
            scored_documents = []
            for (doc, _), score in zip(documents, rerank_scores):
                scored_documents.append((doc, float(score)))

            # Sort by rerank score descending
            scored_documents.sort(key=lambda x: x[1], reverse=True)

            # Return top k if specified
            if k is not None:
                scored_documents = scored_documents[:k]

            logger.info(f"Reranked {len(documents)} documents, returning {len(scored_documents)}")
            return scored_documents

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original scores
            return documents[:k] if k else documents

class MetadataReranker:
    """Reranker that boosts scores based on metadata preferences."""

    def __init__(self, boost_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata reranker.

        Args:
            boost_rules: Dictionary mapping metadata keys to boost values
                        e.g., {"source": {"policy.pdf": 1.2}, "type": {"policy": 1.1}}
        """
        self.boost_rules = boost_rules or {}

    def rerank(self, documents: List[Tuple[Dict[str, Any], float]],
              k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents based on metadata boosts.

        Args:
            documents: List of tuples (document, score)
            k: Number of top documents to return (if None, return all)

        Returns:
            List of tuples (document, boosted_score) sorted by boosted score descending
        """
        if not documents:
            return []

        boosted_documents = []
        for doc, score in documents:
            boosted_score = score

            # Apply metadata boosts
            metadata = doc.get("metadata", {})
            for key, rules in self.boost_rules.items():
                if key in metadata:
                    value = metadata[key]
                    if isinstance(rules, dict) and value in rules:
                        boosted_score *= rules[value]
                    elif not isinstance(rules, dict):
                        # Assume rules is a multiplier
                        boosted_score *= rules

            boosted_documents.append((doc, boosted_score))

        # Sort by boosted score descending
        boosted_documents.sort(key=lambda x: x[1], reverse=True)

        # Return top k if specified
        if k is not None:
            boosted_documents = boosted_documents[:k]

        return boosted_documents

# Convenience functions
def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """Create a reranker instance."""
    return Reranker(model_name)

def create_metadata_reranker(boost_rules: Optional[Dict[str, Any]] = None) -> MetadataReranker:
    """Create a metadata reranker instance."""
    return MetadataReranker(boost_rules)