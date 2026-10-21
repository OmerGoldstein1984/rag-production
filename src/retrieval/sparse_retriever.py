"""
Sparse retriever using BM25 algorithm.
"""

from typing import List, Dict, Any, Tuple
import re
import math
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class BM25Retriever:
    """BM25 based sparse retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 retriever.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = []  # List of document dictionaries
        self.tokenized_docs = []  # List of tokenized documents
        self.doc_lengths = []  # List of document lengths
        self.avg_doc_length = 0.0  # Average document length
        self.word_doc_count = defaultdict(int)  # Number of documents containing each word
        self.total_docs = 0  # Total number of documents

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on non-alphanumeric characters.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of document dictionaries with 'content' field
        """
        self.documents.extend(documents)

        # Tokenize new documents
        new_tokenized = []
        new_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc["content"])
            new_tokenized.append(tokens)
            new_lengths.append(len(tokens))

            # Update word document counts (using set to count each word once per doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.word_doc_count[token] += 1

        self.tokenized_docs.extend(new_tokenized)
        self.doc_lengths.extend(new_lengths)
        self.total_docs = len(self.documents)

        # Recalculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        logger.info(f"Added {len(documents)} documents to BM25 index. Total: {self.total_docs}")

    def get_score(self, query: str, doc_index: int) -> float:
        """
        Calculate BM25 score for a query and document.

        Args:
            query: Query string
            doc_index: Index of document in the collection

        Returns:
            BM25 score
        """
        if doc_index >= len(self.documents):
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_index]
        doc_length = self.doc_lengths[doc_index]

        # Count term frequencies in the document
        term_freq = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freq:
                continue

            # BM25 formula components
            tf = term_freq[term]
            df = self.word_doc_count[term]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top k documents for a query using BM25.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of tuples (document, score) sorted by score descending
        """
        if not self.documents:
            return []

        # Calculate scores for all documents
        scores = []
        for i in range(len(self.documents)):
            score = self.get_score(query, i)
            scores.append((self.documents[i], score))

        # Sort by score descending and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# Convenience function
def create_bm25_retriever(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """Create a BM25 retriever instance."""
    return BM25Retriever(k1=k1, b=b)