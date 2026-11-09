"""
Sparse retriever using BM25 algorithm.

FILE HEADER
===========
Business problem:
  Semantic (dense) search can miss exact keyword matches — for example a question
  about a specific acronym like "TAC" or "IMEI". BM25 is a classical term-frequency
  / inverse-document-frequency scorer that surfaces chunks containing the exact
  query terms, complementing dense retrieval so neither signal has to work alone.

Place in data flow:
  Sits in G4 Retrieval as the sparse half of hybrid retrieval. It is built
  independently of the vector store (no embeddings needed) and is queried by
  HybridRetriever / HybridRetrieverRRF alongside the dense retriever, then the two
  result lists are fused. Input: a list of document dicts (to index) and a query
  string. Output: ranked (document, score) tuples.

What calls it:
  src/retrieval/hybrid.py (HybridRetrieverRRF.retrieve) and
  src/retrieval/retriever.py (HybridRetriever.retrieve) call BM25Retriever.retrieve().
  Ingestion would call add_documents() to build the index from stored chunks.

What it calls:
  Only Python standard library (re, math, collections) — no external services,
  no network, no disk I/O beyond what the caller passes in.

What breaks if removed:
  Sparse/keyword retrieval disappears. Any question answered by an exact term that
  dense similarity under-ranks will return weaker or no results, and hybrid/RRF
  fusion loses its keyword signal entirely.

Tags:
  # GAP: persistence / concurrency — the index lives only in RAM
         (self.documents, self.tokenized_docs). A restart wipes it and there is
         no locking, so concurrent add_documents() / retrieve() calls race.
  # BUSINESS: protects recall for term-specific questions (acronyms, part numbers,
              error codes) that semantic similarity alone often misses.
  # INTERVIEW: say-it-out-loud — "BM25 scores a query against each document using
               term frequency and inverse document frequency. It's the keyword
               counterpart to vector search; combine both for robust retrieval."
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
        WHAT:  Configure BM25 tuning parameters and prepare empty index state.
        WHY:   k1 controls term-frequency saturation (how quickly repeated terms
               stop adding score) and b controls length normalization (how much
               short vs. long documents are favoured). Both are knobs a production
               system would tune against a golden eval set.
        INPUTS: k1 (float): 1.5, b (float): 0.75 — standard BM25 defaults.
        OUTPUTS: None (instance initialized with empty index).
        SIDE EFFECTS: None.
        FAILURE MODES: None at construction.
        ALTERNATIVE: Use a library BM25 (e.g. rank_bm25) for an optimized, tested
                     implementation rather than this educational from-scratch one.
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
        WHAT:  Lowercase the text and split it into alphanumeric tokens.
        WHY:   BM25 operates on tokens, not raw text; normalization (lowercasing,
               punctuation stripping) makes "LTE" and "lte" match the same index
               entry, which matters for telecom acronyms.
        INPUTS: text (str): e.g. "What is the TAC?"
        OUTPUTS: List[str] — e.g. ["what", "is", "the", "tac"]
        SIDE EFFECTS: None (pure function).
        FAILURE MODES: None; re.findall always returns a list (possibly empty).
        ALTERNATIVE: Use a proper tokenizer (spaCy, NLTK, or the LLM's tokenizer)
                     to handle hyphenated terms and domain acronyms better.
        """
        # Convert to lowercase and split on non-alphanumeric characters.
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        WHAT:  Index a batch of documents: tokenize each, record lengths, and
               update per-word document-frequency counts.
        WHY:   BM25's idf component needs to know, for every word, how many
               documents contain it — that statistic is built incrementally here.
        INPUTS: documents (List[Dict[str, Any]]): each dict must have a 'content'
                field holding the chunk text.
        OUTPUTS: None (mutates the index in place).
        SIDE EFFECTS: Extends self.documents, self.tokenized_docs, self.doc_lengths;
                      updates self.word_doc_count and self.total_docs; recomputes
                      self.avg_doc_length; logs an info line (sparse_retriever.py:78).
        FAILURE MODES: A document missing 'content' raises KeyError at
                       sparse_retriever.py:61 — callers must guarantee the field.
        ALTERNATIVE: Rebuild the index from scratch when documents change, since
                     this implementation only supports appending, not updates/deletes.
        """
        self.documents.extend(documents)

        # Tokenize new documents and collect their lengths.
        new_tokenized = []
        new_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc["content"])
            new_tokenized.append(tokens)
            new_lengths.append(len(tokens))

            # Update word document counts (using set to count each word once per doc).
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.word_doc_count[token] += 1

        self.tokenized_docs.extend(new_tokenized)
        self.doc_lengths.extend(new_lengths)
        self.total_docs = len(self.documents)

        # Recalculate average document length used for BM25 length normalization.
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        logger.info(f"Added {len(documents)} documents to BM25 index. Total: {self.total_docs}")

    def get_score(self, query: str, doc_index: int) -> float:
        """
        WHAT:  Compute the BM25 relevance score between a query and one document.
        WHY:   The score combines term frequency (how often a query term appears in
               the doc) with inverse document frequency (how rare the term is across
               the corpus) — rare matching terms score higher.
        INPUTS: query (str): e.g. "TAC handover"; doc_index (int): position in
                self.documents.
        OUTPUTS: float — BM25 score (higher = more relevant; 0.0 if index out of
                 range or no terms match).
        SIDE EFFECTS: None (read-only).
        FAILURE MODES: Returns 0.0 for an out-of-range doc_index rather than
                       raising (sparse_retriever.py:91), which silently hides a
                       caller bug instead of surfacing it.
        ALTERNATIVE: Use a vectorized BM25 implementation for batch scoring of
                     many documents at once.
        """
        if doc_index >= len(self.documents):
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_index]
        doc_length = self.doc_lengths[doc_index]

        # Count term frequencies in the document.
        term_freq = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freq:
                continue

            # BM25 formula components.
            tf = term_freq[term]
            df = self.word_doc_count[term]
            # idf: rarer terms (low df) get a bigger boost.
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 score component: tf scaled by saturation and length normalization.
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Score every indexed document for the query and return the top k.
        WHY:   Retrieval needs a ranked candidate list; scoring all documents is
               simple and exact (no approximate index), fine for modest corpora.
        INPUTS: query (str): e.g. "eSIM provisioning steps"; k (int): 5.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] sorted by score descending.
        SIDE EFFECTS: None (read-only).
        FAILURE MODES: Returns [] when no documents are indexed (sparse_retriever.py:130)
                       instead of raising — an empty index is treated as "no results".
        ALTERNATIVE: For large corpora, use an inverted index so only documents
                     sharing query terms are scored, rather than scanning all.
        """
        if not self.documents:
            return []

        # Calculate scores for all documents.
        scores = []
        for i in range(len(self.documents)):
            score = self.get_score(query, i)
            scores.append((self.documents[i], score))

        # Sort by score descending and return top k.
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# Convenience function
def create_bm25_retriever(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """
    WHAT:  Build and return a BM25Retriever instance.
    WHY:   Factory convenience so callers don't construct and configure manually.
    INPUTS: k1 (float): 1.5, b (float): 0.75.
    OUTPUTS: BM25Retriever.
    SIDE EFFECTS: None.
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate BM25Retriever(k1, b) directly.
    """
    return BM25Retriever(k1=k1, b=b)
