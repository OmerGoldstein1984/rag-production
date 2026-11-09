"""
Document retrieval utilities for RAG systems.

FILE HEADER
===========
Business problem:
  Given a user's natural-language question (e.g. an FDE interview candidate asking
  "How does LTE handover work?"), find the most relevant chunks from the ingested
  telecom document corpus so an LLM can ground its answer. Without this layer the
  system would have no way to locate source material and would either hallucinate
  or be forced to scan the whole corpus on every question.

Place in data flow:
  Sits in G4 Retrieval, after ingestion has embedded and stored chunks
  (src/ingestion/*) and before generation answers the user (src/generation/*).
  Input: a query string + optional filters. Output: a ranked list of
  (document_dict, similarity_score) tuples.

What calls it:
  The retrieval orchestration layer (agent/tools/chat in G5) calls retrieve()
  and the hybrid/RRF variants to fetch candidate chunks for an answer.

What it calls:
  embedder.embed_text(query) to turn the query into a vector;
  vector_store.search(...) to run the nearest-neighbour lookup.

What breaks if removed:
  Retrieval collapses: no chunks are surfaced for the question, so answers lose
  their grounding and the system cannot cite document sections.

Tags:
  # GAP: async / retries / backoff — retrieve() is synchronous and makes a single
         vector-store call with no retry, so a transient store outage fails the
         whole request instead of backing off. A production system would retry
         with exponential backoff and surface a degraded mode.
  # BUSINESS: protects answer accuracy — retrieved chunks are the evidence an
              LLM cites; wrong or missing chunks mean wrong interview answers.
  # INTERVIEW: say-it-out-loud — "Retrieval turns a question into a vector, asks
               the vector store for nearest neighbours, then trims by score and
               count. If you remove it, the RAG system has no evidence to ground
               answers."
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant documents from a vector store.

    WHAT:  Given a query string, embeds it and asks the vector store for the
           nearest-neighbour chunks, then applies an optional similarity threshold
           and caps the result at k documents.
    WHY:   The RAG system needs ranked source chunks to ground answers; raw
           vector-store results need thresholding and trimming before they are
           safe to hand to a generator.
    INPUTS:
        query (str): e.g. "What is the tracking area code in LTE?"
        k (int): 5 — how many chunks to return.
        metadata_filters (Optional[Dict[str, Any]]): e.g. {"department": "Human Resources"}
            or {"is_current": True}
        similarity_threshold (float): 0.0 — drop chunks below this score.
        current_version_only (bool): False — when True, only return chunks flagged
            is_current == True in metadata.
    OUTPUTS: List[Tuple[Dict[str, Any], float]] — e.g.
        [({"content": "TAC ...", "metadata": {...}}, 0.87),
         ({"content": "...", "metadata": {...}}, 0.81)]
    SIDE EFFECTS: Logs an info line with the number of results and the query
                  prefix (retriever.py:63). No mutation of inputs.
    FAILURE MODES: If embedder.embed_text() raises, the exception propagates
                   (no retry). If vector_store.search() returns fewer than k,
                   fewer results come back — callers must handle empty lists.
    ALTERNATIVE: Use HybridRetriever (dense + BM25) or HybridRetrieverRRF when
                 lexical keyword matching matters as much as semantic similarity.
    """

    def __init__(self, vector_store: Any, embedder: Any):
        """
        WHAT:  Store the vector store and embedder this retriever will use.
        WHY:   Dependencies are injected so the same class works with any
               vector-store / embedder implementation (testable with fakes).
        INPUTS: vector_store (Any): object exposing search(query_vector=..., k=...,
               filter_metadata=...)
                embedder (Any): object exposing embed_text(text) -> vector.
        OUTPUTS: None (instance is initialized).
        SIDE EFFECTS: None.
        FAILURE MODES: None at construction; failures surface later in retrieve().
        ALTERNATIVE: Subclass and override retrieve() to use a different search
                     strategy (e.g. a SQL full-text index).
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5,
                 metadata_filters: Optional[Dict[str, Any]] = None,
                 similarity_threshold: float = 0.0,
                 current_version_only: bool = False) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Return the k most relevant documents for a query, optionally
               filtered by metadata and a similarity threshold.
        WHY:   Retrieval must surface evidence chunks and discard low-score noise
               so the generator only sees relevant context.
        INPUTS:
            query (str): e.g. "How does eSIM provisioning work?"
            k (int): 5.
            metadata_filters (Optional[Dict[str, Any]]): e.g. {"is_current": True}
            similarity_threshold (float): 0.0 (0-1 range expected).
            current_version_only (bool): False.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] of (document, score) meeting
                 filters and threshold, capped at k.
        SIDE EFFECTS: Logs an info line (retriever.py:63).
        FAILURE MODES: Propagates exceptions from embedder/vector store; no retry,
                       no circuit breaker — a store outage fails the request.
        ALTERNATIVE: Call retrieve_with_version_filter() or retrieve_by_department()
                     for the common filtered cases instead of passing filters manually.
        """
        # Embed the query into a vector so the store can run nearest-neighbour search.
        query_embedding = self.embedder.embed_text(query)

        # When only current versions are wanted and no filters were supplied, inject
        # an is_current=True filter so stale document versions are excluded.
        if current_version_only and metadata_filters is None:
            metadata_filters = {"is_current": True}

        # Search the vector store; fetch 3x k when filtering because filtered-out
        # items would otherwise leave us with fewer than k survivors.
        results = self.vector_store.search(
            query_vector=query_embedding,
            k=k*3 if metadata_filters else k,  # Get extra if filtering to account for removed items
            filter_metadata=metadata_filters
        )

        # Apply similarity threshold: drop any chunk whose score is below the bar.
        if similarity_threshold > 0.0:
            results = [(doc, score) for doc, score in results if score >= similarity_threshold]

        # Cap the list at k results for the caller.
        results = results[:k]

        logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...' "
                   f"(threshold: {similarity_threshold}, filters: {bool(metadata_filters)})")
        return results

    def retrieve_with_version_filter(self, query: str,
                                     k: int = 5,
                                     current_version_only: bool = True,
                                     similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Retrieve documents, defaulting to only the latest/current version
               of each document (useful for policy documents).
        WHY:   Policy documents get versioned; old versions carry is_current=False
               and must not shadow the current answer.
        INPUTS: query (str), k (int): 5, current_version_only (bool): True,
                similarity_threshold (float): 0.0.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] — see retrieve().
        SIDE EFFECTS: Delegates to retrieve(); logs via that path.
        FAILURE MODES: Same as retrieve(); no extra failure surface.
        ALTERNATIVE: Call retrieve() directly with {"is_current": True} in
                     metadata_filters to avoid this wrapper.
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
        WHAT:  Retrieve documents restricted to a single department.
        WHY:   Telecom knowledge is split by department (HR vs. Roaming vs. SIM
               Security); a roaming query should not return HR policy chunks.
        INPUTS: query (str), department (str): e.g. "Human Resources", k (int): 5,
                similarity_threshold (float): 0.0.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] — see retrieve().
        SIDE EFFECTS: Delegates to retrieve().
        FAILURE MODES: Same as retrieve(); an unknown department simply yields no
                       matches rather than an error.
        ALTERNATIVE: Pass metadata_filters={"department": ...} to retrieve()
                     directly.
        """
        metadata_filters = {"department": department}
        return self.retrieve(query, k, metadata_filters, similarity_threshold)


class HybridRetriever:
    """Combines dense and sparse retrieval methods."""

    def __init__(self, dense_retriever: Retriever, sparse_retriever: Any = None):
        """
        WHAT:  Hold a dense retriever and an optional sparse (BM25) retriever.
        WHY:   Dense (semantic) search finds conceptually related chunks; sparse
               (keyword) search finds exact-term matches — combining them covers
               both kinds of relevance.
        INPUTS: dense_retriever (Retriever): vector-store backed retriever.
                sparse_retriever (Any): BM25-style retriever with a retrieve()
                        method, or None to use dense only.
        OUTPUTS: None.
        SIDE EFFECTS: None.
        FAILURE MODES: None at construction.
        ALTERNATIVE: Use HybridRetrieverRRF (hybrid.py) which fuses with RRF
                     instead of weighted score blending.
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
        WHAT:  Retrieve using both dense and sparse search, blend the scores by
               weight, and return the top k by combined score.
        WHY:   Some questions are answered by keyword hits (e.g. a specific acronym)
               that semantic similarity alone misses; weighting both signals
               improves recall.
        INPUTS:
            query (str): e.g. "What is the TAC?"
            k (int): 5.
            dense_weight (float): 0.7 — importance of the dense score.
            sparse_weight (float): 0.3 — importance of the sparse score.
            metadata_filters (Optional[Dict[str, Any]]): filters applied to dense.
            similarity_threshold (float): 0.0.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] of (document, combined_score).
        SIDE EFFECTS: None (read-only retrieval).
        FAILURE MODES: If sparse_retriever is None it silently falls back to dense
                       only (by design). Uses id(doc) as a temporary key, which
                       does NOT match the same chunk across dense and sparse lists
                       (see _doc_key note) — so cross-list de-duplication is weak
                       here; prefer HybridRetrieverRRF for correct fusion.
        ALTERNATIVE: HybridRetrieverRRF with Reciprocal Rank Fusion handles
                     cross-retriever identity correctly via _doc_key.
        """
        # Fetch more than k when filtering so that removed items don't starve the result set.
        search_k = k * 3 if metadata_filters else k * 2
        dense_results = self.dense_retriever.retrieve(
            query, k=search_k,
            metadata_filters=metadata_filters,
            similarity_threshold=0.0  # We'll apply threshold after combining
        )

        # If no sparse retriever was configured, just return the dense top k.
        if self.sparse_retriever is None:
            results = dense_results[:k]
            if similarity_threshold > 0.0:
                results = [(doc, score) for doc, score in results if score >= similarity_threshold]
            return results

        # Get sparse (keyword) results for the same query.
        sparse_results = self.sparse_retriever.retrieve(query, k=search_k)

        # Combine scores per document, keyed by object id (see GAP note above).
        doc_scores = {}

        # Add dense scores to the running tally.
        for doc, score in dense_results:
            doc_id = id(doc)  # Use object ID as temporary identifier
            doc_scores[doc_id] = {
                "document": doc,
                "dense_score": score,
                "sparse_score": 0.0
            }

        # Add sparse scores, merging into entries that also appear in dense.
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

        # Calculate combined scores as a weighted sum of dense and sparse scores.
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = (dense_weight * scores["dense_score"] +
                              sparse_weight * scores["sparse_score"])
            combined_results.append((scores["document"], combined_score))

        # Sort by combined score descending so the best blended chunk comes first.
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Apply similarity threshold and return top k.
        if similarity_threshold > 0.0:
            combined_results = [(doc, score) for doc, score in combined_results if score >= similarity_threshold]

        return combined_results[:k]


def _doc_key(doc: Dict[str, Any]) -> str:
    """
    WHAT:  Produce a stable identity string for a retrieved chunk.
    WHY:   Dense and sparse retrievers return distinct dict objects for the same
           underlying chunk, so Python's id() never matches across lists — we need
           an explicit key (doc_id, then source#chunk_index, then content hash) to
           recognise the same chunk and fuse its scores correctly.
    INPUTS: doc (Dict[str, Any]): a retrieved document with optional metadata.
    OUTPUTS: str — e.g. "doc_123", "policy.pdf#5", or "h:<md5 hexdigest>".
    SIDE EFFECTS: Imports hashlib lazily inside the fallback branch (retriever.py:216).
    FAILURE MODES: Never raises; falls back to an MD5 content hash if no identifying
                   metadata is present, so distinct contents get distinct keys.
    ALTERNATIVE: If every chunk carried a guaranteed-unique doc_id in metadata,
                 this helper would be unnecessary.
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
    WHAT:  Fuse several ranked document lists into one using Reciprocal Rank Fusion.
    WHY:   RRF ranks by position rather than raw score, so it fairly combines lists
           scored on different scales (dense cosine vs. BM25) without re-normalising.
    INPUTS:
        rank_lists (List[List[Tuple[Dict[str, Any], float]]]): e.g. [[dense_results], [sparse_results]]
        k (int): 60 — RRF parameter that dampens the influence of lower-ranked items.
    OUTPUTS: List[Tuple[Dict[str, Any], float]] fused by descending RRF score.
    SIDE EFFECTS: None.
    FAILURE MODES: Empty input lists yield an empty result; never raises.
    ALTERNATIVE: Weighted score blending (HybridRetriever.retrieve) is simpler but
                 requires scores to be on comparable scales.
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank_list in rank_lists:
        for rank, (doc, _) in enumerate(rank_list):
            key = _doc_key(doc)
            if key not in doc_scores:
                doc_scores[key] = 0.0
                doc_map[key] = doc
            # RRF formula: 1 / (k + rank + 1) — earlier ranks contribute more.
            doc_scores[key] += 1.0 / (k + rank + 1)

    # Sort by score descending so the highest-fused chunk comes first.
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[key], score) for key, score in sorted_docs]


# Convenience function
def create_retriever(vector_store: Any, embedder: Any) -> Retriever:
    """
    WHAT:  Build and return a Retriever instance.
    WHY:   Centralises construction so callers don't wire dependencies manually.
    INPUTS: vector_store (Any), embedder (Any).
    OUTPUTS: Retriever.
    SIDE EFFECTS: None.
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate Retriever(vector_store, embedder) directly.
    """
    return Retriever(vector_store, embedder)


def create_hybrid_retriever(dense_retriever: Retriever,
                           sparse_retriever: Any = None) -> HybridRetriever:
    """
    WHAT:  Build and return a HybridRetriever instance.
    WHY:   Factory convenience for the dense+sparse blended retriever.
    INPUTS: dense_retriever (Retriever), sparse_retriever (Any): optional BM25 retriever.
    OUTPUTS: HybridRetriever.
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: Instantiate HybridRetriever(...) directly.
    """
    return HybridRetriever(dense_retriever, sparse_retriever)
