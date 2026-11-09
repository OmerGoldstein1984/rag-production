"""
Hybrid retrieval using BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF).

FILE HEADER
===========
Business problem:
  Dense (semantic) and sparse (keyword) search each have blind spots. Hybrid
  retrieval fuses both ranked lists with Reciprocal Rank Fusion so a question is
  answered by whichever signal — meaning or exact terms — surfaces the right chunk.

Place in data flow:
  Sits in G4 Retrieval, called after the dense and sparse retrievers are built and
  before re-ranking / answer generation. Input: a query string (+ optional filters).
  Output: fused ranked (document, score) tuples.

What calls it:
  The retrieval orchestration layer (agent/tools/chat in G5) calls
  HybridRetrieverRRF.retrieve() or create_hybrid_retriever_rrf().

What it calls:
  dense_retriever.retrieve(...) (src/retrieval/retriever.py),
  sparse_retriever.retrieve(...) (src/retrieval/sparse_retriever.py),
  _doc_key(...) (src/retrieval/retriever.py:203) to match the same chunk across lists.

What breaks if removed:
  Hybrid/RRF retrieval disappears — callers fall back to dense-only or sparse-only,
  losing recall for questions that need both semantic and keyword matching.

Tags:
  # GAP: observability — no metrics track fusion score distribution, per-list
         contribution, or latency; you cannot tell whether dense or sparse is
         driving answers in production.
  # BUSINESS: protects recall — fusing both signals means neither a purely
              semantic nor a purely keyword question leaves the system empty-handed.
  # INTERVIEW: say-it-out-loud — "RRF fuses ranked lists by position, not raw
               score, so dense cosine scores and BM25 scores — which live on
               different scales — combine fairly without re-normalisation."
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np

from .sparse_retriever import BM25Retriever
from .reranker import Reranker
from .retriever import _doc_key

logger = logging.getLogger(__name__)


class HybridRetrieverRRF:
    """Hybrid retriever that combines BM25 and dense retrieval using Reciprocal Rank Fusion."""

    def __init__(self, dense_retriever: Any, sparse_retriever: BM25Retriever,
                 k: int = 60, weight_dense: float = 0.5, weight_sparse: float = 0.5):
        """
        WHAT:  Wire up the dense and sparse retrievers plus RRF tuning parameters.
        WHY:   The dense retriever finds conceptually related chunks, the sparse
               retriever finds exact-term matches, and RRF fuses their rankings
               without requiring comparable score scales.
        INPUTS: dense_retriever (Any): object with a retrieve() method.
                sparse_retriever (BM25Retriever): the BM25 retriever.
                k (int): 60 — RRF parameter dampening lower-ranked items.
                weight_dense (float): 0.5, weight_sparse (float): 0.5 — weights
                stored on the instance (note: the current _reciprocal_rank_fusion
                does not actually apply these weights — see GAP note).
        OUTPUTS: None (instance initialized).
        SIDE EFFECTS: Logs an info line with the configuration (hybrid.py:37).
        FAILURE MODES: None at construction.
        ALTERNATIVE: Use the simpler weighted-score HybridRetriever (retriever.py)
                     if RRF's position-based fusion is not needed.
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
        WHAT:  Retrieve dense and sparse results for the query, fuse them with RRF,
               optionally threshold, and return the top k.
        WHY:   Combining both signals gives better recall than either alone; RRF
               merges rankings fairly despite different score scales.
        INPUTS: query (str): e.g. "What is the TAC in LTE?"
                k (int): 5 — results to return.
                metadata_filters (Optional[Dict[str, Any]]): applied to dense only.
                similarity_threshold (float): 0.0 — applied to fused scores.
                current_version_only (bool): True — passed to dense retrieve.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] of (document, fused_score),
                 sorted descending, capped at k.
        SIDE EFFECTS: None (read-only retrieval).
        FAILURE MODES: Propagates exceptions from dense/sparse retrieve(); no retry.
                       similarity_threshold is compared against RRF scores (which are
                       small position-based values, not 0-1 similarity), so a
                       non-zero threshold may filter out everything — a semantics trap.
        ALTERNATIVE: Use HybridRetriever.retrieve() (weighted blending) when you
                     want explicit score weighting rather than position-based fusion.
        """
        # Get dense results (get more for better fusion).
        search_k = max(k * 3, 10)  # Get extra for RRF to work well
        dense_results = self.dense_retriever.retrieve(
            query, k=search_k,
            metadata_filters=metadata_filters,
            similarity_threshold=0.0,  # Apply threshold after fusion
            current_version_only=current_version_only
        )

        # Get sparse results.
        sparse_results = self.sparse_retriever.retrieve(query, k=search_k)

        # Apply Reciprocal Rank Fusion.
        fused_results = self._reciprocal_rank_fusion([dense_results, sparse_results])

        # Apply similarity threshold if specified.
        if similarity_threshold > 0.0:
            fused_results = [(doc, score) for doc, score in fused_results
                             if score >= similarity_threshold]

        # Return top k results.
        return fused_results[:k]

    def _reciprocal_rank_fusion(self, rank_lists: List[List[Tuple[Dict[str, Any], float]]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Fuse multiple ranked document lists into one using Reciprocal Rank Fusion.
        WHY:   RRF assigns each document 1/(k+rank+1) per list and sums, so a chunk
               ranked highly by either dense or sparse rises to the top even if its
               raw score in the other list is low.
        INPUTS: rank_lists (List[List[Tuple[Dict[str, Any], float]]]): e.g.
                [[dense_results], [sparse_results]].
        OUTPUTS: List[Tuple[Dict[str, Any], float]] fused by descending RRF score.
        SIDE EFFECTS: None (read-only).
        FAILURE MODES: Empty lists yield an empty result; never raises.
        ALTERNATIVE: Weighted score blending (retriever.py HybridRetriever) is
                     simpler but needs scores on comparable scales.
        """
        doc_scores = {}
        doc_map = {}  # Map stable doc key back to document object

        for rank_list in rank_lists:
            for rank, (doc, _) in enumerate(rank_list):
                # Stable key (doc_id / source#chunk / content hash) so the same
                # chunk returned by dense AND sparse lists is fused correctly.
                key = _doc_key(doc)
                if key not in doc_scores:
                    doc_scores[key] = 0.0
                    doc_map[key] = doc
                # RRF formula: 1 / (k + rank + 1)
                doc_scores[key] += 1.0 / (self.k + rank + 1)

        # Sort by score descending.
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents in order with their fused scores.
        return [(doc_map[key], score) for key, score in sorted_docs]


def create_hybrid_retriever_rrf(dense_retriever: Any, sparse_retriever: BM25Retriever,
                               k: int = 60, weight_dense: float = 0.5, weight_sparse: float = 0.5) -> HybridRetrieverRRF:
    """
    WHAT:  Build and return a HybridRetrieverRRF instance.
    WHY:   Factory convenience for the RRF-based hybrid retriever.
    INPUTS: dense_retriever (Any), sparse_retriever (BM25Retriever), k (int): 60,
            weight_dense (float): 0.5, weight_sparse (float): 0.5.
    OUTPUTS: HybridRetrieverRRF.
    SIDE EFFECTS: None.
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate HybridRetrieverRRF(...) directly.
    """
    return HybridRetrieverRRF(dense_retriever, sparse_retriever, k, weight_dense, weight_sparse)
