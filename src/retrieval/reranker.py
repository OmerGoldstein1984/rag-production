"""
Document reranking utilities for improving retrieval quality.

FILE HEADER
===========
Business problem:
  Initial retrieval returns a ranked candidate list, but the top results are not
  always the most useful. A reranker re-scores query–document pairs with a more
  expressive model (a cross-encoder) so the most relevant chunks rise to the top
  before the generator reads them.

Place in data flow:
  Sits in G4 Retrieval, after initial retrieval (dense/sparse/hybrid) and before
  answer generation. Input: query + list of (document, score) tuples. Output:
  re-scored and re-sorted (document, score) tuples.

What calls it:
  The retrieval / agent layer (G5) calls Reranker.rerank() or
  MetadataReranker.rerank() to refine the candidate set before generating an answer.

What it calls:
  sentence_transformers.CrossEncoder (optional dependency) for the cross-encoder
  model; falls back to passthrough (original scores) when unavailable.

What breaks if removed:
  Re-ranking disappears; candidates are used in their initial retrieval order, which
  may surface less relevant chunks first and degrade answer quality.

Tags:
  # GAP: optional dependency / graceful degradation — sentence_transformers is
         imported in a try/except and the reranker degrades to passthrough scores
         when missing; production should declare the dependency and gate features.
  # BUSINESS: protects answer quality — better-ranked chunks mean the generator
              grounds its answer in the most relevant evidence.
  # INTERVIEW: say-it-out-loud — "A cross-encoder looks at the query and document
               together (unlike a bi-encoder which encodes each separately), so it
               scores relevance more accurately but is slower; use it only on the
               top-k candidates, not the whole corpus."
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # sentence-transformers optional at import time; reranker degrades to passthrough
    CrossEncoder = None


class Reranker:
    """Reranker using cross-encoder models for improved relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        WHAT:  Record the model name and load the cross-encoder model.
        WHY:   A cross-encoder jointly encodes the query and each document, giving
               a more accurate relevance score than the bi-encoder used at index
               time — at the cost of per-pair inference latency.
        INPUTS: model_name (str): "cross-encoder/ms-marco-MiniLM-L-6-v2".
        OUTPUTS: None (instance initialized; model loaded lazily via _load_model).
        SIDE EFFECTS: Calls _load_model(), which may load a model from disk / hub
                      (reranker.py:28) and logs success or failure.
        FAILURE MODES: If sentence_transformers is missing or the model fails to
                       load, self.model is set to None and rerank() degrades to
                       passthrough (no exception is raised to the caller).
        ALTERNATIVE: Use MetadataReranker (cheap, model-free) when a cross-encoder
                     is too slow or unavailable.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        WHAT:  Load the cross-encoder model if the optional dependency is present.
        WHY:   Keeping sentence_transformers optional lets this module import in
               environments without it, degrading gracefully instead of crashing.
        INPUTS: None (uses self.model_name).
        OUTPUTS: None (sets self.model to the CrossEncoder or None).
        SIDE EFFECTs: May download/load model weights; logs warning or error.
        FAILURE MODES: ImportError → self.model = None with a warning; any other
                       exception → logged error and self.model = None (passthrough).
        ALTERNATIVE: Make sentence_transformers a hard dependency and fail fast at
                     import time if you want guaranteed reranking.
        """
        if CrossEncoder is None:
            logger.warning("sentence-transformers not installed; reranker disabled (passthrough scores).")
            self.model = None
            return
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str,
               documents: List[Tuple[Dict[str, Any], float]],
               k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Re-score documents using the cross-encoder and return them sorted
               by the new relevance score (top k if requested).
        WHY:   Retrieval scores (cosine / BM25) are approximate; a cross-encoder
               re-scores the top candidates more accurately so the best evidence
               is presented to the generator first.
        INPUTS:
            query (str): e.g. "How does eSIM provisioning work?"
            documents (List[Tuple[Dict[str, Any], float]]): initial (doc, score)
                candidates from retrieval.
            k (Optional[int]): how many to return; None returns all.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] of (doc, rerank_score), sorted
                 descending by rerank score.
        SIDE EFFECTS: Logs info/warning/error lines depending on model availability
                      and outcome (reranker.py:63, 62, 88).
        FAILURE MODES: Empty documents → returns []. Model unavailable → returns
                       original scores (passthrough). Prediction exception → caught,
                       logged, and falls back to original scores (no exception propagates).
        ALTERNATIVE: MetadataReranker reranks by cheap metadata boosts with no model
                     and no latency.
        """
        if not documents:
            return []

        if self.model is None:
            # Return original scores if no model available.
            logger.warning("No reranker model available, returning original scores")
            return documents[:k] if k else documents

        try:
            # Prepare query-document pairs for the cross-encoder.
            pairs = [[query, doc["content"]] for doc, _ in documents]

            # Get reranker scores for each pair.
            rerank_scores = self.model.predict(pairs)

            # Create scored document list with the new scores.
            scored_documents = []
            for (doc, _), score in zip(documents, rerank_scores):
                scored_documents.append((doc, float(score)))

            # Sort by rerank score descending so the most relevant chunk is first.
            scored_documents.sort(key=lambda x: x[1], reverse=True)

            # Return top k if specified.
            if k is not None:
                scored_documents = scored_documents[:k]

            logger.info(f"Reranked {len(documents)} documents, returning {len(scored_documents)}")
            return scored_documents

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original scores.
            return documents[:k] if k else documents


class MetadataReranker:
    """Reranker that boosts scores based on metadata preferences."""

    def __init__(self, boost_rules: Optional[Dict[str, Any]] = None):
        """
        WHAT:  Store rules that multiply a document's score based on its metadata.
        WHY:   Some documents should be preferred for non-relevance reasons — for
               example a current policy version should outrank a stale one without
               re-running an expensive model.
        INPUTS: boost_rules (Optional[Dict[str, Any]]): e.g.
                {"source": {"policy.pdf": 1.2}, "type": {"policy": 1.1}}.
        OUTPUTS: None (instance initialized).
        SIDE EFFECTS: None.
        FAILURE MODES: None at construction.
        ALTERNATIVE: Express the same preference as a metadata filter in retrieval
                     so unwanted documents never enter the candidate set.
        """
        self.boost_rules = boost_rules or {}

    def rerank(self, documents: List[Tuple[Dict[str, Any], float]],
               k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        WHAT:  Multiply each document's score by applicable metadata boosts and
               return the re-sorted top k.
        WHY:   Cheap, model-free way to promote documents whose metadata matches
               business preferences (e.g. current versions, specific sources).
        INPUTS: documents (List[Tuple[Dict[str, Any], float]]): (doc, score)
                candidates; k (Optional[int]): how many to return.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] of (doc, boosted_score), sorted
                 descending by boosted score.
        SIDE EFFECTS: None (read-only).
        FAILURE MODES: Empty documents → returns []. Missing metadata keys are
                       simply skipped (no boost applied) — no error raised.
        ALTERNATIVE: Apply the preference as a retrieval-time metadata filter
                     instead of post-hoc boosting.
        """
        if not documents:
            return []

        boosted_documents = []
        for doc, score in documents:
            boosted_score = score

            # Apply metadata boosts.
            metadata = doc.get("metadata", {})
            for key, rules in self.boost_rules.items():
                if key in metadata:
                    value = metadata[key]
                    if isinstance(rules, dict) and value in rules:
                        boosted_score *= rules[value]
                    elif not isinstance(rules, dict):
                        # Assume rules is a multiplier.
                        boosted_score *= rules

            boosted_documents.append((doc, boosted_score))

        # Sort by boosted score descending.
        boosted_documents.sort(key=lambda x: x[1], reverse=True)

        # Return top k if specified.
        if k is not None:
            boosted_documents = boosted_documents[:k]

        return boosted_documents


# Convenience functions
def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """
    WHAT:  Build and return a Reranker instance.
    WHY:   Factory convenience for the cross-encoder reranker.
    INPUTS: model_name (str).
    OUTPUTS: Reranker.
    SIDE EFFECTS: None at construction (model loads in Reranker.__init__).
    FAILURE MODES: None at construction.
    ALTERNATIVE: Instantiate Reranker(model_name) directly.
    """
    return Reranker(model_name)


def create_metadata_reranker(boost_rules: Optional[Dict[str, Any]] = None) -> MetadataReranker:
    """
    WHAT:  Build and return a MetadataReranker instance.
    WHY:   Factory convenience for the metadata-based reranker.
    INPUTS: boost_rules (Optional[Dict[str, Any]]).
    OUTPUTS: MetadataReranker.
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: Instantiate MetadataReranker(boost_rules) directly.
    """
    return MetadataReranker(boost_rules)
