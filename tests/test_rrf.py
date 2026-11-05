"""
Unit tests for Reciprocal Rank Fusion (RRF).

RRF lives in src.retrieval.retriever.reciprocal_rank_fusion and operates on
ranked lists of (document, score) tuples.
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.retrieval.retriever import reciprocal_rank_fusion


def docs(*ids):
    """Create distinct document dicts keyed by id."""
    return [({"metadata": {"doc_id": i}}, 1.0) for i in ids]


def test_rrf_basic():
    list1 = docs("doc1", "doc2", "doc3", "doc4", "doc5")
    list2 = docs("doc3", "doc1", "doc4", "doc2", "doc5")

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 5
    scores = [score for _, score in fused]
    # scores must be descending
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_single_list_scores():
    list1 = docs("doc1", "doc2", "doc3")
    fused = reciprocal_rank_fusion([list1], k=60)
    # standard RRF: score = 1/(k + rank) with 1-based rank
    expected = [1 / (60 + i + 1) for i in range(3)]
    actual = [score for _, score in fused]
    assert all(abs(a - e) < 1e-10 for a, e in zip(actual, expected))


def test_rrf_overlap_ranks_higher():
    list1 = docs("doc1", "doc2", "doc3")
    list2 = docs("doc2", "doc3", "doc4")
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    score_by_id = {doc["metadata"]["doc_id"]: score for doc, score in fused}
    assert set(score_by_id) == {"doc1", "doc2", "doc3", "doc4"}
    # docs appearing in both lists outrank doc1 (only in list1, at rank 0)
    assert score_by_id["doc2"] > score_by_id["doc1"] or score_by_id["doc3"] > score_by_id["doc1"]


def test_rrf_k_parameter():
    list1 = docs("doc1", "doc2")
    list2 = docs("doc2", "doc1")
    fused_k1 = reciprocal_rank_fusion([list1, list2], k=1)
    fused_k100 = reciprocal_rank_fusion([list1, list2], k=100)
    s_k1 = dict((d["metadata"]["doc_id"], s) for d, s in fused_k1)
    s_k100 = dict((d["metadata"]["doc_id"], s) for d, s in fused_k100)
    # lower k => higher scores
    assert s_k1["doc1"] >= s_k100["doc1"]


def test_rrf_no_overlap():
    list1 = docs("doc1", "doc2", "doc3")
    list2 = docs("doc4", "doc5", "doc6")
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    ids = {doc["metadata"]["doc_id"] for doc, _ in fused}
    assert ids == {"doc1", "doc2", "doc3", "doc4", "doc5", "doc6"}


if __name__ == "__main__":
    pytest.main([__file__])
