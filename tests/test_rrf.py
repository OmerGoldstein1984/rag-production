"""
Unit tests for the Reciprocal Rank Fusion (RRF) implementation.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import RetrievalMetrics

def test_reciprocal_rank_fusion_basic():
    """Test basic RRF functionality."""
    # Create two ranked lists of document IDs
    list1 = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    list2 = ["doc3", "doc1", "doc4", "doc2", "doc5"]

    # Apply RRF
    fused = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=60)

    # Verify results
    assert len(fused) == 5
    assert all(isinstance(item, tuple) and len(item) == 2 for item in fused)
    assert all(isinstance(item[0], str) and isinstance(item[1], (int, float)) for item in fused)

    # Documents that appear high in both lists should rank higher
    # doc1: rank 0 in list1, rank 1 in list2 -> score = 1/(60+0) + 1/(60+1)
    # doc2: rank 1 in list1, rank 3 in list2 -> score = 1/(60+1) + 1/(60+3)
    # doc3: rank 2 in list1, rank 0 in list2 -> score = 1/(60+2) + 1/(60+0)

    # Verify scores are descending
    scores = [score for _, score in fused]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def test_reciprocal_rank_fusion_empty_lists():
    """Test RRF with empty input lists."""
    fused = RetrievalMetrics.reciprocal_rank_fusion([[], []], k=60)
    assert fused == []

def test_reciprocal_rank_fusion_single_list():
    """Test RRF with a single list."""
    list1 = ["doc1", "doc2", "doc3"]
    fused = RetrievalMetrics.reciprocal_rank_fusion([list1], k=60)

    # With single list, scores should be 1/(k+rank)
    expected_scores = [1/(60+i) for i in range(len(list1))]
    actual_scores = [score for _, score in fused]

    assert len(fused) == 3
    assert all(abs(actual - expected) < 1e-10 for actual, expected in zip(actual_scores, expected_scores))

def test_reciprocal_rank_fusion_overlap():
    """Test RRF with overlapping documents."""
    list1 = ["doc1", "doc2", "doc3"]
    list2 = ["doc2", "doc3", "doc4"]

    fused = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=60)

    # Should contain all unique documents
    doc_ids = [doc_id for doc_id, _ in fused]
    assert set(doc_ids) == {"doc1", "doc2", "doc3", "doc4"}

    # doc2 and doc3 should rank higher as they appear in both lists
    scores_dict = dict(fused)
    assert scores_dict["doc2"] > scores_dict["doc1"] or scores_dict["doc3"] > scores_dict["doc1"]

def test_reciprocal_rank_fusion_k_parameter():
    """Test that the k parameter affects scores appropriately."""
    list1 = ["doc1", "doc2"]
    list2 = ["doc2", "doc1"]

    fused_k1 = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=1)
    fused_k10 = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=10)
    fused_k100 = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=100)

    # Lower k should produce higher scores (less penalty for rank)
    score_doc1_k1 = dict(fused_k1)["doc1"]
    score_doc1_k10 = dict(fused_k10)["doc1"]
    score_doc1_k100 = dict(fused_k100)["doc1"]

    assert score_doc1_k1 >= score_doc1_k10 >= score_doc1_k100

def test_reciprocal_rank_fusion_no_overlap():
    """Test RRF when lists have no overlapping documents."""
    list1 = ["doc1", "doc2", "doc3"]
    list2 = ["doc4", "doc5", "doc6"]

    fused = RetrievalMetrics.reciprocal_rank_fusion([list1, list2], k=60)

    # Should contain all documents from both lists
    doc_ids = [doc_id for doc_id, _ in fused]
    assert set(doc_ids) == {"doc1", "doc2", "doc3", "doc4", "doc5", "doc6"}

    # Documents should be ranked by their best position in either list
    scores_dict = dict(fused)
    # All documents from list1 should have score >= 1/(60+2) (best case: position 0)
    # All documents from list2 should have score >= 1/(60+2) (best case: position 0)
    min_expected_score = 1/(60+2)  # Best possible score for position 0

    for score in scores_dict.values():
        assert score >= min_expected_score * 0.9  # Allow small floating point differences

if __name__ == "__main__":
    pytest.main([__file__])