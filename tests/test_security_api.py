"""
Security + eval-repair tests: ingest path guard, bearer-token auth, and the
repaired evaluation runner (hybrid retrieval with real BM25, judge wiring).
"""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import resolve_safe_ingest_dir, verify_api_token
from src.evaluation.run_eval import RAGEvaluationPipeline, make_llm_judge


# --- ingest path guard ----------------------------------------------------

def test_ingest_dir_allows_project_data():
    p = resolve_safe_ingest_dir("data/raw")
    assert p.name == "raw" and "data" in str(p)


def test_ingest_dir_blocks_traversal():
    with pytest.raises(HTTPException) as exc:
        resolve_safe_ingest_dir("../../etc")
    assert exc.value.status_code == 400


def test_ingest_dir_blocks_absolute_outside():
    with pytest.raises(HTTPException):
        resolve_safe_ingest_dir("C:/Windows/System32")


# --- bearer token auth ----------------------------------------------------

def test_token_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert verify_api_token(authorization=None) is None  # dev mode: allowed


def test_token_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-token-123")
    from src.config import get_settings
    # bypass lru? get_settings isn't cached here (plain function) — env read each call
    with pytest.raises(HTTPException) as e1:
        verify_api_token(authorization=None)
    assert e1.value.status_code == 401
    with pytest.raises(HTTPException) as e2:
        verify_api_token(authorization="Bearer wrong")
    assert e2.value.status_code == 401
    assert verify_api_token(authorization="Bearer secret-token-123") is None


# --- eval runner repairs --------------------------------------------------

def test_make_llm_judge_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    judge = make_llm_judge()
    out = judge("score this 0-10")
    assert isinstance(out, str)


def test_eval_pipeline_hybrid_retrieve_uses_bm25(tmp_path):
    # Build a tiny pipeline with a fake corpus and real BM25
    docs = [
        {"content": "Remote work allowed 3 days per week with manager approval.",
         "metadata": {"source": "data/raw/hr_remote_work_policy.md", "chunk_index": 0, "doc_id": "d1"}},
        {"content": "Passwords must be 12 characters and changed every 90 days.",
         "metadata": {"source": "data/raw/it_security_policy.md", "chunk_index": 0, "doc_id": "d2"}},
    ]
    pipe = RAGEvaluationPipeline(use_hybrid=True, use_reranker=False, use_rrf=True)
    # bypass full ingestion; inject components directly
    from src.retrieval.sparse_retriever import create_bm25_retriever
    from src.retrieval.retriever import _doc_key

    class FakeRetriever:
        def retrieve(self, q, k=10, current_version_only=True, **kw):
            return [(docs[0], 0.9), (docs[1], 0.1)][:k]

    pipe.retriever = FakeRetriever()
    pipe.sparse_retriever = create_bm25_retriever()
    pipe.sparse_retriever.add_documents(docs)
    pipe._corpus = docs
    pipe.is_ready = True

    results = pipe._hybrid_retrieve("remote work days", k=2)
    assert len(results) > 0
    # RRF should surface the remote-work doc first for this query
    top = results[0][0]
    assert _doc_key(top) == _doc_key(docs[0])


def test_eval_single_question_runs_with_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    docs = [
        {"content": "Remote work allowed 3 days per week with manager approval.",
         "metadata": {"source": "data/raw/hr_remote_work_policy.md", "chunk_index": 0, "doc_id": "d1"}},
    ]
    pipe = RAGEvaluationPipeline(use_hybrid=False, use_reranker=False)
    from src.retrieval.sparse_retriever import create_bm25_retriever

    class FakeRetriever:
        def retrieve(self, q, k=10, current_version_only=False, **kw):
            return [(docs[0], 0.9)][:k]

    class FakeStore:
        pass

    pipe.retriever = FakeRetriever()
    pipe.vector_store = FakeStore()
    pipe.is_ready = True

    q = {"id": "q1", "question": "How many remote days?",
         "expected_answer_contains": ["3 days"], "expected_sources": ["hr_remote_work_policy.md"]}
    res = pipe.evaluate_single_question(q)
    assert "retrieval" in res and "generation" in res
    assert "answer" not in res.get("generation", {}) or isinstance(res["generation"], dict)
