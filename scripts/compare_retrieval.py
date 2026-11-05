"""
Retrieval experiment runner: compares chunking x retrieval configs on the
golden set and writes a measured markdown report.

Usage:
  python scripts/compare_retrieval.py                 # default config grid
  python scripts/compare_retrieval.py --data-dir data/raw --golden-set src/evaluation/golden_set.json

No API key needed: generation is mocked; only RETRIEVAL metrics are measured
(hybrid BM25/dense/RRF/reranker effects). Reports actual measured numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import Embedder
from src.store.vector_store import VectorStore, add_documents_to_store
from src.retrieval.retriever import create_retriever, reciprocal_rank_fusion, _doc_key
from src.retrieval.sparse_retriever import create_bm25_retriever
from src.retrieval.reranker import create_reranker
from src.evaluation.metrics import RetrievalMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_index(data_dir: str, chunk_strategy: str, chunk_size: int, chunk_overlap: int):
    """Load, clean, chunk, embed, and index the corpus. Returns (store, embedder, corpus)."""
    raw = load_documents_from_directory(data_dir)
    cleaned = clean_documents(raw)
    chunked = chunk_documents(cleaned, strategy=chunk_strategy,
                              chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # assign stable doc ids so metrics can match sources
    for i, c in enumerate(chunked):
        c["metadata"].setdefault("doc_id", f"{c['metadata'].get('filename', 'doc')}_{i}")
    embedder = Embedder()
    embedded = embedder.embed_documents(chunked)
    store = VectorStore(persist_directory=str(project_root / "data" / "chroma_db"))
    add_documents_to_store(store, embedded)
    return store, embedder, embedded


def retrieve_config(store, embedder, corpus, query, *, use_hybrid, use_rrf, use_reranker, hybrid_alpha, k=10):
    dense = create_retriever(store, embedder)
    dense_results = dense.retrieve(query, k=max(k * 3, 10))
    if not use_hybrid:
        results = dense_results
    else:
        bm25 = create_bm25_retriever()
        bm25.add_documents(corpus)
        sparse = bm25.retrieve(query, k=max(k * 3, 10))
        if use_rrf:
            results = reciprocal_rank_fusion([dense_results, sparse], k=60)[:k * 2]
        else:
            def norm(rs):
                if not rs:
                    return rs
                ss = [s for _, s in rs]
                lo, hi = min(ss), max(ss)
                span = (hi - lo) or 1.0
                return [(d, (s - lo) / span) for d, s in rs]
            dn, sn = norm(dense_results), norm(sparse)
            comb = {}
            for d, s in dn:
                comb.setdefault(_doc_key(d), [d, 0.0, 0.0])[1] = s
            for d, s in sn:
                comb.setdefault(_doc_key(d), [d, 0.0, 0.0])[2] = s
            a = hybrid_alpha
            results = sorted(((v[0], a * v[1] + (1 - a) * v[2]) for v in comb.values()),
                             key=lambda x: x[1], reverse=True)[:k * 2]
    if use_reranker and results:
        reranker = create_reranker()
        results = reranker.rerank(query, results, k=k)
    else:
        results = results[:k]
    return results


def relevant_ids_for(question: dict, corpus) -> set:
    """Relevant doc keys = chunks whose source filename is in expected_sources."""
    wanted = {s.replace("\\", "/").split("/")[-1].lower() for s in question.get("expected_sources", [])}
    ids = set()
    for c in corpus:
        fn = str(c["metadata"].get("filename", "")).lower()
        if fn in wanted:
            ids.add(_doc_key(c))
    return ids


def evaluate_config(config, data_dir, questions, out_dir):
    store, embedder, corpus = build_index(
        data_dir, config["chunk_strategy"], config["chunk_size"], config["chunk_overlap"]
    )
    m = RetrievalMetrics()
    per_q = []
    for q in questions:
        rel = relevant_ids_for(q, corpus)
        results = retrieve_config(
            store, embedder, corpus, q["question"],
            use_hybrid=config["use_hybrid"], use_rrf=config["use_rrf"],
            use_reranker=config["use_reranker"], hybrid_alpha=config.get("hybrid_alpha", 0.7),
            k=config.get("k", 10),
        )
        docs = [d for d, _ in results]
        per_q.append({
            "hit5": m.hit_rate_at_k(docs, rel, 5),
            "hit10": m.hit_rate_at_k(docs, rel, 10),
            "recall5": m.recall_at_k(docs, rel, 5),
            "recall10": m.recall_at_k(docs, rel, 10),
            "mrr": m.mrr(docs, rel),
            "ndcg10": m.ndcg_at_k(docs, rel, 10),
        })
    n = len(per_q) or 1
    agg = {k: sum(p[k] for p in per_q) / n for k in per_q[0]}
    return agg, per_q


def write_report(all_results: Dict[str, dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Configuration Comparison (measured)",
        "",
        "| Config | chunk | hybrid | RRF | rerank | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in all_results.items():
        cfg = r["config"]
        a = r["agg"]
        lines.append(
            f"| {name} | {cfg['chunk_strategy']}/{cfg['chunk_size']} | "
            f"{'yes' if cfg['use_hybrid'] else 'no'} | {'yes' if cfg['use_rrf'] else 'no'} | "
            f"{'yes' if cfg['use_reranker'] else 'no'} | "
            f"{a['hit5']:.3f} | {a['hit10']:.3f} | {a['recall5']:.3f} | {a['recall10']:.3f} | "
            f"{a['mrr']:.3f} | {a['ndcg10']:.3f} |"
        )
    lines += [
        "",
        "_Metrics computed from scratch (src/evaluation/metrics.py). Relevant set = "
        "golden-set expected_sources. No API key required (generation mocked; retrieval only)._",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"report written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--golden-set", default="src/evaluation/golden_set.json")
    ap.add_argument("--out", default="evaluation/retrieval_comparison.md")
    args = ap.parse_args()

    questions = json.loads((project_root / args.golden_set).read_text(encoding="utf-8"))["questions"]
    logger.info(f"{len(questions)} golden questions")

    grid = [
        {"name": "dense_fixed500", "chunk_strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": False, "use_rrf": False, "use_reranker": False},
        {"name": "dense_recursive500", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": False, "use_rrf": False, "use_reranker": False},
        {"name": "dense_structure500", "chunk_strategy": "structure_aware", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": False, "use_rrf": False, "use_reranker": False},
        {"name": "hybrid_weighted", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": True, "use_rrf": False, "use_reranker": False, "hybrid_alpha": 0.7},
        {"name": "hybrid_rrf", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": True, "use_rrf": True, "use_reranker": False},
        {"name": "hybrid_rrf_rerank", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50,
         "use_hybrid": True, "use_rrf": True, "use_reranker": True},
        {"name": "hybrid_rrf_chunk800", "chunk_strategy": "recursive", "chunk_size": 800, "chunk_overlap": 100,
         "use_hybrid": True, "use_rrf": True, "use_reranker": True},
    ]

    results = {}
    for cfg in grid:
        logger.info(f"evaluating {cfg['name']}")
        agg, per_q = evaluate_config(cfg, args.data_dir, questions, project_root)
        results[cfg["name"]] = {"config": cfg, "agg": agg}

    write_report(results, project_root / args.out)
    print(f"\nDone. Report: {args.out}")


if __name__ == "__main__":
    main()
