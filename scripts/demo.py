"""
Live demo script — runs the full system with LLM_PROVIDER=mock (no API key
needed). For a REAL NVIDIA demo, set LLM_PROVIDER=nvidia and NVIDIA_API_KEYS
in your environment and rerun; /provider will show the real model.

Usage:
  python scripts/demo.py                # full guided demo
  python scripts/demo.py --real         # asserts a real provider is configured
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="fail unless a real (non-mock) provider is configured")
    args = ap.parse_args()

    os.environ.setdefault("LLM_PROVIDER", "mock")

    from src.config import get_settings
    from src.api.main import initialize_rag_pipeline, rag_pipeline  # noqa: F401
    import src.api.main as api_main
    from src.observability import metrics

    settings = get_settings()
    banner(f"RAG Academy demo — provider={settings.llm_provider}")
    if args.real and settings.llm_provider == "mock":
        print("ERROR: --real requires LLM_PROVIDER=nvidia with NVIDIA_API_KEYS set")
        return 2

    banner("1. Ingest + index data/raw (load -> clean -> chunk -> embed -> ChromaDB)")
    api_main.initialize_rag_pipeline("data/raw")
    store = api_main.rag_pipeline["vector_store"]
    stats = store.get_stats()
    print(f"   vector store: {stats.get('storage_type')} chunks={stats.get('total_vectors') or stats.get('total_documents')}")
    print(f"   sources: {len(store.list_sources())} files")

    banner("2. Provider status (a mock can never hide as real)")
    prov = api_main.rag_pipeline["llm_provider"].get_model_info()
    print(f"   provider={prov['provider']} model={prov['model_name']} streaming={prov.get('supports_streaming')}")

    banner("3. Ask 3 golden-set questions (grounded answers + citations + usage)")
    questions = [
        "How many remote days per week are allowed?",
        "What are the password requirements?",
        "What is the procedure for reporting a security incident?",
    ]
    gen = api_main.rag_pipeline["answer_generator"]
    retr = api_main.rag_pipeline["retriever"]
    for q in questions:
        results = retr.retrieve(q, k=5)
        docs = [d for d, _ in results][:3]
        res = gen.generate_answer_with_context(q, docs)
        print(f"\n   Q: {q}")
        print(f"   A: {res.answer[:220]}")
        print(f"   provider={res.metadata.get('provider')} model={res.metadata.get('model_used')} "
              f"usage={res.metadata.get('usage')} latency={res.metadata.get('answer_generation_latency_ms'):.0f}ms")
        print(f"   sources: {[s['source_filename'] for s in res.sources]}")

    banner("4. Agent with controlled tool calls")
    from src.agent import RAGAgent, ToolContext
    ctx = ToolContext(
        retriever=retr, vector_store=store, doc_manager=api_main.doc_manager,
        answer_generator=gen, is_admin=True,
        collection_metas=(lambda: store.collection.get(include=["metadatas"])["metadatas"]
                          if store.collection is not None else []),
    )
    agent = RAGAgent(ctx, use_llm_planner=False)
    res = agent.run("Compare versions of hr_leave_policy.md")
    print(f"   question: {res.question}")
    print(f"   answer: {res.answer[:200]}")
    for tc in res.tool_calls:
        print(f"   tool call step {tc['step']}: {tc['tool']} ok={tc['ok']} "
              f"authorized={tc['authorized']} {tc['duration_ms']:.0f}ms")

    banner("5. Observability: metrics snapshot")
    snap = metrics.snapshot()
    print(json.dumps({k: snap[k] for k in ("counters", "tokens", "error_rate")}, indent=2)[:600])

    banner("6. Key pool status (fingerprints only, never full keys)")
    from src.key_pool import KeyPool
    try:
        pool = KeyPool.from_env()
        for s in pool.status():
            print(f"   {s['fingerprint']} status={s['status']} expires={s.get('expires_at')}")
    except Exception as e:
        print(f"   (no NVIDIA_API_KEYS configured: {e})")

    banner("Demo complete. For the real model: export NVIDIA_API_KEYS=... and LLM_PROVIDER=nvidia, then rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
