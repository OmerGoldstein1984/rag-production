"""
Tests for the controlled tool-calling agent: allow-list enforcement, pydantic
validation, authorization, repeat-cap, step-cap, tracing, and the /agent/ask
endpoint.
"""

import os
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent import RAGAgent, ToolContext, ToolRegistry, build_default_tools
from src.agent.tools import ToolSpec, ToolNotAuthorized, ToolNotFound
from src.store.vector_store import VectorStore


class DummyInput(BaseModel):
    x: int


def make_registry():
    reg = ToolRegistry()

    def t_echo(args, ctx):
        return {"echo": args["x"]}

    def t_admin(args, ctx):
        return {"secret": True}

    reg.register(ToolSpec("echo", "echo back x", DummyInput, t_echo))
    reg.register(ToolSpec("admin_only", "admin tool", DummyInput, t_admin, requires_admin=True))
    return reg


# --- registry enforcement --------------------------------------------------

def test_tool_not_in_allowlist_rejected():
    reg = make_registry()
    call = reg.execute("rm_rf", {"x": 1}, ToolContext())
    assert not call.ok and not call.authorized
    assert "not in the allow-list" in call.error


def test_invalid_args_rejected_by_pydantic():
    reg = make_registry()
    call = reg.execute("echo", {"x": "not-an-int"}, ToolContext())
    assert not call.ok
    assert "validation" in call.error.lower() or "int" in call.error.lower()


def test_admin_tool_requires_admin():
    reg = make_registry()
    ctx = ToolContext(is_admin=False)
    call = reg.execute("admin_only", {"x": 1}, ctx)
    assert not call.ok and not call.authorized
    assert "admin" in call.error
    # with admin it works
    call2 = reg.execute("admin_only", {"x": 1}, ToolContext(is_admin=True))
    assert call2.ok and call2.authorized and call2.result == {"secret": True}


def test_tool_timeout_recorded():
    import time
    reg = ToolRegistry()

    def slow(args, ctx):
        time.sleep(1.0)
        return "done"

    reg.register(ToolSpec("slow", "slow", DummyInput, slow, timeout_s=0.05))
    call = reg.execute("slow", {"x": 1}, ToolContext())
    assert not call.ok and "timed out" in call.error


def test_history_records_every_call():
    reg = make_registry()
    reg.execute("echo", {"x": 1}, ToolContext())
    reg.execute("echo", {"x": "bad"}, ToolContext())
    assert len(reg.history) == 2
    assert reg.history[0].ok and not reg.history[1].ok


# --- agent behavior --------------------------------------------------------

def _ctx_with_docs(store):
    from src.retrieval.retriever import create_retriever
    from src.embedding.embedder import Embedder
    docs = [{"content": "Remote work allowed 3 days per week with manager approval.",
             "metadata": {"source": "uploads/policy.md", "filename": "policy.md", "doc_id": "d1"}}]
    from src.store.vector_store import add_documents_to_store
    emb = Embedder()
    embedded = emb.embed_documents(docs)
    add_documents_to_store(store, embedded)
    retr = create_retriever(store, emb)
    return ToolContext(retriever=retr, vector_store=store,
                       collection_metas=lambda: [d["metadata"] for d in embedded])


def test_agent_searches_and_answers(tmp_path):
    os.environ["LLM_PROVIDER"] = "mock"
    store = VectorStore(persist_directory=str(tmp_path / "c"))
    ctx = _ctx_with_docs(store)
    from src.generation.answer import create_answer_generator
    ctx.answer_generator = create_answer_generator()
    agent = RAGAgent(ctx, tools=build_default_tools(), use_llm_planner=False)
    res = agent.run("How many remote days are allowed?")
    assert res.used_retrieval
    assert res.steps >= 1
    assert res.tool_calls[0]["tool"] == "search_documents"
    assert res.tool_calls[0]["ok"]
    assert isinstance(res.answer, str) and len(res.answer) > 0
    assert res.sources  # sources attached


def test_agent_repeat_cap(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "c"))
    ctx = _ctx_with_docs(store)
    reg = build_default_tools()
    agent = RAGAgent(ctx, tools=reg, use_llm_planner=False)
    # force the planner to always pick the same tool
    agent.plan = lambda q: {"tool": "check_document_available", "args": {"filename": "policy.md"}}
    res = agent.run("anything")
    same = [c for c in res.tool_calls if c["tool"] == "check_document_available"]
    assert len(same) <= 2  # MAX_REPEAT_SAME_CALL


def test_agent_rejects_non_allowlisted_planner_choice(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "c"))
    ctx = _ctx_with_docs(store)
    reg = build_default_tools()
    agent = RAGAgent(ctx, tools=reg, use_llm_planner=False)
    agent.plan = lambda q: {"tool": "os_system", "args": {}}
    res = agent.run("do something bad")
    assert res.tool_calls[0]["ok"] is False
    assert "not in the allow-list" in res.tool_calls[0]["error"]


def test_agent_admin_tool_gated(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "c"))
    ctx = _ctx_with_docs(store)
    ctx.is_admin = False
    reg = build_default_tools()
    agent = RAGAgent(ctx, tools=reg, use_llm_planner=False)
    agent.plan = lambda q: {"tool": "generate_report", "args": {"topic": "x"}}
    res = agent.run("give me a report")
    assert res.tool_calls[0]["ok"] is False and "admin" in res.tool_calls[0]["error"]


# --- endpoint --------------------------------------------------------------

def test_agent_endpoint(tmp_path, monkeypatch):
    os.environ["LLM_PROVIDER"] = "mock"
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    initialize_rag_pipeline("data/raw")
    c = TestClient(app)
    H = {"X-Access-Level": "1"}
    r = c.post("/agent/ask", json={"question": "How many remote days are allowed?"}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "tool_calls" in body and "sources" in body
    assert body["tool_calls"][0]["tool"] == "search_documents"
