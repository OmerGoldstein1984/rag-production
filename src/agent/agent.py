"""
Controlled RAG agent.

Workflow (deterministic first, per design):
  1. receive question
  2. decide whether retrieval is needed (planner)
  3. propose an allow-listed tool call
  4. validate + authorize + execute the tool (registry does this)
  5. evaluate returned info
  6. generate a grounded answer
  7. return answer + sources + tool-call trace

Safety rails: max steps, cap on repeated identical tool calls, all tool I/O
validated by Pydantic, admin-only tools gated, everything traced.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tools import ToolContext, ToolRegistry, build_default_tools

logger = logging.getLogger(__name__)

MAX_STEPS = 5
MAX_REPEAT_SAME_CALL = 2


@dataclass
class AgentResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    used_retrieval: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer,
            "sources": self.sources, "tool_calls": self.tool_calls,
            "steps": self.steps, "used_retrieval": self.used_retrieval,
        }


class RAGAgent:
    """Deterministic-first tool-using RAG agent."""

    def __init__(self, ctx: ToolContext, tools: Optional[ToolRegistry] = None,
                 provider: Any = None, use_llm_planner: bool = False):
        self.ctx = ctx
        self.tools = tools or build_default_tools()
        self.provider = provider
        self.use_llm_planner = use_llm_planner and provider is not None

    # --- planner ----------------------------------------------------------

    def plan(self, question: str) -> Dict[str, Any]:
        """Decide the first tool call. LLM planner if enabled, else heuristic."""
        if self.use_llm_planner:
            try:
                return self._plan_with_llm(question)
            except Exception as e:
                logger.warning(f"LLM planner failed, using heuristic: {e}")
        return self._plan_heuristic(question)

    def _plan_heuristic(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        # metadata/version questions -> document tools
        if any(w in q for w in ("version", "superseded", "current version", "compare")):
            m = re.search(r"([\w\-\.]+\.(?:md|pdf|docx|txt))", q)
            fn = m.group(1) if m else ""
            return {"tool": "compare_document_versions", "args": {"filename": fn}}
        if any(w in q for w in ("available", "indexed", "exists", "do you have")):
            m = re.search(r"([\w\-\.]+\.(?:md|pdf|docx|txt))", q)
            return {"tool": "check_document_available", "args": {"filename": m.group(1) if m else ""}}
        if any(w in q for w in ("metadata", "department", "size", "when was")):
            m = re.search(r"([\w\-\.]+\.(?:md|pdf|docx|txt))", q)
            return {"tool": "get_document_metadata", "args": {"filename": m.group(1) if m else ""}}
        if any(w in q for w in ("report", "summarize", "summary of")):
            return {"tool": "generate_report", "args": {"topic": question}}
        # default: retrieval
        return {"tool": "search_documents", "args": {"query": question, "k": 5}}

    def _plan_with_llm(self, question: str) -> Dict[str, Any]:
        """Ask the LLM to pick a tool from the allow-list; validate its choice."""
        tool_desc = self.tools.describe()
        prompt = (
            "You can call ONE tool from this allow-list to help answer the user question. "
            "Reply ONLY with JSON: {\"tool\": name, \"args\": {...}}.\n"
            f"Tools: {json.dumps(tool_desc)}\nQuestion: {question}"
        )
        raw = self.provider.generate(prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"planner returned no JSON: {raw[:100]}")
        choice = json.loads(m.group(0))
        if choice.get("tool") not in self.tools.names():
            raise ValueError(f"planner picked non-allow-listed tool: {choice.get('tool')}")
        return choice

    # --- run ---------------------------------------------------------------

    def run(self, question: str) -> AgentResult:
        result = AgentResult(question=question, answer="")
        seen_calls: Dict[tuple, int] = {}

        plan = self.plan(question)
        result.used_retrieval = plan.get("tool") == "search_documents"

        for step in range(MAX_STEPS):
            tool = plan.get("tool")
            args = plan.get("args") or {}
            key = (tool, json.dumps(args, sort_keys=True, default=str))
            if seen_calls.get(key, 0) >= MAX_REPEAT_SAME_CALL:
                logger.warning(f"capping repeated tool call {tool}")
                break
            seen_calls[key] = seen_calls.get(key, 0) + 1

            call = self.tools.execute(tool, args, self.ctx, step=step + 1)
            result.tool_calls.append(call.to_dict())
            result.steps = step + 1

            if not call.ok:
                # on failure, stop and report (deterministic, observable)
                result.answer = (f"I couldn't complete that: tool '{tool}' failed "
                                 f"({call.error}). Please rephrase or ask something else.")
                return result

            if tool == "search_documents":
                # evaluate: gather docs for grounding
                docs = self._docs_from_search(call.result)
                if docs:
                    answer = self._grounded_answer(question, docs)
                    result.answer = answer.answer
                    result.sources = answer.sources
                else:
                    result.answer = ("I don't have enough information to answer this question "
                                     "based on the provided documents.")
                return result
            else:
                # non-search tool: after one metadata-style call, generate answer mentioning it
                result.answer = self._answer_with_tool_info(question, call)
                return result

        result.answer = "I reached the maximum number of tool steps without an answer."
        return result

    # --- helpers ----------------------------------------------------------

    def _docs_from_search(self, search_result) -> List[Dict[str, Any]]:
        """Re-fetch full docs for the snippets returned by search_documents."""
        docs = []
        if self.ctx.retriever is not None:
            # search result carries filename; do a real retrieve for full content
            # (the tool returned snippets; here we keep it simple and re-retrieve by query is
            #  not possible, so we reconstruct minimal docs from snippets)
            for item in search_result or []:
                docs.append({"content": item.get("snippet", ""),
                             "metadata": {"source": item.get("source"), "filename": item.get("filename"),
                                          "score": item.get("score")}})
        return docs

    def _grounded_answer(self, question: str, docs: List[Dict[str, Any]]):
        if self.ctx.answer_generator is not None:
            return self.ctx.answer_generator.generate_answer_with_context(question, docs)
        from src.generation.answer import GenerationResult
        return GenerationResult(query=question, answer="(no generator)", context="",
                                sources=[], retrieved_chunks=docs, latency_ms=0,
                                token_counts={}, prompt_version="v2", metadata={})

    def _answer_with_tool_info(self, question: str, call) -> str:
        return (f"Tool '{call.tool}' returned: {json.dumps(call.result, default=str)[:400]}. "
                f"(Answered deterministically from tool output.)")
