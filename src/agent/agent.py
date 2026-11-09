"""
Controlled RAG agent.

BUSINESS PROBLEM: This file is the agent layer that decides how to answer a
user question — whether to retrieve documents, look up metadata, compare
versions, or generate a report — using only allow-listed, validated tools. It
is the "brain" that orchestrates the controlled tool interface
(src/agent/tools.py) and the generation stack (src/generation/answer.py).

PLACE IN DATA FLOW: Sits between the user/CLI/API and the data-access tools.
Receives a question, plans which tool to call, executes via ToolRegistry
(validates → authorizes → runs under timeout → records), evaluates the result,
and either generates a grounded answer (for search) or reports tool output
(for metadata/version/report tools). Returns an AgentResult with answer,
sources, steps, and the full tool-call trace.

WHAT CALLS IT: src/generation/chat.py does NOT currently use it (the CLI uses
RAGChat directly). The agent is designed to be imported by an API endpoint or
UI layer; tests construct it directly. It is called by anything that needs a
tool-using reasoning loop over the RAG corpus.

WHAT IT CALLS: src/agent/tools.py (ToolContext, ToolRegistry,
build_default_tools); src.generation.answer (GenerationResult fallback,
AnswerGenerator via ctx). Standard library: json, logging, re, dataclasses.

WHAT BREAKS IF REMOVED: The tool-using reasoning loop disappears. The system
loses the ability to plan and execute metadata/version/report queries through
the allow-listed tool path — only direct RAGChat retrieval remains.

# GAP: The LLM planner path (use_llm_planner=True) requires a provider, but the
heuristic planner is fragile keyword matching — a question like "what's the
current version of the roaming policy" matches "version" but the regex may
fail to extract a filename, leading to an empty filename tool call that fails.
# BUSINESS: Guarantees the agent never runs unvalidated, unauthorized, or
unbounded tool calls — the core safety guarantee for an LLM-driven agent.
# INTERVIEW: Say it out loud — "Deterministic first: we try keyword heuristics
before asking the LLM to pick a tool. Every tool call is validated, authorized,
timed, and audited. We cap steps at 5 and repeated identical calls at 2 so the
agent can't loop forever."
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
    """
    The result of one agent run.

    WHAT: Bundles the question, final answer, sources, tool-call trace, step
    count, and whether retrieval was used into a single serializable object.
    WHY: Callers (API, UI, tests) need a consistent shape with both the answer
    and its provenance — sources and every tool call made along the way.
    INPUTS (constructor): question: str; answer: str; sources: List[dict]
        (default []); tool_calls: List[dict] (default []); steps: int (default
        0); used_retrieval: bool (default False).
    OUTPUTS: None (data holder); to_dict() returns a plain dict for JSON
    serialization.
    SIDE EFFECTS: None.
    FAILURE MODES: None — a plain dataclass with defaults.
    ALTERNATIVE: Return a plain dict (less structured, no defaults).
    """

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    used_retrieval: bool = False

    def to_dict(self) -> dict:
        """
        Serialize the AgentResult to a plain dict.

        WHAT: Convert all fields into a JSON-friendly dict.
        WHY: The result must be returned over HTTP/API or logged without
        exposing the dataclass type.
        INPUTS: None (self).
        OUTPUTS: dict — e.g.
            {"question": "What is 5G?", "answer": "5G uses OFDM...",
             "sources": [...], "tool_calls": [...], "steps": 1,
             "used_retrieval": true}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Use dataclasses.asdict() (equivalent here).
        """
        return {
            "question": self.question, "answer": self.answer,
            "sources": self.sources, "tool_calls": self.tool_calls,
            "steps": self.steps, "used_retrieval": self.used_retrieval,
        }


class RAGAgent:
    """
    Deterministic-first tool-using RAG agent.
    """

    def __init__(self, ctx: ToolContext, tools: Optional[ToolRegistry] = None,
                 provider: Any = None, use_llm_planner: bool = False):
        """
        Initialize the agent with its context, tool registry, and optional LLM planner.

        WHAT: Store the ToolContext (services tools can reach), the ToolRegistry
        (allow-listed tools), an optional LLM provider for the planner, and
        whether to use the LLM planner (only if a provider is present).
        WHY: Dependency injection lets tests pass a fake registry/provider while
        production uses the real default tools. The LLM planner is opt-in and
        degrades to the heuristic planner if no provider is given.
        INPUTS: ctx: ToolContext — services (retriever, vector_store,
            answer_generator) plus is_admin flag; tools: Optional[ToolRegistry]
            (default: build_default_tools()); provider: Any — LLM provider for
            the LLM planner (default None); use_llm_planner: bool — enable LLM
            planner only when provider is not None.
        OUTPUTS: None (instance).
        SIDE EFFECTS: Calls build_default_tools() (creates a ThreadPoolExecutor)
        when no tools are passed.
        FAILURE MODES: None.
        ALTERNATIVE: Require all four arguments with no defaults (more explicit,
        more verbose for callers).
        """
        self.ctx = ctx
        self.tools = tools or build_default_tools()
        self.provider = provider
        self.use_llm_planner = use_llm_planner and provider is not None

    # --- planner ----------------------------------------------------------

    def plan(self, question: str) -> Dict[str, Any]:
        """
        Decide the first tool call. LLM planner if enabled, else heuristic.

        WHAT: Choose which tool to call first for the given question, trying the
        LLM planner when enabled and falling back to the heuristic planner on any
        failure.
        WHY: The planner is the agent's decision point — it maps a user question
        to a concrete tool call. Trying the LLM planner first (with heuristic
        fallback) balances flexibility with reliability.
        INPUTS: question: str — e.g. "What is the current version of the roaming
            policy?"
        OUTPUTS: Dict[str, Any] — e.g. {"tool": "compare_document_versions",
            "args": {"filename": "roaming.md"}} or
            {"tool": "search_documents", "args": {"query": "...", "k": 5}}.
        SIDE EFFECTS: Logs a WARNING if the LLM planner fails and falls back.
        FAILURE MODES: If the LLM planner is enabled but raises, the exception is
        caught and the heuristic planner is used (graceful degradation).
        ALTERNATIVE: Always use the LLM planner (no fallback) — less reliable
        when the provider is flaky.
        """
        if self.use_llm_planner:
            try:
                return self._plan_with_llm(question)
            except Exception as e:
                logger.warning(f"LLM planner failed, using heuristic: {e}")
        return self._plan_heuristic(question)

    def _plan_heuristic(self, question: str) -> Dict[str, Any]:
        """
        Pick a tool via keyword matching on the question.

        WHAT: Match keywords in the question (version/compare → compare
        document versions; available → check document available;
        metadata → get document metadata; report → generate_report; default →
        search_documents) and extract an optional filename.
        WHY: A deterministic, provider-free way to route common question types
        to the right tool without an LLM call — fast, free, and predictable.
        INPUTS: question: str — e.g. "Is the roaming policy indexed?".
        OUTPUTS: Dict[str, Any] — e.g.
            {"tool": "check_document_available", "args": {"filename": ""}}
            when no filename is found, or
            {"tool": "search_documents", "args": {"query": question, "k": 5}}.
        SIDE EFFECTS: None.
        FAILURE MODES: The filename regex may return "" when no document name is
        in the question, producing a tool call with an empty filename that will
        fail at execution (recorded as a failed ToolCall).
        ALTERNATIVE: Use the LLM planner for all routing (smarter but slower,
        costs tokens, less deterministic).
        """
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
        """
        Ask the LLM to pick a tool from the allow-list; validate its choice.

        WHAT: Build a prompt listing all registered tools (with JSON schemas),
        ask the LLM to return one JSON tool call, parse it, and verify the
        chosen tool is allow-listed.
        WHY: Lets the LLM handle ambiguous question routing (smarter than keyword
        matching) while the allow-list validation ensures it can never choose an
        unauthorized tool.
        INPUTS: question: str — e.g. "Compare the roaming policy across versions".
        OUTPUTS: Dict[str, Any] — the parsed LLM choice, e.g.
            {"tool": "compare_document_versions", "args": {"filename": "roaming.md"}}.
        SIDE EFFECTS: Calls self.provider.generate() (network call to the LLM).
        FAILURE MODES: Raises ValueError if the LLM returns no JSON, invalid JSON,
        or a non-allow-listed tool name — caught by plan() which falls back to
        the heuristic.
        ALTERNATIVE: Skip validation and trust the LLM (unsafe — defeats the
        allow-list).
        # GAP: JSON extraction uses a greedy brace-regex which can over-match
        # if the LLM's response contains multiple or nested braces.
        """
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
        """
        Run the agent: plan, execute one tool, evaluate, and produce an answer.

        WHAT: Plan the first tool call, then loop up to MAX_STEPS executing
        tools (capped repeats), evaluate the result, and generate a final answer
        with sources and a full tool-call trace.
        WHY: This is the agent's main reasoning loop — bounded (max 5 steps, max
        2 repeats of the same call) so it cannot loop forever, and fully traced
        so every step is observable.
        INPUTS: question: str — e.g. "What is 5G?".
        OUTPUTS: AgentResult — answer string, sources list, tool_calls trace,
            steps count, used_retrieval flag.
        SIDE EFFECTs: Calls self.tools.execute() (may reach the vector store,
            retriever, or LLM — network/disk); logs WARNING on repeated-call cap.
        FAILURE MODES: If a tool call fails (call.ok is False), returns
        immediately with an error answer. If MAX_STEPS is reached without an
        answer, returns "I reached the maximum number of tool steps...".
        ALTERNATIVE: Allow unlimited steps/retries (risks infinite loops and
        unbounded latency/cost).
        # GAP: After one tool call the agent always returns (the loop breaks via
        `return` in both branches) — so MAX_STEPS and the repeat cap never
        actually come into play; multi-step reasoning is not implemented.
        """
        result = AgentResult(question=question, answer="")
        seen_calls: Dict[tuple, int] = {}

        plan = self.plan(question)
        result.used_retrieval = plan.get("tool") == "search_documents"

        for step in range(MAX_STEPS):
            tool = plan.get("tool")
            args = plan.get("args") or {}
            # Build a dedup key so identical repeated calls are capped
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
        """
        Re-fetch full docs for the snippets returned by search_documents.

        WHAT: Convert the search tool's result (list of {source, filename,
        score, snippet} dicts) into minimal document dicts with content + metadata.
        WHY: The search tool returns snippets, but _grounded_answer needs
        full-ish documents with content; this reconstructs documents from
        snippets so they can be grounded.
        INPUTS: search_result — list of dicts from the search_documents tool.
        OUTPUTS: List[Dict[str, Any]] — e.g.
            [{"content": "LTE uses OFDM...", "metadata": {"source": "lte.md",
            "filename": "lte.md", "score": 0.85}}].
        SIDE EFFECTS: None.
        FAILURE MODES: Returns [] when ctx.retriever is None or search_result is
        empty/None (safe degradation).
        ALTERNATIVE: Re-retrieve by query to get full content (not possible
        here since snippets lack the original query context).
        """
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
        """
        Generate a grounded answer from retrieved docs via the answer generator.

        WHAT: Delegate to ctx.answer_generator.generate_answer_with_context if
        available; otherwise return a GenerationResult with a "(no generator)"
        placeholder.
        WHY: Turns retrieved documents into a final, sourced answer — the core
        value of the agent for retrieval questions.
        INPUTS: question: str; docs: List[Dict[str, Any]] — retrieved documents.
        OUTPUTS: GenerationResult (or placeholder) with answer, sources,
            latency, tokens.
        SIDE EFFECTs: Calls the answer generator (may reach the LLM — network).
        FAILURE MODES: If ctx.answer_generator is None, returns a placeholder
        GenerationResult instead of raising (safe degradation).
        ALTERNATIVE: Generate the answer inline here (duplicates answer.py
        logic).
        """
        if self.ctx.answer_generator is not None:
            return self.ctx.answer_generator.generate_answer_with_context(question, docs)
        from src.generation.answer import GenerationResult
        return GenerationResult(query=question, answer="(no generator)", context="",
                                sources=[], retrieved_chunks=docs, latency_ms=0,
                                token_counts={}, prompt_version="v2", metadata={})

    def _answer_with_tool_info(self, question: str, call) -> str:
        """
        Build a short answer from a non-search tool's result.

        WHAT: Format the tool call result (truncated to 400 chars) into a plain
        sentence for metadata/version/report tool outcomes.
        WHY: Non-search tools (metadata, version check, availability) return
        structured data rather than documents, so the answer is just a summary
        of that data.
        INPUTS: question: str (unused, kept for signature consistency); call —
            a ToolCall with .tool and .result.
        OUTPUTS: str — e.g. "Tool 'get_document_metadata' returned: {...}.
            (Answered deterministically from tool output.)".
        SIDE EFFECTS: None (json.dumps with default=str is safe).
        FAILURE MODES: None — json.dumps(default=str) handles non-serializable
        values.
        ALTERNATIVE: Use the answer generator to summarize tool output (slower,
        unnecessary for structured data).
        """
        return (f"Tool '{call.tool}' returned: {json.dumps(call.result, default=str)[:400]}. "
                f"(Answered deterministically from tool output.)")
