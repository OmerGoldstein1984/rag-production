"""
Controlled tool interface for the agent.

Rules (per the agent design):
- Every tool has an explicit name, description, and a Pydantic input schema.
- The LLM may PROPOSE a tool call; the application validates + authorizes it
  before execution. The LLM never gets raw filesystem/DB/OS access.
- Execution is wrapped in a timeout and recorded (history + trace).
- Repeated identical calls are capped; total steps are capped.

BUSINESS PROBLEM: This file is the agent's "sandbox boundary." It lets an LLM
propose tool calls (search documents, get metadata, compare versions, etc.) but
never lets the LLM execute arbitrary code or touch the filesystem/DB directly.
Every call is validated against a Pydantic schema, authorized against admin
flags, executed under a timeout, and recorded in an audit trail — so a
compromised or hallucinating LLM can't do damage.

PLACE IN DATA FLOW: The agent (src/agent/agent.py) proposes a tool call, then
calls ToolRegistry.execute() here. The tool reads from retriever/vector_store
(via ToolContext) and returns structured results back to the agent, which uses
them to ground its answer. This is the read-only data-access layer for the
agent.

WHAT CALLS IT: src/agent/agent.py (RAGAgent.run calls self.tools.execute);
build_default_tools() is called by RAGAgent.__init__ when no registry is passed.

WHAT IT CALLS: pydantic (BaseModel, ValidationError); concurrent.futures
(ThreadPoolExecutor, TimeoutError); standard library (logging, threading,
dataclasses). The tool handlers call ctx.retriever, ctx.vector_store, etc.

WHAT BREAKS IF REMOVED: The agent has no controlled way to access data — any
tool-calling agent would need to be rebuilt from scratch without validation,
authorization, timeouts, or audit tracing. The agent design's core safety rail
disappears.

# GAP: ThreadPoolExecutor is never shut down (no close()/shutdown() on the
registry) — long-running processes leak the executor and its threads.
# BUSINESS: Protects against prompt injection / unauthorized tool use — the LLM
proposes, but the app validates + authorizes + timeouts + records every call.
# INTERVIEW: Say it out loud — "The LLM never touches disk or DB directly. It
proposes a tool call; we validate the input, check authorization, run with a
timeout, and log everything. That's the trust boundary."
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """
    Tool execution failed.

    WHAT: Base exception raised when a tool's handler raises or the tool cannot
    run (not for auth/not-found — those have their own subclasses).
    WHY: Callers (ToolRegistry.execute) catch ToolError to distinguish a real
    execution failure from an authorization or not-found condition.
    INPUTS: Message string passed to RuntimeError.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: None beyond being raised.
    ALTERNATIVE: Use a single generic Exception for all tool problems (loses
    granularity).
    """


class ToolNotAuthorized(ToolError):
    """
    Tool requires permissions the caller doesn't have.

    WHAT: Raised when a tool requires admin (requires_admin=True) but the
    ToolContext is not admin.
    WHY: Enforces least privilege — sensitive tools (e.g. generate_report) are
    gated behind is_admin so a non-admin user/session can't invoke them.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Caught by ToolRegistry.execute and recorded as a failed call
    (call.ok=False), not propagated.
    ALTERNATIVE: Check authorization inside each handler (duplication, easy to
    forget).
    """


class ToolNotFound(ToolError):
    """
    Tool name not in the allow-list.

    WHAT: Raised when execute() is asked to run a tool name that was never
    registered.
    WHY: The allow-list is the core safety mechanism — an unknown tool name
    means the LLM is hallucinating or probing, and it must be rejected.
    INPUTS: Message string.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: Caught by ToolRegistry.execute and recorded as a failed call.
    ALTERNATIVE: Return None for unknown names (silently drops suspicious calls
    — harder to audit).
    """


@dataclass
class ToolCall:
    """
    One executed (or rejected) tool call — the audit trail.

    WHAT: A record of a single tool invocation: name, args, whether it was
    authorized, whether it succeeded, the result/error, duration, and step
    number.
    WHY: Every call is appended to ToolRegistry.history, giving the system an
    immutable audit trail of what the agent did — essential for debugging and
    compliance.
    INPUTS (constructor): tool: str; args: Dict[str, Any]; authorized: bool;
        ok: bool; result: Any (default None); error: str (default "");
        duration_ms: float (default 0.0); step: int (default 0).
    OUTPUTS: None (data holder); to_dict() returns a serializable dict.
    SIDE EFFECTS: None.
    FAILURE MODES: to_dict() rounds duration_ms; result may not be JSON-
    serializable (callers must handle).
    ALTERNATIVE: Log calls directly instead of storing structured records (harder
    to query/analyze later).
    """

    tool: str
    args: Dict[str, Any]
    authorized: bool
    ok: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    step: int = 0

    def to_dict(self) -> dict:
        """
        Serialize the ToolCall to a plain dict.

        WHAT: Convert all fields into a JSON-friendly dict (duration rounded to
        2 decimals).
        WHY: The dict can be returned to the agent/UI/logs without exposing the
        dataclass type.
        INPUTS: None (self).
        OUTPUTS: dict — e.g.
            {"tool": "search_documents", "args": {"query": "5G", "k": 5},
             "authorized": true, "ok": true, "result": [...], "error": "",
             "duration_ms": 120.5, "step": 1}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Use dataclasses.asdict() (doesn't round duration_ms).
        """
        return {
            "tool": self.tool, "args": self.args, "authorized": self.authorized,
            "ok": self.ok, "result": self.result, "error": self.error,
            "duration_ms": round(self.duration_ms, 2), "step": self.step,
        }


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Type[BaseModel]
    handler: Callable[[Dict[str, Any], "ToolContext"], Any]
    requires_admin: bool = False
    timeout_s: float = 30.0


@dataclass
class ToolContext:
    """
    What tools are allowed to touch — injected, never global.

    WHAT: A bundle of services (retriever, vector_store, doc_manager,
    answer_generator) plus an is_admin flag and a collection_metas callable,
    handed to each tool handler.
    WHY: Dependency injection keeps tools decoupled from globals, testable with
    fakes, and lets the caller control exactly what each tool can reach.
    INPUTS (constructor): retriever: Any (default None); vector_store: Any
        (default None); doc_manager: Any (default None); answer_generator: Any
        (default None); is_admin: bool (default False);
        collection_metas: Any (default None) — callable returning chunk metadata.
    OUTPUTS: None (data holder).
    SIDE EFFECTS: None.
    FAILURE MODES: Handlers raise ToolError when a needed service is None
        (e.g. "no retriever available").
    ALTERNATIVE: Pass individual arguments to each handler (more verbose,
        less extensible).
    """

    retriever: Any = None          # dense/hybrid retriever
    vector_store: Any = None
    doc_manager: Any = None
    answer_generator: Any = None
    is_admin: bool = False
    collection_metas: Any = None   # callable -> list of all chunk metadatas (for version compare)


class ToolRegistry:
    """
    Allow-list of tools. Execution = validate (pydantic) -> authorize -> run (timeout) -> record.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.history: List[ToolCall] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    def register(self, spec: ToolSpec) -> None:
        """
        Register a tool in the allow-list.

        WHAT: Store a ToolSpec keyed by its name so execute() can look it up.
        WHY: Registration is how tools enter the allow-list; unregistered names
        are rejected at execution time (ToolNotFound).
        INPUTS: spec: ToolSpec — e.g. ToolSpec("search_documents", "Semantic
            search...", SearchInput, t_search).
        OUTPUTS: None.
        SIDE EFFECTS: Mutates self._tools.
        FAILURE MODES: Registering two specs with the same name silently
        overwrites the first (no duplicate-name guard).
        ALTERNATIVE: Require unique names and raise on collision.
        # GAP: No duplicate-name guard — re-registering a name silently replaces
        the previous spec, which can mask bugs.
        """
        self._tools[spec.name] = spec

    def names(self) -> List[str]:
        """
        Return sorted registered tool names.

        WHAT: List all allow-listed tool names in sorted order.
        WHY: Used by error messages (to show valid names on ToolNotFound) and by
        the LLM planner (agent.py _plan_with_llm lists available tools).
        INPUTS: None.
        OUTPUTS: List[str] — e.g. ["check_document_available",
            "compare_document_versions", "generate_report", ...].
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Return unsorted dict keys.
        """
        return sorted(self._tools)

    def describe(self) -> List[Dict[str, Any]]:
        """
        Tool descriptions for the planner (JSON-schema of inputs).

        WHAT: Build a list of {name, description, input_schema} dicts where
        input_schema is the Pydantic model's JSON schema.
        WHY: The LLM planner (agent.py) needs this to know which tools exist and
        what arguments they take, so it can propose a valid call.
        INPUTS: None.
        OUTPUTS: List[Dict[str, Any]] — e.g.
            [{"name": "search_documents", "description": "Semantic/hybrid
            search...", "input_schema": {...json schema...}}].
        SIDE EFFECTS: Calls model_json_schema() on each tool's schema.
        FAILURE MODES: None.
        ALTERNATIVE: Hand-authored descriptions instead of Pydantic schemas.
        """
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema.model_json_schema()}
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: Dict[str, Any], ctx: ToolContext, step: int = 0) -> ToolCall:
        """
        Validate + authorize + run a tool. Never raises — records the outcome.

        WHAT: Look up the tool, validate args against its Pydantic schema,
        authorize (admin check), run the handler under a timeout via the thread
        pool, and always record a ToolCall in history.
        WHY: This is the single choke point for every tool call — it guarantees
        validation, authorization, timeout, and audit recording regardless of
        what the LLM proposes. "Never raises" means the agent loop always gets a
        ToolCall back and can decide what to do.
        INPUTS: name: str — e.g. "search_documents"; args: Dict[str, Any] —
            untrusted LLM output e.g. {"query": "5G", "k": 5}; ctx: ToolContext;
            step: int (default 0).
        OUTPUTS: ToolCall — always returned, with ok/authorized/error set
            appropriately.
        SIDE EFFECTS: Mutates self.history (appends call); may submit work to
            the thread pool; logs INFO on completion; handler may call external
            services (vector store, retriever, LLM).
        FAILURE MODES: ToolNotFound (unknown name), ValidationError (bad args),
            ToolNotAuthorized (not admin), ToolError (handler failure/timeout)
            — all caught and recorded as call.ok=False with call.error set;
            never propagates out of execute(). Non-ToolError exceptions are
            caught by the generic `except Exception` and recorded.
        # GAP: Tool errors are swallowed (returned as failed ToolCall) — the
        caller must inspect call.ok to detect failure; there's no way to
        distinguish "tool returned an error result" from "tool crashed."
        ALTERNATIVE: Raise on failure and let the caller catch (more explicit
        but changes the "never raises" contract).
        # BUSINESS: Guarantees every tool call is validated, authorized, timed,
        and audited — the agent's core safety guarantee.
        """
        import time as _time

        call = ToolCall(tool=name, args=args, authorized=False, ok=False, step=step)
        t0 = _time.perf_counter()
        try:
            spec = self._tools.get(name)
            if spec is None:
                raise ToolNotFound(f"tool {name!r} is not in the allow-list: {self.names()}")
            # 1. validate input against the pydantic schema (LLM output is untrusted)
            validated = spec.input_schema.model_validate(args)
            # 2. authorize
            if spec.requires_admin and not ctx.is_admin:
                raise ToolNotAuthorized(f"tool {name!r} requires admin")
            call.authorized = True
            # 3. run with timeout
            fut = self._executor.submit(spec.handler, validated.model_dump(), ctx)
            try:
                result = fut.result(timeout=spec.timeout_s)
            except FutTimeout:
                raise ToolError(f"tool {name!r} timed out after {spec.timeout_s}s")
            call.ok = True
            call.result = result
            return call
        except (ValidationError, ToolNotFound, ToolNotAuthorized, ToolError) as e:
            call.error = str(e)[:500]
            return call
        except Exception as e:
            call.error = f"{type(e).__name__}: {e}"[:500]
            return call
        finally:
            call.duration_ms = (_time.perf_counter() - t0) * 1000.0
            self.history.append(call)
            logger.info("tool_call %s ok=%s ms=%.1f err=%s", name, call.ok, call.duration_ms, call.error[:80])


# --- input schemas ---------------------------------------------------------


class SearchInput(BaseModel):
    query: str
    k: int = 5


class FilenameInput(BaseModel):
    filename: str


class CompareVersionsInput(BaseModel):
    filename: str


class SectionsInput(BaseModel):
    filename: str
    max_sections: int = 5


class ReportInput(BaseModel):
    topic: str


# --- default tools ---------------------------------------------------------


def build_default_tools() -> ToolRegistry:
    """
    Build the default allow-listed tool registry.

    WHAT: Construct a ToolRegistry and register six tools: search_documents,
    get_document_metadata, compare_document_versions, get_document_sections,
    check_document_available, generate_report (admin-only).
    WHY: Gives the agent a ready-to-use set of data-access tools; the agent
    calls this in __init__ when no registry is supplied.
    INPUTS: None.
    OUTPUTS: ToolRegistry with all default tools registered.
    SIDE EFFECTS: Creates a ThreadPoolExecutor (owned by the registry).
    FAILURE MODES: None at registration time; individual tools fail at execute()
    if services are missing.
    ALTERNATIVE: Let the caller register tools individually (more verbose).
    # GAP: ThreadPoolExecutor never shut down — resource leak in long-running
    processes (see ToolRegistry.__init__).
    """
    reg = ToolRegistry()

    def t_search(args: Dict[str, Any], ctx: ToolContext):
        if ctx.retriever is None:
            raise ToolError("no retriever available")
        results = ctx.retriever.retrieve(args["query"], k=args["k"])
        return [
            {"source": d.get("metadata", {}).get("source"),
             "filename": d.get("metadata", {}).get("filename"),
             "score": round(float(s), 4),
             "snippet": d.get("content", "")[:300]}
            for d, s in results
        ]

    def t_metadata(args: Dict[str, Any], ctx: ToolContext):
        if ctx.vector_store is None:
            raise ToolError("no vector store available")
        for row in ctx.vector_store.list_sources():
            if row["filename"] == args["filename"]:
                return row
        raise ToolError(f"document {args['filename']!r} not found")

    def t_compare_versions(args: Dict[str, Any], ctx: ToolContext):
        if ctx.vector_store is None:
            raise ToolError("no vector store available")
        if ctx.collection_metas is None:
            raise ToolError("no metadata access")
        versions = {}
        for m in ctx.collection_metas():
            if m.get("filename") == args["filename"]:
                versions.setdefault(m.get("doc_version"), {"is_current": False, "chunks": 0})
                versions[m["doc_version"]]["chunks"] += 1
                if m.get("is_current"):
                    versions[m["doc_version"]]["is_current"] = True
        if not versions:
            raise ToolError(f"no versions for {args['filename']!r}")
        return {"filename": args["filename"], "versions": versions}

    def t_sections(args: Dict[str, Any], ctx: ToolContext):
        if ctx.vector_store is None:
            raise ToolError("no vector store available")
        out = []
        for row in ctx.vector_store.list_sources():
            if row["filename"] == args["filename"]:
                out.append(row)
        # also pull chunk contents
        chunks = []
        if ctx.collection_metas is None:
            raise ToolError("no chunk access")
        # use a raw search by filename via retriever if possible
        if ctx.retriever is not None:
            res = ctx.retriever.retrieve(f"content of {args['filename']}", k=args["max_sections"] * 3)
            for d, s in res:
                if d.get("metadata", {}).get("filename") == args["filename"]:
                    chunks.append({"chunk_index": d["metadata"].get("chunk_index"),
                                   "content": d.get("content", "")[:500], "score": round(float(s), 3)})
                    if len(chunks) >= args["max_sections"]:
                        break
        return {"filename": args["filename"], "sections": chunks}

    def t_available(args: Dict[str, Any], ctx: ToolContext):
        if ctx.vector_store is None:
            raise ToolError("no vector store available")
        for row in ctx.vector_store.list_sources():
            if row["filename"] == args["filename"]:
                return {"filename": args["filename"], "available": True, **row}
        return {"filename": args["filename"], "available": False}

    def t_report(args: Dict[str, Any], ctx: ToolContext):
        """Generate a structured report on a topic using the LLM (grounded in retrieved ctx)."""
        if ctx.answer_generator is None:
            raise ToolError("no answer generator available")
        results = ctx.retriever.retrieve(args["topic"], k=5) if ctx.retriever else []
        docs = [d for d, _ in results]
        res = ctx.answer_generator.generate_answer_with_context(args["topic"], docs)
        return {"topic": args["topic"], "report": res.answer,
                "sources": [s["source_filename"] for s in res.sources]}

    reg.register(ToolSpec("search_documents", "Semantic/hybrid search over the corpus.", SearchInput, t_search))
    reg.register(ToolSpec("get_document_metadata", "Metadata + version info for one document.", FilenameInput, t_metadata))
    reg.register(ToolSpec("compare_document_versions", "Compare versions of a document (which is current).", CompareVersionsInput, t_compare_versions))
    reg.register(ToolSpec("get_document_sections", "Retrieve the top sections/chunks of one document.", SectionsInput, t_sections))
    reg.register(ToolSpec("check_document_available", "Check whether a document is indexed.", FilenameInput, t_available))
    reg.register(ToolSpec("generate_report", "Generate a grounded structured report on a topic.", ReportInput, t_report, requires_admin=True))
    return reg
