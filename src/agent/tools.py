"""
Controlled tool interface for the agent.

Rules (per the agent design):
- Every tool has an explicit name, description, and a Pydantic input schema.
- The LLM may PROPOSE a tool call; the application validates + authorizes it
  before execution. The LLM never gets raw filesystem/DB/OS access.
- Execution is wrapped in a timeout and recorded (history + trace).
- Repeated identical calls are capped; total steps are capped.
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
    """Tool execution failed."""


class ToolNotAuthorized(ToolError):
    """Tool requires permissions the caller doesn't have."""


class ToolNotFound(ToolError):
    """Tool name not in the allow-list."""


@dataclass
class ToolCall:
    """One executed (or rejected) tool call — the audit trail."""

    tool: str
    args: Dict[str, Any]
    authorized: bool
    ok: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    step: int = 0

    def to_dict(self) -> dict:
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
    """What tools are allowed to touch — injected, never global."""

    retriever: Any = None          # dense/hybrid retriever
    vector_store: Any = None
    doc_manager: Any = None
    answer_generator: Any = None
    is_admin: bool = False
    collection_metas: Any = None   # callable -> list of all chunk metadatas (for version compare)


class ToolRegistry:
    """Allow-list of tools. Execution = validate (pydantic) -> authorize -> run (timeout) -> record."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.history: List[ToolCall] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> List[Dict[str, Any]]:
        """Tool descriptions for the planner (JSON-schema of inputs)."""
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema.model_json_schema()}
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: Dict[str, Any], ctx: ToolContext, step: int = 0) -> ToolCall:
        """Validate + authorize + run a tool. Never raises — records the outcome."""
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
