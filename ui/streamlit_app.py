"""
ui/streamlit_app.py - Streamlit UI for the RAG Academy telecom QA system.

Tabs: Ask (optionally streaming) | Search | Agent (tool calls) | Traces & Metrics | Documents.
Shows answers with citations, provider/model, token usage, and live traces.

BUSINESS PROBLEM: Provides a browser-based front end so FDE interview-prep users
can ask telecom-domain questions, inspect retrieved chunks, run the controlled
agent, check observability traces/metrics, and manage documents - all by calling
the FastAPI backend (src/api/main.py) over HTTP.

WHERE IT SITS IN THE DATA FLOW: The presentation layer (client). It sends user
input to the API and renders answers, citations, traces, and document listings.
It holds no RAG logic itself.

WHAT CALLS IT: streamlit run ui/streamlit_app.py (dev), or a Streamlit host in
production.

WHAT IT CALLS: the FastAPI backend at API_BASE_URL (default
http://localhost:8000): /ask, /ask/stream, /search, /agent/ask, /health,
/provider, /metrics, /traces, /documents, /documents/upload, /documents/status,
/documents/{source}/status. Uses the requests library for HTTP.

WHAT BREAKS IF REMOVED: Users lose the browser interface; the API still runs
but is only reachable via curl / programmatic clients.
"""

import json
import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

# Base URL of the FastAPI backend (src/api/main.py), overridable via env.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def call(path: str, method: str = "GET", payload: dict | None = None,
         headers: dict | None = None, timeout: int = 60) -> Dict[str, Any]:
    """Make a JSON request to the FastAPI backend, returning parsed JSON or an error dict.

    WHAT: Thin wrapper around requests.request that sets Content-Type: application/json,
    raises for HTTP errors, and returns r.json(); on any RequestException returns
    {"error": str(e)} so callers can render Streamlit errors instead of crashing.
    WHY: Centralizes backend HTTP so every endpoint call shares error handling,
    timeout, and JSON handling in one place.
    INPUTS: path (e.g. "/ask"), method ("GET"/"POST"), optional payload dict,
        optional headers dict, timeout int seconds (default 60).
    OUTPUTS: parsed JSON dict on success, or {"error": str} on failure.
    SIDE EFFECTS: performs an HTTP call to the backend (network).
    FAILURE MODES: connection refused / timeout / non-2xx all become
        {"error": ...}; invalid JSON from the server raises inside r.json() and
        is caught by the RequestException handler only if requests raises it
        (note: response JSON decode errors are NOT RequestExceptions, so they
        would propagate).
    ALTERNATIVE: use a generated FastAPI client / httpx with retry logic.
    # GAP: resilience - no retry/backoff or circuit breaker; a transient backend
        hiccup shows an error to the user instead of retrying.
    # GAP: validation - response shape is not validated against the expected
        model; a changed API schema would surface as KeyError downstream.
    # BUSINESS: is the single conduit from the Streamlit UI to the telecom QA
        backend; if it misbehaves, every tab breaks.
    # INTERVIEW: "Why return an error dict instead of raising?" - so Streamlit
        tabs can call st.error(...) and keep rendering instead of killing the
        whole app session on one failed request.
    """
    h = {"Content-Type": "application/json", **(headers or {})}
    try:
        r = requests.request(method, f"{API_BASE_URL}{path}", json=payload,
                             headers=h, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def call_ask(query: str, access_level: int, stream: bool, top_k: int) -> Any:
    """Ask the backend, either streaming (SSE) or as a single JSON response.

    Args:
        query: The user's question
        access_level: The caller's access level (0=guest..3=admin)
        stream: Whether to use the streaming /ask/stream endpoint
        top_k: Number of chunks to retrieve

    WHAT: Calls /ask/stream (SSE) when stream=True, yielding token text chunks
    then the final done event; otherwise calls /ask once and returns the parsed
    JSON response.
    WHY: Lets the UI choose between incremental token streaming (lower perceived
    latency) and a one-shot JSON answer (simpler client handling).
    INPUTS: query string, access_level int, stream bool, top_k int.
    OUTPUTS: a generator yielding str token chunks then the done-event dict
    (streaming), or a parsed JSON dict (non-streaming).
    SIDE EFFECTS: performs HTTP to the backend (network); streaming keeps the
        connection open until [DONE].
    FAILURE MODES: requests exceptions propagate (call/raise_for_status) rather
        than being converted to an error dict here; callers wrap with try/except
        and st.error.
    ALTERNATIVE: always use non-streaming /ask and render the full answer at
        once (simpler, slower first byte).
    # BUSINESS: powers the Ask tab's answer rendering with optional streaming.
    # INTERVIEW: "Why does the streaming path yield 'done' as a separate value
        instead of merging it into the token stream?" - so the caller can
        distinguish final metadata (sources, usage) from token text.
    """
    headers = {"X-Access-Level": str(access_level)}
    payload = {"query": query, "top_k": top_k}
    if stream:
        r = requests.post(f"{API_BASE_URL}/ask/stream", json=payload, headers=headers,
                          timeout=120, stream=True)
        r.raise_for_status()
        tokens, done = [], None
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("type") == "token":
                tokens.append(ev["text"])
                yield ev["text"]
            elif ev.get("type") == "done":
                done = ev
        yield done  # final event
        return
    return call("/ask", "POST", payload, headers)


def main():
    """Render the Streamlit UI: Ask, Search, Agent, Traces & Metrics, Documents tabs.

    WHAT: Builds the full Streamlit interface: page config, a settings sidebar
    (access level, top-k, streaming toggle, health/provider check), and five tabs
    that call the FastAPI backend and render answers, citations, tool calls,
    traces/metrics, and document listings.
    WHY: Gives FDE interview-prep users a browser front end for the telecom QA
    system without writing code or calling curl.
    INPUTS: none (reads query params / widgets via Streamlit session state).
    OUTPUTS: none (renders UI; writes to Streamlit frontend).
    SIDE EFFECTS: makes HTTP calls to the backend on every button press; mutates
        Streamlit session state (checkbox keys, expander state).
    FAILURE MODES: if the backend is down, every call returns {"error": ...}
        and the UI renders st.error; the app itself keeps running.
    ALTERNATIVE: a custom React/HTML frontend calling the same API.
    # GAP: resilience - no retry/backoff or offline cache; a backend outage
        makes all tabs show errors with no fallback.
    # GAP: validation - API responses are accessed with .get() in most places
        but some (e.g. res["results"], res["steps"]) assume the schema;
        a changed API shape raises KeyError.
    # BUSINESS: is the user-facing surface for the telecom QA system; clarity
        and correctness of citations matter for interview prep.
    # INTERVIEW: "Why does the UI call the API directly instead of embedding the
        RAG logic?" - separation of concerns: the API can be reused by curl,
        tests, and other clients while the UI owns presentation only.
    """
    st.set_page_config(page_title="RAG Academy", page_icon="📚", layout="wide")
    st.title("📚 RAG Academy — Telecom/Company Knowledge Base")

    with st.sidebar:
        st.header("⚙️ Settings")
        # Access level drives the X-Access-Level header sent to every endpoint;
        # backend maps it to a role (guest/employee/manager/admin).
        access_level = st.selectbox("Access level", [0, 1, 2, 3],
                                    format_func=lambda v: {0: "Guest", 1: "Employee", 2: "Manager", 3: "Admin"}[v],
                                    index=1)
        top_k = st.slider("Top-k retrieved", 1, 10, 5)
        stream_on = st.checkbox("Stream answers (SSE)", value=True)
        st.divider()
        st.subheader("🏥 System")
        # One-click health check: calls /health and /provider to confirm the
        # backend is up and whether the LLM provider is real or mock.
        if st.button("Health & provider"):
            health = call("/health")
            prov = call("/provider")
            if "error" in health:
                st.error(health["error"])
            else:
                st.success(f"healthy v{health.get('version')}")
            if prov.get("provider"):
                real = "✅ REAL" if prov.get("is_real") else "⚠️ MOCK"
                st.info(f"provider={prov['provider']} model={prov['model']} {real}")

    tabs = st.tabs(["❓ Ask", "🔍 Search", "🤖 Agent", "📊 Traces & Metrics", "📁 Documents"])

    # --- Ask ---
    with tabs[0]:
        st.header("Ask a question")
        q = st.text_area("Question", placeholder="e.g. How many remote days are allowed?", height=80)
        if st.button("Ask", type="primary", disabled=not q.strip()):
            headers = {"X-Access-Level": str(access_level)}
            if stream_on:
                st.subheader("Answer (streaming)")
                out = st.empty()
                acc = ""
                final = None
                # Iterate token chunks then the final done event from the SSE stream.
                for chunk in call_ask(q.strip(), access_level, True, top_k):
                    if isinstance(chunk, str):
                        acc += chunk
                        out.markdown(acc)
                    else:
                        final = chunk
                meta = final or {}
            else:
                res = call("/ask", "POST", {"query": q.strip(), "top_k": top_k}, headers)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.subheader("Answer")
                    st.write(res["answer"])
                    meta = res
            if meta:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{meta.get('latency_ms', res.get('latency_ms', 0)):.0f} ms")
                c2.metric("Provider", str(meta.get("provider", res.get("provider", "?"))))
                c3.metric("Model", str(meta.get("model", res.get("model", "?"))))
                usage = meta.get("usage") or res.get("usage") or {}
                c4.metric("Tokens", usage.get("total_tokens", "est."))
                sources = meta.get("sources", res.get("sources", []))
                st.subheader(f"Sources ({len(sources)})")
                for i, s in enumerate(sources, 1):
                    with st.expander(f"{i}. {s.get('source_filename', s.get('source', '?'))}"):
                        st.write(s.get("content_snippet", ""))
                        if st.checkbox(f"metadata {i}", key=f"am{i}"):
                            st.json(s.get("metadata", {}))

    # --- Search ---
    with tabs[1]:
        st.header("Search (retrieval only)")
        sq = st.text_input("Search query")
        if st.button("Search", disabled=not sq.strip()):
            res = call("/search", "POST", {"query": sq, "top_k": top_k},
                       {"X-Access-Level": str(access_level)})
            if "error" in res:
                st.error(res["error"])
            else:
                st.write(f"{len(res['results'])} results in {res['latency_ms']:.0f} ms")
                for i, r in enumerate(res["results"], 1):
                    with st.expander(f"{i}. {r.get('source', '?')} — score {r.get('score', 0):.3f}"):
                        st.write(r.get("content", "")[:600])

    # --- Agent ---
    with tabs[2]:
        st.header("Agent (controlled tool calls)")
        aq = st.text_area("Agent question", placeholder="e.g. Is hr_leave_policy.md available? Compare versions of hr_leave_policy.md")
        if st.button("Run agent", disabled=not aq.strip()):
            res = call("/agent/ask", "POST",
                       {"question": aq.strip(), "use_llm_planner": False},
                       {"X-Access-Level": str(access_level)})
            if "error" in res:
                st.error(res["error"])
            else:
                st.subheader("Answer")
                st.write(res["answer"])
                st.caption(f"steps={res['steps']} used_retrieval={res['used_retrieval']}")
                if res.get("sources"):
                    st.write("**Sources:**", [s.get("source_filename") for s in res["sources"]])
                st.subheader("Tool calls")
                for tc in res.get("tool_calls", []):
                    status = "✅" if tc["ok"] else "❌"
                    with st.expander(f"{status} step {tc['step']}: {tc['tool']} ({tc['duration_ms']:.0f} ms)"):
                        st.json({"args": tc["args"], "authorized": tc["authorized"],
                                 "result": tc.get("result"), "error": tc.get("error")})

    # --- Traces & metrics ---
    with tabs[3]:
        st.header("Observability")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh metrics"):
                m = call("/metrics", headers={"X-Access-Level": str(access_level)})
                st.json(m)
        with col2:
            if st.button("Refresh traces"):
                t = call("/traces?limit=10", headers={"X-Access-Level": str(access_level)})
                for tr in t.get("traces", []):
                    with st.expander(f"{tr['request_id']} — {tr['total_ms']:.0f} ms"):
                        st.json(tr["stages"])

    # --- Documents ---
    with tabs[4]:
        st.header("Documents")
        if st.button("List documents"):
            res = call("/documents", headers={"X-Access-Level": str(access_level)})
            if "error" in res:
                st.error(res["error"])
            else:
                for d in res.get("documents", []):
                    st.write(f"- **{d['filename']}** — chunks={d['chunk_count']} "
                             f"version={d.get('doc_version')} current={d.get('is_current')}")
        up = st.file_uploader("Upload document (pdf/docx/md/txt, ≤10MB)", type=["pdf", "docx", "md", "txt"])
        if up and st.button("Upload & index"):
            if access_level < 3:
                st.error("Admin (level 3) required to upload")
            else:
                files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
                r = requests.post(f"{API_BASE_URL}/documents/upload", files=files,
                                  headers={"X-Access-Level": str(access_level)}, timeout=120)
                st.json(r.json())


if __name__ == "__main__":
    main()
