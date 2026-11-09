"""
src/api/main.py - FastAPI application for the RAG Academy telecom QA system.

BUSINESS PROBLEM: Exposes the RAG pipeline as a production HTTP API so that the
Streamlit UI, curl clients, and external services can ask telecom-domain questions,
ingest documents, search, delete/reindex documents, submit feedback, and run a
controlled tool-calling agent - all with auth, rate limiting, caching, tracing, and
role-based access control.

WHERE IT SITS IN THE DATA FLOW: Top entry point of the system. Receives raw HTTP
requests from callers, authenticates/rate-limits them, runs retrieval -> (optional)
rerank -> LLM answer generation through the initialized rag_pipeline, caches repeated
queries, records observability traces/metrics, and returns structured JSON responses.

WHAT CALLS IT: uvicorn (direct `python src/api/main.py`), the Streamlit UI
(ui/streamlit_app.py via requests to localhost:8000), external API clients, and
FastAPI's test client.

WHAT IT CALLS: src.ingestion.* (loaders, cleaner, chunker, DocumentManager),
src.embedding.embedder, src.store.vector_store, src.retrieval.* (retriever, reranker,
hybrid, sparse_retriever), src.generation.* (answer, prompt),
src.observability (metrics, Trace, record_trace, new_request_id, get_request_id,
setup_logging), src.config.get_settings, providers.llm.create_llm_provider_from_env,
src.agent.RAGAgent / ToolContext.

WHAT BREAKS IF REMOVED: No HTTP surface exists for the RAG system at all - the UI
cannot ask questions, ingest documents, search, or fetch traces/metrics. The
RAG pipeline components still exist but are unreachable over the network.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging
import sys
import time
import uuid
from pathlib import Path
from collections import defaultdict
import hashlib
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embedding.embedder import embed_documents
from src.store.vector_store import create_vector_store, add_documents_to_store
from src.retrieval.retriever import create_retriever
from src.retrieval.reranker import create_reranker
from src.retrieval.hybrid import create_hybrid_retriever_rrf
from src.generation.answer import create_answer_generator
from src.generation.prompt import format_context, PromptEngineer

logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    use_hybrid: bool = True
    use_rrf: bool = False
    current_version_only: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    query: str
    latency_ms: float
    feedback_id: str
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    use_rrf: bool = False
    current_version_only: bool = True

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    latency_ms: float

class IngestRequest(BaseModel):
    data_directory: str = "data/raw"
    chunk_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_existing_vector_store: bool = False

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int
    latency_ms: float

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: int  # 1 for thumbs up, 0 for thumbs down
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str

class HealthResponse(BaseModel):
    status: str
    message: str
    vector_store_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    version: str

# Simple in-memory cache for repeated queries
# GAP: caching - an LRU/TTL cache would evict stale entries and bound memory better
# than this simple FIFO dict; there is no TTL so a cached wrong answer persists
# until the cache fills and rotates.
class QueryCache:
    """In-memory query cache keyed by a hash of request parameters.

    WHAT: Caches QueryResponse objects so repeated identical queries skip the
    full RAG pipeline (retrieve + rerank + LLM generation), saving latency and
    LLM token cost.
    WHY: Repeating the same telecom question should not re-run the whole
    pipeline; caching cuts cost and latency for hot queries.
    INPUTS: request dict e.g. {"query": "What is EPS bearers?", "top_k": 5,
        "use_reranker": True, ...}; value is a QueryResponse object.
    OUTPUTS: get() returns the cached QueryResponse or None; set() stores it.
    SIDE EFFECTS: mutates self.cache (dict), self.hits / self.misses counters;
        may evict the oldest entry when full (FIFO eviction).
    FAILURE MODES: hash collisions (MD5) are astronomically unlikely; unbounded
        growth is prevented by max_size FIFO eviction; no TTL means stale cache
        entries never expire on their own.
    ALTERNATIVE: Redis or disk-backed cache with TTL would survive restarts and
        share state across API workers; this in-memory dict is per-process only.
    # BUSINESS: protects LLM token budget and keeps the telecom QA UI snappy for
        repeat questions from FDE interview prep users.
    # INTERVIEW: "Why cache by request hash rather than query string alone?" -
        because two requests with the same query but different top_k or reranker
        flags must not return each other's cached results.
    """

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _get_key(self, request: dict) -> str:
        # Hash the full request (query + all flags) so cache keys differ when any
        # parameter that affects the answer changes, e.g. top_k or use_reranker.
        request_str = json.dumps(request, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()

    def get(self, request: dict) -> Optional[Any]:
        key = self._get_key(request)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, request: dict, value: Any):
        key = self._get_key(request)
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            # Remove first item (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

# Rate limiting
# GAP: rate limiting - sliding-window per client_id is in-memory only, so it
# resets on API restart and does not share state across multiple API workers
# (each worker has its own counter). No distributed backing store (Redis).
class RateLimiter:
    """Sliding-window rate limiter keyed by client_id.

    WHAT: Tracks request timestamps per client in a dict of lists and rejects
    requests once a client exceeds max_requests within window_seconds.
    WHY: Prevents a single client (or abusive bot) from flooding the RAG
    pipeline and exhausting LLM token budget or vector-store capacity.
    INPUTS: client_id string e.g. "192.168.1.42"; max_requests=30,
        window_seconds=60.
    OUTPUTS: is_allowed() returns True/False; get_stats() returns a stats dict.
    SIDE EFFECTS: mutates self.requests (appends timestamps, prunes old ones).
    FAILURE MODES: in-memory only - state is lost on restart and is not shared
        across uvicorn workers, so a client could effectively multiply its limit
        by the number of workers. Memory grows with the number of distinct
        clients (no client eviction).
    ALTERNATIVE: Redis-backed sliding window with INCR/EXPIRE would share state
        across workers and survive restarts.
    # BUSINESS: protects the LLM provider budget and keeps the telecom QA service
        available to all interview-prep users under load.
    # INTERVIEW: "Why sliding window instead of fixed window?" - fixed windows
        allow a burst at the boundary (2x max_requests across two adjacent
        windows); sliding window smooths that out.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]

        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True

    def get_stats(self) -> Dict[str, Any]:
        total_blocked = sum(
            max(0, len(reqs) - self.max_requests)
            for reqs in self.requests.values()
        )
        return {
            "max_requests_per_window": self.max_requests,
            "window_seconds": self.window_seconds,
            "total_blocked_requests": total_blocked,
            "active_clients": len(self.requests)
        }

# Global RAG pipeline components
# rag_pipeline holds the fully wired retriever/reranker/embedder/vector-store/
# answer-generator dict; it is None until startup or /ingest runs
# initialize_rag_pipeline. Every request handler reads it, so if it is None the
# API returns 503 instead of answering.
rag_pipeline = None
from src.ingestion.manager import DocumentManager  # noqa: E402
doc_manager = DocumentManager()  # Document lifecycle: upload, delete, reindex, status log
query_cache = QueryCache(max_size=200)  # Shared in-memory query cache for all requests
rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 req/min per client
feedback_store = []  # In-memory list of feedback dicts; prod would use a database

# User roles and permissions (simplified)
# access_level drives can_ingest / can_delete. Enforced by verify_user_role and
# re-checked in /ingest, DELETE /documents/{source}, /documents/upload,
# /documents/reindex, /agent/ask.
USER_ROLES = {
    "admin": {"access_level": 3, "can_ingest": True, "can_delete": True},
    "manager": {"access_level": 2, "can_ingest": False, "can_delete": False},
    "employee": {"access_level": 1, "can_ingest": False, "can_delete": False},
    "guest": {"access_level": 0, "can_ingest": False, "can_delete": False}
}


def verify_api_token(authorization: Optional[str] = Header(None)) -> None:
    """If API_TOKEN is configured, require 'Authorization: Bearer <token>'.

    Without a configured token (local dev) this is a no-op. The token is
    compared in constant time and never logged.

    WHAT: FastAPI dependency that enforces bearer-token auth when API_TOKEN is
    set; permits all requests when unset (dev mode).
    WHY: Protects the RAG API from unauthenticated callers in production while
    staying usable without a token during local development.
    INPUTS: authorization header e.g. "Bearer abc123" or None.
    OUTPUTS: None on success; raises HTTPException(401) on missing/invalid token.
    SIDE EFFECTS: lazy-imports hmac and src.config.get_settings at call time.
    FAILURE MODES: raises 401 if header missing, not Bearer-prefixed, or token
        mismatch (constant-time compare via hmac.compare_digest).
    ALTERNATIVE: OAuth2 / JWT with scoped claims would be more granular than one
        shared bearer token.
    # GAP: auth - single shared bearer token with no scopes, rotation, or expiry;
        X-Access-Level (used by verify_user_role) is client-controlled unless
        bearer auth is enabled.
    # BUSINESS: blocks unauthenticated access to telecom document QA and ingestion.
    # INTERVIEW: "Why hmac.compare_digest instead of ==" - to avoid timing
        side-channels that leak the token byte by byte.
    """
    import hmac
    from src.config import get_settings

    expected = get_settings().api_token
    if not expected:
        return  # dev mode: no token configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization[len("Bearer "):].strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid token")


def verify_user_role(access_level: int = Header(0, alias="X-Access-Level")) -> dict:
    """Verify user access level and return role information.

    Note: the access level is only trustworthy when API_TOKEN auth is enabled
    (see verify_api_token) — otherwise the header is client-controlled.

    WHAT: FastAPI dependency mapping X-Access-Level to a role dict with role,
    access_level, can_ingest, can_delete.
    WHY: Enables role-based gating so only privileged roles can ingest/delete
    documents while guests/employees can only ask and search.
    INPUTS: access_level int from header (default 0 = guest).
    OUTPUTS: dict {"role": str, "access_level": int, "can_ingest": bool,
        "can_delete": bool}.
    SIDE EFFECTS: none (pure lookup).
    FAILURE MODES: unknown access levels fall back to "guest" (no privileges);
        the header is spoofable when API_TOKEN is not configured.
    ALTERNATIVE: derive roles from a verified JWT subject rather than trusting
        a client header.
    # GAP: auth - X-Access-Level is client-controlled unless bearer auth is on,
        so a caller can self-assert admin in dev mode without any token.
    # BUSINESS: enforces least-privilege on write operations in the telecom
        document corpus.
    # INTERVIEW: "Why is trust-on-first-use unsafe here?" - because without
        API_TOKEN anyone can send X-Access-Level: 3 and act as admin.
    """
    # Find role based on access level
    role = "guest"
    for role_name, role_info in USER_ROLES.items():
        if role_info["access_level"] == access_level:
            role = role_name
            break

    return {
        "role": role,
        "access_level": access_level,
        "can_ingest": USER_ROLES[role]["can_ingest"],
        "can_delete": USER_ROLES[role]["can_delete"]
    }


# Directories that ingestion is allowed to read from (resolved, absolute)
_ALLOWED_INGEST_ROOTS = [
    (Path(__file__).resolve().parent.parent.parent / "data").resolve(),
    (Path(__file__).resolve().parent.parent.parent / "uploads").resolve(),
]


def resolve_safe_ingest_dir(requested: str) -> Path:
    """Resolve a requested ingest directory and ensure it stays within allowed roots.

    Blocks path traversal (e.g. '../../etc' or absolute paths outside the project).

    WHAT: Resolves the caller's data_directory relative to the project root and
    verifies it lies under project data/ or uploads/ before allowing ingestion.
    WHY: Prevents path-traversal attacks where a caller points ingestion at
    arbitrary filesystem locations outside the intended directories.
    INPUTS: requested string e.g. "data/raw" or "../../etc".
    OUTPUTS: resolved absolute Path inside an allowed root.
    SIDE EFFECTS: none.
    FAILURE MODES: raises HTTPException(400) when the resolved path is outside
        both allowed roots.
    ALTERNATIVE: os.path.realpath + commonprefix checks, or an allow-list of
        absolute paths configured in settings.
    # GAP: validation - only blocks traversal; does not confirm the directory
        exists or contains ingestible files.
    # BUSINESS: confines document ingestion to sanctioned project folders and
        away from host-system files.
    # INTERVIEW: "Why resolve() before relative_to()?" - resolve() collapses
        '..' and symlinks so an attacker cannot bypass the check with
        'data/../../etc'.
    """
    candidate = (Path(__file__).resolve().parent.parent.parent / requested).resolve()
    for root in _ALLOWED_INGEST_ROOTS:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail="data_directory must be inside the project 'data/' or 'uploads/' directory",
    )

def initialize_rag_pipeline(data_directory: str = "data/raw",
                          chunk_strategy: str = "recursive",
                          chunk_size: int = 500,
                          chunk_overlap: int = 50,
                          use_hybrid: bool = True,
                          use_reranker: bool = True,
                          use_rrf: bool = False):
    """Build and wire the full RAG pipeline into the module-level rag_pipeline.

    WHAT: Loads documents, cleans/chunks/embeds them, creates a vector store plus
    dense and sparse retrievers, optionally combines them into a hybrid retriever,
    attaches an optional reranker and an LLM answer generator, and stores the
    result in the global rag_pipeline dict.
    WHY: Centralizes one-time pipeline construction so the API endpoints can rely
    on a single, pre-warmed retriever/generator instead of rebuilding per request.
    INPUTS: data_directory="data/raw"; chunk_strategy="recursive"; chunk_size=500;
        chunk_overlap=50; use_hybrid=True; use_reranker=True; use_rrf=False.
    OUTPUTS: None (mutates the global rag_pipeline). Logs start/finish.
    SIDE EFFECTS: reads the filesystem (data_directory), builds vector-store
        indexes, instantiates an LLM provider from env (may hit network), and
        rebinds the global rag_pipeline. Not thread-safe against concurrent
        calls (two /ingest requests racing could clobber each other).
    FAILURE MODES: raises if document loading/embedding fails or the LLM provider
        cannot be instantiated; rag_pipeline may be left partially assigned or
        None if an exception occurs before the final assignment.
    ALTERNATIVE: dependency injection (passing components as arguments) would be
        more testable than a module-level global.
    # GAP: concurrency - no lock around rebinding rag_pipeline; concurrent
        /ingest calls could race. Also no retry/backoff around LLM provider
        creation or document embedding.
    # BUSINESS: produces the search/answer backbone that every query endpoint
        depends on; telecom answers are only as good as this wiring.
    # INTERVIEW: "Why rebuild the whole pipeline on /ingest instead of
        incrementally adding chunks?" - simplicity and consistency; incremental
        adds risk of index drift between the vector store and BM25 corpus.
    """
    global rag_pipeline

    logger.info("Initializing RAG pipeline for API...")

    from src.embedding.embedder import Embedder
    from src.retrieval.sparse_retriever import create_bm25_retriever

    # Load and process documents: ingest pipeline stages (load -> clean -> chunk -> embed)
    raw_docs = load_documents_from_directory(data_directory)
    cleaned_docs = clean_documents(raw_docs)
    chunked_docs = chunk_documents(
        cleaned_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy
    )
    embedded_docs = embed_documents(chunked_docs)

    # Create vector store and add documents
    vector_store = create_vector_store()
    add_documents_to_store(vector_store, embedded_docs)

    # Embedder is required to embed queries at retrieval time
    embedder = Embedder()
    dense_retriever = create_retriever(vector_store, embedder)

    # Sparse (BM25) retriever over the same chunks for hybrid search
    sparse_retriever = create_bm25_retriever()
    sparse_retriever.add_documents(embedded_docs)

    # Single retriever facade: hybrid RRF when enabled, else dense-only
    if use_hybrid:
        retriever = create_hybrid_retriever_rrf(dense_retriever, sparse_retriever)
    else:
        retriever = dense_retriever

    reranker = create_reranker() if use_reranker else None
    # Real provider from env (NVIDIA in prod; mock only if LLM_PROVIDER=mock)
    from providers.llm import create_llm_provider_from_env
    llm_provider = create_llm_provider_from_env()
    answer_generator = create_answer_generator(llm_provider=llm_provider)

    rag_pipeline = {
        "llm_provider": llm_provider,
        "vector_store": vector_store,
        "embedder": embedder,
        "retriever": retriever,
        "dense_retriever": dense_retriever,
        "sparse_retriever": sparse_retriever,
        "reranker": reranker,
        "answer_generator": answer_generator,
        "config": {
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_hybrid": use_hybrid,
            "use_reranker": use_reranker,
            "use_rrf": use_rrf
        }
    }

    logger.info("RAG pipeline initialized successfully")

# Create FastAPI app
# app is the ASGI application served by uvicorn. Every route and the logging
# middleware are registered on it below.
app = FastAPI(
    title="RAG Academy API",
    description="API for the RAG Academy telecom domain question answering system",
    version="2.0.0"
)

# Middleware for request logging (structured, with request id + metrics)
# Runs on every HTTP request: assigns a request id, wraps the handler in a trace,
# counts requests/latency/errors, and emits an X-Request-Id header.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Assign a request id, log JSON request/response, record metrics + trace.

    WHAT: Wraps each request with a unique id, starts/stops an observability
    trace, counts requests and latency, and records errors.
    WHY: Gives every API call a traceable id for debugging and surfaces request
    rate/latency/error metrics for observability.
    INPUTS: request (FastAPI Request), call_next (the downstream handler).
    OUTPUTS: the downstream response with an X-Request-Id header added.
    SIDE EFFECTS: increments metrics counters, records a trace, logs start/done.
    FAILURE MODES: if call_next raises, the error is recorded on the trace and
        re-raised; the response still gets an X-Request-Id header before
        re-raising (header set before raise, so it's on the error response).
    ALTERNATIVE: structured logging could be delegated to a sidecar / log
        shipper instead of in-process metrics.
    # GAP: observability - metrics are in-memory only (reset on restart); no
        distributed tracing context propagation (e.g. W3C traceparent) across
        services.
    # BUSINESS: lets operators tell whether the telecom QA API is up, slow, or
        erroring, and correlate logs by request id.
    # INTERVIEW: "Why set X-Request-Id on the error path too?" - so a client
        can correlate a failed response with its server-side trace/log entry.
    """
    from src.observability import new_request_id, metrics, Trace, record_trace

    request_id = new_request_id()
    trace = Trace(request_id=request_id)
    start_time = time.time()
    metrics.inc("requests_total")

    logger.info(f"request start {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as e:
        metrics.inc_error(type(e).__name__)
        trace.error = str(e)
        record_trace(trace)
        raise

    process_ms = (time.time() - start_time) * 1000.0
    metrics.observe_latency("request_ms", process_ms)
    if response.status_code >= 400:
        metrics.inc_error(f"http_{response.status_code}")
    response.headers["X-Request-Id"] = request_id
    logger.info(f"request done {request.method} {request.url.path} "
                f"status={response.status_code} ms={process_ms:.1f}")
    record_trace(trace)
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize structured logging and the RAG pipeline when the server starts.

    WHAT: Configures JSON logging and calls initialize_rag_pipeline() so the
    API is ready to serve queries as soon as it starts listening.
    WHY: Warms up the retriever/generator at boot rather than on the first user
    request (which would add cold-start latency to the first query).
    INPUTS: none (reads settings.log_level).
    OUTPUTS: none.
    SIDE EFFECTS: calls setup_logging and initialize_rag_pipeline (reads
        filesystem, may hit network for LLM provider).
    FAILURE MODES: if initialize_rag_pipeline raises, the app may start but all
        endpoints return 503 (pipeline not initialized).
    ALTERNATIVE: lazy-initialize on first request to speed up boot at the cost
        of a slow first query.
    # GAP: resilience - startup failure is not retried; a transient embedding
        or provider error at boot leaves the API in a 503 state until restart.
    # BUSINESS: ensures the telecom QA backend is warm and queryable immediately
        after deploy.
    # INTERVIEW: "Why initialize at startup instead of lazily?" - to avoid a
        multi-second cold start on the first FDE interview question.
    """
    from src.config import get_settings
    from src.observability import setup_logging

    setup_logging(level=get_settings().log_level, structured=True)
    logger.info("structured JSON logging enabled")
    initialize_rag_pipeline()


@app.get("/metrics")
async def get_metrics(_token: None = Depends(verify_api_token)):
    """Return an in-memory metrics snapshot: counters, latency percentiles, tokens, errors.

    WHAT: FastAPI GET endpoint exposing the observability metrics snapshot
    (counters, latency percentiles, token counts, error counts).
    WHY: Lets operators and the Streamlit UI check request rate, latency, and
    error rate for the telecom QA API.
    INPUTS: none (auth dependency _token enforces bearer token when configured).
    OUTPUTS: dict from metrics.snapshot().
    SIDE EFFECTS: reads the in-process metrics store (read-only).
    FAILURE MODES: returns 401 if API_TOKEN auth is configured and no valid
        bearer token is supplied; metrics are in-memory and reset on restart.
    ALTERNATIVE: scrape Prometheus-format metrics from a sidecar exporter
        instead of an in-process JSON snapshot.
    # GAP: observability - snapshot is in-memory only; no persistent metrics
        store or alerting thresholds.
    # BUSINESS: surfaces the health and cost signals (latency, token usage) that
        FDE interview-prep usage depends on.
    # INTERVIEW: "Why not expose metrics without auth?" - because latency/token
        counts can reveal usage patterns and model costs.
    """
    from src.observability import metrics

    return metrics.snapshot()


@app.get("/traces")
async def get_traces(limit: int = 20, _token: None = Depends(verify_api_token)):
    """Return recent per-request traces (pipeline stages with timings).

    WHAT: GET endpoint returning the most recent observability traces, each
    containing per-stage timings (retrieve, rerank, llm).
    WHY: Lets developers/operators debug where a slow query spent its time.
    INPUTS: limit int (default 20) capping how many traces to return.
    OUTPUTS: {"traces": [...]}.
    SIDE EFFECTS: reads the in-process trace store (read-only).
    FAILURE MODES: returns 401 without valid bearer token when API_TOKEN is set.
    ALTERNATIVE: query a distributed tracing backend (Jaeger/Tempo) by trace id.
    # GAP: observability - traces live in memory and are lost on restart; no
        trace-id correlation in logs across service boundaries.
    # BUSINESS: helps pinpoint whether retrieve/rerank/llm is the bottleneck for
        telecom QA latency.
    # INTERVIEW: "Why cap the number of traces returned?" - to bound memory and
        response size for the caller.
    """
    from src.observability import get_recent_traces

    return {"traces": get_recent_traces(limit)}


@app.get("/provider")
async def provider_status(_token: None = Depends(verify_api_token)):
    """Reveal which LLM provider/model is active — makes a mock impossible to hide.

    WHAT: GET endpoint reporting the active LLM provider, model name, and
    whether it is a real provider vs the mock.
    WHY: Guarantees transparency about whether answers come from a real LLM or
    a mock, which matters for trust in telecom QA answers.
    INPUTS: none.
    OUTPUTS: {"provider": str|None, "model": str|None, "is_real": bool|None}.
    SIDE EFFECTS: reads rag_pipeline (may be None) and calls get_model_info().
    FAILURE MODES: returns {"provider": "none", ...} when rag_pipeline is None
        (not initialized); returns 401 without valid bearer token.
    ALTERNATIVE: read the LLM_PROVIDER env var directly instead of querying the
        provider object at runtime.
    # GAP: observability - reports the provider but not per-request model cost or
        token latency breakdown.
    # BUSINESS: prevents the mock provider from silently substituting for a real
        telecom-knowledge model.
    # INTERVIEW: "Why is is_real computed as provider != 'mock'?" - because any
        non-mock provider is treated as real; there is no separate real/mock
        flag beyond the provider name.
    """
    prov = rag_pipeline.get("llm_provider") if rag_pipeline else None
    if prov is None:
        return {"provider": "none", "model": None, "note": "pipeline not initialized"}
    info = prov.get_model_info()
    return {"provider": info.get("provider"), "model": info.get("model_name"),
            "is_real": info.get("provider") != "mock"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with enhanced metrics.

    WHAT: GET endpoint reporting whether the RAG pipeline is operational,
    including vector-store stats, cache stats, and rate-limiter stats.
    WHY: Load balancers and operators poll this to decide whether to route
    traffic to this API instance.
    INPUTS: none.
    OUTPUTS: HealthResponse(status, message, vector_store_stats, cache_stats,
        version).
    SIDE EFFECTS: reads rag_pipeline, query_cache, rate_limiter (read-only).
    FAILURE MODES: returns 503 if rag_pipeline is None (not initialized) or if
        any of the stats calls raise (wrapped as 503 with the error message).
    ALTERNATIVE: a lightweight TCP/http health check that does not touch the
        vector store (faster but less informative).
    # GAP: observability - health depends on vector_store.get_stats() which may
        be slow or fail; no dependency-check circuit breaker.
    # BUSINESS: tells routing layers whether this instance can answer telecom QA
        queries at all.
    # INTERVIEW: "Why does /health not require a bearer token?" - health checks
        are typically unauthenticated so load balancers can probe without
        credentials.
    """
    global rag_pipeline, query_cache, rate_limiter

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        stats = rag_pipeline["vector_store"].get_stats()
        cache_stats = query_cache.get_stats()
        rate_limit_stats = rate_limiter.get_stats()

        return HealthResponse(
            status="healthy",
            message="RAG pipeline is operational",
            vector_store_stats=stats,
            cache_stats=cache_stats,
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    client_request: Request,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
    x_forwarded_for: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
):
    """
    Ask a question to the RAG system with caching, rate limiting, and feedback tracking.

    Args:
        request: Query request containing the question and parameters
        background_tasks: FastAPI background tasks
        client_request: The raw request object
        user_info: User information from access level header

    Returns:
        Query response with answer, sources, and metadata

    WHAT: POST endpoint that takes a user question, rate-limits the caller,
    checks the in-memory query cache, runs retrieve -> (optional rerank) -> LLM
    answer generation, records metrics/traces, caches the result, and returns
    the answer with sources and a feedback id.
    WHY: This is the primary user-facing query path for the telecom QA system;
    it balances speed (cache), cost (rate limit, cached answers), and quality
    (rerank + real LLM).
    INPUTS: QueryRequest {query, top_k, use_reranker, use_hybrid, use_rrf,
        current_version_only} plus auth headers.
    OUTPUTS: QueryResponse {answer, sources, context_used, query, latency_ms,
        feedback_id, provider, model, usage}.
    SIDE EFFECTS: mutates query_cache (stores result), feedback_store via
        background task, metrics counters; reads rag_pipeline (global).
    FAILURE MODES: 429 if rate-limited; 503 if pipeline not initialized; 500
        on any internal error (retrieve/rerank/llm failure) with detail.
    ALTERNATIVE: the streaming variant /ask/stream returns tokens incrementally
        instead of waiting for the full answer.
    # GAP: resilience - no retry/backoff around the LLM call; a transient
        provider failure turns into a 500 instead of a retry.
    # GAP: concurrency - the cache read/set and feedback_store append are not
        locked; concurrent requests could interleave (GIL makes dict ops mostly
        atomic but not transactional).
    # BUSINESS: is the core telecom question-answering path; answers must be
        grounded in retrieved documents per the RAG teaching rules.
    # INTERVIEW: "Why retrieve top_k*2 when reranking is on?" - to give the
        reranker a larger candidate pool so it can pick the truly top-k most
        relevant chunks rather than reranking only the initial top-k.
    """
    global rag_pipeline, query_cache, rate_limiter, feedback_store

    # Rate limiting: reject callers over their per-minute quota before doing any work.
    client_id = x_forwarded_for or (client_request.client.host if client_request.client else "unknown")
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

    # Check cache first: build a cache key from the full request so that any
    # parameter affecting the answer (query, top_k, reranker flag, access level)
    # yields a distinct cached entry.
    cache_request = {
        "query": request.query,
        "top_k": request.top_k,
        "use_reranker": request.use_reranker,
        "use_hybrid": request.use_hybrid,
        "use_rrf": request.use_rrf,
        "current_version_only": request.current_version_only,
        "user_access_level": user_info["access_level"]
    }

    cached_result = query_cache.get(cache_request)
    if cached_result is not None:
        logger.info(f"Returning cached result for query: {request.query[:50]}...")
        # Cached responses report minimal latency (no pipeline re-run)
        return cached_result.model_copy(update={"latency_ms": 1.0})

    start_time = time.time()

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    from src.observability import get_request_id, metrics, Trace, record_trace
    trace = Trace(request_id=get_request_id())

    try:
        # Generate feedback ID for tracking this query's user feedback later.
        feedback_id = str(uuid.uuid4())

        logger.info(f"Processing query from user with access level: {user_info['access_level']} ({user_info['role']})")

        # Retrieve documents (traced): fetch more candidates when reranking so the
        # reranker can choose from a larger pool.
        with trace.stage("retrieve", k=request.top_k, hybrid=request.use_hybrid):
            retrieved_docs_with_scores = rag_pipeline["retriever"].retrieve(
                request.query,
                k=request.top_k * 2 if request.use_reranker else request.top_k,
                current_version_only=request.current_version_only
            )
        metrics.observe_latency("retrieve_ms", (time.time() - start_time) * 1000.0)
        retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
        scores = [score for _, score in retrieved_docs_with_scores]
        metrics.inc("retrieved_chunks", len(retrieved_docs))

        # Rerank if requested and available (traced): cross-encoder reorders the
        # retrieved chunks for relevance before answer generation.
        if request.use_reranker and rag_pipeline["reranker"] is not None:
            with trace.stage("rerank", k=request.top_k):
                reranked_docs_with_scores = rag_pipeline["reranker"].rerank(
                    request.query,
                    list(zip(retrieved_docs, scores)),
                    k=request.top_k
                )
            final_docs = [doc for doc, _ in reranked_docs_with_scores]
            final_scores = [score for _, score in reranked_docs_with_scores]
        else:
            # No reranker: take the top-k documents straight from retrieval.
            final_docs = retrieved_docs[:request.top_k]
            final_scores = scores[:request.top_k]

        # Generate answer with the real provider (traced); record token usage.
        with trace.stage("llm", provider=rag_pipeline.get("llm_provider", {}).get_model_info().get("provider") if rag_pipeline.get("llm_provider") else "none"):
            answer_result = rag_pipeline["answer_generator"].generate_answer_with_context(
                request.query, final_docs)
        answer_text = answer_result.answer
        usage = answer_result.metadata.get("usage")
        if usage:
            metrics.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        metrics.inc("answers_generated")
        metrics.observe_latency("llm_ms", answer_result.metadata.get("answer_generation_latency_ms", 0.0))
        record_trace(trace)

        # Format sources for response: one entry per final doc with a 200-char snippet.
        sources = []
        for doc in final_docs:
            metadata = doc.get("metadata", {})
            source_info = {
                "source": metadata.get("source", "Unknown"),
                "content_snippet": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "metadata": metadata
            }
            sources.append(source_info)

        # Format context used: the joined context string sent to the LLM.
        context_used = format_context(final_docs)

        # Calculate latency for the full request (retrieve + rerank + llm).
        latency_ms = (time.time() - start_time) * 1000

        # Prepare response
        response = QueryResponse(
            answer=answer_text,
            sources=sources,
            context_used=context_used,
            query=request.query,
            latency_ms=round(latency_ms, 2),
            feedback_id=feedback_id,
            provider=answer_result.metadata.get("provider"),
            model=answer_result.metadata.get("model_used"),
            usage=answer_result.metadata.get("usage"),
        )

        # Cache the result so an identical repeat query skips the pipeline.
        query_cache.set(cache_request, response)

        # Log feedback request in background so it does not block the response.
        background_tasks.add_task(
            log_feedback_request,
            feedback_id=feedback_id,
            query=request.query,
            user_info=user_info,
            client_id=client_id
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/ask/stream")
async def ask_stream(
    request: QueryRequest,
    client_request: Request,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
    x_forwarded_for: Optional[str] = Header(None),
):
    """Streaming RAG answer (SSE). Emits token events then a final done event
    with sources/usage/provider. Retrieval + validation + tracing still apply.

    WHAT: POST endpoint that streams LLM answer tokens as Server-Sent Events
    (text/event-stream) after retrieve + (optional rerank), ending with a done
    event and [DONE] marker.
    WHY: Lets the UI render the answer incrementally as tokens arrive, cutting
    perceived latency for long telecom answers.
    INPUTS: QueryRequest plus auth headers; no background_tasks here (stream
    handler keeps the connection open).
    OUTPUTS: StreamingResponse yielding "data: {...json...}\n\n" SSE frames.
    SIDE EFFECTS: increments ask_stream_total/ask_stream_done counters, records
        a trace, adds token usage on the done event.
    FAILURE MODES: 503 if pipeline not initialized; 429 if rate-limited; on
        exception emits an error event then [DONE] so the client is not left
        hanging mid-stream.
    ALTERNATIVE: the non-streaming /ask endpoint returns the complete answer in
    one JSON response (simpler for clients, slower perceived first byte).
    # GAP: resilience - a mid-stream LLM failure yields an error event but the
        connection still closes with [DONE]; clients must handle partial answers.
    # GAP: caching - streaming answers are not cached (each stream is unique),
        so repeated streamed queries pay full pipeline cost each time.
    # BUSINESS: improves perceived responsiveness for the telecom QA chat UI.
    # INTERVIEW: "Why yield [DONE] even on the error path?" - so the client's
        event loop always gets a terminal signal instead of waiting forever on
        an open SSE connection.
    """
    from fastapi.responses import StreamingResponse
    from src.observability import get_request_id, metrics, Trace, record_trace

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    client_id = x_forwarded_for or (client_request.client.host if client_request.client else "unknown")
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    trace = Trace(request_id=get_request_id())
    metrics.inc("ask_stream_total")

    async def event_gen():
        import json as _json
        try:
            with trace.stage("retrieve", k=request.top_k):
                retrieved = rag_pipeline["retriever"].retrieve(
                    request.query,
                    k=request.top_k * 2 if request.use_reranker else request.top_k,
                    current_version_only=request.current_version_only)
            docs = [d for d, _ in retrieved]
            scores = [s for _, s in retrieved]
            if request.use_reranker and rag_pipeline["reranker"] is not None:
                with trace.stage("rerank", k=request.top_k):
                    reranked = rag_pipeline["reranker"].rerank(
                        request.query, list(zip(docs, scores)), k=request.top_k)
                final_docs = [d for d, _ in reranked]
            else:
                final_docs = docs[:request.top_k]

            with trace.stage("llm"):
                gen = rag_pipeline["answer_generator"].stream_answer_with_context(
                    request.query, final_docs)
                done = None
                for event in gen:
                    if event.get("type") == "done":
                        done = event
                    yield f"data: {_json.dumps(event, default=str)}\n\n"
            record_trace(trace)
            metrics.inc("ask_stream_done")
            if done and done.get("usage"):
                metrics.add_tokens(done["usage"].get("prompt_tokens", 0),
                                   done["usage"].get("completion_tokens", 0))
            yield "data: [DONE]\n\n"
        except Exception as e:
            metrics.inc_error(type(e).__name__)
            trace.error = str(e)
            record_trace(trace)
            yield f"data: {_json.dumps({'type': 'error', 'error': str(e)[:300]})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"X-Request-Id": get_request_id()})


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    client_request: Request,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
    x_forwarded_for: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
):
    """
    Search for documents without generating an answer (retrieval only).

    Args:
        request: Search request containing the query and parameters
        client_request: The raw request object
        user_info: User information from access level header

    Returns:
        Search response with retrieved documents

    WHAT: POST endpoint that runs retrieval only (no LLM answer generation)
    and returns the matched documents with scores.
    WHY: Lets users inspect what the retriever would surface for a query
    without paying for an LLM call; useful for debugging retrieval quality.
    INPUTS: SearchRequest {query, top_k, use_hybrid, use_rrf,
        current_version_only} plus auth headers.
    OUTPUTS: SearchResponse {results: [{content, score, metadata, source}],
        query, latency_ms}.
    SIDE EFFECTS: reads rag_pipeline (read-only); applies rate limiting.
    FAILURE MODES: 429 if rate-limited; 503 if pipeline not initialized; 500
        on retrieval error.
    ALTERNATIVE: the /ask endpoint adds answer generation on top of the same
        retrieval step.
    # GAP: observability - search results are not traced as deeply as /ask
        (no llm stage), and no caching is applied to search.
    # BUSINESS: lets FDE learners verify which telecom document chunks match a
        query before committing to an answer.
    # INTERVIEW: "Why does search not call format_context or an LLM?" - because
        it is retrieval-only; the caller can re-rank or inspect chunks directly
        without generation cost.
    """
    global rag_pipeline, rate_limiter

    # Rate limiting
    client_id = x_forwarded_for or (client_request.client.host if client_request.client else "unknown")
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

    start_time = time.time()

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        logger.info(f"Processing search from user with access level: {user_info['access_level']}")

        # Retrieve documents (no answer generation)
        retrieved_docs_with_scores = rag_pipeline["retriever"].retrieve(
            request.query,
            k=request.top_k,
            current_version_only=request.current_version_only
        )

        # Format results for response: score cast to float for JSON serialization.
        results = []
        for doc, score in retrieved_docs_with_scores:
            metadata = doc.get("metadata", {})
            result_info = {
                "content": doc["content"],
                "score": float(score),
                "metadata": metadata,
                "source": metadata.get("source", "Unknown")
            }
            results.append(result_info)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            query=request.query,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """
    Ingest new documents into the system.

    Security: requires bearer token when API_TOKEN is set, admin role, and the
    data_directory is confined to the project's data/ or uploads/ directory
    (path-traversal guard).

    Args:
        request: Ingest request containing document source and processing parameters
        background_tasks: FastAPI background tasks
        user_info: User information from access level header

    Returns:
        Ingest response with processing statistics

    WHAT: POST endpoint that loads, cleans, chunks, embeds, and indexes documents
    from a directory (guarded to project data/ or uploads/), rebuilding or
    extending the vector store, and returns processing statistics.
    WHY: Populates the telecom document corpus that the QA system answers from;
    without ingestion there is nothing to retrieve.
    INPUTS: IngestRequest {data_directory, chunk_strategy, chunk_size,
        chunk_overlap, use_existing_vector_store} plus auth headers.
    OUTPUTS: IngestResponse {message, documents_processed, chunks_created,
        latency_ms}.
    SIDE EFFECTS: mutates rag_pipeline (rebuilds or extends it), writes to the
        vector store; reads the filesystem.
    FAILURE MODES: 403 if caller lacks can_ingest; 400 if data_directory escapes
        allowed roots; 503 if pipeline not initialized (when extending); 500 on
        ingestion error.
    ALTERNATIVE: the /documents/upload endpoint ingests a single file rather
        than a whole directory.
    # GAP: concurrency - concurrent /ingest calls race on rag_pipeline rebinding
        (no lock); one could clobber another's index.
    # GAP: idempotency - re-ingesting the same directory duplicates chunks
        unless use_existing_vector_store logic dedupes (delegated to
        DocumentManager in the extend path).
    # BUSINESS: keeps the telecom document corpus current so answers reflect the
        latest policy/procedure documents.
    # INTERVIEW: "Why force use_hybrid=True and use_reranker=False during
        ingestion?" - ingestion only needs to build the index (hybrid search is
        configured for query time); reranking is a query-time concern, not an
        indexing concern.
    """
    global rag_pipeline

    # Check permissions: only roles with can_ingest (admin) may ingest.
    if not user_info["can_ingest"]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions for document ingestion"
        )

    # Path traversal guard: only allow dirs under project data/ or uploads/
    safe_dir = resolve_safe_ingest_dir(request.data_directory)

    start_time = time.time()

    try:
        logger.info(f"Starting document ingestion for user: {user_info['role']}")

        # If not using existing vector store, reinitialize everything
        if not request.use_existing_vector_store or rag_pipeline is None:
            # Initialize with new parameters
            initialize_rag_pipeline(
                data_directory=str(safe_dir),
                chunk_strategy=request.chunk_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                use_hybrid=True,  # Default to hybrid for ingestion
                use_reranker=False,  # No reranker needed for ingestion
                use_rrf=False
            )
            # Reload the pipeline to get the updated components
            rag_pipeline = rag_pipeline  # Already set by initialize_rag_pipeline
        else:
            # Add to existing vector store: load/clean/chunk/embed then append.
            raw_docs = load_documents_from_directory(str(safe_dir))
            cleaned_docs = clean_documents(raw_docs)
            chunked_docs = chunk_documents(
                cleaned_docs,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                strategy=request.chunk_strategy
            )
            embedded_docs = embed_documents(chunked_docs)

            # Add to existing store
            add_documents_to_store(rag_pipeline["vector_store"], embedded_docs)

        # Get statistics from the vector store.
        stats = rag_pipeline["vector_store"].get_stats()

        latency_ms = (time.time() - start_time) * 1000

        return IngestResponse(
            message="Document ingestion completed successfully",
            documents_processed=stats.get("total_documents", 0),
            chunks_created=stats.get("total_chunks", 0),
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during document ingestion: {str(e)}")

@app.delete("/documents/{source}")
async def delete_documents(
    source: str,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """
    Delete documents by source.

    Args:
        source: The source identifier of documents to delete
        user_info: User information from access level header

    Returns:
        Success message

    WHAT: DELETE endpoint that removes all chunks for a given source from the
    vector store and deletes the source file on disk (cascade delete).
    WHY: Lets admins remove outdated or incorrect telecom documents from the
    corpus so they are no longer retrieved.
    INPUTS: source path identifier (path param) plus auth headers.
    OUTPUTS: the doc_manager.delete result dict with an augmented message.
    SIDE EFFECTS: deletes vector-store chunks and the file on disk via
        doc_manager.delete.
    FAILURE MODES: 403 if caller lacks can_delete; 503 if pipeline not
        initialized; 500 on deletion error.
    ALTERNATIVE: soft-delete (mark superseded rather than remove) to preserve
        version history.
    # GAP: resilience - document deletion cascades to disk; if the file is
        already missing the underlying doc_manager may raise or partially fail.
    # BUSINESS: keeps the telecom document corpus accurate by removing stale
        documents that would otherwise be retrieved.
    # INTERVIEW: "Why delete both vector chunks and the on-disk file?" - cascade
        delete ensures the document cannot be re-indexed from its file on the
        next startup; orphaned chunks and files would otherwise diverge.
    """
    global rag_pipeline

    # Check permissions: only roles with can_delete (admin) may delete.
    if not user_info["can_delete"]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions for document deletion"
        )

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        logger.info(f"Deleting documents from source: {source} for user: {user_info['role']}")

        # Delete vector-store chunks AND the file on disk (cascade)
        result = doc_manager.delete(source, rag_pipeline["vector_store"])
        result["message"] = f"Deleted {result['chunks_deleted']} chunks and file for '{source}'"
        return result

    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """Upload one document (pdf/docx/md/txt, size-capped). Validates, dedupes by
    content hash, versions on filename collision, and indexes it immediately.

    WHAT: POST endpoint that accepts a single uploaded document file, validates,
    deduplicates by content hash, versions on filename collision, and indexes it
    into the vector store immediately.
    WHY: Lets admins add individual telecom documents to the corpus without
    re-ingesting an entire directory.
    INPUTS: file (UploadFile) plus auth headers.
    OUTPUTS: doc_manager.upload record as a dict.
    SIDE EFFECTS: reads the uploaded file content, writes/indexes into the
        vector store via doc_manager.upload.
    FAILURE MODES: 403 if caller lacks can_ingest; 503 if pipeline not
        initialized; 400 if doc_manager.upload raises (validation/dedup error).
    ALTERNATIVE: /ingest for bulk directory ingestion instead of single-file
        upload.
    # BUSINESS: provides the single-document on-ramp for adding new telecom
        policy/procedure docs to the QA corpus.
    # INTERVIEW: "Why dedupe by content hash rather than filename?" - because
        the same document under two different filenames should not create
        duplicate chunks in the vector store.
    """
    if not user_info["can_ingest"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions to upload")
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    content = await file.read()
    try:
        rec = doc_manager.upload(file.filename or "unnamed", content, rag_pipeline["vector_store"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return rec.to_dict()


@app.get("/documents")
async def list_documents(
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """List all indexed source documents with chunk counts and version info.

    WHAT: GET endpoint listing every indexed source document with chunk count,
    version, and current-version flag.
    WHY: Lets users see what telecom documents are in the corpus before asking
    or searching.
    INPUTS: none (auth headers only).
    OUTPUTS: {"documents": [...]} from doc_manager.list_documents.
    SIDE EFFECTS: reads rag_pipeline and the vector store (read-only).
    FAILURE MODES: 503 if pipeline not initialized; 401 without bearer token.
    ALTERNATIVE: query the vector store metadata directly rather than via
        doc_manager.
    # BUSINESS: gives visibility into the telecom document corpus the QA system
        draws answers from.
    # INTERVIEW: "Why list documents at all if users can just ask questions?" -
        so users can verify coverage (e.g., confirm a specific policy doc is
        indexed) before relying on answers.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return {"documents": doc_manager.list_documents(rag_pipeline["vector_store"])}


@app.get("/documents/status")
async def ingest_status(
    limit: int = 50,
    _token: None = Depends(verify_api_token),
):
    """Recent ingestion operations (ok/duplicate/superseded/failed).

    WHAT: GET endpoint returning the most recent ingestion status records.
    WHY: Lets operators audit ingestion outcomes (which docs succeeded, were
    duplicated, superseded, or failed).
    INPUTS: limit int (default 50) capping records returned.
    OUTPUTS: {"records": [...]} from doc_manager.status_log.
    SIDE EFFECTS: reads the doc_manager status log (read-only).
    FAILURE MODES: 401 without bearer token when API_TOKEN is configured.
    ALTERNATIVE: query a persistent ingestion-audit table instead of in-memory
        log.
    # BUSINESS: provides an audit trail of what was ingested into the telecom
        corpus and whether it succeeded.
    # INTERVIEW: "Why cap the status log at 50 records?" - to bound the
        response size since the in-memory log grows unboundedly otherwise.
    """
    return {"records": doc_manager.status_log(limit)}


@app.get("/documents/{source}/status")
async def document_status(
    source: str,
    _token: None = Depends(verify_api_token),
):
    """Latest ingest status for one document.

    WHAT: GET endpoint returning the latest ingestion status for a single
    source document.
    WHY: Lets a caller check whether a specific document was indexed successfully
    or failed.
    INPUTS: source document identifier (path param).
    OUTPUTS: the status record dict, or 404 if no ingest record exists.
    SIDE EFFECTS: reads the doc_manager status log (read-only).
    FAILURE MODES: 404 when no ingest record exists for the source; 401 without
        bearer token.
    ALTERNATIVE: scan the full status log client-side and filter by source.
    # BUSINESS: confirms a particular telecom document is present and current in
        the QA corpus.
    # INTERVIEW: "Why 404 instead of returning an empty status?" - because
        absence of any ingest record is meaningfully different from a recorded
        success/failure.
    """
    st = doc_manager.get_status(source)
    if st is None:
        raise HTTPException(status_code=404, detail="no ingest record for this document")
    return st


@app.post("/documents/reindex")
async def reindex_documents(
    data_dir: str = "data/raw",
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """Rebuild the vector store from uploads/ + data dir. Admin only.

    WHAT: POST endpoint that rebuilds the vector store from the uploads/ and
    data directories via doc_manager.reindex (admin only).
    WHY: Provides a full corpus refresh when many documents have changed, rather
    than incremental ingests.
    INPUTS: data_dir path (default "data/raw") plus auth headers.
    OUTPUTS: the doc_manager.reindex result dict.
    SIDE EFFECTS: rebuilds the vector store (destructive rewrite of the index).
    FAILURE MODES: 403 if caller lacks can_ingest; 503 if pipeline not
        initialized; 400 if data_dir escapes allowed roots.
    ALTERNATIVE: /ingest for incremental addition without a full rebuild.
    # BUSINESS: allows a clean-slate rebuild of the telecom document index when
        the corpus has changed substantially.
    # INTERVIEW: "Why is reindex admin-only?" - rebuilding the index is
        destructive and could temporarily remove documents, so it is restricted
        to trusted admins.
    """
    if not user_info["can_ingest"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions to reindex")
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    safe_dir = resolve_safe_ingest_dir(data_dir)
    result = doc_manager.reindex(rag_pipeline["vector_store"], data_dir=str(safe_dir))
    return result


class AgentRequest(BaseModel):
    question: str
    use_llm_planner: bool = False
    is_admin: bool = False  # caller asserts admin; enforced server-side via role below


@app.post("/agent/ask")
async def agent_ask(
    request: AgentRequest,
    user_info: dict = Depends(verify_user_role),
    _token: None = Depends(verify_api_token),
):
    """Controlled tool-calling agent: validated, allow-listed, traced, step-capped.

    WHAT: POST endpoint that runs a controlled RAG agent (tool-calling with an
    allow-list, validation, authorization, tracing, and step cap) to answer a
    question that may require multiple tool invocations.
    WHY: Lets the system answer questions that need more than a single retrieve
    (e.g., comparing document versions, checking file availability) by delegating
    to a tool-using agent with guardrails.
    INPUTS: AgentRequest {question, use_llm_planner, is_admin} plus auth headers.
    OUTPUTS: result.to_dict() containing the answer, steps, tool calls, sources.
    SIDE EFFECTS: constructs a ToolContext, runs the agent, increments
        agent_requests_total/agent_tool_calls metrics, records a trace.
    FAILURE MODES: 503 if pipeline not initialized; 401 without bearer token;
        agent tool errors are surfaced in the result rather than raising 500.
    ALTERNATIVE: /ask for single-shot retrieve+answer without tool use.
    # GAP: resilience - agent steps are capped internally by RAGAgent but there
        is no per-request timeout; a long-running agent run blocks the handler.
    # BUSINESS: handles multi-step telecom queries (version comparison,
        document availability) that a single retrieval cannot answer.
    # INTERVIEW: "Why is is_admin derived from can_ingest rather than the
        request field?" - because the caller's is_admin claim is untrusted;
        server-side role enforcement prevents privilege escalation.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    from src.agent import RAGAgent, ToolContext
    from src.observability import metrics, get_request_id, Trace, record_trace

    store = rag_pipeline["vector_store"]
    ctx = ToolContext(
        retriever=rag_pipeline["retriever"],
        vector_store=store,
        doc_manager=doc_manager,
        answer_generator=rag_pipeline["answer_generator"],
        is_admin=bool(user_info.get("can_ingest")),  # admin role required for admin tools
        collection_metas=(lambda: store.collection.get(include=["metadatas"])["metadatas"]
                          if store.collection is not None
                          else [d.get("metadata", {}) for d in store._fallback_documents]),
    )
    agent = RAGAgent(ctx, provider=rag_pipeline.get("llm_provider"),
                     use_llm_planner=request.use_llm_planner)

    trace = Trace(request_id=get_request_id())
    metrics.inc("agent_requests_total")
    with trace.stage("agent", question=request.question[:80]):
        result = agent.run(request.question)
    metrics.inc("agent_tool_calls", len(result.tool_calls))
    record_trace(trace)
    return result.to_dict()


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    _token: None = Depends(verify_api_token),
):
    """
    Submit feedback for a previous query response.

    Args:
        request: Feedback request with rating and optional comment
        background_tasks: FastAPI background tasks

    Returns:
        Feedback response confirming receipt

    WHAT: POST endpoint that records a thumbs-up/down rating (plus optional
    comment) for a prior query, keyed by feedback_id, and queues analytics
    processing in the background.
    WHY: Collects user judgment on answer quality so the team can identify weak
    telecom answers and improve the corpus/prompts.
    INPUTS: FeedbackRequest {feedback_id, rating (0 or 1), comment} plus auth.
    OUTPUTS: FeedbackResponse {message, feedback_id}.
    SIDE EFFECTS: appends to feedback_store (in-memory list) and schedules
        process_feedback as a background task.
    FAILURE MODES: 400 if rating is not 0 or 1; 500 on storage error; 401
        without bearer token.
    ALTERNATIVE: write feedback directly to a database/analytics pipeline
        instead of an in-memory list.
    # GAP: reliability - feedback_store is an in-memory list; feedback is lost
        on restart and not shared across API workers.
    # BUSINESS: surfaces which telecom answers users found helpful or wrong,
        driving corpus improvement.
    # INTERVIEW: "Why process feedback in a background task?" - so the client
        gets an immediate confirmation without waiting for analytics side
        effects that are not needed for the response.
    """
    global feedback_store

    try:
        # Validate rating: only thumbs-down (0) or thumbs-up (1) accepted.
        if request.rating not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail="Rating must be 0 (thumbs down) or 1 (thumbs up)"
            )

        # Store feedback (in production, this would go to a database)
        feedback_entry = {
            "feedback_id": request.feedback_id,
            "rating": request.rating,
            "comment": request.comment,
            "timestamp": time.time()
        }
        feedback_store.append(feedback_entry)

        # Log feedback in background for analytics (non-blocking for the client).
        background_tasks.add_task(
            process_feedback,
            feedback_entry=feedback_entry
        )

        return FeedbackResponse(
            message="Feedback received successfully",
            feedback_id=request.feedback_id
        )

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


# Background task functions
def log_feedback_request(feedback_id: str, query: str, user_info: dict, client_id: str):
    """Log feedback request for analytics.

    WHAT: Logs a one-line record that feedback was requested for a query.
    WHY: Provides a lightweight audit trail of which queries users chose to
    rate.
    INPUTS: feedback_id, query (truncated to 50 chars in the log), user_info,
        client_id.
    OUTPUTS: none.
    SIDE EFFECTS: writes to the application log.
    FAILURE MODES: logging failure would raise, but this runs in a background
        task so it does not fail the client request.
    ALTERNATIVE: write the feedback-request event to an analytics store.
    # BUSINESS: tracks engagement (how often users rate answers) for the telecom
        QA system.
    # INTERVIEW: "Why truncate the query to 50 chars in the log?" - to avoid
        bloating logs with long user questions and to reduce PII exposure.
    """
    logger.info(
        f"Feedback requested: ID={feedback_id}, Query='{query[:50]}...', "
        f"User={user_info['role']}, Client={client_id}"
    )


def process_feedback(feedback_entry: dict):
    """Process feedback for analytics and model improvement.

    WHAT: Logs the processed feedback entry (rating + comment) for analytics.
    WHY: Centralizes where feedback is recorded so it can later be routed to
    model-weight updates or response-flagging.
    INPUTS: feedback_entry dict {feedback_id, rating, comment, timestamp}.
    OUTPUTS: none.
    SIDE EFFECTS: writes to the application log.
    FAILURE MODES: logging failure raises, but runs in a background task so it
        does not fail the client request.
    ALTERNATIVE: persist feedback to a database and trigger retraining jobs.
    # BUSINESS: converts user thumbs-up/down into signal for improving telecom
        answer quality.
    # INTERVIEW: "Why does process_feedback only log instead of updating model
        weights?" - because this is a study-aid prototype; production would
        connect the feedback loop to actual model fine-tuning or prompt
        adjustment.
    """
    logger.info(
        f"Feedback processed: ID={feedback_entry['feedback_id']}, "
        f"Rating={feedback_entry['rating']}, Comment={feedback_entry['comment']}"
    )
    # In production, this would update model weights, flag problematic responses, etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)