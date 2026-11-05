"""
FastAPI application for the RAG system - Enhanced for production deployment.
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
class QueryCache:
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _get_key(self, request: dict) -> str:
        # Create a hash of the request parameters for caching
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
class RateLimiter:
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
rag_pipeline = None
from src.ingestion.manager import DocumentManager  # noqa: E402
doc_manager = DocumentManager()
query_cache = QueryCache(max_size=200)
rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 requests per minute
feedback_store = []  # In production, this would be a database

# User roles and permissions (simplified)
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
    """Initialize the RAG pipeline components."""
    global rag_pipeline

    logger.info("Initializing RAG pipeline for API...")

    from src.embedding.embedder import Embedder
    from src.retrieval.sparse_retriever import create_bm25_retriever

    # Load and process documents
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
app = FastAPI(
    title="RAG Academy API",
    description="API for the RAG Academy telecom domain question answering system",
    version="2.0.0"
)

# Middleware for request logging (structured, with request id + metrics)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Assign a request id, log JSON request/response, record metrics + trace."""
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
    """Initialize the RAG pipeline on startup."""
    from src.config import get_settings
    from src.observability import setup_logging

    setup_logging(level=get_settings().log_level, structured=True)
    logger.info("structured JSON logging enabled")
    initialize_rag_pipeline()


@app.get("/metrics")
async def get_metrics(_token: None = Depends(verify_api_token)):
    """In-memory metrics snapshot: counters, latency percentiles, tokens, errors."""
    from src.observability import metrics

    return metrics.snapshot()


@app.get("/traces")
async def get_traces(limit: int = 20, _token: None = Depends(verify_api_token)):
    """Recent per-request traces (pipeline stages with timings)."""
    from src.observability import get_recent_traces

    return {"traces": get_recent_traces(limit)}


@app.get("/provider")
async def provider_status(_token: None = Depends(verify_api_token)):
    """Which LLM provider/model is active — makes a mock impossible to hide."""
    prov = rag_pipeline.get("llm_provider") if rag_pipeline else None
    if prov is None:
        return {"provider": "none", "model": None, "note": "pipeline not initialized"}
    info = prov.get_model_info()
    return {"provider": info.get("provider"), "model": info.get("model_name"),
            "is_real": info.get("provider") != "mock"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with enhanced metrics."""
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
    """
    global rag_pipeline, query_cache, rate_limiter, feedback_store

    # Rate limiting
    client_id = x_forwarded_for or (client_request.client.host if client_request.client else "unknown")
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

    # Check cache first
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
        # Cached responses report minimal latency
        return cached_result.model_copy(update={"latency_ms": 1.0})

    start_time = time.time()

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    from src.observability import get_request_id, metrics, Trace, record_trace
    trace = Trace(request_id=get_request_id())

    try:
        # Generate feedback ID for tracking
        feedback_id = str(uuid.uuid4())

        logger.info(f"Processing query from user with access level: {user_info['access_level']} ({user_info['role']})")

        # Retrieve documents (traced)
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

        # Rerank if requested and available (traced)
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
            final_docs = retrieved_docs[:request.top_k]
            final_scores = scores[:request.top_k]

        # Generate answer with the real provider (traced); record usage
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

        # Format sources for response
        sources = []
        for doc in final_docs:
            metadata = doc.get("metadata", {})
            source_info = {
                "source": metadata.get("source", "Unknown"),
                "content_snippet": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "metadata": metadata
            }
            sources.append(source_info)

        # Format context used
        context_used = format_context(final_docs)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Prepare response
        response = QueryResponse(
            answer=answer_text,
            sources=sources,
            context_used=context_used,
            query=request.query,
            latency_ms=round(latency_ms, 2),
            feedback_id=feedback_id
        )

        # Cache the result
        query_cache.set(cache_request, response)

        # Log feedback request in background
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

        # Format results for response
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
    """
    global rag_pipeline

    # Check permissions
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
            # Add to existing vector store
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

        # Get statistics
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
    """
    global rag_pipeline

    # Check permissions
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
    content hash, versions on filename collision, and indexes it immediately."""
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
    """List all indexed source documents with chunk counts and version info."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return {"documents": doc_manager.list_documents(rag_pipeline["vector_store"])}


@app.get("/documents/status")
async def ingest_status(
    limit: int = 50,
    _token: None = Depends(verify_api_token),
):
    """Recent ingestion operations (ok/duplicate/superseded/failed)."""
    return {"records": doc_manager.status_log(limit)}


@app.get("/documents/{source}/status")
async def document_status(
    source: str,
    _token: None = Depends(verify_api_token),
):
    """Latest ingest status for one document."""
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
    """Rebuild the vector store from uploads/ + data dir. Admin only."""
    if not user_info["can_ingest"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions to reindex")
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    safe_dir = resolve_safe_ingest_dir(data_dir)
    result = doc_manager.reindex(rag_pipeline["vector_store"], data_dir=str(safe_dir))
    return result


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
    """
    global feedback_store

    try:
        # Validate rating
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

        # Log feedback in background for analytics
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
    """Log feedback request for analytics."""
    logger.info(
        f"Feedback requested: ID={feedback_id}, Query='{query[:50]}...', "
        f"User={user_info['role']}, Client={client_id}"
    )

def process_feedback(feedback_entry: dict):
    """Process feedback for analytics and model improvement."""
    logger.info(
        f"Feedback processed: ID={feedback_entry['feedback_id']}, "
        f"Rating={feedback_entry['rating']}, Comment={feedback_entry['comment']}"
    )
    # In production, this would update model weights, flag problematic responses, etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)