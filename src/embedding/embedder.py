"""
TEXT EMBEDDING LAYER — rag-academy RAG system

Business problem this file solves:
    Turns raw telecom document text (LTE/5G, Roaming, SIM Security, eSIM policy chunks)
    into fixed-size numeric vectors so the vector store can later retrieve semantically
    similar chunks. Without this layer the retrieval pipeline has nothing to search
    against — every question would return zero results. This is the bridge between
    unstructured text and the vector database.

Where it sits in the data flow:
    INGESTION stage — downstream of the chunker (src/ingestion/chunker.py), upstream of
    the vector store (src/store/vector_store.py). The chunker produces document dicts
    {"content": ..., "metadata": ...}; this file turns "content" into an embedding list
    of floats and hands the enriched document to the store. Also used at query time to
    embed the user's question before retrieval.

What calls it:
    src/ingestion/manager.py (embed_documents on each chunk), src/ingestion/run.py
    (indirectly via the manager), and module-level convenience functions embed_text /
    embed_documents used by tests and one-off scripts. The retriever embeds the query
    vector before calling VectorStore.search.

What it calls:
    sentence_transformers.SentenceTransformer (local model, lazy-loaded), stdlib
    hashlib / json / time / os / pathlib / functools, logging. Falls back to a
    deterministic hash-based mock embedding when the model or package is unavailable.

What breaks if it is removed:
    Every document chunk would have no embedding vector, so ChromaDB/fallback store
    would store nothing searchable; retrieval returns zero chunks; the RAG agent answers
    "I cannot determine from available materials" for every question. The whole Q&A
    system goes silent.

Design notes for the learner (senior QA/automation engineer, weak on production
architecture / concurrency / observability / cost tradeoffs):
    - Caching (content-hash keyed JSON files under data/embedding_cache/) avoids
      re-embedding identical text — the biggest cost and latency win because embedding
      is the most expensive per-call step.
    - Retry with exponential backoff plus rate limiting protect against transient
      failures and against hammering a remote embedding API.
    - Mock fallback keeps the system testable and runnable offline (no model download,
      no network).
    - All failures degrade gracefully: never crash the ingest pipeline; return a mock
      embedding or zero vector instead of raising to the caller.

# GAP: Caching, retries/backoff, rate limiting, graceful degradation — production
  reliability concepts a web/automation engineer often treats as "someone else's layer".
  Here they are hand-rolled (decorators + try/except) rather than using a library such as
  tenacity or a managed cache. A web app usually leans on the framework or a managed
  service; this file builds them from stdlib, so the learner sees the moving parts
  (cache invalidation, backoff without jitter, closure-based state).
# BUSINESS: Protects embedding cost (API dollars), ingestion latency, and offline
  testability; a wrong vector dimension silently poisons search results downstream.
# INTERVIEW: "How do you avoid re-embedding the same paragraph on every ingest run?"
  Say it out loud: content-hash cache keyed by the text itself — identical text hits
  cache, changed text misses and gets re-embedded.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
import logging
import hashlib
import time
import json
import os
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator for retrying functions with exponential backoff.

    WHAT: Wraps a function so that, on any exception, it retries up to max_retries times,
          waiting longer between each attempt (exponential backoff capped at max_delay).
    WHY: Embedding calls (local or remote) can fail transiently — a flaky model load, a
         network blip to a remote embedding API, or a temporary file lock. Retrying
         instead of failing fast keeps the ingest pipeline alive.
    INPUTS:
        max_retries (int): how many retry attempts after the first failure. Example: 3.
        base_delay (float): starting wait in seconds. Example: 1.0.
        max_delay (float): ceiling on the wait in seconds. Example: 60.0.
        func (callable): the wrapped function, e.g. _embed_text_internal(text).
    OUTPUTS: returns func's return value on success; raises the last exception if every
             attempt fails (e.g. raises the original Exception after 4 total attempts).
    SIDE EFFECTS: sleeps the calling thread between attempts (blocks), logs a warning on
                  each retry and an error when all attempts are exhausted.
    FAILURE MODES: if the function keeps raising, the final exception propagates and can
                   crash the caller unless the caller catches it (embed_text catches it).
    ALTERNATIVE: line — use a battle-tested retry library such as tenacity
                 (stop_after_attempt, wait_exponential) instead of hand-rolling this.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):  # first attempt + max_retries retries
                try:
                    return func(*args, **kwargs)  # success: short-circuit out of retry loop
                except Exception as e:
                    last_exception = e  # remember the most recent failure to re-raise later
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)  # exponential: 1s, 2s, 4s... capped
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)  # block the thread before the next attempt
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
            raise last_exception  # every attempt exhausted; surface the last error
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator for rate limiting function calls.

    WHAT: Wraps a function so calls are spaced at least min_interval apart, sleeping the
          thread when a call arrives too soon after the previous one.
    WHY: Remote embedding APIs enforce per-second quotas; without pacing, a batch of
         documents could burst past the limit and get HTTP 429s. Local model calls are
         cheap but the same guard keeps behavior uniform.
    INPUTS:
        calls_per_second (float): max calls allowed per second. Example: 1.0 = one call
                                  per second.
        func (callable): the wrapped function, e.g. _embed_text_internal.
    OUTPUTS: returns func's return value; the wrapper itself is returned by the decorator.
    SIDE EFFECTS: may sleep (block) the calling thread; mutates last_call_time (closure
                  state) on every call — not thread-safe.
    FAILURE MODES: none raised by this decorator itself; if the wrapped function raises,
                   that exception propagates unchanged.
    ALTERNATIVE: line — use a token-bucket library (e.g. limits, bottleneck) or the API
                 client's built-in retry-after handling instead of a home-grown sleep loop.
    """
    min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
    last_call_time = [0.0]  # Use list to make it mutable in closure (mutable default trick)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time[0]  # time since the previous call
            left_to_wait = min_interval - elapsed  # how much longer to hold before calling
            if left_to_wait > 0:
                time.sleep(left_to_wait)  # pace the call to respect the quota
            ret = func(*args, **kwargs)
            last_call_time[0] = time.time()  # stamp when this call finished
            return ret
        return wrapper
    return decorator


class Embedder:
    """Handles text embedding with caching and incremental capabilities."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "./data/embedding_cache",
                 enable_cache: bool = True,
                 requests_per_minute: int = 60):
        """
        Initialize the embedder.

        WHAT: Builds an Embedder: loads (or mock-stubs) the sentence-transformer model,
              prepares the cache directory, wires rate limiting onto _embed_text_internal,
              and resets the statistics counters.
        WHY: Construction is the single place where the expensive model is loaded once,
             rather than reloaded per document. It also decides whether caching is on and
             what query rate to pace calls at.
        INPUTS:
            model_name (str): sentence-transformers model identifier. Example:
                              "all-MiniLM-L6-v2" (384-dimensional output).
            cache_dir (str): folder for content-hash JSON cache files. Example:
                             "./data/embedding_cache".
            enable_cache (bool): whether to read/write the disk cache. Example: True.
            requests_per_minute (int): remote-API call ceiling. Example: 60 (=> 1/sec).
        OUTPUTS: None (constructs the instance; self.model, self.cache_dir, self.stats,
                 self._embed_text are set).
        SIDE EFFECTS: may download the model from Hugging Face on first load (network);
                      creates cache_dir on disk; assigns the rate-limited wrapper to
                      self._embed_text (rebinding the method).
        FAILURE MODES: if sentence_transformers is missing or the model fails to load,
                       self.model stays None and the mock embedder is used — no exception
                       escapes the constructor (all caught and logged).
        ALTERNATIVE: line — pre-download and bundle the model weights offline, or use a
                     remote embedding API client (NVIDIA NIM, OpenAI) instead of a local
                     SentenceTransformer.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)  # convert to Path for / operator later
        self.enable_cache = enable_cache
        self.model = None  # placeholder until _load_model runs
        self._load_model()  # load the model now (may set self.model = None on failure)

        # Setup cache directory: create it (and parents) if caching is enabled
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting setup: pace remote/API calls at requests_per_minute per minute
        self.requests_per_minute = requests_per_minute
        self._embed_text = rate_limit(requests_per_minute / 60.0)(self._embed_text_internal)

        # Statistics: cumulative counters for hit/miss/total/error tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "errors": 0
        }

    def _load_model(self):
        """
        Load the sentence transformer model.

        WHAT: Imports sentence_transformers and instantiates SentenceTransformer(model_name),
              storing it in self.model. Logs success; on any failure sets self.model = None.
        WHY: The model is heavy (hundreds of MB) and slow to load, so it is loaded once at
             construction and reused. If it cannot be loaded, the system still runs via the
             mock embedder instead of crashing startup.
        INPUTS: None (uses self.model_name from the instance).
        OUTPUTS: None; sets self.model (SentenceTransformer or None) and logs.
        SIDE EFFECTS: network download of model weights on first use; logs info/warning/error.
        FAILURE MODES: ImportError (package not installed) -> mock; any other Exception ->
                       mock. Callers must tolerate self.model being None.
        ALTERNATIVE: line — lazy-load on first embed_text call instead of at construction,
                     to speed up instantiation when embedding is never used.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using mock embedder.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def _get_content_hash(self, text: str) -> str:
        """Generate a hash for content-based caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cache_path(self, content_hash: str) -> Path:
        """Get the cache file path for a content hash."""
        return self.cache_dir / f"{content_hash}.json"

    def _load_from_cache(self, content_hash: str) -> Optional[List[float]]:
        """Load embedding from cache if it exists."""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(content_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                # Verify the cached content matches
                if cache_data.get("content_hash") == content_hash:
                    self.stats["cache_hits"] += 1
                    return cache_data["embedding"]
            except Exception as e:
                logger.warning(f"Failed to load from cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, content_hash: str, text: str, embedding: List[float]):
        """Save embedding to cache."""
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(content_hash)
        try:
            cache_data = {
                "content_hash": content_hash,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "text_length": len(text),
                "embedding": embedding,
                "model_name": self.model_name,
                "cached_at": time.time()
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_path}: {e}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
    def _embed_text_internal(self, text: str) -> List[float]:
        """Internal embedding method with retry logic."""
        if self.model is not None:
            try:
                embedding = self.model.encode([text])[0]  # encode returns array; take row 0
                return embedding.tolist()  # convert numpy array to plain Python list of floats
            except Exception as e:
                logger.error(f"Error embedding text with model: {e}")
                raise  # let the retry decorator handle it
        else:
            return self._mock_embed(text)

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string with caching.

        WHAT: Returns the embedding vector for one piece of text, checking the disk cache
              first and computing (with retry/rate-limit) only on a cache miss.
        WHY: Single-text embedding is the primitive used by embed_document and at query
             time. Caching avoids paying the model cost for repeated or unchanged chunks.
        INPUTS:
            text (str): the text to embed. Example: "eSIM profile download uses HTTPS."
        OUTPUTS: List[float] — the embedding vector. Example (mock, 384-d): [0.12, -0.05,
                 ...] length 384; empty input returns [0.0]*384.
        SIDE EFFECTS: reads/writes cache files on disk; updates self.stats; may sleep for
                      rate limiting; logs errors on failure.
        FAILURE MODES: if embedding fails even after retries, returns a mock embedding
                       (degraded) rather than raising; empty/None text returns zeros.
        ALTERNATIVE: line — call the model directly without caching for one-off queries
                     where cache disk I/O is more expensive than the model forward pass.
        # GAP: Caching and graceful degradation — the learner may expect every failure to
        #      surface as an exception; here errors are swallowed and replaced with a mock
        #      vector, which is production-safe but can mask real problems (e.g. a bad
        #      model silently producing useless vectors).
        # BUSINESS: Keeps a single-chunk ingest or a user query from failing the whole
        #           request; protects per-call embedding cost via the cache.
        # INTERVIEW: "What happens if you embed an empty string?" Say it out loud: it
        #            short-circuits to a 384-dimensional zero vector, never touches the
        #            model or the cache.
        """
        if not text:
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2; no model work

        # Check cache first: hash the text, read the JSON cache file if present
        content_hash = self._get_content_hash(text)
        cached_embedding = self._load_from_cache(content_hash)
        if cached_embedding is not None:
            return cached_embedding  # cache hit: return stored vector, skip model entirely

        # Compute embedding: rate-limited + retried internal call
        try:
            embedding = self._embed_text(text)
            self.stats["cache_misses"] += 1
            self.stats["total_embeddings"] += 1

            # Save to cache so the next identical text is a hit
            self._save_to_cache(content_hash, text, embedding)

            return embedding
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error embedding text: {e}")
            # Return mock embedding as fallback: degrade gracefully, never raise to caller
            return self._mock_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with caching.

        WHAT: Embeds many texts at once, reusing cached vectors for hits and computing only
              the uncached ones in a single model.encode batch call.
        WHY: Batching is far more efficient than one-by-one calls — the model processes a
             matrix in one forward pass, and only misses go to disk for caching.
        INPUTS:
            texts (List[str]): list of text strings. Example:
                               ["eSIM profile download uses HTTPS.", "Roaming charges apply."]
        OUTPUTS: List[List[float]] — one embedding per input text, same order. Example:
                 [[0.12, ...], [-0.04, ...]] each length 384; empty input returns [].
        SIDE EFFECTS: reads/writes cache files (one per miss); updates self.stats in bulk;
                      may sleep/rate-limit; logs errors on batch failure.
        FAILURE MODES: if the batch model call fails, falls back to per-text mock
                       embeddings (degraded); empty strings yield zero vectors without
                       model calls.
        ALTERNATIVE: line — embed each text individually if cache hit rates are so high
                     that batching the misses is not worth the complexity.
        """
        if not texts:
            return []

        # Check cache for each text: split into cached hits and uncached misses
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text:
                embeddings.append([0.0] * 384)  # empty string -> zero vector, skip cache/model
                continue

            content_hash = self._get_content_hash(text)
            cached_embedding = self._load_from_cache(content_hash)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                self.stats["cache_hits"] += 1
            else:
                embeddings.append(None)  # Placeholder to be filled after batch compute
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts in batch: a single model.forward over the whole miss list
        if uncached_texts:
            try:
                if self.model is not None:
                    batch_embeddings = self.model.encode(uncached_texts)  # numpy array (n, dim)
                    batch_embeddings_list = batch_embeddings.tolist()
                else:
                    batch_embeddings_list = [self._mock_embed(text) for text in uncached_texts]

                # Fill in the results: place each computed vector at its original index
                for idx, embedding in zip(uncached_indices, batch_embeddings_list):
                    embeddings[idx] = embedding
                    # Save to cache: look up the text by its position in uncached_texts
                    content_hash = self._get_content_hash(uncached_texts[uncached_indices.index(idx)])
                    self._save_to_cache(content_hash, uncached_texts[uncached_indices.index(idx)], embedding)

                self.stats["cache_misses"] += len(uncached_texts)
                self.stats["total_embeddings"] += len(uncached_texts)

            except Exception as e:
                self.stats["errors"] += len(uncached_texts)
                logger.error(f"Error embedding batch: {e}")
                # Fallback to mock embeddings for every missed text
                for idx in uncached_indices:
                    embeddings[idx] = self._mock_embed(uncached_texts[uncached_indices.index(idx)])

        return embeddings

    def embed_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Embed a document and return it with embedding vector.

        WHAT: Takes a document dict {"content": ..., "metadata": ...}, embeds "content",
              and returns a new dict with the embedding and enriched metadata.
        WHY: Ingestion needs each chunk carried along with its vector and metadata flags
             (embedded, model, dimension, cached) so the store can persist a complete
             record.
        INPUTS:
            document (Dict[str, Any]): must contain "content" (str) and "metadata" (dict).
                                       Example: {"content": "eSIM uses HTTPS.",
                                                 "metadata": {"filename": "esim.pdf"}}
        OUTPUTS: Dict[str, Any] — the original document plus "embedding" (List[float]) and
                 metadata merged with {"embedded": True, "embedding_model": ...,
                 "embedding_dimension": int, "embedding_cached": bool}.
        SIDE EFFECTS: reads/writes cache via embed_text; does not mutate the input dict
                      (builds a new dict with **document).
        FAILURE MODES: KeyError if "content" or "metadata" is missing from the document;
                       embed_text failures degrade to a mock vector (no raise).
        ALTERNATIVE: line — mutate the input dict in place to avoid copying metadata, if
                     callers do not need the original preserved.
        """
        embedding = self.embed_text(document["content"])

        return {
            **document,
            "embedding": embedding,
            "metadata": {
                **document["metadata"],
                "embedded": True,
                "embedding_model": self.model_name,
                "embedding_dimension": len(embedding),
                "embedding_cached": self._is_cached(document["content"])
            }
        }

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents.

        WHAT: Extracts "content" from every document, embeds them in one batch, and returns
              each document enriched with its vector and metadata flags.
        WHY: This is the ingestion entry point — manager.py calls it per batch of chunks,
             so batching keeps the whole ingest run fast.
        INPUTS:
            documents (List[Dict[str, Any]]): list of document dicts, each with "content"
                                              and "metadata". Example:
                                              [{"content": "...", "metadata": {...}}, ...]
        OUTPUTS: List[Dict[str, Any]] — one enriched document per input, same order.
        SIDE EFFECTS: reads/writes cache via embed_batch; builds new dicts (no input
                      mutation).
        FAILURE MODES: KeyError if any document lacks "content"/"metadata"; batch
                       failures degrade to mock vectors per chunk (no raise to caller).
        ALTERNATIVE: line — embed and store documents one at a time if memory for the
                     full batch is a concern (trades speed for lower peak RAM).
        """
        texts = [doc["content"] for doc in documents]
        embeddings = self.embed_batch(texts)

        embedded_documents = []
        for doc, embedding in zip(documents, embeddings):
            embedded_documents.append({
                **doc,
                "embedding": embedding,
                "metadata": {
                    **doc["metadata"],
                    "embedded": True,
                    "embedding_model": self.model_name,
                    "embedding_dimension": len(embedding),
                    "embedding_cached": self._is_cached(doc["content"])
                }
            })

        return embedded_documents

    def _is_cached(self, text: str) -> bool:
        """Check if text embedding is in cache."""
        if not self.enable_cache or not text:
            return False
        content_hash = self._get_content_hash(text)
        cache_path = self._get_cache_path(content_hash)
        return cache_path.exists()

    def _mock_embed(self, text: str) -> List[float]:
        """
        Generate a mock embedding for testing purposes.

        WHAT: Builds a deterministic 384-dimensional vector by hashing the text and
              spreading its hex bytes across the dimension, padded/truncated to 384.
        WHY: Provides a no-network, no-model fallback so tests and offline runs produce
             stable vectors (same text => same vector) without downloading anything.
        INPUTS:
            text (str): text to hash. Example: "eSIM profile download."
        OUTPUTS: List[float] — exactly 384 floats, each in [-1, 1]. Example:
                 [0.47, -0.12, 0.91, ...].
        SIDE EFFECTS: none (pure function of text).
        FAILURE MODES: none — md5 always succeeds; the padding loop terminates because
                       it re-extends from the existing (non-empty) list.
        ALTERNATIVE: line — use a real small model or random-but-seeded vectors for tests
                     that need statistical properties the hash trick lacks.
        """
        # Simple hash-based mock embedding
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()

        # Convert to list of floats between -1 and 1
        mock_embedding = []
        for i in range(0, len(hash_hex), 2):
            hex_pair = hash_hex[i:i+2]
            value = int(hex_pair, 16) / 255.0 * 2 - 1  # Scale to [-1, 1]
            mock_embedding.append(value)

        # Pad or truncate to 384 dimensions
        while len(mock_embedding) < 384:
            mock_embedding.extend(mock_embedding[:min(len(mock_embedding), 384 - len(mock_embedding))])

        return mock_embedding[:384]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_size = 0
        if self.cache_dir.exists():
            cache_size = len(list(self.cache_dir.glob("*.json")))

        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / max(total_requests, 1)) * 100

        return {
            **self.stats,
            "cache_size": cache_size,
            "hit_rate_percent": round(hit_rate, 2)
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        # Reset stats
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "errors": 0
        }
        logger.info("Embedding cache cleared")


# Convenience functions
def embed_text(text: str,
               model_name: str = "all-MiniLM-L6-v2",
               cache_dir: str = "./data/embedding_cache") -> List[float]:
    """Embed a single text string."""
    embedder = Embedder(model_name=model_name, cache_dir=cache_dir)
    return embedder.embed_text(text)


def embed_documents(documents: List[Dict[str, Any]],
                   model_name: str = "all-MiniLM-L6-v2",
                   cache_dir: str = "./data/embedding_cache") -> List[Dict[str, Any]]:
    """Embed a list of documents."""
    embedder = Embedder(model_name=model_name, cache_dir=cache_dir)
    return embedder.embed_documents(documents)