"""
Text embedding utilities with incremental ingestion capabilities.
Supports batch embedding, retry/backoff, rate-limit handling, and content-hash cache.
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

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
            raise last_exception
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator for rate limiting function calls.

    Args:
        calls_per_second: Maximum calls allowed per second
    """
    min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
    last_call_time = [0.0]  # Use list to make it mutable in closure

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_call_time[0] = time.time()
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

        Args:
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to store embedding cache
            enable_cache: Whether to use content-hash caching
            requests_per_minute: Rate limit for API calls (if using remote models)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.model = None
        self._load_model()

        # Setup cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting setup
        self.requests_per_minute = requests_per_minute
        self._embed_text = rate_limit(requests_per_minute / 60.0)(self._embed_text_internal)

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "errors": 0
        }

    def _load_model(self):
        """Load the sentence transformer model."""
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
                embedding = self.model.encode([text])[0]
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error embedding text with model: {e}")
                raise
        else:
            return self._mock_embed(text)

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text:
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

        # Check cache first
        content_hash = self._get_content_hash(text)
        cached_embedding = self._load_from_cache(content_hash)
        if cached_embedding is not None:
            return cached_embedding

        # Compute embedding
        try:
            embedding = self._embed_text(text)
            self.stats["cache_misses"] += 1
            self.stats["total_embeddings"] += 1

            # Save to cache
            self._save_to_cache(content_hash, text, embedding)

            return embedding
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error embedding text: {e}")
            # Return mock embedding as fallback
            return self._mock_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text:
                embeddings.append([0.0] * 384)
                continue

            content_hash = self._get_content_hash(text)
            cached_embedding = self._load_from_cache(content_hash)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                self.stats["cache_hits"] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts in batch
        if uncached_texts:
            try:
                if self.model is not None:
                    batch_embeddings = self.model.encode(uncached_texts)
                    batch_embeddings_list = batch_embeddings.tolist()
                else:
                    batch_embeddings_list = [self._mock_embed(text) for text in uncached_texts]

                # Fill in the results
                for idx, embedding in zip(uncached_indices, batch_embeddings_list):
                    embeddings[idx] = embedding
                    # Save to cache
                    content_hash = self._get_content_hash(uncached_texts[uncached_indices.index(idx)])
                    self._save_to_cache(content_hash, uncached_texts[uncached_indices.index(idx)], embedding)

                self.stats["cache_misses"] += len(uncached_texts)
                self.stats["total_embeddings"] += len(uncached_texts)

            except Exception as e:
                self.stats["errors"] += len(uncached_texts)
                logger.error(f"Error embedding batch: {e}")
                # Fallback to mock embeddings
                for idx in uncached_indices:
                    embeddings[idx] = self._mock_embed(uncached_texts[uncached_indices.index(idx)])

        return embeddings

    def embed_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Embed a document and return it with embedding vector.

        Args:
            document: Document dictionary with content and metadata

        Returns:
            Document dictionary with added embedding
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

        Args:
            documents: List of document dictionaries

        Returns:
            List of document dictionaries with embeddings
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

        Args:
            text: Text to embed

        Returns:
            Mock embedding vector
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