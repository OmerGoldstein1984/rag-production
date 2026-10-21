"""
Embedding provider interface and implementations.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        pass

class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing and development."""

    def __init__(self, model_name: str = "mock", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        logger.info(f"Initialized MockEmbeddingProvider with model: {model_name}, dimension: {dimension}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a mock embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Mock embedding vector
        """
        if not text:
            return [0.0] * self.dimension

        # Simple hash-based mock embedding
        import hashlib
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()

        # Convert to list of floats between -1 and 1
        mock_embedding = []
        for i in range(0, len(hash_hex), 2):
            if i + 1 < len(hash_hex):
                hex_pair = hash_hex[i:i+2]
                value = int(hex_pair, 16) / 255.0 * 2 - 1  # Scale to [-1, 1]
                mock_embedding.append(value)

        # Pad or truncate to desired dimension
        while len(mock_embedding) < self.dimension:
            mock_embedding.extend(mock_embedding[:min(len(mock_embedding), self.dimension - len(mock_embedding))])

        return mock_embedding[:self.dimension]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of mock embedding vectors
        """
        return [self.embed_text(text) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock embedding model."""
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "type": "embedding",
            "dimension": self.dimension,
            "max_sequence_length": 512
        }

# Placeholder for real embedding providers
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformer embedding provider (placeholder)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # In real implementation: initialize SentenceTransformer model
        self.model_name = model_name
        # Set dimension based on model
        self.dimension = 384 if "MiniLM" in model_name else 768  # Simplified
        logger.warning(f"SentenceTransformerEmbeddingProvider is a placeholder - real implementation would use sentence-transformers")

    def embed_text(self, text: str) -> List[float]:
        # Placeholder implementation
        import hashlib
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()

        mock_embedding = []
        for i in range(0, len(hash_hex), 2):
            if i + 1 < len(hash_hex):
                hex_pair = hash_hex[i:i+2]
                value = int(hex_pair, 16) / 255.0 * 2 - 1
                mock_embedding.append(value)

        # Pad or truncate to dimension
        while len(mock_embedding) < self.dimension:
            mock_embedding.extend(mock_embedding[:min(len(mock_embedding), self.dimension - len(mock_embedding))])

        return mock_embedding[:self.dimension]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "sentence_transformers",
            "model_name": self.model_name,
            "type": "embedding",
            "dimension": self.dimension,
            "max_sequence_length": 512
        }

def create_embedding_provider(provider_type: str = "mock", **kwargs) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider_type: Type of provider to create ("mock", "sentence_transformers")
        **kwargs: Additional arguments for provider initialization

    Returns:
        EmbeddingProvider instance
    """
    providers = {
        "mock": MockEmbeddingProvider,
        "sentence_transformers": SentenceTransformerEmbeddingProvider
    }

    provider_class = providers.get(provider_type.lower())
    if provider_class is None:
        raise ValueError(f"Unsupported provider type: {provider_type}")

    return provider_class(**kwargs)