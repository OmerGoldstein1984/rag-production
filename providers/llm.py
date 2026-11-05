"""
LLM provider interface and implementations.
"""

from typing import List, Dict, Any, Optional
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        pass

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        logger.info(f"Initialized MockLLMProvider with model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a mock response from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters (ignored)

        Returns:
            Mock generated text
        """
        # Simple mock response based on prompt content
        if "context" in prompt.lower() and "question" in prompt.lower():
            # Extract context and question for a more meaningful mock response
            import re

            context_match = re.search(r"Context: (.*?)\n\nQuestion:", prompt, re.DOTALL | re.IGNORECASE)
            question_match = re.search(r"Question: (.*?)\n", prompt, re.DOTALL | re.IGNORECASE)

            context = context_match.group(1).strip() if context_match else ""
            question = question_match.group(1).strip() if question_match else ""

            if context and len(context.strip()) > 10:
                # Generate a mock answer that references the context
                return f"Based on the provided context, I can inform you that the information relates to {question.lower()}. The context contains relevant details that would allow for a comprehensive answer to this question."
            else:
                return "I don't have sufficient information in the provided context to answer this question accurately."
        else:
            return "This is a mock response from the LLM provider."

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model."""
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "type": "mock",
            "max_tokens": 1000,
            "supports_streaming": False
        }

# Placeholder for real LLM providers (would be implemented based on actual APIs)
class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider (placeholder)."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        # In real implementation: initialize OpenAI client
        self.api_key = api_key
        self.model_name = model_name
        logger.warning("OpenAILLMProvider is a placeholder - real implementation would use OpenAI API")

    def generate(self, prompt: str, **kwargs) -> str:
        # Placeholder implementation
        return f"[OpenAI placeholder response for: {prompt[:50]}...]"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "type": "llm",
            "max_tokens": 4000,
            "supports_streaming": True
        }

class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider (placeholder)."""

    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        # In real implementation: initialize Anthropic client
        self.api_key = api_key
        self.model_name = model_name
        logger.warning("AnthropicLLMProvider is a placeholder - real implementation would use Anthropic API")

    def generate(self, prompt: str, **kwargs) -> str:
        # Placeholder implementation
        return f"[Anthropic placeholder response for: {prompt[:50]}...]"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "type": "llm",
            "max_tokens": 4000,
            "supports_streaming": True
        }

def create_llm_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider to create ("mock", "nvidia", "openai", "anthropic")
        **kwargs: Additional arguments for provider initialization

    Returns:
        LLMProvider instance

    "nvidia" is the production provider (NVIDIA NIM, OpenAI-compatible).
    "mock" is for local development and tests only — never use it in prod.
    """
    if provider_type.lower() == "nvidia":
        from .llm_nvidia import NVIDIAProvider  # lazy import: httpx only needed for real calls
        from src.config import get_settings
        from src.key_pool import KeyPool

        settings = get_settings()
        # Prefer the multi-key pool (NVIDIA_API_KEYS); fall back to the single key.
        if settings.nvidia_api_keys:
            pool = KeyPool(keys=list(settings.nvidia_api_keys))
            return NVIDIAProvider(
                model=settings.nvidia_model,
                base_url=settings.nvidia_base_url,
                timeout_s=settings.nvidia_timeout_s,
                max_retries=settings.nvidia_max_retries,
                key_pool=pool,
                **kwargs,
            )
        return NVIDIAProvider(
            api_key=settings.nvidia_api_key or "",
            model=settings.nvidia_model,
            base_url=settings.nvidia_base_url,
            timeout_s=settings.nvidia_timeout_s,
            max_retries=settings.nvidia_max_retries,
            **kwargs,
        )

    providers = {
        "mock": MockLLMProvider,
        "openai": OpenAILLMProvider,
        "anthropic": AnthropicLLMProvider
    }

    provider_class = providers.get(provider_type.lower())
    if provider_class is None:
        raise ValueError(f"Unsupported provider type: {provider_type}")

    return provider_class(**kwargs)


def create_llm_provider_from_env() -> LLMProvider:
    """Create the provider selected by the LLM_PROVIDER env var (default: mock).

    Fails fast with a clear error if provider=nvidia and NVIDIA_API_KEY is missing.
    """
    from src.config import get_settings

    return create_llm_provider(get_settings().llm_provider)