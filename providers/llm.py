"""
LLM provider interface and implementations.

BUSINESS PROBLEM: This file is the abstraction layer that lets the RAG system swap
LLM backends (mock for dev/tests, NVIDIA NIM for production, plus placeholder
OpenAI/Anthropic providers) without touching retrieval, prompt, or answer code.
It defines the LLMProvider contract every backend must satisfy, plus a factory
(create_llm_provider) and an env-driven entry point (create_llm_provider_from_env).

PLACE IN DATA FLOW: Sits at the edge of the generation stack. The agent
(src/agent/agent.py) and the answer generator (src/generation/answer.py) call the
LLMProvider interface; this file returns a concrete provider. The real production
path routes through providers/llm_nvidia.py (NVIDIA NIM). Nothing upstream
(ingestion/retrieval) knows which provider is in use.

WHAT CALLS IT: src/generation/answer.py (create_answer_generator resolves a
provider from env), src/agent/agent.py (LLM planner), and anything importing
create_llm_provider / create_llm_provider_from_env.

WHAT IT CALLS: providers/llm_nvidia.py (NVIDIAProvider, lazy import), src.config
(get_settings), src.key_pool (KeyPool). Standard library: typing, logging, os, abc.

WHAT BREAKS IF REMOVED: All LLM-backed generation dies. The mock provider is the
only one that works without a network or API key; removing it breaks tests and local
dev. The NVIDIA path (production) cannot fall back to anything else.

# GAP: No timeout, retry, or backoff on the mock/placeholder providers — real
production resilience (retries, backoff, key rotation) lives only in
providers/llm_nvidia.py. A learner should not assume every provider is equally
hardened.
# BUSINESS: Protects the system from provider lock-in; lets us run offline tests
(mock) and route to NVIDIA NIM in prod through a single stable interface.
# INTERVIEW: Say it out loud — "The provider interface is a seam. Business logic
never imports a concrete provider directly; it depends on the abstract
LLMProvider, so swapping backends is a one-line factory change."
"""

from typing import List, Dict, Any, Optional
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    WHAT: Defines the contract (generate, get_model_info) every LLM backend must
    implement so the rest of the system can stay provider-agnostic.
    WHY: Lets the RAG system call any backend through one stable interface instead
    of hard-coding OpenAI/NVIDIA/Anthropic calls everywhere.
    INPUTS: None (constructor takes backend-specific config, e.g. api_key,
    model_name).
    OUTPUTS (generate): str — e.g. "Based on the context, LTE uses OFDMA...".
    OUTPUTS (get_model_info): Dict[str, Any] — e.g.
        {"provider": "mock", "model_name": "mock", "type": "mock",
         "max_tokens": 1000, "supports_streaming": True}.
    SIDE EFFECTS: None by itself; concrete subclasses may log or call networks.
    FAILURE MODES: Subclasses that forget to implement a method raise
    TypeError at instantiation (ABC abstractmethod enforcement).
    ALTERNATIVE: Use a protocol (typing.Protocol) instead of ABC for structural
    subtyping without inheritance.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        WHAT: Ask the LLM to complete a single text prompt.
        WHY: Core generation primitive used by answer generation and the agent
        planner; every provider must support it.
        INPUTS: prompt: str — e.g. "Context: 5G uses OFDM...\n\nQuestion: What
            waveform does 5G use?"; **kwargs: provider-specific params such as
            temperature=0.2 or max_tokens=512.
        OUTPUTS: str — e.g. "5G uses OFDM (orthogonal frequency-division
            multiplexing) for the downlink."
        SIDE EFFECTS: None for this abstract declaration; real implementations
            may make a network call and log.
        FAILURE MODES: Returns NotImplementedError if called on the base class
            (abstract); concrete providers may raise on network/auth failure.
        ALTERNATIVE: Use chat(messages) instead of a single prompt for
            multi-turn conversations.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        WHAT: Return metadata describing the backing model (provider, name,
        capabilities).
        WHY: Callers (answer generator, observability) need to record which model
        produced an answer for auditing and cost tracking.
        INPUTS: None.
        OUTPUTS: Dict[str, Any] — e.g.
            {"provider": "nvidia", "model_name": "meta/llama-3.1-70b-instruct",
             "type": "llm", "supports_streaming": True, "timeout_s": 60.0}.
        SIDE EFFECTS: None.
        FAILURE MODES: Abstract — calling on the base class raises
        NotImplementedError.
        ALTERNATIVE: Hard-code model info in config instead of querying the
            provider object.
        """
        pass


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing and development.

    WHAT: Deterministic fake LLM that parses a prompt for "Context"/"Question"
    markers and returns a canned answer, without any network call.
    WHY: Lets tests and local development run without an API key, network, or
    cost; never used in production.
    INPUTS (__init__): model_name: str — e.g. "mock" (recorded only, not used to
        select a real model).
    OUTPUTS (generate): str — e.g. "Based on the provided context, I can inform
        you that the information relates to what is lte..." or
        "I don't have sufficient information in the provided context...".
    SIDE EFFECTS: Logs an INFO line at init ("Initialized MockLLMProvider with
        model: ...").
    FAILURE MODES: Safe — never raises for malformed prompts; degrades to a
        generic mock response when markers are missing.
    ALTERNATIVE: Use a canned JSON fixture or record/replay real LLM responses
        instead of computing answers on the fly.
    """

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        logger.info(f"Initialized MockLLMProvider with model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a mock response from a prompt.

        WHAT: Parse the prompt for Context/Question markers and emit a plausible
        canned answer (or an insufficiency message) without calling any model.
        WHY: Provides a predictable, offline answer for unit tests and local dev
        so they don't depend on an external LLM API.
        INPUTS: prompt: str — e.g. "Context: LTE uses OFDMA...\n\nQuestion: What
            does LTE use?"; **kwargs: ignored.
        OUTPUTS: str — e.g. "Based on the provided context, I can inform you that
            the information relates to what does lte use. The context contains
            relevant details..." when context is present and long enough;
            otherwise an insufficiency message.
        SIDE EFFECTS: None.
        FAILURE MODES: Never raises; on missing/short context it returns an
            insufficiency fallback rather than a wrong answer.
        ALTERNATIVE: Use recorded real responses (cassettes) for more realism.
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

    def stream_chat(self, messages, **kwargs):
        """
        Mock streaming: yields the full mock answer in small chunks.

        WHAT: Pretend to stream by chopping the (synchronously generated) mock
        answer into 24-character slices.
        WHY: Satisfies the streaming interface so callers that expect an iterator
        don't crash in tests/dev.
        INPUTS: messages: List[dict] — e.g.
            [{"role": "user", "content": "What is LTE?"}]; **kwargs: ignored.
        OUTPUTS (yields): str chunks — e.g. "This is a mock response from the ".
        SIDE EFFECTS: Calls self.generate() synchronously (blocks; not real
        streaming).
        FAILURE MODES: None real — if messages is empty it generates from "".
        ALTERNATIVE: Stream token-by-token from a real provider instead of
        slicing a precomputed string.
        # GAP: This is fake streaming — it generates the whole answer first then
        doles it out in chunks. Real streaming (providers/llm_nvidia.py:282)
        yields tokens as the model produces them.
        """
        text = self.generate(messages[-1].get("content", "") if messages else "")
        for i in range(0, len(text), 24):
            yield text[i:i + 24]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the mock model.

        WHAT: Return metadata describing the mock backend.
        WHY: Keeps the mock interchangeable with real providers for callers that
        log model info.
        INPUTS: None.
        OUTPUTS: Dict[str, Any] — e.g.
            {"provider": "mock", "model_name": "mock", "type": "mock",
             "max_tokens": 1000, "supports_streaming": True}.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Read model info from a shared config object.
        """
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "type": "mock",
            "max_tokens": 1000,
            "supports_streaming": True
        }


# Placeholder for real LLM providers (would be implemented based on actual APIs)
class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider (placeholder).

    WHAT: Stub for an OpenAI-backed provider; stores api_key/model_name and logs
    a warning but makes no real API call.
    WHY: Reserves a slot for OpenAI integration; callers can construct it, but it
    only returns placeholder text so it must not be routed to production traffic.
    INPUTS (__init__): api_key: str — e.g. "sk-..."; model_name: str — e.g.
        "gpt-3.5-turbo".
    OUTPUTS (generate): str — e.g. "[OpenAI placeholder response for: What is
        5G...]".
    OUTPUTS (get_model_info): Dict[str, Any] — e.g.
        {"provider": "openai", "model_name": "gpt-3.5-turbo", "type": "llm",
         "max_tokens": 4000, "supports_streaming": True}.
    SIDE EFFECTS: Logs a WARNING at init that this is a placeholder.
    FAILURE MODES: generate() never calls OpenAI, so it cannot surface real auth
    or network errors; it silently returns placeholder text (a correctness risk if
    mistakenly used in prod).
    ALTERNATIVE: Use the official openai Python package instead of a hand-rolled
    stub.
    # GAP: Placeholder only — no real OpenAI client, no retries, no key handling.
    Do not send production traffic here.
    """

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
    """
    Anthropic LLM provider (placeholder).

    WHAT: Stub for an Anthropic-backed provider; stores api_key/model_name and
    logs a warning but makes no real API call.
    WHY: Reserves a slot for Anthropic integration; returns placeholder text and
    must not be routed to production traffic.
    INPUTS (__init__): api_key: str — e.g. "sk-ant-..."; model_name: str — e.g.
        "claude-3-haiku-20240307".
    OUTPUTS (generate): str — e.g. "[Anthropic placeholder response for: What is
        5G...]".
    OUTPUTS (get_model_info): Dict[str, Any] — e.g.
        {"provider": "anthropic", "model_name": "claude-3-haiku-20240307",
         "type": "llm", "max_tokens": 4000, "supports_streaming": True}.
    SIDE EFFECTS: Logs a WARNING at init that this is a placeholder.
    FAILURE MODES: generate() never calls Anthropic, so it cannot surface real
    auth or network errors; silently returns placeholder text (correctness risk
    if mistakenly used in prod).
    ALTERNATIVE: Use the official anthropic Python package instead of a stub.
    # GAP: Placeholder only — no real Anthropic client, no retries, no key
    handling. Do not send production traffic here.
    """

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

    WHAT: Build and return the right LLMProvider subclass based on provider_type.
    WHY: Centralizes provider selection so callers (answer generator, agent) don't
    instantiate concrete classes directly — keeps the system provider-agnostic.
    INPUTS: provider_type: str — e.g. "mock", "nvidia", "openai", "anthropic";
        **kwargs: forwarded to the chosen provider's constructor (api_key,
        model_name, base_url, timeout_s, max_retries, key_pool, etc.).
    OUTPUTS: LLMProvider instance — e.g. a MockLLMProvider, NVIDIAProvider,
        OpenAILLMProvider, or AnthropicLLMProvider.
    SIDE EFFECTS: For "nvidia", imports providers.llm_nvidia and src.config at
    call time (lazy import) and reads settings; may construct a KeyPool.
    FAILURE MODES: Raises ValueError(f"Unsupported provider type: {provider_type}")
    for unknown types; NVIDIA path may raise ProviderAuthError if no API key is
    configured (see providers/llm_nvidia.py:108).
    ALTERNATIVE: Let callers instantiate providers directly (loses central
    configuration and the nvidia/key-pool wiring).
    # GAP: The "nvidia" branch is the production path but lives behind an
    if-check here rather than being in the providers dict; it's special-cased.
    # BUSINESS: Keeps provider selection in one place, prevents hard-coded
    backend choices across the codebase.
    # INTERVIEW: Say it out loud — "The factory is the single seam for picking an
    LLM backend; production uses nvidia (NVIDIA NIM), dev/tests use mock."
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
    """
    Create the provider selected by the LLM_PROVIDER env var (default: mock).

    WHAT: Read LLM_PROVIDER from settings/env and delegate to create_llm_provider.
    WHY: Lets operations pick the backend via configuration (env var) without code
    changes — the standard 12-factor way to select a service implementation.
    INPUTS: None (reads settings.llm_provider internally).
    OUTPUTS: LLMProvider instance — e.g. a MockLLMProvider (default) or
        NVIDIAProvider when LLM_PROVIDER=nvidia.
    SIDE EFFECTS: Imports src.config and calls get_settings(); may construct a
    KeyPool for NVIDIA.
    FAILURE MODES: Propagates ProviderAuthError from create_llm_provider if
    provider=nvidia and NVIDIA_API_KEY is missing (fails fast with a clear error).
    ALTERNATIVE: Read os.environ["LLM_PROVIDER"] directly instead of going
    through get_settings().
    # BUSINESS: Lets ops switch backend via env (12-factor config) without
    redeploying code changes.
    # INTERVIEW: Say it out loud — "Env-driven provider selection means we can run
    the same code in local (mock) and production (nvidia) just by flipping an env
    var."
    """
    from src.config import get_settings

    return create_llm_provider(get_settings().llm_provider)
