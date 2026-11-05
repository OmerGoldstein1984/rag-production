"""
Answer generation utilities for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import time
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Result of answer generation."""
    query: str
    answer: str
    context: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: float
    token_counts: Dict[str, int]
    prompt_version: str
    metadata: Dict[str, Any]


class AnswerGenerator:
    """Generates answers using language models with retrieved context.

    Uses a real LLMProvider (e.g. NVIDIA NIM) when one is supplied; otherwise
    falls back to the deterministic mock (tests / local dev only).
    """

    def __init__(self, model_name: str = "mock", llm_provider: Any = None):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the language model to use (recorded in metadata)
            llm_provider: An LLMProvider instance (e.g. NVIDIAProvider). If None,
                the mock generator is used (never in production).
        """
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.model = llm_provider  # kept for backward-compat attribute name

        # Token estimation constants (rough approximations)
        self.chars_per_token = 4.0  # Average for English text

    def _load_model(self):
        """Kept for backward compatibility; provider is injected via __init__."""
        logger.info(f"Initialized answer generator with model: {self.model_name} "
                    f"(provider={'real' if self.llm_provider else 'mock'})")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0
        return max(1, len(text) // self.chars_per_token)

    def _build_context_within_budget(self,
                                   documents: List[Dict[str, Any]],
                                   max_tokens: int = 2000) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Build context string respecting token budget.

        Args:
            documents: List of retrieved document dictionaries
            max_tokens: Maximum tokens for context

        Returns:
            Tuple of (context_string, used_documents, token_stats)
        """
        if not documents:
            return "", [], {"estimated_tokens": 0, "available_tokens": max_tokens, "used_tokens": 0}

        used_documents = []
        context_parts = []
        used_tokens = 0

        for doc in documents:
            content = doc.get("content", "")

            # Format document with source information
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            source_filename = source.split("/")[-1].split("\\")[-1] if source else "Unknown"
            chunk_info = f" (Chunk {metadata.get('chunk_index', '?')}" + \
                        (f"/{metadata.get('total_chunks', '?')}" if 'total_chunks' in metadata else "") + ")"

            formatted_doc = f"[Source: {source_filename}{chunk_info}]\n{content}\n"

            # Estimate tokens for this document
            doc_tokens = self._estimate_tokens(formatted_doc)

            # Check if adding this document would exceed budget
            if used_tokens + doc_tokens > max_tokens:
                logger.info(f"Context budget limit reached: {used_tokens}/{max_tokens} tokens")
                break

            context_parts.append(formatted_doc)
            used_documents.append(doc)
            used_tokens += doc_tokens

        context = "\n---\n".join(context_parts)

        # Calculate final token stats
        token_stats = {
            "estimated_tokens": used_tokens,
            "available_tokens": max_tokens,
            "used_tokens": used_tokens,
            "remaining_tokens": max_tokens - used_tokens
        }

        return context, used_documents, token_stats

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer from a prompt.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated answer string
        """
        if self.llm_provider is not None:
            try:
                return self.llm_provider.generate(prompt)
            except Exception as e:
                logger.error(f"LLM provider failed, returning insufficient-evidence fallback: {e}")
                return ("I don't have enough information to answer this question based on the "
                        "provided documents. (LLM provider error)")
        return self._mock_generate(prompt)

    def generate_answer_with_context(self,
                                    query: str,
                                    context_documents: List[Dict[str, Any]],
                                    max_context_tokens: int = 2000) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query: User query string
            context_documents: List of retrieved document dictionaries
            max_context_tokens: Maximum tokens to use for context

        Returns:
            GenerationResult containing answer and metadata
        """
        start_time = time.time()

        # Build context within token budget
        context, used_documents, token_stats = self._build_context_within_budget(
            context_documents, max_context_tokens)

        # Create grounded prompt
        from .prompt import create_grounded_prompt
        prompt = create_grounded_prompt(context, query, "v2")

        # Estimate prompt tokens
        prompt_tokens = self._estimate_tokens(prompt)
        token_stats["prompt_tokens"] = prompt_tokens
        token_stats["estimated_total_tokens"] = token_stats["estimated_tokens"] + prompt_tokens

        # Generate answer
        answer_start = time.time()
        answer = self.generate_answer(prompt)
        answer_latency = (time.time() - answer_start) * 1000

        # Capture provider-reported usage when available (real providers only)
        provider_info: Dict[str, Any] = {"provider": "mock", "model": self.model_name}
        usage = None
        if self.llm_provider is not None:
            info = self.llm_provider.get_model_info()
            provider_info = {"provider": info.get("provider", "unknown"),
                             "model": info.get("model_name", self.model_name)}
            last = getattr(self.llm_provider, "last_result", None)
            if last is not None and getattr(last, "usage", None):
                usage = {
                    "prompt_tokens": last.usage.prompt_tokens,
                    "completion_tokens": last.usage.completion_tokens,
                    "total_tokens": last.usage.total_tokens,
                    "source": "provider-reported",
                }

        # Extract sources from context documents
        sources = []
        for doc in used_documents:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            source_filename = source.split("/")[-1].split("\\")[-1] if source else "Unknown"

            source_info = {
                "source": source,
                "source_filename": source_filename,
                "content_snippet": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "metadata": metadata,
                "chunk_index": metadata.get("chunk_index"),
                "total_chunks": metadata.get("total_chunks")
            }
            sources.append(source_info)

        total_latency = (time.time() - start_time) * 1000

        # Create result
        result = GenerationResult(
            query=query,
            answer=answer,
            context=context,
            sources=sources,
            retrieved_chunks=used_documents,
            latency_ms=total_latency,
            token_counts=token_stats,
            prompt_version="v2",
            metadata={
                "model_used": provider_info["model"],
                "provider": provider_info["provider"],
                "num_context_docs": len(used_documents),
                "context_length": len(context),
                "answer_generation_latency_ms": answer_latency,
                **({"usage": usage} if usage else {}),
            }
        )

        logger.info(f"Generated answer in {total_latency:.2f}ms "
                   f"({token_stats.get('prompt_tokens', 0)} prompt + "
                   f"{token_stats.get('estimated_tokens', 0)} context tokens) "
                   f"provider={provider_info['provider']} model={provider_info['model']}")

        return result

    def stream_answer_with_context(self,
                                   query: str,
                                   context_documents: List[Dict[str, Any]],
                                   max_context_tokens: int = 2000):
        """Streaming RAG answer. Yields dict events:
        {"type":"token","text":...} ... then a final
        {"type":"done","sources":...,"usage":...,"latency_ms":...,"provider":...,"model":...}.
        Falls back to one-shot if the provider can't stream.
        """
        import json as _json
        context, used_documents, token_stats = self._build_context_within_budget(
            context_documents, max_context_tokens)
        from .prompt import create_grounded_prompt
        prompt = create_grounded_prompt(context, query, "v2")
        prompt_tokens = self._estimate_tokens(prompt)
        token_stats["prompt_tokens"] = prompt_tokens

        provider_info = {"provider": "mock", "model": self.model_name}
        streamer = getattr(self.llm_provider, "stream_chat", None) if self.llm_provider else None

        answer_start = time.time()
        if streamer is not None:
            full = []
            for chunk in streamer([{"role": "user", "content": prompt}]):
                full.append(chunk)
                yield {"type": "token", "text": chunk}
            answer = "".join(full)
        else:
            answer = self.generate_answer(prompt)
            yield {"type": "token", "text": answer}
        latency_ms = (time.time() - answer_start) * 1000

        if self.llm_provider is not None:
            info = self.llm_provider.get_model_info()
            provider_info = {"provider": info.get("provider", "unknown"),
                             "model": info.get("model_name", self.model_name)}

        sources = []
        for doc in used_documents:
            md = doc.get("metadata", {})
            src = md.get("source", "Unknown")
            fn = src.replace("\\", "/").split("/")[-1] if src else "Unknown"
            sources.append({"source": src, "source_filename": fn,
                            "chunk_index": md.get("chunk_index"),
                            "content_snippet": (doc.get("content", "")[:200] + "...")
                            if len(doc.get("content", "")) > 200 else doc.get("content", "")})

        usage = token_stats
        last = getattr(self.llm_provider, "last_result", None) if self.llm_provider else None
        if last is not None and getattr(last, "usage", None):
            usage = {"prompt_tokens": last.usage.prompt_tokens,
                     "completion_tokens": last.usage.completion_tokens,
                     "total_tokens": last.usage.total_tokens,
                     "source": "provider-reported"}

        yield {"type": "done", "answer": answer, "sources": sources, "usage": usage,
               "latency_ms": round(latency_ms, 2), **provider_info}

    def generate_multiple_answers(self,
                                 queries: List[str],
                                 contexts_list: List[List[Dict[str, Any]]],
                                 max_context_tokens: int = 2000) -> List[GenerationResult]:
        """
        Generate answers for multiple query-context pairs.

        Args:
            queries: List of query strings
            contexts_list: List of lists of retrieved document dictionaries
            max_context_tokens: Maximum tokens to use for context

        Returns:
            List of GenerationResult objects
        """
        if len(queries) != len(contexts_list):
            raise ValueError("Number of queries must match number of context lists")

        results = []
        for query, contexts in zip(queries, contexts_list):
            result = self.generate_answer_with_context(query, contexts, max_context_tokens)
            results.append(result)

        return results

    def _mock_generate(self, prompt: str) -> str:
        """
        Mock answer generation for testing purposes.

        Args:
            prompt: Input prompt

        Returns:
            Mock answer string
        """
        # Extract query from prompt for mock generation
        query_match = re.search(r"Question: (.*?)\n", prompt, re.DOTALL | re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else "unknown question"

        # Extract context from prompt
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL | re.IGNORECASE)
        context = context_match.group(1).strip() if context_match else ""

        # Check if we have sufficient context
        if context and len(context.strip()) > 20:
            # Simulate extracting answer from context with citations
            sentences = re.split(r'[.!?]+', context)
            if sentences:
                # Take first few meaningful sentences
                meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
                if meaningful_sentences:
                    # Add mock citations
                    cited_sentences = []
                    for i, sentence in enumerate(meaningful_sentences):
                        cited_sentences.append(f"{sentence} [source_document_{i+1}.md]")
                    answer = " ".join(cited_sentences) + "."
                else:
                    answer = "Based on the provided context, I can provide general information but lack specific details to answer this question precisely."
            else:
                answer = "The context contains some information but not in a format that allows me to extract a specific answer."
        else:
            # Check for the specific refusal pattern in the prompt
            if "I don't have enough information" in prompt:
                return "I don't have enough information to answer this question based on the provided documents."
            else:
                answer = "I don't have sufficient information in the provided context to answer this question accurately."

        return answer


class GroundedAnswer(BaseModel):
    """Structured, validated answer schema returned by the LLM."""

    answer: str = Field(min_length=1)
    citations: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    model_config = {"extra": "ignore"}


def create_answer_generator(model_name: str = "mock", llm_provider: Any = None) -> AnswerGenerator:
    """Create an answer generator instance.

    If llm_provider is None, the provider is resolved from the environment via
    create_llm_provider_from_env() (NVIDIA in prod, mock only if explicitly set).
    """
    if llm_provider is None:
        try:
            from providers.llm import create_llm_provider_from_env
            llm_provider = create_llm_provider_from_env()
            model_name = llm_provider.get_model_info().get("model_name", model_name)
        except Exception as e:
            logger.warning(f"Falling back to mock answer generator: {e}")
            llm_provider = None
    return AnswerGenerator(model_name, llm_provider=llm_provider)