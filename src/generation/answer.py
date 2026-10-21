"""
Answer generation utilities for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import time
from dataclasses import dataclass, asdict

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
    """Generates answers using language models with retrieved context."""

    def __init__(self, model_name: str = "mock"):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the language model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

        # Token estimation constants (rough approximations)
        self.chars_per_token = 4.0  # Average for English text

    def _load_model(self):
        """Load the language model."""
        # In a real implementation, this would load an LLM
        # For now, we'll use a mock generator
        logger.info(f"Initialized answer generator with model: {self.model_name}")
        self.model = None  # Mock model

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
        if self.model is None:
            return self._mock_generate(prompt)

        # In a real implementation, this would call the LLM
        # For now, return a placeholder
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
                "model_used": self.model_name,
                "num_context_docs": len(used_documents),
                "context_length": len(context),
                "answer_generation_latency_ms": answer_latency
            }
        )

        logger.info(f"Generated answer in {total_latency:.2f}ms "
                   f"({token_stats.get('prompt_tokens', 0)} prompt + "
                   f"{token_stats.get('estimated_tokens', 0)} context tokens)")

        return result

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

    # Convenience function
def create_answer_generator(model_name: str = "mock") -> AnswerGenerator:
    """Create an answer generator instance."""
    return AnswerGenerator(model_name)