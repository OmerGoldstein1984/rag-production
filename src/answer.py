from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Try to import Anthropic, fallback to OpenAI if needed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not installed. Install with: pip install anthropic")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Install with: pip install openai")

def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    model_provider: str = "anthropic",
    model_name: str = None,
    api_key: str = None,
    temperature: float = 0.2,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Generate a grounded answer based on query and retrieved context.

    Args:
        query: The user's question.
        context_chunks: List of relevant chunks with metadata.
        model_provider: Either "anthropic" or "openai".
        model_name: Specific model to use (defaults to provider's recommended).
        api_key: API key for the provider (if not set via env var).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens to generate.

    Returns:
        Dictionary with keys:
        - "answer": str, the generated answer
        - "citations": List[Dict], citation information
        - "model_used": str, which model was used
        - "prompt_tokens": int, approximate token count
        - "completion_tokens": int, approximate token count
    """
    # Prepare context
    from .retrieve import get_context_for_answer
    context = get_context_for_answer(context_chunks)

    if not context.strip():
        return {
            "answer": "I don't have enough information to answer that question. Please try uploading relevant documents or rephrasing your query.",
            "citations": [],
            "model_used": model_provider,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

    # Build prompt
    prompt = f"""Based on the following context, answer the question accurately and concisely.
If the answer cannot be determined from the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Answer:"""

    # Generate answer based on provider
    if model_provider.lower() == "anthropic" and ANTHROPIC_AVAILABLE:
        return _generate_with_anthropic(prompt, model_name, api_key, temperature, max_tokens)
    elif model_provider.lower() == "openai" and OPENAI_AVAILABLE:
        return _generate_with_openai(prompt, model_name, api_key, temperature, max_tokens)
    else:
        # Fallback or error
        available = []
        if ANTHROPIC_AVAILABLE:
            available.append("anthropic")
        if OPENAI_AVAILABLE:
            available.append("openai")
        if not available:
            raise RuntimeError("No LLM providers available. Install anthropic or openai.")

        provider = available[0]
        logger.warning(f"Requested provider {model_provider} not available, falling back to {provider}")
        if provider == "anthropic":
            return _generate_with_anthropic(prompt, model_name, api_key, temperature, max_tokens)
        else:
            return _generate_with_openai(prompt, model_name, api_key, temperature, max_tokens)

def _generate_with_anthropic(
    prompt: str,
    model_name: str = None,
    api_key: str = None,
    temperature: float = 0.2,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """Generate answer using Anthropic API."""
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    model = model_name or "claude-3-haiku-20240307"  # default to faster/cheaper model

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = message.content[0].text

        # Extract citations from context (simplified - in practice you'd track which chunks contributed)
        # For now, we'll return empty citations and let the UI handle it based on retrieved chunks

        return {
            "answer": answer.strip(),
            "citations": [],  # Will be populated by caller using context_chunks
            "model_used": f"anthropic:{model}",
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens
        }
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "model_used": f"anthropic:{model}",
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

def _generate_with_openai(
    prompt: str,
    model_name: str = None,
    api_key: str = None,
    temperature: float = 0.2,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """Generate answer using OpenAI API."""
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    model = model_name or "gpt-3.5-turbo"  # default to faster/cheaper model

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer.strip(),
            "citations": [],  # Will be populated by caller using context_chunks
            "model_used": f"openai:{model}",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "model_used": f"openai:{model}",
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

def format_answer_with_citations(
    answer_result: Dict[str, Any],
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format the answer with proper citations based on which chunks were used.
    This is a simplified version - in practice you'd track token attribution.

    Args:
        answer_result: Output from generate_answer.
        context_chunks: The chunks used to generate context.

    Returns:
        Enhanced answer result with citations.
    """
    # Simple approach: cite all chunks that contributed to context
    # In a more sophisticated system, you'd track which specific sentences/chunks
    # contributed to the answer (e.g., using retrieval augmentation with citations)

    citations = []
    for chunk in context_chunks:
        metadata = chunk["metadata"]
        citation = {
            "source": metadata["source"],
            "page": metadata["page"] + 1,  # Convert to 1-based for user
            "text_snippet": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
            "relevance_score": chunk.get("similarity_score", 0.0)
        }
        citations.append(citation)

    # Sort by relevance score descending
    citations.sort(key=lambda x: x["relevance_score"], reverse=True)

    answer_result["citations"] = citations
    return answer_result