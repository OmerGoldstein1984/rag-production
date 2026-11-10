"""
Answer generation utilities for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import random
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

from pydantic import BaseModel, Field

# Prompt-injection defense: sanitize user queries before they enter the prompt.
# prompt_guard is the canonical module (fail-closed: raises on injection).
from src.security.prompt_guard import sanitize_user_input, PromptInjectionBlocked  # noqa: E402

logger = logging.getLogger(__name__)

# Canned insufficiency messages the grounded prompt instructs the LLM to emit when
# evidence is missing. Answers that are neither these nor source-cited are treated
# as ungrounded and flagged.
INSUFFICIENCY_MESSAGES = (
    "I don't have enough information to answer this question based on the provided documents.",
    "I don't have sufficient information in the provided context to answer this question accurately.",
)


class CircuitBreaker:
    """Circuit breaker for the LLM provider.

    WHAT: Tracks consecutive failures and, once a threshold is reached, opens the
    circuit so subsequent calls fail fast without hitting the provider. After a
    cooldown it enters half-open state and allows a single trial call.
    WHY: A flaky or overloaded LLM provider should not drain the retry budget or
    block every request indefinitely; failing fast protects latency and cost.
    STATES: closed (calls allowed), open (calls rejected fast), half_open
        (single trial call after cooldown).
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self.failures = 0
        self.state: str = "closed"  # closed | open | half_open
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    def allow_request(self) -> bool:
        """Return True if a call may be attempted right now."""
        now = time.time()
        if self.state == "open":
            if self.last_failure_time is not None and (
                now - self.last_failure_time >= self.recovery_timeout_s
            ):
                self.state = "half_open"
                self.half_open_calls = 0
                return True
            return False
        if self.state == "half_open":
            return self.half_open_calls < self.half_open_max_calls
        return True  # closed

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        self.half_open_calls = 0

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state = "open"


@dataclass
class GenerationResult:
    """
    Result of answer generation.

    WHAT: A dataclass bundling everything produced by one answer generation:
    the original query, the answer text, the context used, the sources, the
    retrieved chunks, latency, token counts, prompt version, and extra metadata.
    WHY: Callers (RAGChat.query, the agent, the Streamlit UI) need a consistent
    result shape with answer + provenance (sources) + observability (latency,
    tokens).
    INPUTS (constructor): query: str; answer: str; context: str;
        sources: List[Dict[str, Any]]; retrieved_chunks: List[Dict[str, Any]];
        latency_ms: float; token_counts: Dict[str, int]; prompt_version: str;
        metadata: Dict[str, Any].
    OUTPUTS: None (data holder).
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: A plain dict — less type-safe and no field documentation.
    """

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
    """
    Generates answers using language models with retrieved context.

    Uses a real LLMProvider (e.g. NVIDIA NIM) when one is supplied; otherwise
    falls back to the deterministic mock (tests / local dev only).
    """

    def __init__(self, model_name: str = "mock", llm_provider: Any = None):
        """
        Initialize the answer generator.

        WHAT: Record model_name and store the injected LLMProvider (or None for
        mock); set up token-estimation constants.
        WHY: Dependency injection lets the answer generator work with the mock
        in tests and a real provider (NVIDIA) in production, without knowing how
        providers are constructed.
        INPUTS: model_name: str — e.g. "mock" or "meta/llama-3.1-70b-instruct"
            (recorded in metadata only); llm_provider: Any — an LLMProvider
            instance or None.
        OUTPUTS: None (instance).
        SIDE EFFECTS: None at construction (logging happens in _load_model).
        FAILURE MODES: None.
        ALTERNATIVE: Construct the provider internally from env instead of
        requiring injection.
        """
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.model = llm_provider  # kept for backward-compat attribute name

        # Token estimation constants (rough approximations)
        self.chars_per_token = 4.0  # Average for English text

        # Resilience: circuit breaker, retry + backoff, per-call timeout.
        self.circuit_breaker = CircuitBreaker()
        self.max_retries = 3
        self.base_backoff_s = 1.0
        self.request_timeout_s = 60.0
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _load_model(self):
        """
        Kept for backward compatibility; provider is injected via __init__.

        WHAT: No-op-ish method that logs the configured model/provider.
        WHY: Retained so older code that calls _load_model() doesn't break;
        real provider setup happens in __init__ / create_answer_generator.
        INPUTS: None.
        OUTPUTS: None.
        SIDE EFFECTS: Logs an INFO line.
        FAILURE MODES: None.
        ALTERNATIVE: Remove entirely (dead method kept for compatibility).
        """
        logger.info(f"Initialized answer generator with model: {self.model_name} "
                    f"(provider={'real' if self.llm_provider else 'mock'})")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.

        WHAT: Approximate tokens as len(text) / chars_per_token (floor at 1).
        WHY: A cheap, dependency-free way to size context budgets without a real
        tokenizer; used to keep prompts within the model's context window.
        INPUTS: text: str — e.g. "LTE uses OFDM for the downlink."
        OUTPUTS: int — e.g. 9 for a ~36-char string (36 / 4.0 = 9).
        SIDE EFFECTS: None.
        FAILURE MODES: Returns 0 for empty text; over/under-estimates vs a real
        tokenizer (chars-per-token is language-dependent).
        # GAP: Character-based estimation is inaccurate — real tokenizers use
        subword units, so a 2000-char "budget" may admit far more or fewer actual
        tokens than intended.
        ALTERNATIVE: Use the model's actual tokenizer (e.g. tiktoken for GPT
        models) for exact counts.
        """
        if not text:
            return 0
        return max(1, len(text) // self.chars_per_token)

    def _build_context_within_budget(self,
                                     documents: List[Dict[str, Any]],
                                     max_tokens: int = 2000) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Build context string respecting token budget.

        WHAT: Iterate documents, format each with [Source: filename] header,
        estimate tokens, and stop before exceeding max_tokens.
        WHY: Ensures the context fed to the LLM fits the model's context window,
        prioritizing the highest-ranked retrieved chunks first.
        INPUTS: documents: List[Dict[str, Any]] — retrieved docs ranked by
            relevance; max_tokens: int (default 2000).
        OUTPUTS: Tuple[str, List[Dict[str, Any]], Dict[str, int]] —
            (context_string, used_documents, token_stats), e.g.
            ("[Source: lte.md (Chunk 0/3)]\nLTE uses OFDM...\n---\n...",
             [doc0, doc1], {"estimated_tokens": 1500, "available_tokens": 2000,
             "used_tokens": 1500, "remaining_tokens": 500}).
        SIDE EFFECTS: Logs INFO when the budget limit is reached.
        FAILURE MODES: Returns ("", [], zero-stats) for empty documents — safe,
        no crash. Documents are taken in list order (assumes caller pre-sorted by
        relevance).
        ALTERNATIVE: Use a real tokenizer and reserve tokens for the prompt
        itself, not just context.
        # GAP: max_tokens only bounds the context, not the full prompt
        (instructions + context + question), so the total prompt can still
        exceed the model's window.
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

        WHAT: Call the real LLM provider's generate(), or fall back to the mock,
        returning the answer text.
        WHY: The single generation entry point — supports both production
        (NVIDIA via provider) and offline (mock) paths with a uniform interface.
        INPUTS: prompt: str — e.g. "CONTEXT: ...\nQUESTION: What is 5G?\n\nRESPONSE:".
        OUTPUTS: str — e.g. "5G uses orthogonal frequency-division multiplexing."
        SIDE EFFECTs: May make a network call (real provider) or return a canned
        response (mock).
        FAILURE MODES: If the provider raises, logs an error and returns an
        insufficiency fallback message instead of crashing (graceful degradation).
        ALTERNATIVE: Let the exception propagate so the caller knows generation
        failed (less forgiving).
        # BUSINESS: The fallback ensures the system always returns *something*
        rather than a 500 when the LLM is down — critical for a study aid that
        must stay available.
        """
        if self.llm_provider is not None:
            last_error = None
            for attempt in range(self.max_retries + 1):
                if not self.circuit_breaker.allow_request():
                    logger.warning("LLM provider circuit breaker is open; refusing call")
                    return (
                        "I don't have enough information to answer this question based on the "
                        "provided documents. (LLM provider unavailable)"
                    )
                try:
                    return self._call_with_timeout(prompt)
                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure()
                    if attempt < self.max_retries:
                        sleep_for = self.base_backoff_s * (2**attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}; "
                            f"retrying in {sleep_for:.2f}s"
                        )
                        time.sleep(sleep_for)
            logger.error(
                f"LLM provider failed after {self.max_retries + 1} attempts, returning fallback: {last_error}"
            )
            return (
                "I don't have enough information to answer this question based on the "
                "provided documents. (LLM provider error)"
            )
        return self._mock_generate(prompt)

    def _call_with_timeout(self, prompt: str) -> str:
        """Call the provider under a per-request timeout.

        WHAT: Submit the blocking provider.generate() to a single-thread executor
        and wait with a hard timeout.
        WHY: A slow or hung provider must not block the API handler indefinitely.
        """
        fut = self._executor.submit(self.llm_provider.generate, prompt)
        try:
            return fut.result(timeout=self.request_timeout_s)
        except FutTimeout:
            self.circuit_breaker.record_failure()
            raise TimeoutError(
                f"LLM provider call timed out after {self.request_timeout_s}s"
            )

    def _verify_grounding(self,
                          answer: str,
                          used_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify that the generated answer is grounded in retrieved context.

        WHAT: Heuristically checks whether the answer either (a) states insufficient
        evidence (a canned refusal) or (b) cites at least one source filename drawn
        from the retrieved documents. Returns a dict with `grounded` (bool) and
        `reason` (str).
        WHY: Prompt injection can trick the model into emitting ungrounded claims or
        leaking instructions; this is a last-line output filter that flags answers
        not traceable to the source corpus so callers can downgrade or block them.
        INPUTS: answer: str — the generated answer text;
            used_documents: List[Dict] — the retrieved docs used as context.
        OUTPUTS: dict {"grounded": bool, "reason": str}.
        FAILURE MODES: Never raises; defaults to grounded=False with a reason when
        the check cannot be completed.
        """
        result = {"grounded": False, "reason": "no source citation or insufficiency message found"}
        if not answer:
            return result

        # (a) Insufficiency: the model correctly refused due to missing evidence.
        for msg in INSUFFICIENCY_MESSAGES:
            if msg in answer:
                return {"grounded": True, "reason": "insufficiency message (no evidence)"}

        # (b) Source citation: answer references a source drawn from context.
        cited_sources = set()
        for doc in used_documents:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "")
            source_filename = source.split("/")[-1].split("\\")[-1] if source else ""
            if source_filename and source_filename in answer:
                cited_sources.add(source_filename)
            # Also match bracketed citation styles like [source_filename] or [Source: file]
            if source_filename and f"[{source_filename}]" in answer:
                cited_sources.add(source_filename)

        if cited_sources:
            return {"grounded": True, "reason": f"cited sources: {sorted(cited_sources)}"}

        return result

    def generate_answer_with_context(self,
                                     query: str,
                                     context_documents: List[Dict[str, Any]],
                                     max_context_tokens: int = 2000) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context.

        WHAT: Build context within token budget, create a grounded prompt,
        estimate prompt tokens, generate the answer, collect sources, and
        assemble a GenerationResult with latency and token metadata.
        WHY: The main RAG answer path used by RAGChat.query and the agent —
        combines retrieval output with generation and observability.
        INPUTS: query: str — e.g. "What is 5G?"; context_documents:
            List[Dict[str, Any]] — retrieved docs; max_context_tokens: int
            (default 2000).
        OUTPUTS: GenerationResult — e.g.
            GenerationResult(query="What is 5G?", answer="5G uses OFDM...",
            context="...", sources=[{...}], retrieved_chunks=[...],
            latency_ms=950.0, token_counts={...}, prompt_version="v2",
            metadata={...}).
        SIDE EFFECTs: Logs INFO with latency and token counts; may make a network
        call (LLM provider).
        FAILURE MODES: generate_answer degrades to a fallback message on provider
        failure (doesn't raise). Empty context_documents yields an empty context
        and an insufficiency-style answer.
        ALTERNATIVE: Skip token budgeting and stuff all retrieved docs (risks
        exceeding the context window).
        # INTERVIEW: Say it out loud — "We budget context tokens so the prompt
        fits the model's window, then record latency + tokens per answer for
        cost and SLA tracking."
        """
        start_time = time.time()

        # Prompt-injection defense: fail-closed on instruction-override / jailbreak.
        injection_detected = False
        try:
            safe_query = sanitize_user_input(query)
        except PromptInjectionBlocked:
            injection_detected = True
            logger.warning("Prompt injection blocked in user query")
            safe_query = ""

        # Build context within token budget
        context, used_documents, token_stats = self._build_context_within_budget(
            context_documents, max_context_tokens)

        # Create grounded prompt (user input is delimiter-framed in prompt.py)
        from .prompt import create_grounded_prompt
        prompt = create_grounded_prompt(context, safe_query, "v2")

        # Estimate prompt tokens
        prompt_tokens = self._estimate_tokens(prompt)
        token_stats["prompt_tokens"] = prompt_tokens
        token_stats["estimated_total_tokens"] = token_stats["estimated_tokens"] + prompt_tokens

        # Generate answer
        answer_start = time.time()
        answer = self.generate_answer(prompt)
        answer_latency = (time.time() - answer_start) * 1000

        # Output grounding verification: fail-closed on ungrounded answers or
        # detected injection attempts.
        grounding = self._verify_grounding(answer, used_documents)
        if not grounding["grounded"] or injection_detected:
            logger.warning(f"Answer not grounded/injection blocked: {grounding['reason']}")
            answer = ("I don't have enough information to answer this question based on the "
                      "provided documents.")

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
                "grounded": grounding["grounded"],
                "grounding_reason": grounding["reason"],
                "injection_detected": injection_detected,
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
        """
        Streaming RAG answer. Yields dict events:
        {"type":"token","text":...} ... then a final
        {"type":"done","sources":...,"usage":...,"latency_ms":...,"provider":...,"model":...}.
        Falls back to one-shot if the provider can't stream.

        WHAT: Build the grounded prompt, then stream text deltas from the
        provider's stream_chat() (or fall back to one-shot generate), yielding
        token events and a final "done" event with sources/usage/latency.
        WHY: Enables the Streamlit UI to show the answer incrementally as it is
        generated, improving perceived latency.
        INPUTS: query: str; context_documents: List[Dict[str, Any]];
            max_context_tokens: int (default 2000).
        OUTPUTS (yields): dict token events {"type": "token", "text": chunk}
            and a final {"type": "done", "answer": ..., "sources": [...],
            "usage": {...}, "latency_ms": ..., "provider": ..., "model": ...}.
        SIDE EFFECTs: Opens a streaming HTTP connection (real provider) or runs
        one-shot generate (mock); logs nothing itself.
        FAILURE MODES: If the provider lacks stream_chat, falls back to one-shot
        generate (graceful degradation). Provider errors propagate to the caller.
        ALTERNATIVE: Always use one-shot generate and deliver the whole answer
        in a single "done" event (simpler, no incremental UI).
        # GAP: The done-event "usage" falls back to estimated token_stats when
        provider-reported usage isn't available — mixing estimated and actual
        metrics inconsistently.
        """
        import json as _json
        # Prompt-injection defense: fail-closed on instruction-override / jailbreak.
        injection_detected = False
        try:
            safe_query = sanitize_user_input(query)
        except PromptInjectionBlocked:
            injection_detected = True
            logger.warning("Prompt injection blocked in streamed user query")
            safe_query = ""
        context, used_documents, token_stats = self._build_context_within_budget(
            context_documents, max_context_tokens)
        from .prompt import create_grounded_prompt
        prompt = create_grounded_prompt(context, safe_query, "v2")
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

        WHAT: Loop over (query, contexts) pairs, calling generate_answer_with_context
        for each; requires equal-length lists.
        WHY: Batch generation for offline evaluation of many questions (e.g.
        the golden eval set) without writing a custom loop per caller.
        INPUTS: queries: List[str]; contexts_list: List[List[Dict[str, Any]]]
            (must match length of queries); max_context_tokens: int (default 2000).
        OUTPUTS: List[GenerationResult] — one result per pair.
        SIDE EFFECTs: Makes one LLM call per pair (network, sequential — no
        parallelism).
        FAILURE MODES: Raises ValueError if len(queries) != len(contexts_list).
        ALTERNATIVE: Call generate_answer_with_context in a comprehension from
        the caller.
        # GAP: Strictly sequential — no batching or parallelism, so N queries
        take N times the single-call latency (costly for large eval sets).
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

        WHAT: Parse the prompt for Context/Question markers and produce a
        plausible canned answer (with mock citations) or an insufficiency
        message — no model involved.
        WHY: Gives tests a deterministic, offline answer so they don't depend on
        an external LLM API.
        INPUTS: prompt: str — e.g. "CONTEXT:\n[Source: lte.md] ...\n\nQUESTION:
            What is 5G?\n\nRESPONSE:".
        OUTPUTS: str — a canned answer with mock [source_document_N.md]
            citations when context is present and long enough; otherwise an
            insufficiency message.
        SIDE EFFECTS: None.
        FAILURE MODES: Never raises; degrades to canned insufficiency text when
        markers are missing or context is too short.
        ALTERNATIVE: Use recorded real LLM responses (cassettes) for more
        realism in tests.
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
    """
    Structured, validated answer schema returned by the LLM.

    WHAT: A Pydantic model with answer (min_length=1), citations (list of str),
    and insufficient_evidence (bool), used by chat_json to get typed, validated
    structured output.
    WHY: Guarantees the LLM's output conforms to a known shape so downstream code
    can rely on answer/citations/insufficient_evidence fields instead of parsing
    free text.
    INPUTS: answer: str (required, min 1 char); citations: List[str]
        (default []); insufficient_evidence: bool (default False).
    OUTPUTS: None (Pydantic model instance).
    SIDE EFFECTS: model_config = {"extra": "ignore"} — unknown fields are
    silently dropped rather than rejected.
    FAILURE MODES: Pydantic ValidationError if answer is missing/empty or
    citations is not a list of str.
    ALTERNATIVE: Parse free-form text with regex (fragile, no guarantee).
    """

    answer: str = Field(min_length=1)
    citations: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    model_config = {"extra": "ignore"}


def create_answer_generator(model_name: str = "mock", llm_provider: Any = None) -> AnswerGenerator:
    """
    Create an answer generator instance.

    WHAT: Factory that builds an AnswerGenerator, resolving a real provider from
    env when none is injected (falling back to mock on any error).
    WHY: Lets production code get an NVIDIA-backed generator automatically from
    LLM_PROVIDER env, while tests can inject a mock or pass None for the mock.
    INPUTS: model_name: str (default "mock"); llm_provider: Any (optional).
    OUTPUTS: AnswerGenerator — backed by a real provider (NVIDIA) or mock.
    SIDE EFFECTs: Imports providers.llm and calls create_llm_provider_from_env()
    when llm_provider is None (reads settings, may construct KeyPool).
    FAILURE MODES: If provider resolution raises, logs a WARNING and falls back
    to mock (graceful degradation — never crashes the app).
    ALTERNATIVE: Always require an explicit provider (no env auto-resolution).
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
