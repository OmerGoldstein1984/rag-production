"""
Prompt engineering utilities for RAG systems.

BUSINESS PROBLEM: This file turns retrieved document chunks + a user question
into a tightly controlled prompt that forces the LLM to answer only from the
provided context, cite sources, and refuse when evidence is insufficient. It is
the guardrail that prevents hallucination in a RAG system.

PLACE IN DATA FLOW: Between retrieval (which returns ranked chunks) and
generation (src/generation/answer.py calls create_grounded_prompt and
format_context). Retrieval outputs documents; this file formats them into context
and wraps them in a grounded prompt template; the LLM then answers.

WHAT CALLS IT: src/generation/answer.py (generate_answer_with_context,
stream_answer_with_context use create_grounded_prompt); src/generation/chat.py
(imports create_grounded_prompt, format_context); tests.

WHAT IT CALLS: Nothing project-specific — only typing, logging, re standard
library. (It is pure: no network, no vector store.)

WHAT BREAKS IF REMOVED: The LLM would receive raw context with no grounding
instructions — hallucination risk rises, citations vanish, and the "I don't have
enough information" refusal behavior is lost. Answers would no longer be
traceable to source documents.

# GAP: No template versioning registry, no A/B testing of prompts, no
prompt-eval integration — changing a template is an untracked code edit.
# BUSINESS: Protects answer accuracy — the core requirement of a study-aid RAG
system (CLAUDE.md: all answers must be grounded in retrieved chunks, no
speculation).
# INTERVIEW: Say it out loud — "The prompt is the trust boundary: it tells the
model to answer only from context, cite sources, and say 'I don't know' when
evidence is missing. Without it, the LLM will hallucinate."
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Template for generating prompts with context and query.

    WHAT: Holds a template string with {context} and {query} placeholders plus a
    version string, and renders it via str.format.
    WHY: Lets the system swap prompt wording (grounded_v1 vs grounded_v2 vs
    citation_focused) while tracking which version produced an answer.
    INPUTS (__init__): template_str: str — e.g.
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"; version: str
        (default "1.0").
    OUTPUTS (format): str — e.g. "Context:\nLTE uses OFDM...\n\nQuestion: What
        is 5G?\n\nAnswer:".
    OUTPUTS (get_version): str — e.g. "2.0".
    SIDE EFFECTS: None.
    FAILURE MODES: format() raises KeyError if the template references a
    placeholder other than {context} or {query}; raises KeyError if either is
    missing.
    ALTERNATIVE: Use Jinja2 or string.Template for richer templating (loops,
    conditionals).
    """

    def __init__(self, template_str: str, version: str = "1.0"):
        """
        Initialize the prompt template.

        WHAT: Store the template string and its version label.
        WHY: Version tracking lets us know which prompt variant generated a given
        answer (recorded in GenerationResult.prompt_version).
        INPUTS: template_str: str; version: str — e.g. "2.0".
        OUTPUTS: None (instance).
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Derive version from a template registry key instead of
        storing it on each instance.
        """
        self.template_str = template_str
        self.version = version

    def format(self, context: str, query: str) -> str:
        """
        Format the prompt with context and query.

        WHAT: Substitute {context} and {query} into the template string.
        WHY: Produces the final prompt text sent to the LLM, injecting retrieved
        context and the user's question.
        INPUTS: context: str — e.g. "[Source: lte.md] LTE uses OFDM...";
            query: str — e.g. "What waveform does 5G use?".
        OUTPUTS: str — e.g.
            "Context:\n[Source: lte.md] LTE uses OFDM...\n\nQuestion: What
            waveform does 5G use?\n\nAnswer:".
        SIDE EFFECTS: None.
        FAILURE MODES: KeyError if template uses placeholders besides
        {context}/{query}; IndexError for malformed braces (e.g. single {).
        ALTERNATIVE: Use re.sub or Jinja2.render for more control.
        """
        return self.template_str.format(context=context, query=query)

    def get_version(self) -> str:
        """
        Get the version of this prompt template.

        WHAT: Return the version string recorded at construction.
        WHY: Callers log prompt_version per answer so results can be grouped by
        which prompt variant was used (eval/regression analysis).
        INPUTS: None.
        OUTPUTS: str — e.g. "2.0".
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Store version externally and look it up by template id.
        """
        return self.version


class PromptEngineer:
    """
    Engineers prompts for RAG systems with various strategies.

    WHAT: Holds a registry of named PromptTemplate instances (grounded_v1,
    grounded_v2, citation_focused) and provides methods to render prompts by
    name, by custom template string, or in batch.
    WHY: Centralizes prompt strategy selection so the answer generator and UI
    can pick a template by name without knowing its text.
    INPUTS (__init__): None (builds the built-in template registry).
    OUTPUTS: None (instance).
    SIDE EFFECTS: None at construction.
    FAILURE MODES: create_prompt falls back to grounded_v2 if the requested name
    is missing (silent fallback — may hide a typo in the template name).
    ALTERNATIVE: Load templates from external files/DB instead of hard-coding.
    """

    def __init__(self):
        """
        Initialize with grounded prompt templates.

        WHAT: Populate self.templates with three built-in grounded templates.
        WHY: Gives callers immediate, safe prompt strategies out of the box
        (grounded_v2 is the default used by answer generation).
        INPUTS: None.
        OUTPUTS: None (instance; self.templates is a dict of name -> PromptTemplate).
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Inject templates via constructor for testability.
        """
        self.templates = {
            "grounded_v1": PromptTemplate(
                "You are a helpful assistant that answers questions based ONLY on the provided context. "
                "Follow these rules strictly:\n"
                "1. Answer the question using only information from the context below\n"
                "2. If the context does not contain sufficient information to answer the question, "
                "you MUST say \"I don't have enough information to answer this question based on the provided documents.\"\n"
                "3. When making factual statements, cite the source using [source_filename] notation\n"
                "4. Be concise and direct in your answer\n"
                "5. Do not add information not present in the context\n"
                "6. Do not speculate or make assumptions beyond what is explicitly stated\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer:",
                version="1.0"
            ),
            "grounded_v2": PromptTemplate(
                "You are an AI assistant specialized in answering questions from company documentation. "
                "Your responses must be grounded in the provided source material.\n\n"
                "RULES:\n"
                "- ANSWER ONLY FROM CONTEXT: Use information exclusively from the provided context\n"
                "- CITE YOUR SOURCES: Reference statements with [source_filename] where the information appears\n"
                "- KNOW YOUR LIMITS: If context lacks sufficient information, respond exactly with: "
                "\"I don't have enough information to answer this question based on the provided documents.\"\n"
                "- BE PRECISE: Provide direct answers without unnecessary elaboration\n"
                "- NO HALLUCINATION: Never invent information not present in the source documents\n\n"
                "CONTEXT:\n{context}\n\n"
                "QUESTION: {query}\n\n"
                "RESPONSE:",
                version="2.0"
            ),
            "citation_focused": PromptTemplate(
                "Answer the question based solely on the following context. "
                "Each sentence in your answer should end with a citation to its source.\n"
                "Format citations as [source_filename]. If you cannot answer based on context, "
                "say: \"I don't have enough information to answer this question based on the provided documents.\"\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer with citations:",
                version="1.0"
            )
        }

    def create_prompt(self,
                      context: str,
                      query: str,
                      template_name: str = "grounded_v2") -> str:
        """
        Create a prompt using a named template.

        WHAT: Look up a template by name and render it with the given context and
        query; fall back to grounded_v2 if the name is unknown.
        WHY: The standard entry point used by answer generation (create_grounded_prompt)
        and the CLI chat — keeps template selection at one call site.
        INPUTS: context: str — e.g. "[Source: lte.md] LTE uses OFDM...";
            query: str — e.g. "What is 5G?"; template_name: str — e.g.
            "grounded_v2" (default).
        OUTPUTS: str — the rendered prompt, e.g.
            "CONTEXT:\n[Source: lte.md] ...\n\nQUESTION: What is 5G?\n\nRESPONSE:".
        SIDE EFFECTS: None.
        FAILURE MODES: Silent fallback to grounded_v2 when template_name is not
        registered (a misspelled name won't raise).
        ALTERNATIVE: Require the caller to pass the template object directly.
        """
        template = self.templates.get(template_name, self.templates["grounded_v2"])
        return template.format(context, query)

    def create_prompt_from_template(self,
                                    template_str: str,
                                    context: str,
                                    query: str) -> str:
        """
        Create a prompt from a custom template string.

        WHAT: Build a one-off PromptTemplate from an arbitrary string and render
        it with context/query.
        WHY: Lets callers use ad-hoc prompt wording without registering a named
        template.
        INPUTS: template_str: str — e.g. "Answer from context: {context}\nQ:
            {query}"; context: str; query: str.
        OUTPUTS: str — the rendered prompt.
        SIDE EFFECTS: None.
        FAILURE MODES: PromptTemplate.format raises KeyError/IndexError on bad
        placeholders (propagates to caller).
        ALTERNATIVE: Register the custom template in self.templates for reuse.
        """
        template = PromptTemplate(template_str)
        return template.format(context, query)

    def create_batch_prompts(self,
                             contexts: List[str],
                             queries: List[str],
                             template_name: str = "grounded_v2") -> List[str]:
        """
        Create prompts for multiple context-query pairs.

        WHAT: Render one prompt per (context, query) pair using the named
        template; requires equal-length lists.
        WHY: Efficiently builds prompts for bulk/offline evaluation of many
        questions at once.
        INPUTS: contexts: List[str]; queries: List[str] (must match length of
            contexts); template_name: str (default "grounded_v2").
        OUTPUTS: List[str] — one rendered prompt per pair.
        SIDE EFFECTS: None.
        FAILURE MODES: Raises ValueError if len(contexts) != len(queries).
        ALTERNATIVE: Call create_prompt in a loop from the caller.
        """
        if len(contexts) != len(queries):
            raise ValueError("Number of contexts must match number of queries")

        prompts = []
        for context, query in zip(contexts, queries):
            prompt = self.create_prompt(context, query, template_name)
            prompts.append(prompt)

        return prompts

    def get_prompt_version(self, template_name: str = "grounded_v2") -> str:
        """
        Get the version of a named template.

        WHAT: Return the version string for the named template (falling back to
        grounded_v2 if unknown).
        WHY: Lets callers record which template version was used without
        instantiating PromptEngineer twice.
        INPUTS: template_name: str (default "grounded_v2").
        OUTPUTS: str — e.g. "2.0".
        SIDE EFFECTS: None.
        FAILURE MODES: Silent fallback to grounded_v2 for unknown names.
        ALTERNATIVE: Read version from self.templates directly.
        """
        template = self.templates.get(template_name, self.templates["grounded_v2"])
        return template.get_version()


def format_context(documents: List[Dict[str, Any]],
                   include_metadata: bool = True,
                   max_context_length: Optional[int] = None,
                   deduplicate: bool = True) -> str:
    """
    Format retrieved documents into a context string.

    WHAT: Turn a list of retrieved document dicts into a single context string
    with optional [Source: filename] headers, deduplication, and truncation.
    WHY: Retrieval returns structured docs; the LLM needs a flat context block.
    This function is the single place that decides how documents are quoted,
    cited, deduplicated, and size-bounded.
    INPUTS: documents: List[Dict[str, Any]] — each with "content" and optional
        "metadata" (source, chunk_index, total_chunks); include_metadata: bool
        (default True); max_context_length: Optional[int]; deduplicate: bool
        (default True).
    OUTPUTS: str — e.g.
        "[Source: lte.md (Chunk 0/5)]\nLTE uses OFDM...\n---\n[Source: nr.md
        (Chunk 2/3)]\n5G NR uses..."; "" when documents is empty.
    SIDE EFFECTS: None.
    FAILURE MODES: None raised — missing "content" defaults to "", missing
    metadata defaults gracefully; truncation can cut a document mid-sentence.
    ALTERNATIVE: Return the list of documents and let the LLM/template format
    them (less control over citations/size).
    # GAP: max_context_length truncates by character count, not token count — a
    poor proxy for the model's actual token budget (chars-per-token varies).
    """
    if not documents:
        return ""

    # Deduplicate documents if requested
    if deduplicate:
        seen_content = set()
        unique_documents = []
        for doc in documents:
            content = doc.get("content", "").strip()
            if content not in seen_content and content:
                seen_content.add(content)
                unique_documents.append(doc)
        documents = unique_documents

    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.get("content", "")

        if include_metadata:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            # Extract just the filename for cleaner citation
            source_filename = source.split("/")[-1].split("\\")[-1] if source else "Unknown"
            chunk_info = f" (Chunk {metadata.get('chunk_index', '?')}" + \
                        (f"/{metadata.get('total_chunks', '?')}" if 'total_chunks' in metadata else "") + ")"

            part = f"[Source: {source_filename}{chunk_info}]\n{content}\n"
        else:
            part = f"[Document {i+1}]\n{content}\n"

        context_parts.append(part)

    context = "\n---\n".join(context_parts)

    # Truncate if exceeding max length
    if max_context_length and len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    return context


# Convenience functions
def create_grounded_prompt(context: str, query: str, version: str = "v2") -> str:
    """
    Create a grounded RAG prompt that forces answer-from-context behavior.

    WHAT: Build a PromptEngineer, look up the grounded_{version} template, and
    render it with context/query.
    WHY: The most commonly used entry point — src/generation/answer.py calls this
    to produce the prompt sent to the LLM, enforcing grounded answering.
    INPUTS: context: str — e.g. "[Source: lte.md] LTE uses OFDM...";
        query: str — e.g. "What is 5G?"; version: str (default "v2").
    OUTPUTS: str — e.g. "You are an AI assistant specialized in answering
        questions from company documentation...\nCONTEXT: ...\nQUESTION: What is
        5G?\n\nRESPONSE:".
    SIDE EFFECTS: Instantiates a new PromptEngineer on every call (rebuilds the
    template registry — minor overhead).
    FAILURE MODES: KeyError if f"grounded_{version}" is not a registered
    template (Propagates; no fallback here unlike create_prompt).
    ALTERNATIVE: Reuse a single PromptEngineer instance instead of constructing
    one per call.
    """
    template_name = f"grounded_{version}"
    engineer = PromptEngineer()
    return engineer.create_prompt(context, query, template_name)


def create_citation_prompt(context: str, query: str) -> str:
    """
    Create a prompt focused on source citations.

    WHAT: Build a PromptEngineer and render the "citation_focused" template,
    which requires every sentence to end with a [source_filename] citation.
    WHY: For answers where traceability to source documents is critical (e.g.
    compliance/regulatory telecom answers).
    INPUTS: context: str; query: str.
    OUTPUTS: str — the citation-focused rendered prompt.
    SIDE EFFECTS: Instantiates a new PromptEngineer on every call.
    FAILURE MODES: None beyond template rendering errors.
    ALTERNATIVE: Use grounded_v2 (less strict about per-sentence citations).
    """
    engineer = PromptEngineer()
    return engineer.create_prompt(context, query, "citation_focused")
