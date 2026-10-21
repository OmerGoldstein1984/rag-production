"""
Prompt engineering utilities for RAG systems.
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class PromptTemplate:
    """Template for generating prompts with context and query."""

    def __init__(self, template_str: str, version: str = "1.0"):
        """
        Initialize the prompt template.

        Args:
            template_str: Template string with placeholders {context} and {query}
            version: Version of the prompt template
        """
        self.template_str = template_str
        self.version = version

    def format(self, context: str, query: str) -> str:
        """
        Format the prompt with context and query.

        Args:
            context: Retrieved context text
            query: User query

        Returns:
            Formatted prompt string
        """
        return self.template_str.format(context=context, query=query)

    def get_version(self) -> str:
        """Get the version of this prompt template."""
        return self.version


class PromptEngineer:
    """Engineers prompts for RAG systems with various strategies."""

    def __init__(self):
        """Initialize with grounded prompt templates."""
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

        Args:
            context: Retrieved context text
            query: User query
            template_name: Name of template to use

        Returns:
            Formatted prompt string
        """
        template = self.templates.get(template_name, self.templates["grounded_v2"])
        return template.format(context, query)

    def create_prompt_from_template(self,
                                   template_str: str,
                                   context: str,
                                   query: str) -> str:
        """
        Create a prompt from a custom template string.

        Args:
            template_str: Template string with {context} and {query} placeholders
            context: Retrieved context text
            query: User query

        Returns:
            Formatted prompt string
        """
        template = PromptTemplate(template_str)
        return template.format(context, query)

    def create_batch_prompts(self,
                            contexts: List[str],
                            queries: List[str],
                            template_name: str = "grounded_v2") -> List[str]:
        """
        Create prompts for multiple context-query pairs.

        Args:
            contexts: List of context strings
            queries: List of query strings
            template_name: Name of template to use

        Returns:
            List of formatted prompt strings
        """
        if len(contexts) != len(queries):
            raise ValueError("Number of contexts must match number of queries")

        prompts = []
        for context, query in zip(contexts, queries):
            prompt = self.create_prompt(context, query, template_name)
            prompts.append(prompt)

        return prompts

    def get_prompt_version(self, template_name: str = "grounded_v2") -> str:
        """Get the version of a named template."""
        template = self.templates.get(template_name, self.templates["grounded_v2"])
        return template.get_version()


def format_context(documents: List[Dict[str, Any]],
                   include_metadata: bool = True,
                   max_context_length: Optional[int] = None,
                   deduplicate: bool = True) -> str:
    """
    Format retrieved documents into a context string.

    Args:
        documents: List of document dictionaries
        include_metadata: Whether to include metadata in the context
        max_context_length: Maximum length of context string (truncate if exceeded)
        deduplicate: Whether to remove duplicate content

    Returns:
        Formatted context string
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
    """Create a grounded RAG prompt that forces answer-from-context behavior."""
    template_name = f"grounded_{version}"
    engineer = PromptEngineer()
    return engineer.create_prompt(context, query, template_name)


def create_citation_prompt(context: str, query: str) -> str:
    """Create a prompt focused on source citations."""
    engineer = PromptEngineer()
    return engineer.create_prompt(context, query, "citation_focused")