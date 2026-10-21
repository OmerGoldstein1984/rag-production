"""
Text chunking utilities for preparing documents for embedding.
Implements 3 strategies: fixed-size, recursive, and structure-aware.
"""

from typing import List, Dict, Any
import re
from enum import Enum


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    STRUCTURE_AWARE = "structure_aware"


class TextChunker:
    """Splits text into chunks with configurable strategies."""

    def __init__(self,
                 strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the chunker.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Target size of each chunk (meaning depends on strategy)
            chunk_overlap: Number of characters/units to overlap between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the selected strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(text)
        elif self.strategy == ChunkingStrategy.STRUCTURE_AWARE:
            return self._chunk_structure_aware(text)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Original fixed-size chunking with overlap and sentence boundary awareness."""
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # If we're not at the end of text, try to break at a sentence boundary
            if end < text_length:
                # Look for sentence boundary within the last 100 characters
                search_start = max(start, end - 100)
                chunk_segment = text[search_start:end]

                # Find last sentence ending
                last_period = chunk_segment.rfind('.')
                last_exclamation = chunk_segment.rfind('!')
                last_question = chunk_segment.rfind('?')

                boundary = max(last_period, last_exclamation, last_question)

                if boundary != -1:
                    # Adjust end to be at the sentence boundary
                    end = search_start + boundary + 1

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk (with overlap)
            start = end - self.chunk_overlap

            # Ensure we make progress
            if start >= text_length:
                break

        return chunks

    def _chunk_recursive(self, text: str) -> List[str]:
        """
        Recursive chunking: first try paragraphs, then sentences, then fixed-size.
        This preserves semantic units better than pure fixed-size.
        """
        if not text:
            return []

        # First, split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, apply fixed-size chunking to it
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Sub-chunk this oversized chunk using fixed-size strategy
                sub_chunker = TextChunker(
                    ChunkingStrategy.FIXED_SIZE,
                    self.chunk_size,
                    self.chunk_overlap
                )
                sub_chunks = sub_chunker._chunk_fixed_size(chunk)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _chunk_structure_aware(self, text: str) -> List[str]:
        """
        Structure-aware chunking: split by markdown headings first.
        This preserves document structure and section boundaries.
        """
        if not text:
            return []

        # Split by markdown headings (lines starting with #)
        # We want to keep the heading with its content
        sections = re.split(r'(\n#{1,6}\s+.+)', text)

        # Reconstruct sections with their headings
        reconstructed_sections = []
        current_section = ""

        for i, section in enumerate(sections):
            if re.match(r'\n#{1,6}\s+.+', section):
                # This is a heading
                if current_section.strip():
                    reconstructed_sections.append(current_section.strip())
                current_section = section
            else:
                # This is content
                current_section += section

        # Don't forget the last section
        if current_section.strip():
            reconstructed_sections.append(current_section.strip())

        # Filter out empty sections
        sections = [s for s in reconstructed_sections if s.strip()]

        if not sections:
            return []

        # Now process each section - if it's small enough, keep it as-is
        # If it's too large, apply recursive chunking within the section
        final_chunks = []

        for section in sections:
            if len(section) <= self.chunk_size:
                final_chunks.append(section)
            else:
                # Apply recursive chunking to this section
                recursive_chunker = TextChunker(
                    ChunkingStrategy.RECURSIVE,
                    self.chunk_size,
                    self.chunk_overlap
                )
                sub_chunks = recursive_chunker._chunk_recursive(section)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document and return chunks with metadata.

        Args:
            document: Document dictionary with content and metadata

        Returns:
            List of chunk dictionaries
        """
        chunks = self.chunk_text(document["content"])

        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **document["metadata"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "is_chunk": True,
                "chunking_strategy": self.strategy.value,
                "chunk_size_config": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }

            chunked_documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })

        return chunked_documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunked document dictionaries
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks


# Convenience function
def chunk_documents(documents: List[Dict[str, Any]],
                   strategy: str = "fixed_size",
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents with specified parameters.

    Args:
        documents: List of document dictionaries
        strategy: Chunking strategy ("fixed_size", "recursive", "structure_aware")
        chunk_size: Target size of each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked document dictionaries
    """
    strategy_map = {
        "fixed_size": ChunkingStrategy.FIXED_SIZE,
        "recursive": ChunkingStrategy.RECURSIVE,
        "structure_aware": ChunkingStrategy.STRUCTURE_AWARE
    }

    chunking_strategy = strategy_map.get(strategy.lower(), ChunkingStrategy.FIXED_SIZE)
    chunker = TextChunker(chunking_strategy, chunk_size, chunk_overlap)
    return chunker.chunk_documents(documents)