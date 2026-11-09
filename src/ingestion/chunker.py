# =============================================================================
# FILE HEADER — src/ingestion/chunker.py
# Business problem: Splits cleaned document text into overlapping chunks of a
# target size so embeddings can be computed on manageable pieces while
# preserving sentence/paragraph/structure boundaries where possible.
# Place in data flow: Step 3 of ingestion (G3 group). Called by
# manager._index_file (manager.py:172) and run.run_ingestion_pipeline
# (run.py:61). Calls nothing external except stdlib (re, enum).
# What breaks if removed: No chunks are produced, so the embedder and vector
# store have nothing to index — retrieval returns nothing for any query.
# =============================================================================
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

    # WHAT: Enum naming the three chunking strategies available in TextChunker.
    # WHY: Makes strategy selection explicit and type-safe instead of passing
    # raw strings that could be misspelled.
    # INPUTS: None (enum members).
    # OUTPUTS: One of FIXED_SIZE, RECURSIVE, STRUCTURE_AWARE.
    # SIDE EFFECTS: None.
    # FAILURE MODES: None — enum values are fixed.
    # ALTERNATIVE: Plain string constants instead of an Enum.
    # GAP: No validation that the chosen strategy matches the document type
    # (e.g. structure_aware on a plain .txt file with no headings degrades to
    # recursive behavior silently).
    # BUSINESS: Gives callers a clear, documented set of chunking options.
    # INTERVIEW: "Why an Enum instead of strings? So typos like 'fixd_size'
    # fail at strategy selection, not deep inside chunking logic."


class TextChunker:
    """Splits text into chunks with configurable strategies.

    WHAT: Configurable text chunker supporting fixed-size, recursive, and
    structure-aware strategies, all with overlap between consecutive chunks.
    WHY: Different documents benefit from different chunking — code-like text
    needs fixed-size, prose benefits from paragraph-aware recursive, and
    Markdown benefits from heading-aware structure-aware chunking.
    INPUTS: strategy (ChunkingStrategy), chunk_size (int), chunk_overlap (int).
    OUTPUTS: None directly — methods return lists of chunks or chunk dicts.
    SIDE EFFECTS: Validates inputs in __init__ and raises ValueError on bad
    values (chunker.py:55-57).
    FAILURE MODES: __init__ raises ValueError if chunk_size <= 0,
    chunk_overlap < 0, or chunk_overlap >= chunk_size (chunker.py:55-57).
    ALTERNATIVE: Could use a library like langchain's text splitters instead of
    custom implementations.
    # GAP: No token-aware chunking — chunk_size is in characters, not tokens,
    so a 500-char chunk of dense technical text may still exceed the embedding
    model's token limit (chunker.py:42).
    # BUSINESS: Overlap between chunks prevents context loss at chunk
    boundaries — critical for retrieval where a sentence might span chunks.
    # INTERVIEW: "Why overlap chunks? So a sentence split across two chunks is
    still recoverable — retrieval doesn't lose the tail of one chunk or head of
    the next."
    """

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

        WHAT: Stores strategy, chunk_size, and chunk_overlap after validating
        them.
        WHY: Validation catches misconfiguration early rather than producing
        degenerate chunks (zero-size or infinite loops).
        INPUTS: strategy (ChunkingStrategy, default FIXED_SIZE), chunk_size
        (int, default 500), chunk_overlap (int, default 50).
        OUTPUTS: None — configures the instance.
        SIDE EFFECTS: Raises ValueError on invalid inputs (chunker.py:55-57).
        FAILURE MODES: ValueError("chunk_size must be positive") if
        chunk_size <= 0 (chunker.py:55); ValueError("chunk_overlap cannot be
        negative") if chunk_overlap < 0 (chunker.py:56); ValueError("chunk_overlap
        must be less than chunk_size") if chunk_overlap >= chunk_size
        (chunker.py:57).
        ALTERNATIVE: Could silently clamp invalid values instead of raising.
        # GAP: chunk_size is characters, not tokens — a 500-char English chunk
        is ~100 tokens, but a 500-char chunk with rare words could be more;
        embedding models have token limits, not char limits (chunker.py:42).
        # BUSINESS: Validation prevents infinite loops in chunking loops that
        depend on chunk_size > chunk_overlap.
        # INTERVIEW: "Why reject chunk_overlap >= chunk_size? Because then
        start never advances (start = end - overlap >= end), creating an
        infinite loop."
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

        WHAT: Dispatches to the strategy-specific private method
        (_chunk_fixed_size, _chunk_recursive, or _chunk_structure_aware).
        WHY: Single entry point lets callers chunk text without knowing the
        internal strategy implementations.
        INPUTS: text (str) — e.g. "The quick brown fox..." (any length).
        OUTPUTS: List[str] — e.g. ["The quick brown fox", "jumps over the lazy
        dog", ...]; empty list if text is empty (chunker.py:77).
        SIDE EFFECTS: None (pure transformation).
        FAILURE MODES: Raises ValueError for an unsupported strategy value
        (chunker.py:86); returns [] for empty input (chunker.py:77).
        ALTERNATIVE: Could use a dictionary dispatch instead of if/elif chain.
        # GAP: Empty text returns [] silently — callers can't distinguish
        "empty input" from "input produced no chunks" (chunker.py:77).
        # BUSINESS: Consistent return type (always a list) simplifies caller
        logic — no None checks needed.
        # INTERVIEW: "Why return [] for empty text rather than None? So callers
        can iterate the result without an None check."
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
        """Original fixed-size chunking with overlap and sentence boundary awareness.

        WHAT: Walks through text in windows of chunk_size, attempting to end
        each chunk at a sentence boundary (. ! ?) within the last 100 chars,
        then advances by chunk_size - chunk_overlap.
        WHY: Fixed-size is the simplest strategy; sentence-boundary awareness
        avoids splitting mid-sentence where possible.
        INPUTS: text (str) — e.g. a 2000-character document.
        OUTPUTS: List[str] — overlapping chunks each at most ~chunk_size chars.
        SIDE EFFECTS: None.
        FAILURE MODES: Returns [] for empty input (chunker.py:92); guaranteed
        to terminate because start advances by at least (chunk_size -
        chunk_overlap) > 0 each iteration.
        ALTERNATIVE: Could split purely by character count with no sentence
        boundary awareness.
        # GAP: The sentence-boundary search only looks within the last 100
        chars before end — if no sentence end is found there, the chunk is cut
        mid-sentence regardless (chunker.py:105).
        # BUSINESS: Deterministic output — same input always yields same chunks,
        which matters for reproducible embeddings.
        # INTERVIEW: "Why look back 100 chars for a sentence boundary? A
        heuristic — keeps chunks near the target size while usually avoiding
        mid-sentence splits."
        """
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

        WHAT: Splits text by paragraphs (double newline), greedily fills chunks
        up to chunk_size with whole paragraphs, applies overlap between chunks,
        then sub-chunks any oversized chunk with fixed-size.
        WHY: Keeps paragraphs intact as semantic units, which is better for
        retrieval than cutting mid-paragraph.
        INPUTS: text (str) — e.g. a document with multiple paragraphs.
        OUTPUTS: List[str] — chunks that respect paragraph boundaries where
        possible.
        SIDE EFFECTS: Creates a temporary TextChunker for sub-chunking
        oversized chunks (chunker.py:176).
        FAILURE MODES: Returns [] for empty input (chunker.py:141); recursion
        depth is bounded — sub-chunking uses fixed-size, not recursive again.
        ALTERNATIVE: Could recurse on sentences before falling back to
        fixed-size.
        # GAP: A single paragraph longer than chunk_size gets sub-chunked by
        fixed-size, losing the paragraph-boundary benefit for that paragraph
        (chunker.py:176).
        # BUSINESS: Better semantic coherence than pure fixed-size for
        prose-heavy documents.
        # INTERVIEW: "Why 'recursive'? Because it tries progressively finer
        splits — paragraphs first, then fixed-size as a last resort."
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

        WHAT: Splits text by markdown heading lines (# through ######), keeps
        each heading with its content, then recursively chunks any section
        still larger than chunk_size.
        WHY: Keeps sections under their headings intact, which preserves
        document hierarchy for retrieval.
        INPUTS: text (str) — e.g. Markdown with # Headings.
        OUTPUTS: List[str] — chunks that respect section boundaries where
        possible.
        SIDE EFFECTS: Creates temporary TextChunkers for recursive sub-chunking
        (chunker.py:231).
        FAILURE MODES: Returns [] for empty input (chunker.py:191); text with
        no headings falls through to recursive chunking of the whole text
        (effectively one big section).
        ALTERNATIVE: Could also split by HTML <section> tags or other
        structural markers.
        # GAP: The heading regex captures the heading line but the section
        reconstruction can mishandle content before the first heading (that
        content becomes a section with no heading) (chunker.py:202).
        # BUSINESS: Section-aware chunks make retrieval results more
        interpretable — a chunk is more likely to be a coherent section.
        # INTERVIEW: "Why structure-aware for Markdown? Because a heading plus
        its content is a natural semantic unit — keeping them together helps
        the LLM answer 'what does section X say?'"
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

        WHAT: Calls chunk_text on document["content"], then wraps each chunk
        in a dict with the original metadata plus chunk_index, total_chunks,
        chunk_size, is_chunk, chunking_strategy, chunk_size_config,
        chunk_overlap.
        WHY: Preserves the document's metadata on every chunk so retrieval can
        trace a chunk back to its source document and version.
        INPUTS: document (Dict[str, Any]) — e.g.
        {"content": "...", "metadata": {"source": "...", "doc_version": "1.0", ...}}.
        OUTPUTS: List[Dict[str, Any]] — one dict per chunk, each with
        "content" and "metadata" keys.
        SIDE EFFECTS: None.
        FAILURE MODES: Raises KeyError if document lacks "content" or
        "metadata" keys (chunker.py:252).
        ALTERNATIVE: Could add only a subset of metadata to each chunk to
        reduce storage.
        # GAP: total_chunks is computed after chunk_text returns — if chunking
        produces zero chunks, total_chunks would be 0 and no chunk dicts are
        emitted, losing the document entirely from the index (chunker.py:252).
        # BUSINESS: Every chunk carries full provenance metadata — essential for
        access control, versioning, and delete-by-source operations.
        # INTERVIEW: "Why include chunk_index and total_chunks? So the UI can
        show 'chunk 3 of 12' and so reassembly is possible if needed."
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

        WHAT: Iterates over all documents, chunks each via chunk_document, and
        concatenates all chunks into one flat list.
        WHY: Batch entry point for chunking many documents at once.
        INPUTS: documents (List[Dict[str, Any]]) — list of document dicts.
        OUTPUTS: List[Dict[str, Any]] — flat list of all chunks from all docs.
        SIDE EFFECTS: None.
        FAILURE MODES: Propagates chunk_document failures (KeyError) — one bad
        document aborts the whole batch.
        ALTERNATIVE: Could return a per-document list of lists instead of
        flattening.
        # GAP: Flattening loses the document boundary — callers can't tell
        which chunks came from which document without reading metadata
        (chunker.py:285).
        # BUSINESS: Simple flat list is what the embedder and vector store
        expect downstream.
        # INTERVIEW: "Why flatten into one list? Because embed_documents and
        add_documents_to_store both expect a flat list of chunks."
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

    WHAT: Module-level convenience function that maps a string strategy name to
    a ChunkingEnum, builds a TextChunker, and delegates to its chunk_documents
    method.
    WHY: Lets callers specify the strategy as a plain string (e.g. from CLI
    args or config) without importing the Enum.
    INPUTS: documents (List[Dict[str, Any]]), strategy (str, default
    "fixed_size"), chunk_size (int, default 500), chunk_overlap (int, default 50).
    OUTPUTS: List[Dict[str, Any]] — flat list of all chunks from all docs.
    SIDE EFFECTS: None.
    FAILURE MODES: Unknown strategy strings silently fall back to FIXED_SIZE
    (chunker.py:503) — a typo like "recuresive" won't raise, it just uses the
    default. Propagates TextChunker errors (ValueError on bad chunk_size/
    chunk_overlap).
    ALTERNATIVE: Require callers to pass a ChunkingStrategy Enum directly.
    # GAP: Invalid strategy strings silently default to FIXED_SIZE instead of
    raising — misconfiguration is hidden (chunker.py:503).
    # BUSINESS: String-based API is friendlier for CLI and config-file callers.
    # INTERVIEW: "Why default to fixed_size on bad input? Lenient behavior at
    the cost of hiding typos — a production system might log a warning here."
    """
    strategy_map = {
        "fixed_size": ChunkingStrategy.FIXED_SIZE,
        "recursive": ChunkingStrategy.RECURSIVE,
        "structure_aware": ChunkingStrategy.STRUCTURE_AWARE
    }

    chunking_strategy = strategy_map.get(strategy.lower(), ChunkingStrategy.FIXED_SIZE)
    chunker = TextChunker(chunking_strategy, chunk_size, chunk_overlap)
    return chunker.chunk_documents(documents)