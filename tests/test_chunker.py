"""
Unit tests for the document chunker module.

Documents are plain dicts: {"content": str, "metadata": dict}.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.chunker import chunk_documents


def make_doc(content: str) -> dict:
    return {"content": content, "metadata": {"source": "test.txt"}}


def test_chunk_documents_fixed_size():
    """Test fixed size chunking strategy."""
    test_docs = [make_doc("This is a test document. " * 100)]  # ~3000 chars

    chunks = chunk_documents(test_docs, chunk_size=100, chunk_overlap=20, strategy="fixed_size")

    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    # chunks should not exceed size + small boundary tolerance
    assert all(len(chunk["content"]) <= 200 for chunk in chunks)
    # metadata is propagated and chunk info added
    assert chunks[0]["metadata"]["source"] == "test.txt"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_chunk_documents_recursive():
    """Test recursive chunking strategy preserves paragraph structure."""
    content = "First paragraph.\n\nSecond paragraph with more content. " * 20 + "\n\nThird paragraph."
    chunks = chunk_documents([make_doc(content)], chunk_size=100, chunk_overlap=20, strategy="recursive")

    assert len(chunks) > 0
    # recursive chunking splits on paragraphs first, so chunks should contain paragraph text
    assert any("paragraph" in c["content"].lower() for c in chunks)


def test_chunk_documents_structure_aware():
    """Test structure-aware chunking keeps headings with their content."""
    content = "# Header 1\n\nContent under header 1.\n\n## Header 2\n\nContent under header 2."
    chunks = chunk_documents([make_doc(content)], chunk_size=100, chunk_overlap=20, strategy="structure_aware")

    assert len(chunks) > 0
    # heading should travel with its section
    assert any("Header 1" in c["content"] for c in chunks)


def test_chunk_documents_empty_input():
    chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_chunk_documents_single_character():
    chunks = chunk_documents([make_doc("A")], chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "A"


def test_different_strategies_produce_results():
    content = "Sentence one. Sentence two. Sentence three. " * 20
    fixed_chunks = chunk_documents([make_doc(content)], chunk_size=50, chunk_overlap=10, strategy="fixed_size")
    recursive_chunks = chunk_documents([make_doc(content)], chunk_size=50, chunk_overlap=10, strategy="recursive")
    assert len(fixed_chunks) > 0
    assert len(recursive_chunks) > 0


def test_invalid_chunk_params_raise():
    from src.ingestion.chunker import TextChunker
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    pytest.main([__file__])
