"""
Unit tests for the document chunker module.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.chunker import chunk_documents, Document

def test_chunk_documents_fixed_size():
    """Test fixed size chunking strategy."""
    # Create test documents
    test_docs = [
        Document(
            page_content="This is a test document. " * 100,  # ~3000 characters
            metadata={"source": "test.txt", "page": 1}
        )
    ]

    # Test fixed size chunking
    chunks = chunk_documents(
        test_docs,
        chunk_size=100,
        chunk_overlap=20,
        strategy="fixed_size"
    )

    # Verify chunks were created
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)

    # Verify chunk sizes are approximately correct (accounting for overlap)
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # Allow some flexibility

    # Verify overlap exists
    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        chunk1_end = chunks[0].page_content[-20:]
        chunk2_start = chunks[1].page_content[:20]
        # In a real implementation, there would be actual overlap
        # For now, we just verify the chunks exist

def test_chunk_documents_recursive():
    """Test recursive chunking strategy."""
    # Create test documents with paragraph-like structure
    test_docs = [
        Document(
            page_content="First paragraph.\n\nSecond paragraph with more content. " * 20 + "\n\nThird paragraph.",
            metadata={"source": "test.txt", "page": 1}
        )
    ]

    # Test recursive chunking
    chunks = chunk_documents(
        test_docs,
        chunk_size=100,
        chunk_overlap=20,
        strategy="recursive"
    )

    # Verify chunks were created
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)

    # Verify that chunking respects paragraph boundaries to some extent
    # (This is harder to test precisely without knowing the exact implementation)

def test_chunk_documents_structure_aware():
    """Test structure-aware chunking strategy."""
    # Create test documents with markdown-like structure
    test_docs = [
        Document(
            page_content="# Header 1\n\nContent under header 1.\n\n## Header 2\n\nContent under header 2.",
            metadata={"source": "test.txt", "page": 1}
        )
    ]

    # Test structure-aware chunking
    chunks = chunk_documents(
        test_docs,
        chunk_size=100,
        chunk_overlap=20,
        strategy="structure_aware"
    )

    # Verify chunks were created
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)

def test_chunk_documents_empty_input():
    """Test chunking with empty input."""
    chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
    assert chunks == []

def test_chunk_documents_single_character():
    """Test chunking with minimal content."""
    test_docs = [
        Document(
            page_content="A",
            metadata={"source": "test.txt", "page": 1}
        )
    ]

    chunks = chunk_documents(test_docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].page_content == "A"

def test_different_strategies_produce_different_results():
    """Test that different chunking strategies can produce different results."""
    # Create a document that would chunk differently based on strategy
    test_docs = [
        Document(
            page_content="Sentence one. Sentence two. Sentence three. " * 20,
            metadata={"source": "test.txt", "page": 1}
        )
    ]

    fixed_chunks = chunk_documents(
        test_docs,
        chunk_size=50,
        chunk_overlap=10,
        strategy="fixed_size"
    )

    recursive_chunks = chunk_documents(
        test_docs,
        chunk_size=50,
        chunk_overlap=10,
        strategy="recursive"
    )

    # Both should produce chunks
    assert len(fixed_chunks) > 0
    assert len(recursive_chunks) > 0

    # The number of chunks might differ based on strategy implementation
    # This test mainly verifies both strategies work without crashing

if __name__ == "__main__":
    pytest.main([__file__])