"""
Unit tests for chunk.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunk import chunk_text

def test_chunk_text_basic():
    """Test basic chunking functionality."""
    # Mock pages data
    pages = [
        {
            "text": "This is a test sentence. " * 50,  # ~1000 characters
            "metadata": {
                "source": "test.pdf",
                "page": 0,
                "file_path": "/fake/path/test.pdf"
            }
        }
    ]

    # Chunk with small size for testing
    chunks = chunk_text(pages, chunk_size=100, chunk_overlap=20)

    # Should have multiple chunks
    assert len(chunks) > 1

    # Each chunk should have text and metadata
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test.pdf"
        assert chunk["metadata"]["page"] == 0
        assert "chunk_index" in chunk["metadata"]
        assert "total_chunks" in chunk["metadata"]

    # Check that chunks have reasonable size
    for chunk in chunks:
        assert len(chunk["text"]) <= 120  # chunk_size + some overlap tolerance

    # Check that total_chunks is consistent
    total_chunks = chunks[0]["metadata"]["total_chunks"]
    for chunk in chunks:
        assert chunk["metadata"]["total_chunks"] == total_chunks

    # Check chunk indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk_index"] == i

def test_chunk_text_empty():
    """Test chunking with empty text."""
    pages = [
        {
            "text": "",
            "metadata": {
                "source": "empty.pdf",
                "page": 0,
                "file_path": "/fake/path/empty.pdf"
            }
        }
    ]

    chunks = chunk_text(pages)
    assert len(chunks) == 0

def test_chunk_text_multiple_pages():
    """Test chunking across multiple pages."""
    pages = [
        {
            "text": "Page one content. " * 30,
            "metadata": {
                "source": "multi.pdf",
                "page": 0,
                "file_path": "/fake/path/multi.pdf"
            }
        },
        {
            "text": "Page two content. " * 30,
            "metadata": {
                "source": "multi.pdf",
                "page": 1,
                "file_path": "/fake/path/multi.pdf"
            }
        }
    ]

    chunks = chunk_text(pages, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 0

    # Check that we have chunks from both pages
    pages_found = set(chunk["metadata"]["page"] for chunk in chunks)
    assert 0 in pages_found
    assert 1 in pages_found

    # Check that chunk indices are sequential across pages
    indices = [chunk["metadata"]["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))

def test_chunk_parameters():
    """Test that chunk parameters work correctly."""
    pages = [
        {
            "text": "A" * 200,  # 200 'A' characters
            "metadata": {
                "source": "param.pdf",
                "page": 0,
                "file_path": "/fake/path/param.pdf"
            }
        }
    ]

    # Test with chunk_size=50, no overlap
    chunks = chunk_text(pages, chunk_size=50, chunk_overlap=0)
    assert len(chunks) == 4  # 200 / 50 = 4

    # Test with overlap
    chunks = chunk_text(pages, chunk_size=50, chunk_overlap=10)
    # With overlap, we should get more chunks
    # 200 chars, chunk_size=50, overlap=10
    # First chunk: 0-50
    # Second: 40-90 (50-10 overlap)
    # Third: 80-130
    # Fourth: 120-170
    # Fifth: 160-200
    # So 5 chunks
    assert len(chunks) >= 4  # At least 4, possibly 5 with overlap

if __name__ == "__main__":
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_chunk_text_multiple_pages()
    test_chunk_parameters()
    print("All tests passed!")