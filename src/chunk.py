from typing import List, Dict, Any
import re

def chunk_text(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split text from pages into overlapping chunks.

    Args:
        pages: List of page dictionaries from load_pdf_document.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dictionaries, each with:
        {
            "text": str,
            "metadata": {
                "source": str,
                "page": int,          # zero-based page number where chunk starts
                "chunk_index": int,   # index of chunk within the document
                "total_chunks": int,  # total chunks in document (will be set later)
                "file_path": str,
            }
        }
    """
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_index = 0

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        source = metadata["source"]
        page_num = metadata["page"]
        file_path = metadata["file_path"]

        # Clean text: remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            continue

        # Split into chunks
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Create chunk
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "chunk_index": global_chunk_index,
                    "total_chunks": 0,  # placeholder, will update later
                    "file_path": file_path,
                }
            }
            all_chunks.append(chunk)
            global_chunk_index += 1

            # Move start for next chunk, accounting for overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

    # Update total_chunks for each chunk (same for all chunks in the same document?)
    # Actually, total_chunks is per document. We'll set it per document group.
    # For simplicity, we'll set the same total_chunks for all chunks (total number of chunks)
    total_chunks = len(all_chunks)
    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = total_chunks

    return all_chunks