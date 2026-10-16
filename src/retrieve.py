from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(
    query: str,
    n_results: int = 5,
    distance_threshold: float = 0.7,
    filter_metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a query using similarity search.

    Args:
        query: The query text.
        n_results: Number of results to return.
        distance_threshold: Maximum distance to consider a result relevant
                           (lower = more similar, depends on distance metric).
        filter_metadata: Optional metadata filter.

    Returns:
        List of relevant chunk dictionaries with added similarity score.
    """
    from .index import search_similar

    # Search for similar chunks
    results = search_similar(
        query=query,
        n_results=n_results,
        filter_metadata=filter_metadata
    )

    # Filter by distance threshold and convert distance to similarity score
    relevant_chunks = []
    for result in results:
        distance = result["distance"]
        # Convert distance to similarity score (assuming cosine distance: 0=identical, 2=opposite)
        # For sentence-transformers with cosine distance, similarity = 1 - distance
        # But we'll keep it generic and let the caller interpret
        if distance <= distance_threshold:
            result_copy = result.copy()
            result_copy["similarity_score"] = 1.0 - distance  # assuming cosine distance
            relevant_chunks.append(result_copy)

    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query (threshold={distance_threshold})")
    return relevant_chunks

def get_context_for_answer(
    chunks: List[Dict[str, Any]],
    max_context_length: int = 2000
) -> str:
    """
    Combine retrieved chunks into a context string for answer generation.

    Args:
        chunks: List of chunk dictionaries from retrieve_relevant_chunks.
        max_context_length: Maximum characters for the context.

    Returns:
        Context string formatted for prompt inclusion.
    """
    if not chunks:
        return ""

    # Sort by relevance (similarity score descending)
    sorted_chunks = sorted(
        chunks,
        key=lambda x: x.get("similarity_score", 0),
        reverse=True
    )

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(sorted_chunks):
        # Format chunk with citation info
        metadata = chunk["metadata"]
        citation = f"[Source: {metadata['source']}, Page: {metadata['page'] + 1}]"  # 1-based for user
        chunk_text = f"{citation}\n{chunk['text']}\n"

        # Check if adding this chunk would exceed max length
        if current_length + len(chunk_text) > max_context_length and context_parts:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    context = "\n---\n".join(context_parts)
    logger.debug(f"Generated context of length {len(context)} from {len(context_parts)} chunks")
    return context