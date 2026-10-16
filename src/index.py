from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger(__name__)

# Global model and client (in practice, you might want to manage this via dependency injection)
_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight and effective
_CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
_COLLECTION_NAME = "telecom_docs"

def get_embedding_model() -> SentenceTransformer:
    """Load and return the sentence transformer model."""
    return SentenceTransformer(_MODEL_NAME)

def get_chroma_client() -> chromadb.ClientAPI:
    """Initialize and return a persistent ChromaDB client."""
    # Ensure the directory exists
    os.makedirs(_CHROMA_PATH, exist_ok=True)
    settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=_CHROMA_PATH,
    )
    return chromadb.Client(settings)

def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the collection for storing documents."""
    try:
        collection = client.get_collection(name=_COLLECTION_NAME)
        logger.info(f"Got existing collection '{_COLLECTION_NAME}'")
    except Exception:
        collection = client.create_collection(name=_COLLECTION_NAME)
        logger.info(f"Created new collection '{_COLLECTION_NAME}'")
    return collection

def index_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Embed and index chunks into ChromaDB.

    Args:
        chunks: List of chunk dictionaries from chunk_text.
    """
    if not chunks:
        logger.warning("No chunks to index.")
        return

    # Initialize model and client
    model = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Prepare data for ChromaDB
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    # Generate IDs: we can use a hash or simple index-based ID
    ids = [f"{meta['source']}_{meta['page']}_{meta['chunk_index']}" for meta in metadatas]

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Add to collection
    # Note: ChromaDB upserts if IDs already exist
    collection.upsert(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    logger.info(f"Indexed {len(chunks)} chunks into collection '{_COLLECTION_NAME}'.")

    # Persist the database
    client.persist()

def search_similar(
    query: str,
    n_results: int = 5,
    filter_metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks given a query.

    Args:
        query: The query text.
        n_results: Number of results to return.
        filter_metadata: Optional metadata filter (e.g., {"source": "esim-spec.pdf"}).

    Returns:
        List of dictionaries with keys: "text", "metadata", "distance".
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Embed the query
    query_embedding = model.encode([query]).tolist()[0]

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filter_metadata,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted_results = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
    return formatted_results