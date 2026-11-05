"""
Vector store for storing and retrieving document embeddings using ChromaDB.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed. Using fallback vector store.")


class VectorStore:
    """Vector store using ChromaDB for storage and retrieval."""

    def __init__(self,
                 collection_name: str = "rag_academy",
                 persist_directory: str = "./data/chroma_db",
                 dimension: int = 384):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            dimension: Dimension of embedding vectors (for validation)
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.dimension = dimension
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available, using fallback storage")
            self._init_fallback()
            return

        try:
            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection (chromadb raises NotFoundError, not ValueError, when absent)
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG Academy document embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.warning("Falling back to simple vector store")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback simple vector store."""
        self.client = None
        self.collection = None
        self._fallback_vectors = []  # List of embedding vectors
        self._fallback_documents = []  # List of document dictionaries
        self._fallback_ids = []  # List of document IDs

    def add_vectors(self,
                   vectors: List[List[float]],
                   documents: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> None:
        """
        Add vectors and their associated documents to the store.

        Args:
            vectors: List of embedding vectors
            documents: List of document dictionaries
            ids: Optional list of document IDs
        """
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")

        if ids is None:
            ids = [f"doc_{len(self._get_existing_ids()) + i}" for i in range(len(documents))]
        # ensure generated ids don't collide with existing ones
        existing = set(self._get_existing_ids())
        if existing.intersection(ids):
            import uuid
            ids = [f"doc_{uuid.uuid4().hex[:12]}" for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Validate vector dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise ValueError(f"Vector {i} has dimension {len(vector)}, expected {self.dimension}")

        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Add to ChromaDB
                self.collection.add(
                    embeddings=vectors,
                    documents=[doc.get("content", "") for doc in documents],
                    metadatas=[doc.get("metadata", {}) for doc in documents],
                    ids=ids
                )
                logger.info(f"Added {len(vectors)} vectors to ChromaDB collection. Total: {self.collection.count()}")
            except Exception as e:
                logger.error(f"Failed to add vectors to ChromaDB: {e}")
                logger.warning("Falling back to simple storage for this operation")
                self._add_to_fallback(vectors, documents, ids)
        else:
            self._add_to_fallback(vectors, documents, ids)

    def list_sources(self) -> List[Dict[str, Any]]:
        """One summary row per source file: filename, source path, chunk count,
        doc_version, is_current, content hash (if recorded)."""
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                data = self.collection.get(include=["metadatas"])
                metas = data.get("metadatas", [])
            except Exception as e:
                logger.error(f"list_sources failed on ChromaDB: {e}")
                return self._fallback_list_sources()
        else:
            metas = [d.get("metadata", {}) for d in self._fallback_documents]

        by_file: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            if not m:
                continue
            fn = m.get("filename") or m.get("source", "?")
            row = by_file.setdefault(fn, {
                "filename": fn,
                "source": m.get("source"),
                "chunk_count": 0,
                "doc_version": m.get("doc_version"),
                "is_current": False,
                "content_hash": m.get("content_hash"),
                "department": m.get("department"),
            })
            row["chunk_count"] += 1
            if m.get("is_current"):
                row["is_current"] = True
        return sorted(by_file.values(), key=lambda r: r["filename"])

    def _fallback_list_sources(self) -> List[Dict[str, Any]]:
        by_file: Dict[str, Dict[str, Any]] = {}
        for d in self._fallback_documents:
            m = d.get("metadata", {})
            fn = m.get("filename") or m.get("source", "?")
            row = by_file.setdefault(fn, {"filename": fn, "source": m.get("source"),
                                           "chunk_count": 0, "doc_version": m.get("doc_version"),
                                           "is_current": False, "content_hash": m.get("content_hash")})
            row["chunk_count"] += 1
            if m.get("is_current"):
                row["is_current"] = True
        return sorted(by_file.values(), key=lambda r: r["filename"])

    def mark_source_current(self, filename: str, current: bool) -> int:
        """Set is_current on every chunk of the given source file. Returns count updated."""
        target = filename.replace("\\", "/").split("/")[-1]
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                data = self.collection.get(include=["metadatas"])
                ids, metas = data.get("ids", []), data.get("metadatas", [])
                upd_ids, upd_metas = [], []
                for doc_id, m in zip(ids, metas):
                    if (m or {}).get("filename") == target or \
                       str((m or {}).get("source", "")).replace("\\", "/").split("/")[-1] == target:
                        new_meta = dict(m or {})
                        new_meta["is_current"] = current
                        upd_ids.append(doc_id)
                        upd_metas.append(new_meta)
                if upd_ids:
                    self.collection.update(ids=upd_ids, metadatas=upd_metas)
                return len(upd_ids)
            except Exception as e:
                logger.error(f"mark_source_current failed on ChromaDB: {e}")
                return self._fallback_mark_current(target, current)
        return self._fallback_mark_current(target, current)

    def _fallback_mark_current(self, filename: str, current: bool) -> int:
        n = 0
        for d in self._fallback_documents:
            m = d.get("metadata", {})
            if m.get("filename") == filename:
                m["is_current"] = current
                n += 1
        return n

    def _get_existing_ids(self) -> List[str]:
        """Return all currently stored document ids (ChromaDB or fallback)."""
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                return self.collection.get(include=[]).get("ids", [])
            except Exception as e:
                logger.error(f"Failed to list ChromaDB ids: {e}")
                return list(self._fallback_ids)
        return list(self._fallback_ids)

    def _add_to_fallback(self,
                        vectors: List[List[float]],
                        documents: List[Dict[str, Any]],
                        ids: List[str]):
        """Add vectors to fallback storage."""
        self._fallback_vectors.extend(vectors)
        self._fallback_documents.extend(documents)
        self._fallback_ids.extend(ids)
        logger.info(f"Added {len(vectors)} vectors to fallback store. Total: {len(self._fallback_vectors)}")

    def search(self,
              query_vector: List[float],
              k: int = 5,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for k nearest neighbors using cosine similarity.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of tuples (document, similarity_score) sorted by similarity descending
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector has dimension {len(query_vector)}, expected {self.dimension}")

        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Build where clause for metadata filtering
                where_clause = None
                if filter_metadata:
                    where_clause = filter_metadata

                # Query ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=k,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )

                # Process results
                search_results = []
                if results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][i]
                        distance = results['distances'][0][i]  # ChromaDB returns distance (lower is better)
                        # Convert distance to similarity score (1 - normalized distance)
                        similarity = 1.0 - min(distance, 1.0)  # Ensure similarity is in [0,1]

                        # Reconstruct document
                        document = {
                            "content": results['documents'][0][i] if results['documents'] else "",
                            "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                        }

                        search_results.append((document, similarity))

                return search_results
            except Exception as e:
                logger.error(f"Failed to search ChromaDB: {e}")
                logger.warning("Falling back to simple search")
                return self._fallback_search(query_vector, k, filter_metadata)
        else:
            return self._fallback_search(query_vector, k, filter_metadata)

    def _fallback_search(self,
                        query_vector: List[float],
                        k: int,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Fallback search using simple cosine similarity."""
        if not self._fallback_vectors:
            return []

        # Convert to numpy arrays for efficient computation
        import numpy as np
        query_np = np.array(query_vector)
        vectors_np = np.array(self._fallback_vectors)

        # Calculate cosine similarity
        dot_products = np.dot(vectors_np, query_np)
        query_norm = np.linalg.norm(query_np)
        vector_norms = np.linalg.norm(vectors_np, axis=1)

        # Avoid division by zero
        norms_product = query_norm * vector_norms
        norms_product = np.where(norms_product == 0, 1e-8, norms_product)

        similarities = dot_products / norms_product

        # Apply metadata filter if provided
        if filter_metadata:
            filtered_indices = []
            for i, doc in enumerate(self._fallback_documents):
                metadata = doc.get("metadata", {})
                match = True
                for key, value in filter_metadata.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)

            if filtered_indices:
                # Only consider filtered indices
                filtered_similarities = similarities[filtered_indices]
                filtered_documents = [self._fallback_documents[i] for i in filtered_indices]
                filtered_ids = [self._fallback_ids[i] for i in filtered_indices]

                # Get top k from filtered results
                if len(filtered_similarities) > 0:
                    top_k_indices = np.argsort(filtered_similarities)[::-1][:min(k, len(filtered_similarities))]
                    results = []
                    for idx in top_k_indices:
                        original_idx = filtered_indices[idx]
                        results.append((self._fallback_documents[original_idx], float(filtered_similarities[idx])))
                    return results
            return []  # No matches found

        # No filter, get top k from all results
        if len(similarities) > 0:
            top_k_indices = np.argsort(similarities)[::-1][:min(k, len(similarities))]
            results = []
            for idx in top_k_indices:
                results.append((self._fallback_documents[idx], float(similarities[idx])))
            return results

        return []

    def search_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document ID to search for

        Returns:
            Document dictionary if found, None otherwise
        """
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                results = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
                if results['ids'] and len(results['ids']) > 0:
                    return {
                        "content": results['documents'][0],
                        "metadata": results['metadatas'][0] if results['metadatas'] else {}
                    }
                return None
            except Exception as e:
                logger.error(f"Failed to search by ID in ChromaDB: {e}")
                return self._fallback_search_by_id(doc_id)
        else:
            return self._fallback_search_by_id(doc_id)

    def _fallback_search_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fallback search by ID."""
        try:
            idx = self._fallback_ids.index(doc_id)
            return self._fallback_documents[idx]
        except ValueError:
            return None

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a source file.

        Args:
            source: Source identifier (path or filename) to match against metadata['source']

        Returns:
            Number of documents deleted
        """
        # Match by exact source path OR by basename (API may pass either)
        target_name = source.replace("\\", "/").split("/")[-1]
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Chroma where-clauses support $eq / $contains-like ops only on exact fields,
                # so fetch candidate ids by querying metadata, then delete.
                all_meta = self.collection.get(include=["metadatas"])
                ids_to_delete = [
                    doc_id
                    for doc_id, meta in zip(all_meta.get("ids", []), all_meta.get("metadatas", []))
                    if (meta or {}).get("source") == source
                    or str((meta or {}).get("filename", "")) == target_name
                ]
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for source {source!r} from ChromaDB")
                return len(ids_to_delete)
            except Exception as e:
                logger.error(f"Failed to delete by source from ChromaDB: {e}")
                return self._fallback_delete_by_source(source)
        return self._fallback_delete_by_source(source)

    def _fallback_delete_by_source(self, source: str) -> int:
        target_name = source.replace("\\", "/").split("/")[-1]
        keep = [
            i for i, doc in enumerate(self._fallback_documents)
            if (doc.get("metadata", {}).get("source") != source
                and doc.get("metadata", {}).get("filename") != target_name)
        ]
        deleted = len(self._fallback_documents) - len(keep)
        self._fallback_vectors = [self._fallback_vectors[i] for i in keep]
        self._fallback_documents = [self._fallback_documents[i] for i in keep]
        self._fallback_ids = [self._fallback_ids[i] for i in keep]
        return deleted

    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document was deleted, False if not found
        """
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                self.collection.delete(ids=[doc_id])
                logger.info(f"Deleted document with ID: {doc_id} from ChromaDB")
                return True
            except Exception as e:
                logger.error(f"Failed to delete from ChromaDB: {e}")
                return self._fallback_delete_by_id(doc_id)
        else:
            return self._fallback_delete_by_id(doc_id)

    def _fallback_delete_by_id(self, doc_id: str) -> bool:
        """Fallback delete by ID."""
        try:
            idx = self._fallback_ids.index(doc_id)
            del self._fallback_vectors[idx]
            del self._fallback_documents[idx]
            del self._fallback_ids[idx]
            logger.info(f"Deleted document with ID: {doc_id} from fallback store")
            return True
        except ValueError:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                count = self.collection.count()
                return {
                    "total_vectors": count,
                    "dimension": self.dimension,
                    "total_documents": count,
                    "is_empty": count == 0,
                    "storage_type": "chromadb",
                    "persist_directory": str(self.persist_directory),
                    "collection_name": self.collection_name
                }
            except Exception as e:
                logger.error(f"Failed to get ChromaDB stats: {e}")
                return self._fallback_get_stats()
        else:
            return self._fallback_get_stats()

    def _fallback_get_stats(self) -> Dict[str, Any]:
        """Get statistics from fallback storage."""
        return {
            "total_vectors": len(self._fallback_vectors),
            "dimension": self.dimension,
            "total_documents": len(self._fallback_documents),
            "is_empty": len(self._fallback_vectors) == 0,
            "storage_type": "fallback",
            "total_vectors": len(self._fallback_vectors)
        }

    def reset(self):
        """Reset/clear the vector store."""
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Delete and recreate collection
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG Academy document embeddings"}
                )
                logger.info(f"Reset collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to reset ChromaDB collection: {e}")
                self._fallback_reset()
        else:
            self._fallback_reset()

    def _fallback_reset(self):
        """Reset fallback storage."""
        self._fallback_vectors = []
        self._fallback_documents = []
        self._fallback_ids = []
        logger.info("Reset fallback vector store")


# Convenience functions
def create_vector_store(collection_name: str = "rag_academy",
                       persist_directory: str = "./data/chroma_db",
                       dimension: int = 384) -> VectorStore:
    """Create a new vector store."""
    return VectorStore(collection_name=collection_name,
                      persist_directory=persist_directory,
                      dimension=dimension)

def add_documents_to_store(store: VectorStore,
                          documents: List[Dict[str, Any]]) -> None:
    """
    Add documents (with embeddings) to a vector store.

    Args:
        store: VectorStore instance
        documents: List of document dictionaries with embeddings
    """
    vectors = [doc["embedding"] for doc in documents if "embedding" in doc]
    valid_docs = [doc for doc in documents if "embedding" in doc]

    if vectors:
        store.add_vectors(vectors, valid_docs)