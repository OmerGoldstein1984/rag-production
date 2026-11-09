"""
VECTOR STORE LAYER — rag-academy RAG system

Business problem this file solves:
    Stores document embedding vectors and their source metadata, then answers
    nearest-neighbor ("which chunks are most like this query?") lookups so the RAG
    agent can ground its answers in retrieved telecom document text. Without this layer
    there is no memory of ingested documents: every question returns "I cannot determine
    from available materials." It is the long-term memory of the system.

Where it sits in the data flow:
    STORAGE/RETRIEVAL stage — downstream of the embedder
    (src/embedding/embedder.py), upstream of the retriever
    (src/retrieval/retriever.py). Ingestion calls add_vectors/add_documents_to_store to
    persist chunks; retrieval calls search (and search_by_id, delete_by_source) to query
    and manage them.

What calls it:
    src/ingestion/manager.py (ingest path adds documents),
    src/retrieval/retriever.py / sparse_retriever.py / hybrid.py (search path queries),
    src/api/main.py (delete-by-source on document removal / reindex), plus module-level
    convenience functions create_vector_store / add_documents_to_store.

What it calls:
    chromadb.PersistentClient + chromadb.config.Settings when ChromaDB is installed;
    otherwise numpy (stdlib-adjacent) for an in-memory fallback cosine-similarity store.
    Logs via the standard logging module.

What breaks if it is removed:
    Ingested vectors have nowhere to live and searches have nothing to query; retrieval
    always returns empty, so no answer can be grounded in source documents. The system
    degrades to pure hallucination risk or total silence.

Design notes for the learner (senior QA/automation engineer, weak on production
architecture / concurrency / observability / cost tradeoffs):
    - Dual-backend design: ChromaDB (persistent, on disk under data/chroma_db) is
      preferred; a plain-Python in-memory fallback keeps everything testable and runnable
      when ChromaDB is missing. This is a graceful-degradation pattern, not a bug.
    - ChromaDB returns distances (lower = closer); this layer converts to similarity
      (1 - distance) so callers always see "higher = more similar."
    - All ChromaDB operations are wrapped in try/except and fall back to the in-memory
      store on failure — the caller never sees a ChromaDB exception.
    - There is no locking or concurrency control: the store is single-process. Multiple
      writers (e.g. two ingest workers) can corrupt interleaved state — see # GAP below.

# GAP: Concurrency / locking and observability — there is no thread lock, file lock, or
  transaction around add/delete/reset, and no metrics or tracing hooks on search latency
  or cache hit rate. A web/automation engineer used to request-scoped transactions may
  assume "the database handles it"; here ChromaDB is embedded (single-process) and the
  fallback is plain Python lists, so concurrent writers and crash consistency are NOT
  handled. Productionizing means adding a lock, health checks, and latency histograms.
# BUSINESS: Protects answer correctness (wrong/empty retrieval => wrong or missing
  answers) and keeps the system runnable offline for demos and tests.
# INTERVIEW: "What happens if ChromaDB isn't installed?" Say it out loud: the module
  catches ImportError at import time and transparently uses an in-memory cosine-similarity
  fallback — same API, no crash, data lost on process exit.
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

        WHAT: Records configuration, then calls _initialize_client to open (or create)
              the ChromaDB collection; if ChromaDB is unavailable or initialization fails,
              falls back to an in-memory store.
        WHY: Construction must succeed regardless of environment so the rest of the app
             can be built and tested without a running database server. Choosing
             collection_name and persist_directory here fixes where data lives for the
             object's lifetime.
        INPUTS:
            collection_name (str): the ChromaDB collection name. Example: "rag_academy".
            persist_directory (str): on-disk folder for the persistent DB.
                                     Example: "./data/chroma_db".
            dimension (int): expected embedding vector size, used to validate incoming
                             vectors. Example: 384 (all-MiniLM-L6-v2 output size).
        OUTPUTS: None; sets self.client, self.collection, self.dimension, and fallback
                 lists (_fallback_vectors / _fallback_documents / _fallback_ids).
        SIDE EFFECTS: creates persist_directory on disk; may open/create a ChromaDB
                      database file; logs info/warning/error.
        FAILURE MODES: if ChromaDB import or client init raises, silently falls back to
                       the in-memory store (no exception escapes the constructor).
        ALTERNATIVE: line — use a client-server ChromaDB (HttpClient) or a managed vector
                     DB (Pinecone, Weaviate cloud) instead of an embedded PersistentClient
                     for multi-service or production deployments.
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)  # Path for / operator and mkdir
        self.dimension = dimension  # used to validate every incoming vector
        self.client = None  # set by _initialize_client
        self.collection = None  # set by _initialize_client
        self._initialize_client()  # open ChromaDB or fall back to in-memory

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available, using fallback storage")
            self._init_fallback()  # no ChromaDB package at all
            return

        try:
            # Ensure persist directory exists before opening the embedded DB
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize client: embedded, persistent, telemetry off, reset allowed
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection: chromadb raises NotFoundError (not ValueError)
            # when a collection is absent, so catch generically and create it.
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
            self._init_fallback()  # ChromaDB present but unusable at runtime

    def _init_fallback(self):
        """Initialize fallback simple vector store."""
        self.client = None  # no ChromaDB client
        self.collection = None  # no ChromaDB collection
        self._fallback_vectors = []  # List of embedding vectors (parallel arrays)
        self._fallback_documents = []  # List of document dictionaries
        self._fallback_ids = []  # List of document IDs

    def add_vectors(self,
                   vectors: List[List[float]],
                   documents: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> None:
        """
        Add vectors and their associated documents to the store.

        WHAT: Validates that vectors and documents line up, resolves IDs (generating
          doc_N ids when missing, re-issuing uuid-based ids on collision), checks every
          vector's dimension, then persists to ChromaDB or the fallback store.
        WHY: This is the ingestion write path. Dimension validation catches mismatched
          embeddings early (a 768-d vector in a 384-d store would silently break search).
          ID collision handling keeps re-ingest / versioning from overwriting old chunks.
        INPUTS:
            vectors (List[List[float]]): embedding vectors. Example:
                                         [[0.12, -0.05, ...], ...] each length 384.
            documents (List[Dict[str, Any]]): document dicts with "content" and
                                              "metadata". Example:
                                              [{"content": "...", "metadata": {...}}]
            ids (Optional[List[str]]): explicit IDs, or None to auto-generate.
                                       Example: ["chunk_1", "chunk_2"].
        OUTPUTS: None (stores in ChromaDB or the fallback lists).
        SIDE EFFECTS: writes to ChromaDB (persisted to disk) or appends to the in-memory
                      fallback lists; logs success/failure; may raise ValueError.
        FAILURE MODES: raises ValueError if lengths mismatch or a vector has the wrong
                       dimension; ChromaDB write failure is caught and routed to the
                       fallback store for this operation only.
        ALTERNATIVE: line — require callers to always supply ids (no auto-generation)
                     to make re-ingest idempotent and collision-free.
        # GAP: Concurrency / idempotency — auto-generated ids ("doc_N") depend on the
        #      current count, so two concurrent add_vectors calls can generate the same
        #      id and overwrite each other; there is no lock. A learner from web/automation
        #      may expect database-style uniqueness; here uniqueness is best-effort.
        # BUSINESS: Guards against silent dimension-mismatch corruption that would make
        #           search return garbage; id handling protects versioned re-ingest.
        # INTERVIEW: "How do you prevent duplicate chunks when re-ingesting the same file?"
        #            Say it out loud: delete_by_source removes all old chunks for that file
        #            before add_vectors writes the new ones — delete-then-write, not upsert.
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

        # Validate vector dimensions: reject anything not matching self.dimension
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
            row["chunk_count"] += 1  # count each chunk belonging to this source file
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
        target = filename.replace("\\", "/").split("/")[-1]  # normalize to basename only
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                data = self.collection.get(include=["metadatas"])
                ids, metas = data.get("ids", []), data.get("metadatas", [])
                upd_ids, upd_metas = [], []
                for doc_id, m in zip(ids, metas):
                    # match by filename metadata OR by basename of the source path
                    if (m or {}).get("filename") == target or \
                       str((m or {}).get("source", "")).replace("\\", "/").split("/")[-1] == target:
                        new_meta = dict(m or {})
                        new_meta["is_current"] = current  # flip the current-version flag
                        upd_ids.append(doc_id)
                        upd_metas.append(new_meta)
                if upd_ids:
                    self.collection.update(ids=upd_ids, metadatas=upd_metas)
                return len(upd_ids)  # how many chunks were flipped
            except Exception as e:
                logger.error(f"mark_source_current failed on ChromaDB: {e}")
                return self._fallback_mark_current(target, current)
        return self._fallback_mark_current(target, current)

    def _fallback_mark_current(self, filename: str, current: bool) -> int:
        n = 0
        for d in self._fallback_documents:
            m = d.get("metadata", {})
            if m.get("filename") == filename:
                m["is_current"] = current  # mutates the stored dict in place
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

        WHAT: Takes a query embedding and returns the k most similar stored documents as
              (document_dict, similarity_score) tuples sorted descending by similarity.
        WHY: This is the core retrieval operation — given an embedded user question,
             find the chunks that most likely contain the answer. Similarity (higher =
             more similar) is returned so callers can threshold or rank uniformly.
        INPUTS:
            query_vector (List[float]): the embedded query. Example: [0.12, -0.05, ...]
                                        length 384.
            k (int): max results to return. Example: 5.
            filter_metadata (Optional[Dict[str, Any]]): metadata key/value constraints.
                Example: {"filename": "esim.pdf"} to scope to one source file.
        OUTPUTS: List[Tuple[Dict[str, Any], float]] — each tuple is
                 ({"content": ..., "metadata": ...}, similarity). Example:
                 [({"content": "eSIM uses HTTPS...", "metadata": {...}}, 0.89), ...]
        SIDE EFFECTS: queries ChromaDB or scans the in-memory vectors; logs errors;
                      no writes, no state mutation.
        FAILURE MODES: raises ValueError if query_vector length != self.dimension;
                       ChromaDB failure is caught and routed to fallback search (no
                       exception to caller).
        ALTERNATIVE: line — use an approximate-nearest index (HNSW, FAISS) or a
                     dedicated vector DB service for large corpora instead of a linear
                     scan.
        # GAP: Search is a full linear scan over all vectors in the fallback (and ChromaDB
        #      here is a small embedded instance). At scale this is O(n) per query with
        #      no index tuning, no re-ranking, and no result caching — fine for a demo,
        #      not for millions of vectors. Observability (latency histograms) is absent.
        # BUSINESS: Retrieval quality directly drives answer correctness; dimension
        #           validation prevents meaningless similarity scores.
        # INTERVIEW: "How does similarity scoring work?" Say it out loud: ChromaDB gives a
        #            distance (lower = closer); this layer converts it to similarity =
        #            1 - min(distance, 1.0), so the caller always sees higher = more
        #            similar, clamped to [0,1].
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector has dimension {len(query_vector)}, expected {self.dimension}")

        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Build where clause for metadata filtering
                where_clause = None
                if filter_metadata:
                    where_clause = filter_metadata

                # Query ChromaDB: nearest neighbors by embedding
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=k,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )

                # Process results: ChromaDB returns distance (lower is better)
                search_results = []
                if results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][i]
                        distance = results['distances'][0][i]  # ChromaDB: lower is better
                        # Convert distance to similarity score (1 - normalized distance)
                        similarity = 1.0 - min(distance, 1.0)  # ensure similarity in [0,1]

                        # Reconstruct document dict from returned fields
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

        # Calculate cosine similarity: dot product divided by norm product
        dot_products = np.dot(vectors_np, query_np)
        query_norm = np.linalg.norm(query_np)
        vector_norms = np.linalg.norm(vectors_np, axis=1)

        # Avoid division by zero: replace zero norms with a tiny epsilon
        norms_product = query_norm * vector_norms
        norms_product = np.where(norms_product == 0, 1e-8, norms_product)

        similarities = dot_products / norms_product

        # Apply metadata filter if provided: keep only matching documents
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
                # Only consider filtered indices for ranking
                filtered_similarities = similarities[filtered_indices]
                filtered_documents = [self._fallback_documents[i] for i in filtered_indices]
                filtered_ids = [self._fallback_ids[i] for i in filtered_indices]

                # Get top k from filtered results (descending similarity)
                if len(filtered_similarities) > 0:
                    top_k_indices = np.argsort(filtered_similarities)[::-1][:min(k, len(filtered_similarities))]
                    results = []
                    for idx in top_k_indices:
                        original_idx = filtered_indices[idx]
                        results.append((self._fallback_documents[original_idx], float(filtered_similarities[idx])))
                    return results
            return []  # No matches found after filtering

        # No filter, get top k from all results (descending similarity)
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

        WHAT: Looks up a single stored document by its exact ID and returns its
              content and metadata, or None if absent.
        WHY: Used to fetch a specific chunk (e.g. after deletion, for editing, or to
             verify a stored record) without running a similarity search.
        INPUTS:
            doc_id (str): the document ID. Example: "doc_7" or a uuid-based id.
        OUTPUTS: Dict[str, Any] with "content" and "metadata", or None if not found.
        SIDE EFFECTS: reads only; logs error on ChromaDB failure.
        FAILURE MODES: ChromaDB failure is caught and routed to fallback lookup (no
                       exception to caller); missing id returns None.
        ALTERNATIVE: line — none needed; direct-ID fetch is the right primitive here.
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
            idx = self._fallback_ids.index(doc_id)  # ValueError if not present
            return self._fallback_documents[idx]
        except ValueError:
            return None

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a source file.

        WHAT: Removes every stored chunk whose metadata matches the given source path or
              filename, returning the number of chunks deleted.
        WHY: Supports document versioning / re-ingest — before re-adding a file's chunks,
             the old ones for that source are wiped so there are no duplicates or stale
             versions. API callers may pass a full path or just a basename; both match.
        INPUTS:
            source (str): source identifier (path or filename) to match against
                          metadata['source'] / metadata['filename'].
                          Example: "uploads/esim_policy.pdf" or "esim_policy.pdf".
        OUTPUTS: int — number of documents deleted. Example: 3 (three chunks).
        SIDE EFFECTS: permanently deletes rows from ChromaDB or the fallback lists;
                      logs the count.
        FAILURE MODES: ChromaDB failure is caught and routed to fallback delete (no
                       exception); deleting a nonexistent source returns 0.
        ALTERNATIVE: line — none needed; this is the correct primitive for source-scoped
                     deletion.
        # BUSINESS: Prevents stale/duplicate chunks from accumulating on re-ingest,
        #           which would otherwise make retrieval return contradictory answers.
        # INTERVIEW: "How do you update a document that has already been ingested?"
        #            Say it out loud: delete_by_source removes all old chunks, then
        #            add_vectors writes the fresh ones — delete-then-write, not upsert.
        """
        # Match by exact source path OR by basename (API may pass either)
        target_name = source.replace("\\", "/").split("/")[-1]
        if CHROMA_AVAILABLE and self.collection is not None:
            try:
                # Chroma where-clauses support only exact-field ops, so fetch candidate
                # ids by reading all metadata, then delete the matching subset.
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

        WHAT: Removes the single stored chunk with the given ID; returns True if it
              existed and was removed, False otherwise.
        WHY: Fine-grained removal for correcting or retracting one chunk without
             touching the rest of the store.
        INPUTS:
            doc_id (str): the document ID to delete. Example: "doc_7".
        OUTPUTS: bool — True if deleted, False if not found.
        SIDE EFFECTS: deletes from ChromaDB or fallback; logs the outcome.
        FAILURE MODES: ChromaDB failure is caught and routed to fallback delete
                       (returns False on failure path via fallback); nonexistent id
                       returns False.
        ALTERNATIVE: line — none needed; direct-ID delete is the right primitive.
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

        WHAT: Returns a summary dict: total vector count, dimension, emptiness, storage
              type, and (for ChromaDB) persist directory and collection name.
        WHY: Used by health endpoints, monitoring, and tests to confirm the store is
             populated and reachable, and to distinguish ChromaDB vs fallback mode.
        INPUTS: None (reads instance state).
        OUTPUTS: Dict[str, Any]. Example ChromaDB:
                 {"total_vectors": 42, "dimension": 384, "total_documents": 42,
                  "is_empty": False, "storage_type": "chromadb",
                  "persist_directory": "./data/chroma_db", "collection_name": "rag_academy"}
                 Example fallback: same shape with storage_type "fallback".
        SIDE EFFECTS: reads only; logs error on ChromaDB failure.
        FAILURE MODES: ChromaDB failure is caught and routed to fallback stats (no
                       exception).
        ALTERNATIVE: line — add per-source chunk counts or a last-updated timestamp if
                     operators need more granular health info.
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
            "total_vectors": len(self._fallback_vectors)  # duplicate key retained from original
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

    WHAT: Pulls the "embedding" vector out of each document that has one, then calls
          store.add_vectors with those vectors and the matching documents.
    WHY: Thin adapter so ingestion code can pass a list of already-embedded document
         dicts straight into the store without extracting vectors itself.
    INPUTS:
        store (VectorStore): the target store instance.
        documents (List[Dict[str, Any]]): embedded document dicts, each with an
                                          "embedding" key. Example:
                                          [{"content": "...", "embedding": [...], ...}]
    OUTPUTS: None (writes into store).
    SIDE EFFECTS: writes vectors into store via add_vectors; logs via add_vectors.
    FAILURE MODES: documents lacking "embedding" are silently skipped (valid_docs filter),
                   so a batch with no embeddings writes nothing and raises nothing;
                   add_vectors may raise ValueError on dimension mismatch.
    ALTERNATIVE: line — validate upfront that every document has an embedding and raise
                 if any is missing, rather than silently dropping incomplete records.
    """
    vectors = [doc["embedding"] for doc in documents if "embedding" in doc]
    valid_docs = [doc for doc in documents if "embedding" in doc]

    if vectors:
        store.add_vectors(vectors, valid_docs)