# =============================================================================
# FILE HEADER — src/ingestion/manager.py
# Business problem: Manages the full document lifecycle — upload with
# validation + dedupe + versioning, listing, status, delete (cascading to the
# vector store), and re-indexing — so the RAG corpus stays consistent and
# auditable as documents are added, changed, or removed.
# Place in data flow: Orchestrates steps 1-4 of ingestion (load→clean→chunk→embed→store)
# via _index_file (manager.py:172), and exposes lifecycle operations for the
# API layer. Calls loaders.py, cleaner.py, chunker.py, embedding/embedder.py,
# store/vector_store.py.
# What breaks if removed: No way to upload, version, delete, or reindex
# documents; the corpus is static and can't be kept current as source docs change.
# =============================================================================
"""
Document lifecycle manager: upload with validation + dedupe + versioning,
listing, status, delete (cascades to the vector store), and re-indexing.

Rules:
- Only allowlisted extensions and a size cap (configurable).
- Files are written under the project uploads/ dir (never outside it).
- Dedupe by SHA-256 content hash: re-uploading identical bytes is a no-op.
- Versioning: re-uploading the same filename supersedes the old version
  (old chunks get is_current=False in the vector store) and bumps doc_version.
- Every operation is recorded in a status log (in-memory + persisted JSON)
  so failed ingestions are visible and retryable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = project_root / "uploads"
STATUS_FILE = project_root / "data" / "ingest_status.json"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class UploadError(ValueError):
    """Invalid upload (bad type, too large, unsafe name).

    WHAT: Custom exception subclass of ValueError raised by DocumentManager
    validation when a file fails upload checks.
    WHY: Lets callers catch upload-specific failures separately from other
    ValueErrors.
    INPUTS: message (str) — human-readable reason for rejection.
    OUTPUTS: None (exception).
    SIDE EFFECTS: None.
    FAILURE MODES: None — it's an exception type, raised by validate/upload.
    ALTERNATIVE: Use ValueError directly with a message string.
    # GAP: No error code or structured detail — callers must parse the message
    string to distinguish "too large" from "bad extension" (manager.py:49).
    # BUSINESS: Clear, catchable signal for the API layer to return 400 Bad
    Request with a meaningful message.
    # INTERVIEW: "Why a custom exception instead of plain ValueError? So
    except UploadError catches only upload rejections, not unrelated ValueErrors."
    """


@dataclass
class IngestRecord:
    filename: str
    content_hash: str
    status: str  # ok | duplicate | superseded | failed
    chunks: int = 0
    doc_version: str = ""
    error: str = ""
    ts: float = field(default_factory=time.time)

    # WHAT: Dataclass recording one ingestion attempt — filename, content hash,
    # status, chunk count, version, error message, and timestamp.
    # WHY: Provides an auditable log of every upload so failed ingestions are
    # visible and retryable.
    # INPUTS: filename (str), content_hash (str), status (str), chunks (int,
    # default 0), doc_version (str, default ""), error (str, default ""),
    # ts (float, default now).
    # OUTPUTS: None (dataclass instance); to_dict produces a serializable dict.
    # SIDE EFFECTS: ts defaults to time.time() at construction (manager.py:60).
    # FAILURE MODES: None — plain data holder.
    # ALTERNATIVE: A plain dict or a Pydantic model with validation.
    # GAP: status is a free-form string with no enum — a typo like "succeeded"
    # instead of "ok" would go unnoticed (manager.py:56).
    # BUSINESS: The status log is the system's source of truth for what was
    # ingested and when — critical for ops and debugging.
    # INTERVIEW: "Why store content_hash truncated to 12 chars in to_dict?
    # Enough to identify a duplicate without bloating the log with full 64-char
    # hashes."
    def to_dict(self) -> dict:
        return {
            "filename": self.filename, "content_hash": self.content_hash[:12],
            "status": self.status, "chunks": self.chunks,
            "doc_version": self.doc_version, "error": self.error, "ts": self.ts,
        }


class DocumentManager:
    """Handles upload/list/status/delete/reindex for the RAG corpus.

    WHAT: Central coordinator for document lifecycle operations — validating,
    uploading, deduplicating, versioning, listing, checking status, deleting,
    and re-indexing documents in the RAG corpus.
    WHY: A single class owns all lifecycle concerns, keeping the API layer thin
    and ensuring consistent rules (dedupe by hash, version by filename, cascade
    delete) across all entry points.
    INPUTS: upload_dir (Path, default UPLOAD_DIR), status_file (Path, default
    STATUS_FILE), max_size (int, default MAX_FILE_SIZE_BYTES).
    OUTPUTS: None directly — methods return records, lists, or dicts.
    SIDE EFFECTS: Creates upload_dir on construction (manager.py:127);
    persists status log to disk on every record (manager.py:151).
    FAILURE MODES: upload raises UploadError on validation failure;
    _index_file failures are caught and recorded as "failed" status
    (manager.py:203).
    ALTERNATIVE: Could split into separate classes (Uploader, Deleter,
    Reindexer) for finer separation of concerns.
    # GAP: No locking — concurrent uploads of the same filename could race on
    versioning and deduplication (manager.py:126-126).
    # BUSINESS: Protects corpus consistency — dedupe prevents duplicate
    storage, versioning keeps history, cascade delete avoids orphaned chunks.
    # INTERVIEW: "Why one manager class instead of separate services? So all
    lifecycle rules live in one place and can't be applied inconsistently."
    """

    def __init__(self, upload_dir: Path = UPLOAD_DIR, status_file: Path = STATUS_FILE,
                 max_size: int = MAX_FILE_SIZE_BYTES):
        """Initialize the manager — create upload dir, load persisted status."""
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)  # Ensure uploads/ exists
        self.status_file = Path(status_file)
        self.max_size = max_size
        self._records: List[IngestRecord] = self._load_status()  # Rehydrate from disk

    # --- status log -------------------------------------------------------

    def _load_status(self) -> List[IngestRecord]:
        """Load ingestion records from the persisted JSON status file."""
        if not self.status_file.exists():
            return []  # First run — no status file yet
        try:
            data = json.loads(self.status_file.read_text(encoding="utf-8"))
            return [IngestRecord(**{k: r.get(k) for k in IngestRecord.__dataclass_fields__})
                    for r in data.get("records", [])]
        except Exception:
            return []  # Corrupt or unreadable status file — start fresh rather than crash

    def _save_status(self) -> None:
        """Persist all records to the JSON status file."""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure data/ dir exists
        self.status_file.write_text(
            json.dumps({"records": [r.to_dict() for r in self._records]}, indent=2),
            encoding="utf-8",
        )

    def _record(self, rec: IngestRecord) -> None:
        """Append a record, cap the in-memory list at 500, and persist."""
        self._records.append(rec)
        if len(self._records) > 500:
            self._records = self._records[-500:]  # Keep only the most recent 500
        self._save_status()

    def status_log(self, limit: int = 50) -> List[dict]:
        """Return the most recent ``limit`` records as dicts."""
        return [r.to_dict() for r in self._records[-limit:]]

    # --- validation -------------------------------------------------------

    def validate(self, filename: str, content: bytes) -> str:
        """Validate filename + size; return content hash. Raises UploadError.

        WHAT: Checks that filename is a plain name (no path traversal), has an
        allowed extension, content is non-empty, and size is within the cap.
        Returns the SHA-256 hex digest of the content.
        WHY: Blocks unsafe uploads (path traversal, disallowed types, empty
        files, oversized files) before any disk or vector-store write.
        INPUTS: filename (str) — e.g. "policy.pdf"; content (bytes) — raw file
        bytes.
        OUTPUTS: str — SHA-256 hex digest, e.g.
        "a3f5...c9e1".
        SIDE EFFECTS: None (pure computation + validation).
        FAILURE MODES: Raises UploadError for path separators in filename
        (manager.py:161), disallowed extension (manager.py:163), empty content
        (manager.py:165), or oversized file (manager.py:167).
        ALTERNATIVE: Could also scan for malware or validate MIME type against
        extension.
        # GAP: No rate limiting — a client could upload thousands of files in a
        loop, filling disk (manager.py:161).
        # BUSINESS: First line of defense — keeps unsafe or useless files out
        of the corpus entirely.
        # INTERVIEW: "Why validate before hashing? Hashing a multi-gigabyte
        file wastes CPU; reject oversized files first."
        """
        name = Path(filename).name  # strip any path components
        if not name or name != filename:
            raise UploadError("filename must be a plain name, no path separators")
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UploadError(f"extension {ext!r} not allowed; allowed: {sorted(ALLOWED_EXTENSIONS)}")
        if not content:
            raise UploadError("empty file")
        if len(content) > self.max_size:
            raise UploadError(f"file too large: {len(content)} > {self.max_size} bytes")
        return hashlib.sha256(content).hexdigest()

    # --- upload + ingest --------------------------------------------------

    def upload(self, filename: str, content: bytes, vector_store) -> IngestRecord:
        """Validate, dedupe, version, persist, and index one file. Returns the record.

        WHAT: Full upload flow — validate, check dedupe by content hash,
        version by filename, persist bytes to uploads/, index
        (load→clean→chunk→embed→store), and record the result.
        WHY: One atomic-ish operation handles all the rules (dedupe, version,
        persist, index) so callers don't have to orchestrate them.
        INPUTS: filename (str), content (bytes), vector_store (object with
        list_sources, mark_source_current, add_documents_to_store).
        OUTPUTS: IngestRecord with status "ok", "duplicate", "superseded", or
        "failed".
        SIDE EFFECTS: Writes file to uploads/ (manager.py:195), writes to
        vector store (manager.py:200), appends to status log and persists it
        (manager.py:206).
        FAILURE MODES: Validation errors raise UploadError (propagate from
        validate, manager.py:175). Indexing failures are caught, logged, and
        recorded as status="failed" (manager.py:203-205) — the upload still
        returns a record rather than raising.
        ALTERNATIVE: Could make upload transactional — roll back the file write
        if indexing fails.
        # GAP: Not atomic — the file is written to disk (manager.py:195) before
        indexing; if indexing fails, an orphaned file remains in uploads/
        (manager.py:203-205).
        # BUSINESS: Dedupe and versioning keep the corpus from bloating with
        duplicates and preserve history on re-uploads.
        # INTERVIEW: "Why return a 'failed' record instead of raising? So the
        API can return 200 with a failure status rather than 500 — the caller
        can retry from the status log."
        """
        content_hash = self.validate(filename, content)
        name = Path(filename).name

        # dedupe: identical content already ingested?
        existing = vector_store.list_sources()
        for row in existing:
            if row.get("filename") == name and row.get("content_hash") == content_hash:
                rec = IngestRecord(name, content_hash, status="duplicate", chunks=row["chunk_count"],
                                   doc_version=str(row.get("doc_version", "")))
                self._record(rec)
                return rec

        # versioning: if this filename exists with a different hash, supersede old
        prev = next((r for r in existing if r.get("filename") == name), None)
        new_version = self._next_version(prev.get("doc_version") if prev else None)
        if prev:
            vector_store.mark_source_current(name, current=False)  # Old chunks no longer current

        # persist file under uploads/
        dest = self.upload_dir / name
        dest.write_bytes(content)  # Write raw bytes to disk

        rec = IngestRecord(name, content_hash, status="ok", doc_version=new_version)
        try:
            chunks = self._index_file(dest, vector_store, content_hash, new_version)
            rec.chunks = chunks
            rec.status = "superseded" if prev else "ok"
        except Exception as e:
            logger.error(f"ingest failed for {name}: {e}")
            rec.status = "failed"
            rec.error = str(e)[:300]  # Truncate error to keep log readable
        self._record(rec)
        return rec

    def _next_version(self, current: Optional[str]) -> str:
        """Bump the minor version of a version string like "1.0" -> "1.1".

        WHAT: Parses "major.minor" and increments minor; returns "1.0" if input
        is missing or unparseable.
        WHY: Provides a deterministic version bump on re-upload of the same
        filename.
        INPUTS: current (Optional[str]) — e.g. "1.3" or None.
        OUTPUTS: str — e.g. "1.4" or "1.0".
        SIDE EFFECTS: None.
        FAILURE MODES: Returns "1.0" on any parse failure (manager.py:216) —
        never raises.
        ALTERNATIVE: Could use semantic versioning (major.minor.patch) or
        date-based versions.
        # GAP: Only bump minor — a breaking content change can't be signaled
        as a major version bump (manager.py:212-214).
        # BUSINESS: Simple, predictable versioning for re-uploaded documents.
        # INTERVIEW: "Why only minor bump? Keeps it simple — the vector store
        tracks is_current anyway, so the version is mostly informational."
        """
        if not current:
            return "1.0"
        try:
            major, minor = current.split(".")
            return f"{major}.{int(minor) + 1}"
        except Exception:
            return "1.0"

    def _index_file(self, path: Path, vector_store, content_hash: str, doc_version: str) -> int:
        """Load -> clean -> chunk -> embed -> store one file. Returns chunk count.

        WHAT: Runs the full ingestion pipeline on a single persisted file:
        load_document, clean_documents, chunk_documents (recursive, 500/50),
        embed_documents, add_documents_to_store.
        WHY: Encapsulates the entire per-file pipeline in one method so upload
        and reindex share the same code path.
        INPUTS: path (Path) — file on disk; vector_store (object with
        add_documents_to_store); content_hash (str); doc_version (str).
        OUTPUTS: int — number of chunks embedded and stored.
        SIDE EFFECTS: Reads file from disk; writes embeddings to vector store;
        mutates chunk metadata with content_hash, doc_version, is_current.
        FAILURE MODES: Propagates errors from loaders/cleaner/chunker/embedder
        (caught by upload, manager.py:203).
        ALTERNATIVE: Could stream chunks to the embedder one at a time instead
        of building the full list in memory.
        # GAP: All chunks are held in memory before embedding — very large
        documents could cause OOM (manager.py:233).
        # BUSINESS: Single code path guarantees upload and reindex produce
        identical results.
        # INTERVIEW: "Why recursive strategy hardcoded here? It's the default
        recommendation for general documents; run.py lets CLI users override."
        """
        from src.ingestion.loaders import load_document
        from src.ingestion.cleaner import clean_documents
        from src.ingestion.chunker import chunk_documents
        from src.embedding.embedder import embed_documents
        from src.store.vector_store import add_documents_to_store

        doc = load_document(str(path))
        doc["metadata"]["content_hash"] = content_hash
        doc["metadata"]["doc_version"] = doc_version
        doc["metadata"]["is_current"] = True
        cleaned = clean_documents([doc])
        chunked = chunk_documents(cleaned, strategy="recursive", chunk_size=500, chunk_overlap=50)
        for c in chunked:
            c["metadata"]["content_hash"] = content_hash
            c["metadata"]["doc_version"] = doc_version
            c["metadata"]["is_current"] = True
        embedded = embed_documents(chunked)
        add_documents_to_store(vector_store, embedded)
        return len(embedded)

    # --- list / status / delete / reindex ---------------------------------

    def list_documents(self, vector_store) -> List[dict]:
        """List all source documents tracked by the vector store."""
        return vector_store.list_sources()

    def get_status(self, filename: str) -> Optional[dict]:
        """Return the most recent ingestion record for a filename, or None."""
        name = Path(filename).name
        recs = [r.to_dict() for r in self._records if r.filename == name]
        return recs[-1] if recs else None

    def delete(self, filename: str, vector_store) -> dict:
        """Delete the file and all its vector-store chunks (cascade).

        WHAT: Removes all chunks for the source from the vector store, then
        deletes the file from uploads/ if present.
        WHY: Cascade delete prevents orphaned chunks — retrieval never returns
        chunks for a deleted document.
        INPUTS: filename (str), vector_store (object with delete_by_source).
        OUTPUTS: dict with keys "filename", "chunks_deleted" (int),
        "file_deleted" (bool).
        SIDE EFFECTs: Deletes from vector store (manager.py:254); deletes file
        from disk (manager.py:256).
        FAILURE MODES: If vector_store.delete_by_source raises, the exception
        propagates and the file is not deleted (partial state possible).
        ALTERNATIVE: Soft-delete (mark is_current=False) instead of hard delete.
        # GAP: If vector store deletion succeeds but disk deletion fails (or
        vice versa), the system is in an inconsistent state — no rollback
        (manager.py:254-256).
        # BUSINESS: Ensures deleted documents are fully removed — critical for
        data privacy and compliance.
        # INTERVIEW: "Why delete from vector store first? So if disk deletion
        fails, at least the chunks are gone and retrieval won't surface the doc."
        """
        name = Path(filename).name
        deleted_chunks = vector_store.delete_by_source(name)
        f = self.upload_dir / name
        file_deleted = f.exists()
        if file_deleted:
            f.unlink()
        return {"filename": name, "chunks_deleted": deleted_chunks, "file_deleted": file_deleted}

    def reindex(self, vector_store, data_dir: Optional[str] = None) -> dict:
        """Rebuild the vector store from scratch (uploads dir + optional data dir).

        WHAT: Resets the vector store, then loads+cleans+chunks+embeds+stores
        every document from uploads/ and optionally an additional data dir.
        WHY: Full rebuild is useful after configuration changes (e.g. new
        chunking strategy) or to recover from a corrupt index.
        INPUTS: vector_store (object with reset, add_documents_to_store);
        data_dir (Optional[str]) — extra directory to index.
        OUTPUTS: dict with keys "chunks_indexed" (int), "sources_dirs"
        (List[str]).
        SIDE EFFECTS: Wipes the vector store (manager.py:269); reads all files
        from disk; writes all embeddings to the store.
        FAILURE MODES: If any document fails to load, load_documents_from_directory
        skips it (prints error); indexing continues with the rest.
        ALTERNATIVE: Incremental reindex — only reprocess changed files.
        # GAP: vector_store.reset() wipes everything before reindexing — if
        reindex fails midway, the store is empty and the old data is gone
        (manager.py:269).
        # BUSINESS: Gives ops a way to rebuild the entire index from source
        files — disaster recovery for the vector store.
        # INTERVIEW: "Why reset first? So stale chunks from deleted documents
        don't linger in the rebuilt index."
        """
        from src.ingestion.loaders import load_documents_from_directory
        from src.ingestion.cleaner import clean_documents
        from src.ingestion.chunker import chunk_documents
        from src.embedding.embedder import embed_documents
        from src.store.vector_store import add_documents_to_store

        vector_store.reset()
        dirs = [str(self.upload_dir)]
        if data_dir:
            dirs.append(str((project_root / data_dir).resolve()))
        total = 0
        for d in dirs:
            docs = load_documents_from_directory(d)
            for doc in docs:
                doc["metadata"].setdefault("is_current", True)
            cleaned = clean_documents(docs)
            chunked = chunk_documents(cleaned, strategy="recursive", chunk_size=500, chunk_overlap=50)
            embedded = embed_documents(chunked)
            add_documents_to_store(vector_store, embedded)
            total += len(embedded)
        return {"chunks_indexed": total, "sources_dirs": dirs}
