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
    """Invalid upload (bad type, too large, unsafe name)."""


@dataclass
class IngestRecord:
    filename: str
    content_hash: str
    status: str  # ok | duplicate | superseded | failed
    chunks: int = 0
    doc_version: str = ""
    error: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename, "content_hash": self.content_hash[:12],
            "status": self.status, "chunks": self.chunks,
            "doc_version": self.doc_version, "error": self.error, "ts": self.ts,
        }


class DocumentManager:
    """Handles upload/list/status/delete/reindex for the RAG corpus."""

    def __init__(self, upload_dir: Path = UPLOAD_DIR, status_file: Path = STATUS_FILE,
                 max_size: int = MAX_FILE_SIZE_BYTES):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = Path(status_file)
        self.max_size = max_size
        self._records: List[IngestRecord] = self._load_status()

    # --- status log -------------------------------------------------------

    def _load_status(self) -> List[IngestRecord]:
        if not self.status_file.exists():
            return []
        try:
            data = json.loads(self.status_file.read_text(encoding="utf-8"))
            return [IngestRecord(**{k: r.get(k) for k in IngestRecord.__dataclass_fields__})
                    for r in data.get("records", [])]
        except Exception:
            return []

    def _save_status(self) -> None:
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.write_text(
            json.dumps({"records": [r.to_dict() for r in self._records]}, indent=2),
            encoding="utf-8",
        )

    def _record(self, rec: IngestRecord) -> None:
        self._records.append(rec)
        if len(self._records) > 500:
            self._records = self._records[-500:]
        self._save_status()

    def status_log(self, limit: int = 50) -> List[dict]:
        return [r.to_dict() for r in self._records[-limit:]]

    # --- validation -------------------------------------------------------

    def validate(self, filename: str, content: bytes) -> str:
        """Validate filename + size; return content hash. Raises UploadError."""
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
        """Validate, dedupe, version, persist, and index one file. Returns the record."""
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
            vector_store.mark_source_current(name, current=False)

        # persist file under uploads/
        dest = self.upload_dir / name
        dest.write_bytes(content)

        rec = IngestRecord(name, content_hash, status="ok", doc_version=new_version)
        try:
            chunks = self._index_file(dest, vector_store, content_hash, new_version)
            rec.chunks = chunks
            rec.status = "superseded" if prev else "ok"
        except Exception as e:
            logger.error(f"ingest failed for {name}: {e}")
            rec.status = "failed"
            rec.error = str(e)[:300]
        self._record(rec)
        return rec

    def _next_version(self, current: Optional[str]) -> str:
        if not current:
            return "1.0"
        try:
            major, minor = current.split(".")
            return f"{major}.{int(minor) + 1}"
        except Exception:
            return "1.0"

    def _index_file(self, path: Path, vector_store, content_hash: str, doc_version: str) -> int:
        """Load -> clean -> chunk -> embed -> store one file. Returns chunk count."""
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
        return vector_store.list_sources()

    def get_status(self, filename: str) -> Optional[dict]:
        name = Path(filename).name
        recs = [r.to_dict() for r in self._records if r.filename == name]
        return recs[-1] if recs else None

    def delete(self, filename: str, vector_store) -> dict:
        """Delete the file and all its vector-store chunks (cascade)."""
        name = Path(filename).name
        deleted_chunks = vector_store.delete_by_source(name)
        f = self.upload_dir / name
        file_deleted = f.exists()
        if file_deleted:
            f.unlink()
        return {"filename": name, "chunks_deleted": deleted_chunks, "file_deleted": file_deleted}

    def reindex(self, vector_store, data_dir: Optional[str] = None) -> dict:
        """Rebuild the vector store from scratch (uploads dir + optional data dir)."""
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
