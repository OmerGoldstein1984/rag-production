"""
Tests for document lifecycle: upload validation, dedupe, versioning/supersede,
delete cascade, list/status, and the API endpoints.
"""

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.manager import DocumentManager, UploadError
from src.store.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma"))


@pytest.fixture
def manager(tmp_path):
    return DocumentManager(upload_dir=tmp_path / "uploads", status_file=tmp_path / "status.json")


MD_V1 = b"# Policy\n\nRemote work allowed 3 days per week with manager approval."
MD_V2 = b"# Policy\n\nRemote work allowed 5 days per week with manager approval."


# --- validation -----------------------------------------------------------

def test_validate_rejects_bad_extension(manager):
    with pytest.raises(UploadError):
        manager.validate("evil.exe", b"MZ")


def test_validate_rejects_oversize(manager):
    manager.max_size = 10
    with pytest.raises(UploadError):
        manager.validate("big.md", b"x" * 11)


def test_validate_rejects_path_in_filename(manager):
    with pytest.raises(UploadError):
        manager.validate("../evil.md", b"hi")


def test_validate_rejects_empty(manager):
    with pytest.raises(UploadError):
        manager.validate("a.md", b"")


# --- upload / dedupe / versioning -----------------------------------------

def test_upload_indexes_and_lists(manager, store):
    rec = manager.upload("policy.md", MD_V1, store)
    assert rec.status == "ok" and rec.chunks >= 1
    docs = manager.list_documents(store)
    assert any(d["filename"] == "policy.md" and d["is_current"] for d in docs)
    # file persisted under uploads/
    assert (manager.upload_dir / "policy.md").exists()


def test_upload_dedupe_same_content(manager, store):
    manager.upload("policy.md", MD_V1, store)
    rec2 = manager.upload("policy.md", MD_V1, store)
    assert rec2.status == "duplicate"
    # still only one current source
    docs = manager.list_documents(store)
    assert sum(1 for d in docs if d["filename"] == "policy.md") == 1


def test_upload_new_version_supersedes_old(manager, store):
    manager.upload("policy.md", MD_V1, store)
    rec2 = manager.upload("policy.md", MD_V2, store)  # different content, same name
    assert rec2.status == "superseded"
    assert rec2.doc_version != "1.0"
    # verify version state at the chunk level (list_sources collapses per file)
    if store.collection is not None:
        rows_meta = store.collection.get(include=["metadatas"])["metadatas"]
    else:
        rows_meta = [d.get("metadata", {}) for d in store._fallback_documents]
    versions = {(m.get("doc_version"), m.get("is_current")) for m in rows_meta if m.get("filename") == "policy.md"}
    assert ("1.0", False) in versions and (rec2.doc_version, True) in versions


def test_failed_ingest_is_recorded(manager, store, monkeypatch):
    # force _index_file to fail
    monkeypatch.setattr(manager, "_index_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rec = manager.upload("bad.md", b"# Bad\n\ntext", store)
    assert rec.status == "failed" and "boom" in rec.error
    st = manager.get_status("bad.md")
    assert st and st["status"] == "failed"


# --- delete cascade --------------------------------------------------------

def test_delete_removes_chunks_and_file(manager, store):
    manager.upload("policy.md", MD_V1, store)
    before = len(manager.list_documents(store))
    res = manager.delete("policy.md", store)
    assert res["chunks_deleted"] >= 1 and res["file_deleted"] is True
    assert not (manager.upload_dir / "policy.md").exists()
    assert all(d["filename"] != "policy.md" for d in manager.list_documents(store))


# --- API endpoints ---------------------------------------------------------

def test_api_upload_list_delete(tmp_path, monkeypatch):
    os.environ["LLM_PROVIDER"] = "mock"
    monkeypatch.setattr("src.api.main.doc_manager",
                        DocumentManager(upload_dir=tmp_path / "uploads",
                                        status_file=tmp_path / "status.json"))
    from fastapi.testclient import TestClient
    from src.api.main import app, initialize_rag_pipeline

    # point pipeline at a temp chroma
    initialize_rag_pipeline("data/raw")
    c = TestClient(app)
    H = {"X-Access-Level": "3"}

    r = c.post("/documents/upload", files={"file": ("note.md", b"# Note\n\nhello world", "text/markdown")}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "duplicate", "superseded") and body["chunks"] >= 1

    r = c.get("/documents", headers=H)
    assert r.status_code == 200
    assert any(d["filename"] == "note.md" for d in r.json()["documents"])

    r = c.get("/documents/status", headers=H)
    assert r.status_code == 200 and any(rec["filename"] == "note.md" for rec in r.json()["records"])

    r = c.delete("/documents/note.md", headers=H)
    assert r.status_code == 200
    assert r.json()["chunks_deleted"] >= 1

    # non-admin cannot upload
    r = c.post("/documents/upload", files={"file": ("x.md", b"x", "text/markdown")}, headers={"X-Access-Level": "1"})
    assert r.status_code == 403

    # bad extension rejected
    r = c.post("/documents/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=H)
    assert r.status_code == 400
