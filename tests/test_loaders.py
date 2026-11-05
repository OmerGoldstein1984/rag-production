"""
Tests for document loaders — including REAL PDF/DOCX extraction (no fabricated text).
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loaders import load_document, load_documents_from_directory


def test_load_markdown_extracts_heading(tmp_path):
    f = tmp_path / "policy.md"
    f.write_text("# Remote Work Policy\n\nBody text here.", encoding="utf-8")
    doc = load_document(str(f))
    assert doc["metadata"]["type"] == "markdown"
    assert "Remote Work Policy" in doc["content"]
    assert doc["metadata"]["title"] == "Remote Work Policy"


def test_load_pdf_real_extraction():
    # Use a real PDF shipped in the repo
    pdf = project_root / "data" / "raw" / "benefits_summary.pdf"
    if not pdf.exists():
        pytest.skip("no sample pdf in repo")
    doc = load_document(str(pdf))
    assert doc["metadata"]["type"] == "pdf"
    assert doc["metadata"]["extracted_ok"] is True
    assert len(doc["content"]) > 50
    # never fabricated placeholder text
    assert "[PDF content from" not in doc["content"]


def test_load_docx_real_extraction(tmp_path):
    # Build a minimal valid .docx (zip with word/document.xml) using stdlib only
    import zipfile

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Quarterly results are strong.</w:t></w:r></w:p></w:body></w:document>'
    )
    f = tmp_path / "note.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml", document_xml)
    doc = load_document(str(f))
    assert doc["metadata"]["type"] == "docx"
    assert doc["metadata"]["extracted_ok"] is True
    assert "Quarterly results" in doc["content"]
    assert "[DOCX content from" not in doc["content"]


def test_load_corrupt_pdf_does_not_fabricate(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a real pdf at all")
    doc = load_document(str(f))
    # must not invent content; should record failure
    assert doc["metadata"]["extracted_ok"] is False
    assert "[PDF content from" not in doc["content"]


def test_load_directory_real_repo_docs():
    raw = project_root / "data" / "raw"
    docs = load_documents_from_directory(str(raw))
    assert len(docs) > 0
    pdfs = [d for d in docs if d["metadata"]["type"] == "pdf"]
    assert len(pdfs) > 0
    for d in pdfs:
        # the critical safety property: never fabricate placeholder text
        assert "[PDF content from" not in d["content"]
        # extraction status is reported honestly
        assert d["metadata"]["extracted_ok"] == bool(d["content"].strip())
    # synthetic real-text PDF extracted fine (covered above); repo PDFs may be
    # image-only, in which case extracted_ok=False is the honest result.
