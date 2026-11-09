# =============================================================================
# FILE HEADER — src/ingestion/loaders.py
# Business problem: Converts raw files (PDF, DOCX, TXT, MD) into a uniform
# {content, metadata} dict so downstream clean/chunk/embed/store steps can
# process them without caring about the original file format.
# Place in data flow: Step 1 of ingestion (G3 group). Called by
# manager._index_file (manager.py:161) and run.run_ingestion_pipeline
# (run.py:47) via load_documents_from_directory. Calls nothing external except
# stdlib plus optional pypdf/PyPDF2/python-docx.
# What breaks if removed: The ingestion pipeline has no way to read documents
# from disk; no content reaches the cleaner, chunker, embedder, or vector
# store. The RAG system cannot ingest any new documents.
# =============================================================================
"""
Document loaders for various file formats.
Supports PDF, DOCX, TXT, and MD formats.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import datetime

class DocumentLoader:
    """Base class for document loaders.

    WHAT: Abstract base defining the loader interface every concrete loader
    (TextLoader, MarkdownLoader, PDFLoader, DocxLoader) must implement.
    WHY: Lets the rest of the pipeline treat all file formats uniformly through
    the same ``load(file_path) -> {content, metadata}`` contract, so adding a
    new format only requires a new subclass, not changes to callers.
    INPUTS: None (constructor takes no arguments).
    OUTPUTS: Subclasses must implement ``load`` returning
    ``{"content": str, "metadata": dict}``.
    SIDE EFFECTS: None at the base class level.
    FAILURE MODES: Calling ``load`` directly on the base class raises
    ``NotImplementedError`` (loaders.py:30).
    ALTERNATIVE: A factory function (get_loader_for_file, loaders.py:253) picks
    the right subclass by extension instead of callers instantiating loaders directly.
    # GAP: No async loading — large PDFs/DOCX block the caller thread; production
    systems would use a thread pool or async file reads so one slow file doesn't
    stall the whole ingestion (G3 Ingestion, loaders.py:25-30).
    # BUSINESS: Protects the pipeline from format-specific coupling — callers
    never need to know whether a doc is PDF or Markdown.
    # INTERVIEW: "Why a base class instead of one big function? So we can add
    new file types without touching existing code — open/closed principle."
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a document and return its content and metadata.

        WHAT: Abstract entry point — every subclass reads the file at
        ``file_path`` and returns content plus metadata.
        WHY: Defines the contract all loaders honor so the pipeline is
        format-agnostic.
        INPUTS: file_path (str) — e.g. "/uploads/5g_roaming_policy.pdf".
        OUTPUTS: Dict[str, Any] with keys "content" (str) and "metadata" (dict),
        e.g. {"content": "...text...", "metadata": {"source": "...",
        "title": "5g Roaming Policy", "extension": ".pdf", ...}}.
        SIDE EFFECTS: Reads the file from disk (I/O).
        FAILURE MODES: Base implementation raises NotImplementedError
        (loaders.py:30); concrete subclasses may raise FileNotFoundError,
        PermissionError, or format-specific parse errors.
        ALTERNATIVE: Callers could use get_loader_for_file (loaders.py:253)
        to pick a loader by extension rather than instantiating a subclass directly.
        """
        raise NotImplementedError

    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata common to all file types.

        WHAT: Builds a metadata dict (source, title, filename, extension, size,
        last_modified, department, access_level, doc_version, is_current) shared
        by every loader.
        WHY: Centralizes common metadata so each loader doesn't reimplement the
        same fields; ensures downstream filtering (e.g. by is_current) works
        uniformly.
        INPUTS: file_path (str) — e.g. "/uploads/hr_policy.txt".
        OUTPUTS: Dict[str, Any] such as
        {"source": "/uploads/hr_policy.txt", "title": "Hr Policy",
        "filename": "hr_policy.txt", "extension": ".txt", "size": 4096,
        "last_modified": "2026-01-15T10:30:00", "department": "Human Resources",
        "access_level": "internal", "doc_version": "1.0", "is_current": True}.
        SIDE EFFECTs: Calls os.stat via Path.stat (loaders.py:35) — touches the
        filesystem; calls _infer_department and _extract_version.
        FAILURE MODES: Raises FileNotFoundError if the path doesn't exist
        (loaders.py:35); PermissionError if unreadable.
        ALTERNATIVE: Callers could pass metadata in explicitly instead of
        deriving it from the filesystem path.
        # GAP: No retry/backoff on stat failure; a transient filesystem hiccup
        fails the whole load (loaders.py:35).
        # BUSINESS: Ensures every document carries consistent, queryable metadata
        for access control and version filtering.
        # INTERVIEW: "Why store is_current=True by default? Because the version
        filter in manager.py:37 relies on it to surface only the latest version
        of a document."
        """
        path_obj = Path(file_path)
        stat = path_obj.stat()  # Touch filesystem to get size + mtime (may raise FileNotFoundError)

        # Try to extract title from filename or content
        title = path_obj.stem.replace('_', ' ').replace('-', ' ').title()

        return {
            "source": file_path,
            "title": title,
            "filename": path_obj.name,
            "extension": path_obj.suffix.lower(),
            "size": stat.st_size,
            "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "department": self._infer_department(file_path),
            "access_level": "internal",  # Default access level
            "doc_version": self._extract_version(file_path) or "1.0",
            "is_current": True,  # version filter relies on this
        }

    def _infer_department(self, file_path: str) -> str:
        """Infer department from file path or name.

        WHAT: Guesses an organizational department (Human Resources, IT,
        Product/Marketing, Finance, Leadership, General) by keyword-matching the
        lowercased file path.
        WHY: Lets the RAG system tag documents with a department for later
        access-control or filtering without manual tagging.
        INPUTS: file_path (str) — e.g. "hr/leave_policy.md".
        OUTPUTS: str — one of "Human Resources", "Information Technology",
        "Product/Marketing", "Finance", "Leadership", "General".
        SIDE EFFECTS: None (pure string matching).
        FAILURE MODES: None — always returns a string (defaults to "General"
        when no keyword matches, loaders.py:66).
        ALTERNATIVE: Read department from an explicit metadata file or a
        directory-based mapping instead of keyword guessing.
        # GAP: Keyword matching is brittle — a file named "travel_report.md"
        about IT travel gets tagged "Finance"; no confidence score or override.
        # BUSINESS: Provides automatic, zero-effort document categorization for
        non-technical uploaders.
        # INTERVIEW: "Why fall back to 'General'? So unclassified docs are still
        ingestible rather than rejected."
        """
        file_path_lower = file_path.lower()
        if any(dept in file_path_lower for dept in ['hr', 'human', 'employee', 'leave', 'policy']):
            return "Human Resources"
        elif any(dept in file_path_lower for dept in ['it', 'tech', 'security', 'software']):
            return "Information Technology"
        elif any(dept in file_path_lower for dept in ['product', 'marketing', 'sales', 'customer']):
            return "Product/Marketing"
        elif any(dept in file_path_lower for dept in ['travel', 'expense', 'finance', 'accounting']):
            return "Finance"
        elif any(dept in file_path_lower for dept in ['leadership', 'executive', 'management']):
            return "Leadership"
        else:
            return "General"

    def _extract_version(self, file_path: str) -> str:
        """Extract version information from file path or name.

        WHAT: Scans the filename stem for version-like patterns (v1.0,
        version_2.1, _2.3, etc.) and returns the first match.
        WHY: Lets documents carry a version derived from their filename, which
        the manager uses for version bumping on re-upload.
        INPUTS: file_path (str) — e.g. "roaming_policy_v2.1.pdf".
        OUTPUTS: str version like "2.1", or None if no pattern matches
        (loaders.py:91).
        SIDE EFFECTS: Imports re lazily inside the method (loaders.py:80).
        FAILURE MODES: Returns None gracefully when no pattern matches — never
        raises.
        ALTERNATIVE: Store version in document metadata at upload time instead
        of parsing it from the filename.
        # GAP: Regex-based filename parsing can misfire (e.g. "section_1.2.txt"
        yields "1.2" as a version); no validation that the extracted string is
        actually a semantic version.
        # BUSINESS: Enables automatic version detection so re-uploads can bump
        versions without user input.
        # INTERVIEW: "Why 'or 1.0' in _extract_basic_metadata? Because
        _extract_version may return None, and the pipeline needs a default."
        """
        import re
        filename = Path(file_path).stem
        # Look for version patterns like v1.0, version_2.1, etc.
        version_patterns = [
            r'v(\d+\.\d+)',
            r'version[_\s-]*(\d+\.\d+)',
            r'[\s_-](\d+\.\d+)[\s_-]',
            r'_(\d+\.\d+)$'
        ]

        for pattern in version_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)
        return None


class TextLoader(DocumentLoader):
    """Loader for plain text files.

    WHAT: Reads a .txt file with UTF-8 encoding and returns its content plus
    metadata (type, line_count, word_count).
    WHY: Provides the simplest loader path — plain text needs no parsing
    library, just a file read.
    INPUTS: file_path (str) — e.g. "/uploads/faq.txt".
    OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} where metadata
    includes basic fields plus {"type": "text", "line_count": int, "word_count": int}.
    Example: {"content": "Hello world", "metadata": {"type": "text",
    "line_count": 1, "word_count": 2, ...}}.
    SIDE EFFECTS: Opens and reads the file from disk (I/O).
    FAILURE MODES: Raises FileNotFoundError if missing, PermissionError if
    unreadable, UnicodeDecodeError if not valid UTF-8 (loaders.py:199).
    ALTERNATIVE: Could use mmap for very large files instead of reading all
    into memory at once.
    # GAP: No encoding detection — assumes UTF-8; a Latin-1 encoded file fails
    with UnicodeDecodeError (loaders.py:199).
    # BUSINESS: Handles the most common document format with zero dependencies.
    # INTERVIEW: "Why read with encoding='utf-8'? So we fail fast on bad
    encoding rather than silently mojibake the text."
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()  # Read entire file into memory (may fail on huge files)

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "text",
            "line_count": len(content.splitlines()),  # Count of newline-separated lines
            "word_count": len(content.split())  # Count of whitespace-separated words
        })

        return {
            "content": content,
            "metadata": metadata
        }


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files.

    WHAT: Reads a .md file, extracts the first heading as the document title,
    and returns content plus metadata (type, line_count, word_count).
    WHY: Markdown files often carry a title in their first heading; using it as
    the document title gives more meaningful metadata than the filename alone.
    INPUTS: file_path (str) — e.g. "/uploads/readme.md".
    OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} where metadata
    includes basic fields plus {"type": "markdown", "line_count": int,
    "word_count": int} and possibly an overridden "title" from the first # heading.
    SIDE EFFECTS: Opens and reads the file from disk (I/O).
    FAILURE MODES: Raises FileNotFoundError if missing, PermissionError if
    unreadable, UnicodeDecodeError if not valid UTF-8 (loaders.py:220).
    ALTERNATIVE: Could use a full Markdown parser (e.g. mistune) to extract
    structured headings instead of a simple line-starts-with-# check.
    # GAP: Only the first heading is used as title; subsequent headings are
    ignored, and no heading hierarchy is captured in metadata.
    # BUSINESS: Gives Markdown documents human-readable titles from their
    content rather than generic filenames.
    # INTERVIEW: "Why override the title from the first heading? Because a file
    named 'doc_final_v2.md' is a worse title than '# 5G Roaming Policy'."
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()  # Read entire file into memory

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "markdown",
            "line_count": len(content.splitlines()),
            "word_count": len(content.split())
        })

        # Try to extract first heading as title
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):  # A Markdown heading line
                # Found a heading, clean it up
                title = line.lstrip('#').strip()
                if title:
                    metadata["title"] = title  # Override filename-derived title
                break  # Only the first heading counts

        return {
            "content": content,
            "metadata": metadata
        }


class PDFLoader(DocumentLoader):
    """Loader for PDF files. Tries pypdf, then PyPDF2; never fabricates text.

    WHAT: Extracts text from a PDF file, trying pypdf first then PyPDF2 as a
    fallback. Returns content plus metadata (type, pages, extraction_errors,
    extracted_ok).
    WHY: PDFs need a parsing library; trying two libraries maximizes the chance
    of successful extraction in environments where only one is installed. Never
    fabricates text — if extraction fails, content is empty and the error is
    recorded.
    INPUTS: file_path (str) — e.g. "/uploads/spec.pdf".
    OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} where metadata
    includes basic fields plus {"type": "pdf", "pages": int,
    "extraction_errors": List[str], "extracted_ok": bool}.
    SIDE EFFECTS: Opens and reads the file from disk (I/O); imports pypdf or
    PyPDF2 at runtime.
    FAILURE MODES: If neither library is installed, records "no PDF library
    installed" error and returns empty content (loaders.py:261). If a page fails
    to extract, records that page's error but continues (loaders.py:272).
    ALTERNATIVE: Could use pdftotext (poppler) as a third fallback via
    subprocess.
    # GAP: No OCR for scanned/image-only PDFs — those return empty content with
    an error rather than being processed (loaders.py:278-280).
    # BUSINESS: Honest about extraction failure rather than producing garbage
    text that would mislead downstream retrieval.
    # INTERVIEW: "Why not fabricate content? Because a wrong answer with high
    confidence is worse than no answer — retrieval would return bogus chunks."
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        content = ""
        pages = 0
        errors: List[str] = []
        reader_cls = None
        for mod_name in ("pypdf", "PyPDF2"):
            try:
                mod = __import__(mod_name)  # Try importing the library
                reader_cls = mod.PdfReader
                break  # Stop at the first library that imports successfully
            except ImportError:
                continue  # Try the next library
        if reader_cls is None:
            errors.append("no PDF library installed (pip install pypdf)")
        else:
            try:
                with open(file_path, "rb") as fh:  # Open PDF in binary mode
                    reader = reader_cls(fh)
                    pages = len(reader.pages)  # Total page count
                    parts = []
                    for i, page in enumerate(reader.pages):
                        try:
                            parts.append(page.extract_text() or "")  # Extract text; None → ""
                        except Exception as e:
                            errors.append(f"page {i + 1}: {e}")  # Record per-page failure, keep going
                    content = "\n\n".join(parts).strip()  # Join all pages with double newline
            except Exception as e:
                errors.append(f"pdf parse error: {e}")  # Whole-file parse failure

        if not content:
            # Do NOT fabricate content — record that extraction failed.
            content = ""
            errors.append("no text extracted")

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "pdf",
            "pages": pages,
            "extraction_errors": errors,
            "extracted_ok": bool(content),  # True only if some text was extracted
        })

        return {
            "content": content,
            "metadata": metadata
        }


class DocxLoader(DocumentLoader):
    """Loader for DOCX files. Uses python-docx if available, else stdlib zipfile+XML.

    DOCX is a zip containing word/document.xml — the stdlib fallback needs no deps.
    Never fabricates text.

    WHAT: Extracts text (paragraphs and tables) from a .docx file, using
    python-docx if installed, otherwise a stdlib zipfile+XML fallback. Returns
    content plus metadata (type, extraction_errors, extracted_ok).
    WHY: DOCX is a zipped XML format; the stdlib fallback guarantees the loader
    works even without third-party dependencies. Never fabricates text.
    INPUTS: file_path (str) — e.g. "/uploads/contract.docx".
    OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} where metadata
    includes basic fields plus {"type": "docx", "extraction_errors": List[str],
    "extracted_ok": bool}.
    SIDE EFFECTS: Opens and reads the file from disk (I/O); imports docx or
    uses zipfile+ElementTree.
    FAILURE MODES: If docx import fails and the stdlib fallback also fails,
    records the error and returns empty content (loaders.py:320-321).
    ALTERNATIVE: Could use antiword or pandoc as external fallbacks.
    # GAP: The stdlib fallback only reads paragraph text from word/document.xml;
    headers, footers, and text boxes are skipped.
    # BUSINESS: Guarantees DOCX ingestion works even in locked-down environments
    without pip access.
    # INTERVIEW: "Why a stdlib fallback? So ingestion doesn't fail just because
    python-docx isn't installed — operational resilience."
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        content = ""
        errors: List[str] = []
        try:
            import docx  # type: ignore  # Try the full-featured library first

            document = docx.Document(file_path)
            parts = [p.text for p in document.paragraphs if p.text.strip()]  # Non-empty paragraphs
            for table in document.tables:  # Also extract table content
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))  # Pipe-separated table row
            content = "\n\n".join(parts).strip()
        except ImportError:
            try:
                content = self._extract_docx_stdlib(file_path)  # Fallback: stdlib only
            except Exception as e:
                errors.append(f"docx stdlib parse error: {e}")
        except Exception as e:
            errors.append(f"docx parse error: {e}")  # python-docx raised something else

        if not content:
            content = ""
            errors.append("no text extracted")

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "docx",
            "extraction_errors": errors,
            "extracted_ok": bool(content),
        })

        return {
            "content": content,
            "metadata": metadata
        }

    @staticmethod
    def _extract_docx_stdlib(file_path: str) -> str:
        """Extract paragraph text from word/document.xml using only the stdlib.

        WHAT: Unzips the .docx, parses word/document.xml with ElementTree, and
        concatenates all paragraph text.
        WHY: Provides a zero-dependency DOCX reader so ingestion works without
        installing python-docx.
        INPUTS: file_path (str) — e.g. "/uploads/contract.docx".
        OUTPUTS: str — all paragraph text joined by double newlines, or "" if
        nothing found.
        SIDE EFFECTS: Opens the file as a zip archive (I/O).
        FAILURE MODES: Raises zipfile.BadZipFile if the file isn't a valid zip;
        raises ET.ParseError if the XML is malformed.
        ALTERNATIVE: Could also extract text from headers/footers XML parts.
        # GAP: Only reads word/document.xml — content in headers, footers,
        textboxes, and comments is silently dropped.
        # BUSINESS: Removes the python-docx dependency requirement for DOCX support.
        # INTERVIEW: "Why is DOCX a zip? Because the Office Open XML format
        packages multiple XML parts inside a ZIP container."
        """
        import zipfile
        import xml.etree.ElementTree as ET

        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"  # WordprocessingML namespace
        with zipfile.ZipFile(file_path) as z:  # Open DOCX as a zip archive
            xml = z.read("word/document.xml")  # The main document part
        root = ET.fromstring(xml)
        parts = []
        for p in root.iter(W + "p"):  # Iterate all paragraph elements
            texts = [t.text or "" for t in p.iter(W + "t")]  # All text runs in the paragraph
            line = "".join(texts).strip()
            if line:
                parts.append(line)
        return "\n\n".join(parts).strip()


def get_loader_for_file(file_path: str) -> DocumentLoader:
    """Return appropriate loader based on file extension.

    WHAT: Factory function that instantiates the right loader subclass for a
    given file's extension (.txt → TextLoader, .md → MarkdownLoader, .pdf →
    PDFLoader, .docx → DocxLoader), defaulting to TextLoader for unknown types.
    WHY: Centralizes extension→loader mapping so callers don't need to know
    which loader handles which format.
    INPUTS: file_path (str) — e.g. "/uploads/spec.pdf".
    OUTPUTS: DocumentLoader — an instance of the matching loader subclass.
    SIDE EFFECTS: None (just object construction; no I/O yet).
    FAILURE MODES: None — unknown extensions fall back to TextLoader
    (loaders.py:478), which will likely fail at load time if the file isn't
    actually text.
    ALTERNATIVE: Could use mimetypes (already imported but unused) to sniff
    the actual file type instead of relying on extension.
    # GAP: mimetypes is imported (loaders.py:20) but never used — the loader
    trusts the extension, so a .txt file that's actually a PDF will be
    mis-routed.
    # BUSINESS: Keeps caller code simple — one function call instead of a
    conditional chain everywhere.
    # INTERVIEW: "Why default to TextLoader for unknown extensions? So we
    attempt ingestion rather than rejecting unfamiliar file types outright."
    """
    extension = Path(file_path).suffix.lower()

    loaders = {
        '.txt': TextLoader(),
        '.md': MarkdownLoader(),
        '.pdf': PDFLoader(),
        '.docx': DocxLoader()
    }

    return loaders.get(extension, TextLoader())  # Default to text loader


def load_document(file_path: str) -> Dict[str, Any]:
    """Load a document using the appropriate loader.

    WHAT: Convenience function that combines get_loader_for_file and
    loader.load into a single call.
    WHY: Provides a simple entry point for callers that just want a document's
    content and metadata without dealing with loader selection.
    INPUTS: file_path (str) — e.g. "/uploads/policy.txt".
    OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} from the
    selected loader.
    SIDE EFFECTS: Opens and reads the file from disk (I/O) via the loader.
    FAILURE MODES: Propagates whatever the loader raises — FileNotFoundError,
    PermissionError, UnicodeDecodeError, or format-specific parse errors.
    ALTERNATIVE: Callers could call get_loader_for_file + loader.load
    separately if they need the loader object for reuse.
    # GAP: No caching — loading the same file twice reads it from disk twice.
    # BUSINESS: Simple, testable entry point for the most common load operation.
    # INTERVIEW: "Why a separate function instead of inlining loader selection?
    So manager.py and run.py both use the same single entry point."
    """
    loader = get_loader_for_file(file_path)
    return loader.load(file_path)


def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    """Load all supported documents from a directory.

    WHAT: Recursively walks ``directory`` and loads every file with a supported
    extension (.txt, .md, .pdf, .docx), returning a list of document dicts.
    Files that fail to load are skipped with a printed error.
    WHY: Enables bulk ingestion of a whole folder of documents in one call.
    INPUTS: directory (str) — e.g. "./data/raw".
    OUTPUTS: List[Dict[str, Any]] — one document dict per successfully loaded
    file. Failed files are omitted (not raised).
    SIDE EFFECTS: Reads every matching file from disk (I/O); prints errors to
    stdout for failed files (loaders.py:498).
    FAILURE MODES: Never raises for individual file failures — each is caught
    and logged via print (loaders.py:497-498). If the directory itself doesn't
    exist, Path.rglob yields nothing and an empty list is returned.
    ALTERNATIVE: Could use os.walk or glob.glob instead of Path.rglob; could
    collect errors into a list instead of printing.
    # GAP: Errors are printed (stdout) rather than logged or collected — a
    caller can't programmatically distinguish "no files found" from "all files
    failed to load" (loaders.py:498).
    # BUSINESS: Bulk load is resilient — one bad file doesn't abort the whole
    directory.
    # INTERVIEW: "Why skip failed files instead of failing fast? Because a
    directory of 1000 docs shouldn't be rejected entirely due to one corrupt PDF."
    """
    documents = []
    supported_extensions = {'.txt', '.md', '.pdf', '.docx'}

    for file_path in Path(directory).rglob('*'):  # Recursively find all files
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(str(file_path))
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")  # Skip bad files, keep going

    return documents