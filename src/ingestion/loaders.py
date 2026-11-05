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
    """Base class for document loaders."""

    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a document and return its content and metadata."""
        raise NotImplementedError

    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata common to all file types."""
        path_obj = Path(file_path)
        stat = path_obj.stat()

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
        """Infer department from file path or name."""
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
        """Extract version information from file path or name."""
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
    """Loader for plain text files."""

    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "text",
            "line_count": len(content.splitlines()),
            "word_count": len(content.split())
        })

        return {
            "content": content,
            "metadata": metadata
        }


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "markdown",
            "line_count": len(content.splitlines()),
            "word_count": len(content.split())
        })

        # Try to extract first heading as title
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                # Found a heading, clean it up
                title = line.lstrip('#').strip()
                if title:
                    metadata["title"] = title
                break

        return {
            "content": content,
            "metadata": metadata
        }


class PDFLoader(DocumentLoader):
    """Loader for PDF files. Tries pypdf, then PyPDF2; never fabricates text."""

    def load(self, file_path: str) -> Dict[str, Any]:
        content = ""
        pages = 0
        errors: List[str] = []
        reader_cls = None
        for mod_name in ("pypdf", "PyPDF2"):
            try:
                mod = __import__(mod_name)
                reader_cls = mod.PdfReader
                break
            except ImportError:
                continue
        if reader_cls is None:
            errors.append("no PDF library installed (pip install pypdf)")
        else:
            try:
                with open(file_path, "rb") as fh:
                    reader = reader_cls(fh)
                    pages = len(reader.pages)
                    parts = []
                    for i, page in enumerate(reader.pages):
                        try:
                            parts.append(page.extract_text() or "")
                        except Exception as e:
                            errors.append(f"page {i + 1}: {e}")
                    content = "\n\n".join(parts).strip()
            except Exception as e:
                errors.append(f"pdf parse error: {e}")

        if not content:
            # Do NOT fabricate content — record that extraction failed.
            content = ""
            errors.append("no text extracted")

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "pdf",
            "pages": pages,
            "extraction_errors": errors,
            "extracted_ok": bool(content),
        })

        return {
            "content": content,
            "metadata": metadata
        }


class DocxLoader(DocumentLoader):
    """Loader for DOCX files. Uses python-docx if available, else stdlib zipfile+XML.

    DOCX is a zip containing word/document.xml — the stdlib fallback needs no deps.
    Never fabricates text.
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        content = ""
        errors: List[str] = []
        try:
            import docx  # type: ignore

            document = docx.Document(file_path)
            parts = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            content = "\n\n".join(parts).strip()
        except ImportError:
            try:
                content = self._extract_docx_stdlib(file_path)
            except Exception as e:
                errors.append(f"docx stdlib parse error: {e}")
        except Exception as e:
            errors.append(f"docx parse error: {e}")

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
        """Extract paragraph text from word/document.xml using only the stdlib."""
        import zipfile
        import xml.etree.ElementTree as ET

        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        with zipfile.ZipFile(file_path) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        parts = []
        for p in root.iter(W + "p"):
            texts = [t.text or "" for t in p.iter(W + "t")]
            line = "".join(texts).strip()
            if line:
                parts.append(line)
        return "\n\n".join(parts).strip()


def get_loader_for_file(file_path: str) -> DocumentLoader:
    """Return appropriate loader based on file extension."""
    extension = Path(file_path).suffix.lower()

    loaders = {
        '.txt': TextLoader(),
        '.md': MarkdownLoader(),
        '.pdf': PDFLoader(),
        '.docx': DocxLoader()
    }

    return loaders.get(extension, TextLoader())  # Default to text loader


def load_document(file_path: str) -> Dict[str, Any]:
    """Load a document using the appropriate loader."""
    loader = get_loader_for_file(file_path)
    return loader.load(file_path)


def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    """Load all supported documents from a directory."""
    documents = []
    supported_extensions = {'.txt', '.md', '.pdf', '.docx'}

    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(str(file_path))
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents