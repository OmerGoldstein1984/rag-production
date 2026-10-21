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
            "doc_version": self._extract_version(file_path) or "1.0"
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
    """Loader for PDF files."""

    def load(self, file_path: str) -> Dict[str, Any]:
        # Placeholder - in real implementation would use PyPDF2 or similar
        # For now, we'll simulate reading a PDF
        try:
            # Try to read as text first (if it's actually a text file with .pdf extension)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If it's a real PDF, we'd extract text here
            # For demo purposes, we'll create some sample content
            content = f"[PDF content from {file_path}]\n\nThis represents extracted text from a PDF document. In a real implementation, we would use PyPDF2 or pdfplumber to extract the actual text content from the PDF file."

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "pdf",
            "estimated_pages": max(1, len(content) // 500)  # Rough estimate
        })

        return {
            "content": content,
            "metadata": metadata
        }


class DocxLoader(DocumentLoader):
    """Loader for DOCX files."""

    def load(self, file_path: str) -> Dict[str, Any]:
        # Placeholder - in real implementation would use python-docx
        try:
            # Try to read as text first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If it's a real DOCX, we'd extract text here
            # For demo purposes, we'll create some sample content
            content = f"[DOCX content from {file_path}]\n\nThis represents extracted text from a Word document. In a real implementation, we would use python-docx to extract the actual text content from the DOCX file."

        metadata = self._extract_basic_metadata(file_path)
        metadata.update({
            "type": "docx"
        })

        return {
            "content": content,
            "metadata": metadata
        }


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