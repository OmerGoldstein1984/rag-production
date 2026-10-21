"""
Text cleaning utilities for document preprocessing.
"""

import re
import html
from typing import List, Dict, Any


class TextCleaner:
    """Cleans and normalizes text content."""

    def __init__(self):
        # Patterns for cleaning
        self.whitespace_pattern = re.compile(r'\s+')
        self.multiple_newlines_pattern = re.compile(r'\n{3,}')
        self.multiple_dots_pattern = re.compile(r'\.{2,}')
        # HTML tag pattern
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        # Table separator pattern (markdown tables)
        self.table_separator_pattern = re.compile(r'^\s*\|[\s\-|]+\|\s*$', re.MULTILINE)
        # Multiple consecutive table separators
        self.multiple_table_seps_pattern = re.compile(r'(\s*\|[\s\-|]+\|\s*\n){2,}', re.MULTILINE)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # HTML escape any HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = self.html_tag_pattern.sub('', text)

        # Handle tables sensibly - convert markdown tables to readable format
        # For now, we'll keep table content but normalize spacing
        lines = text.split('\n')
        cleaned_lines = []

        in_table = False
        for line in lines:
            # Check if this line looks like a table row
            if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
                in_table = True
                # Clean up table row spacing
                cleaned_line = re.sub(r'\s+\|\s+', ' | ', line.strip())
                cleaned_lines.append(cleaned_line)
            elif in_table and ('|' not in line or not line.strip()):
                # End of table
                in_table = False
                cleaned_lines.append('')  # Add blank line after table
                if line.strip():  # If line has content, add it
                    cleaned_lines.append(line.strip())
            elif in_table:
                # Still in table but line doesn't look like table row
                in_table = False
                cleaned_lines.append(line.strip())
            else:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Replace multiple whitespace with single space (but preserve newlines for paragraph structure)
        text = re.sub(r'[ \t]+', ' ', text)  # Only spaces and tabs, not newlines

        # Replace multiple newlines with double newline (paragraph separation)
        text = self.multiple_newlines_pattern.sub('\n\n', text)

        # Replace multiple dots
        text = self.multiple_dots_pattern.sub('...', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a document's content."""
        cleaned_content = self.clean_text(document["content"])

        return {
            "content": cleaned_content,
            "metadata": {
                **document["metadata"],
                "cleaned": True,
                "original_size": document["metadata"].get("size", 0),
                "cleaned_size": len(cleaned_content),
                "cleaning_ratio": len(cleaned_content) / max(len(document["content"]), 1)
            }
        }

    def clean_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a list of documents."""
        cleaner = TextCleaner()
        return [cleaner.clean_document(doc) for doc in documents]


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a list of documents."""
    cleaner = TextCleaner()
    return [cleaner.clean_document(doc) for doc in documents]