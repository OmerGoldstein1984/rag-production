# =============================================================================
# FILE HEADER — src/ingestion/cleaner.py
# Business problem: Removes noise (HTML tags, entities, excess whitespace,
# markdown table separators) from raw document text so downstream chunking and
# embedding operate on clean, normalized content.
# Place in data flow: Step 2 of ingestion (G3 group). Called by
# manager._index_file (manager.py:171) and run.run_ingestion_pipeline
# (run.py:56). Calls nothing external except stdlib (re, html).
# What breaks if removed: Raw HTML tags and entities reach the embedder,
# polluting vector embeddings with markup garbage and hurting retrieval quality.
# =============================================================================
"""
Text cleaning utilities for document preprocessing.
"""

import re
import html
from typing import List, Dict, Any


class TextCleaner:
    """Cleans and normalizes text content.

    WHAT: Holds compiled regex patterns and provides methods to clean a single
    string (clean_text), a single document dict (clean_document), or a list of
    documents (clean_documents).
    WHY: Centralizes all text normalization in one class so cleaning behavior
    is consistent across loaders and callers; patterns are compiled once in
    __init__ for efficiency.
    INPUTS: None (constructor takes no arguments).
    OUTPUTS: None directly — methods return cleaned strings or document dicts.
    SIDE EFFECTS: None at construction time.
    FAILURE MODES: None at the class level; individual methods may raise on bad
    input (e.g. clean_document expects a dict with a "content" key).
    ALTERNATIVE: Could use a dedicated library like bleach or lxml for HTML
    stripping instead of regex.
    # GAP: The class is stateless except for compiled patterns; no configuration
    knobs (e.g. which patterns to apply) — all-or-nothing cleaning.
    # BUSINESS: Ensures every document gets the same normalization, so
    embeddings are computed on comparable text.
    # INTERVIEW: "Why compile regex patterns in __init__? So they're compiled
    once, not on every clean_text call — matters when cleaning thousands of
    docs."
    """

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
        """Clean and normalize text.

        WHAT: Strips HTML entities, removes HTML tags, normalizes markdown
        tables, collapses whitespace and repeated dots/newlines, and trims.
        WHY: Raw document text from loaders may contain HTML markup, entities,
        and inconsistent spacing; cleaning makes it embedding-friendly.
        INPUTS: text (str) — e.g. "Hello   <b>world</b>  &amp;   ..."
        OUTPUTS: str — cleaned text, e.g. "Hello world & ..."
        SIDE EFFECTS: None (pure transformation; returns a new string).
        FAILURE MODES: Returns "" for empty/None input (cleaner.py:39) — never
        raises for missing content.
        ALTERNATIVE: Could use BeautifulSoup for more robust HTML parsing.
        # GAP: Whitespace collapse removes ALL whitespace runs including
        intentional indentation in code blocks or poetry (cleaner.py:69).
        # BUSINESS: Clean text produces better embeddings — markup noise would
        distort semantic similarity.
        # INTERVIEW: "Why unescape HTML entities first? So &amp; becomes &
        before we strip tags, otherwise we'd leave dangling ampersands."
        """
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
        """Clean a document's content.

        WHAT: Calls clean_text on document["content"] and returns a new dict
        with the cleaned content plus metadata augmented with cleaning stats
        (cleaned, original_size, cleaned_size, cleaning_ratio).
        WHY: Lets the pipeline track how much text was removed during cleaning
        and flag documents as cleaned for downstream steps.
        INPUTS: document (Dict[str, Any]) — e.g.
        {"content": "...raw...", "metadata": {"source": "...", "size": 1000, ...}}.
        OUTPUTS: Dict[str, Any] = {"content": str, "metadata": dict} where
        metadata is the original plus {"cleaned": True, "original_size": int,
        "cleaned_size": int, "cleaning_ratio": float}.
        SIDE EFFECTS: None (returns a new dict; doesn't mutate input).
        FAILURE MODES: Raises KeyError if document lacks a "content" key
        (cleaner.py:91); TypeError if document["content"] isn't a string.
        ALTERNATIVE: Could mutate the input dict in place instead of creating a
        new one.
        # GAP: cleaning_ratio divides by max(len(content), 1) so empty original
        yields ratio 0.0 rather than an error — but that masks truly empty
        documents (cleaner.py:101).
        # BUSINESS: Tracks cleaning effectiveness so ops can spot documents
        whose content was mostly stripped away.
        # INTERVIEW: "Why not mutate in place? So the original raw content is
        still available if debugging is needed later."
        """
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
        """Clean a list of documents.

        WHAT: Creates a new TextCleaner instance and cleans every document in
        the list via clean_document.
        WHY: Convenience batch entry point; note it creates a fresh instance
        rather than using self (so patterns are recompiled).
        INPUTS: documents (List[Dict[str, Any]]) — list of document dicts.
        OUTPUTS: List[Dict[str, Any]] — cleaned document dicts, same order.
        SIDE EFFECTS: None (returns a new list).
        FAILURE MODES: Propagates clean_document failures (KeyError, TypeError)
        for any document in the list — one bad document aborts the whole batch.
        ALTERNATIVE: Could use self instead of creating a new instance to avoid
        recompiling patterns.
        # GAP: Creates a new TextCleaner() instead of using self — recompiles
        all regex patterns for every batch call, wasteful at scale (cleaner.py:106).
        # BUSINESS: Batch cleaning lets the pipeline process all loaded docs in
        one call.
        # INTERVIEW: "Why create a new instance instead of using self? Looks
        like a bug — self already has compiled patterns; the new instance
        recompiles them for no benefit."
        """
        cleaner = TextCleaner()
        return [cleaner.clean_document(doc) for doc in documents]


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a list of documents.

    WHAT: Module-level convenience function that creates a TextCleaner and
    cleans every document in the list.
    WHY: Lets callers clean documents without instantiating TextCleaner
    directly; used by manager._index_file (manager.py:171).
    INPUTS: documents (List[Dict[str, Any]]) — list of document dicts.
    OUTPUTS: List[Dict[str, Any]] — cleaned document dicts.
    SIDE EFFECTS: None (returns a new list).
    FAILURE MODES: Propagates TextCleaner.clean_document failures — one bad
    document aborts the whole batch.
    ALTERNATIVE: Callers could instantiate TextCleaner and call
    clean_documents on it for the same result.
    # GAP: Module-level function duplicates the method of the same name on
    TextCleaner (cleaner.py:106) — two code paths for the same behavior, easy
    to diverge during maintenance.
    # BUSINESS: Simple functional entry point preferred by some callers over OO style.
    # INTERVIEW: "Why have both a module function and a method with the same
    name? Historical — the module function came first, the class was added
    later; neither was removed."
    """
    cleaner = TextCleaner()
    return [cleaner.clean_document(doc) for doc in documents]