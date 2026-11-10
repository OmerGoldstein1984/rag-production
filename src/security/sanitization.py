"""
src.security.sanitization - HTML / XSS sanitization for RAG answers.

Defenses:
1. Strip dangerous HTML tags (<script>, <iframe>, <object>, <embed>, <link>, <style>)
   and their contents.
2. Strip on* event handlers (onclick, onerror, onload, etc.).
3. Strip javascript: and data: URLs from href/src attributes.
4. Escape remaining < and > characters that could form tags.

All functions handle None / empty input gracefully.
"""

from __future__ import annotations

import re
from typing import Optional


# Dangerous tags: remove the tag and everything up to its closing tag.
_DANGEROUS_TAGS = re.compile(
    r"<\s*(script|iframe|object|embed|link|style)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Any remaining opening/closing tag for dangerous elements (self-closing or unclosed).
_DANGEROUS_TAG_ONLY = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|link|style)\b[^>]*>",
    re.IGNORECASE,
)

# Event handler attributes: onclick, onerror, onload, onmouseover, etc.
_EVENT_HANDLER = re.compile(r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE)

# javascript: URLs.
_JAVASCRIPT_URL = re.compile(r"\bjavascript\s*:", re.IGNORECASE)

# data: URLs in src/href attributes.
_DATA_URL_ATTR = re.compile(
    r'(src|href)\s*=\s*[\"\']data:', re.IGNORECASE
)

# Any remaining HTML tag (catch-all for stripping).
_ANY_TAG = re.compile(r"<[^>]*>")


def sanitize_html(text: str) -> str:
    """Strip dangerous HTML, event handlers, and dangerous URLs from `text`.

    Removes:
    - <script>, <iframe>, <object>, <embed>, <link>, <style> tags (+ contents)
    - on* event handler attributes
    - javascript: URLs
    - data: URLs in src/href attributes
    """
    if not text:
        return text

    result = text

    # 1. Remove dangerous tags and their contents.
    result = _DANGEROUS_TAGS.sub("", result)
    result = _DANGEROUS_TAG_ONLY.sub("", result)

    # 2. Strip event handler attributes.
    result = _EVENT_HANDLER.sub("", result)

    # 3. Strip javascript: URLs.
    result = _JAVASCRIPT_URL.sub("", result)

    # 4. Strip data: URLs from src/href.
    result = _DATA_URL_ATTR.sub(r"\1=", result)

    return result


def sanitize_answer(answer: str) -> str:
    """Sanitize an LLM answer for safe return to the client.

    Applies HTML sanitization then HTML-escapes any remaining < and > characters
    so no HTML markup can reach the browser.
    """
    if not answer:
        return answer

    result = sanitize_html(answer)
    # Escape remaining angle brackets so no tag can be formed.
    result = result.replace("<", "&lt;").replace(">", "&gt;")
    return result


if __name__ == "__main__":
    # Self-test assertions.
    r1 = sanitize_answer("<script>alert(1)</script>what is lte?")
    assert r1 == "what is lte?", f"Got: {r1}"
    r2 = sanitize_answer("<iframe src='evil.com'></iframe>hello")
    assert r2 == "hello", f"Got: {r2}"
    r3 = sanitize_answer('<img onclick="alert(1)" src="x.png">')
    assert r3 == '<img src="x.png">', f"Got: {r3}"
    r4 = sanitize_answer('<a href="javascript:alert(1)">click</a>')
    assert r4 == '<a>click</a>', f"Got: {r4}"
    assert sanitize_answer(None) is None
    assert sanitize_answer("") == ""
    assert sanitize_answer("plain text") == "plain text"
    r8 = sanitize_answer("<b>bold</b>")
    assert r8 == "&lt;b&gt;bold&lt;/b&gt;", f"Got: {r8}"
    print("All sanitization self-tests passed.")
