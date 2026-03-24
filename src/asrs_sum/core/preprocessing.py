from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")
MULTI_DOT_RE = re.compile(r"\.{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:.!?])")


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_DOT_RE.sub(".", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def safe_truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""

    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()