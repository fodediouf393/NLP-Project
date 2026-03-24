from __future__ import annotations

import re
from dataclasses import dataclass

from .preprocessing import clean_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Sentence:
    text: str
    index: int
    word_count: int


def split_into_sentences(text: str) -> list[Sentence]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    parts = SENTENCE_SPLIT_RE.split(cleaned)

    if len(parts) <= 1:
        parts = re.split(r"\s{2,}|(?<=;)\s+", cleaned)

    sentences: list[Sentence] = []
    for idx, part in enumerate(parts):
        candidate = part.strip()
        if not candidate:
            continue

        word_count = len(candidate.split())
        sentences.append(Sentence(text=candidate, index=idx, word_count=word_count))

    return sentences