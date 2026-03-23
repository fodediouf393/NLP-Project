from __future__ import annotations

from typing import List
import nltk

from .utils import clean_text

# Nécessaire une fois (et safe si déjà téléchargé)
def ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> List[str]:
    ensure_nltk()
    text = clean_text(text)
    if not text:
        return []
    from nltk.tokenize import sent_tokenize  # lazy import
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sents