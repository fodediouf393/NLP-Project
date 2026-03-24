from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .sentence_splitter import Sentence


NUMERIC_RE = re.compile(r"\d")
ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")


@dataclass
class RankedSentence:
    text: str
    index: int
    score: float
    base_score: float
    position_bonus: float
    numeric_bonus: float
    keyword_bonus: float
    word_count: int


def _contains_numeric(text: str) -> bool:
    return bool(NUMERIC_RE.search(text))


def _keyword_density(text: str, keywords: list[str]) -> float:
    lowered = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in lowered)
    if not keywords:
        return 0.0
    return matches / len(keywords)


def _position_score(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - (index / (total - 1))


def rank_sentences(
    sentences: list[Sentence],
    stop_words: str | None = "english",
    ngram_range: tuple[int, int] = (1, 2),
    min_sentence_words: int = 5,
    max_sentence_words: int = 80,
    use_position_bonus: bool = True,
    position_bonus_weight: float = 0.15,
    use_numeric_bonus: bool = True,
    numeric_bonus_weight: float = 0.10,
    use_keyword_bonus: bool = True,
    keyword_bonus_weight: float = 0.15,
    domain_keywords: list[str] | None = None,
) -> list[RankedSentence]:
    if not sentences:
        return []

    domain_keywords = domain_keywords or []

    eligible_sentences = [
        s for s in sentences
        if min_sentence_words <= s.word_count <= max_sentence_words and ALPHANUM_RE.search(s.text)
    ]

    if not eligible_sentences:
        eligible_sentences = sentences

    texts = [s.text for s in eligible_sentences]

    if len(texts) == 1:
        s = eligible_sentences[0]
        return [
            RankedSentence(
                text=s.text,
                index=s.index,
                score=1.0,
                base_score=1.0,
                position_bonus=0.0,
                numeric_bonus=0.0,
                keyword_bonus=0.0,
                word_count=s.word_count,
            )
        ]

    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(texts)

    # Centroid document vector sans np.matrix
    document_vector = np.asarray(matrix.toarray().mean(axis=0), dtype=float).reshape(1, -1)

    # Similarité cosinus calculée manuellement pour éviter tout souci sklearn/np.matrix
    sentence_vectors = matrix.toarray().astype(float)

    sentence_norms = np.linalg.norm(sentence_vectors, axis=1)
    doc_norm = np.linalg.norm(document_vector)

    if doc_norm == 0.0:
        base_scores = np.zeros(len(sentence_vectors), dtype=float)
    else:
        dots = sentence_vectors @ document_vector.ravel()
        denom = sentence_norms * doc_norm
        base_scores = np.divide(
            dots,
            denom,
            out=np.zeros_like(dots, dtype=float),
            where=denom != 0,
        )

    total = len(eligible_sentences)
    ranked: list[RankedSentence] = []

    for i, sentence in enumerate(eligible_sentences):
        position_bonus = position_bonus_weight * _position_score(sentence.index, total) if use_position_bonus else 0.0
        numeric_bonus = numeric_bonus_weight if use_numeric_bonus and _contains_numeric(sentence.text) else 0.0
        keyword_bonus = (
            keyword_bonus_weight * _keyword_density(sentence.text, domain_keywords)
            if use_keyword_bonus else 0.0
        )

        final_score = float(base_scores[i] + position_bonus + numeric_bonus + keyword_bonus)

        ranked.append(
            RankedSentence(
                text=sentence.text,
                index=sentence.index,
                score=final_score,
                base_score=float(base_scores[i]),
                position_bonus=float(position_bonus),
                numeric_bonus=float(numeric_bonus),
                keyword_bonus=float(keyword_bonus),
                word_count=sentence.word_count,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked