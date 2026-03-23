from __future__ import annotations

from typing import List

import numpy as np
import networkx as nx  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from ..preprocess import split_sentences
from ..utils import select_by_word_budget


def summarize_textrank(
    narrative: str,
    max_words: int = 80,
    stop_words: str = "english",
    min_sim: float = 0.0,
) -> str:
    sents: List[str] = split_sentences(narrative)
    if not sents:
        return ""

    vec = TfidfVectorizer(stop_words=stop_words)
    X = vec.fit_transform(sents)

    # Similarité cosinus : X * X^T (TF-IDF approx. normalisé -> ok en baseline)
    sim = (X @ X.T).A
    np.fill_diagonal(sim, 0.0)

    if min_sim > 0:
        sim[sim < min_sim] = 0.0

    g = nx.from_numpy_array(sim)
    # Pagerank (TextRank)
    pr = nx.pagerank(g, weight="weight")
    ranked = sorted(pr.keys(), key=lambda i: pr[i], reverse=True)

    chosen_idx = select_by_word_budget(sents, ranked, max_words=max_words, ensure_one=True)
    return " ".join(sents[i] for i in chosen_idx)from __future__ import annotations

from typing import List

import numpy as np
import networkx as nx  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from ..preprocess import split_sentences
from ..utils import select_by_word_budget


def summarize_textrank(
    narrative: str,
    max_words: int = 80,
    stop_words: str = "english",
    min_sim: float = 0.0,
) -> str:
    sents: List[str] = split_sentences(narrative)
    if not sents:
        return ""

    vec = TfidfVectorizer(stop_words=stop_words)
    X = vec.fit_transform(sents)

    # Similarité cosinus : X * X^T (TF-IDF approx. normalisé -> ok en baseline)
    sim = (X @ X.T).A
    np.fill_diagonal(sim, 0.0)

    if min_sim > 0:
        sim[sim < min_sim] = 0.0

    g = nx.from_numpy_array(sim)
    # Pagerank (TextRank)
    pr = nx.pagerank(g, weight="weight")
    ranked = sorted(pr.keys(), key=lambda i: pr[i], reverse=True)

    chosen_idx = select_by_word_budget(sents, ranked, max_words=max_words, ensure_one=True)
    return " ".join(sents[i] for i in chosen_idx)