from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import clean_text, safe_truncate_words
from .sentence_splitter import split_into_sentences
from .sentence_ranker import RankedSentence


@dataclass
class SummaryResult:
    summary: str
    selected_sentences: list[str]
    selected_indices: list[int]
    ranked_sentences: list[dict]
    original_sentence_count: int
    summary_word_count: int


class TextRankExtractiveSummarizer:
    def __init__(
        self,
        top_k_sentences: int = 2,
        min_sentence_words: int = 5,
        max_sentence_words: int = 60,
        max_summary_words: int = 60,
        redundancy_threshold: float = 0.75,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        damping_factor: float = 0.85,
        similarity_threshold: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
        use_position_bonus: bool = True,
        position_bonus_weight: float = 0.10,
        use_numeric_bonus: bool = True,
        numeric_bonus_weight: float = 0.10,
        use_keyword_bonus: bool = True,
        keyword_bonus_weight: float = 0.20,
        domain_keywords: list[str] | None = None,
    ) -> None:
        self.top_k_sentences = top_k_sentences
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words
        self.max_summary_words = max_summary_words
        self.redundancy_threshold = redundancy_threshold
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.damping_factor = damping_factor
        self.similarity_threshold = similarity_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.use_position_bonus = use_position_bonus
        self.position_bonus_weight = position_bonus_weight
        self.use_numeric_bonus = use_numeric_bonus
        self.numeric_bonus_weight = numeric_bonus_weight
        self.use_keyword_bonus = use_keyword_bonus
        self.keyword_bonus_weight = keyword_bonus_weight
        self.domain_keywords = domain_keywords or []

    def summarize(self, text: str) -> SummaryResult:
        cleaned = clean_text(text)
        if not cleaned:
            return SummaryResult("", [], [], [], 0, 0)

        sentences = split_into_sentences(cleaned)
        if not sentences:
            fallback_summary = safe_truncate_words(cleaned, self.max_summary_words)
            return SummaryResult(
                summary=fallback_summary,
                selected_sentences=[fallback_summary] if fallback_summary else [],
                selected_indices=[0] if fallback_summary else [],
                ranked_sentences=[],
                original_sentence_count=0,
                summary_word_count=len(fallback_summary.split()) if fallback_summary else 0,
            )

        if len(sentences) == 1:
            single = safe_truncate_words(sentences[0].text, self.max_summary_words)
            return SummaryResult(single, [single], [0], [], 1, len(single.split()))

        eligible = [
            s for s in sentences
            if self.min_sentence_words <= s.word_count <= self.max_sentence_words
        ]
        if not eligible:
            eligible = sentences

        ranked = self._rank_sentences(eligible)
        selected = self._select_non_redundant(ranked, self.top_k_sentences)
        selected.sort(key=lambda x: x.index)

        summary_sentences: list[str] = []
        current_words = 0

        for item in selected:
            sent_words = len(item.text.split())
            if summary_sentences and (current_words + sent_words > self.max_summary_words):
                continue

            if not summary_sentences and sent_words > self.max_summary_words:
                truncated = safe_truncate_words(item.text, self.max_summary_words)
                summary_sentences.append(truncated)
                current_words = len(truncated.split())
                break

            summary_sentences.append(item.text)
            current_words += sent_words

        if not summary_sentences and ranked:
            fallback = safe_truncate_words(ranked[0].text, self.max_summary_words)
            summary_sentences = [fallback]
            current_words = len(fallback.split())

        return SummaryResult(
            summary=" ".join(summary_sentences).strip(),
            selected_sentences=summary_sentences,
            selected_indices=[s.index for s in selected[: len(summary_sentences)]],
            ranked_sentences=[asdict(s) for s in ranked],
            original_sentence_count=len(sentences),
            summary_word_count=current_words,
        )

    def _rank_sentences(self, sentences) -> list[RankedSentence]:
        texts = [s.text for s in sentences]

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )
        matrix = vectorizer.fit_transform(texts).toarray().astype(float)

        similarity = self._cosine_similarity_matrix(matrix)
        np.fill_diagonal(similarity, 0.0)

        similarity[similarity < self.similarity_threshold] = 0.0

        pagerank_scores = self._pagerank(similarity)

        ranked: list[RankedSentence] = []
        total = len(sentences)

        for i, sentence in enumerate(sentences):
            base_score = float(pagerank_scores[i])
            position_bonus = self.position_bonus_weight * self._position_score(sentence.index, total) if self.use_position_bonus else 0.0
            numeric_bonus = self.numeric_bonus_weight if self.use_numeric_bonus and self._contains_numeric(sentence.text) else 0.0
            keyword_bonus = self.keyword_bonus_weight * self._keyword_density(sentence.text, self.domain_keywords) if self.use_keyword_bonus else 0.0

            final_score = float(base_score + position_bonus + numeric_bonus + keyword_bonus)

            ranked.append(
                RankedSentence(
                    text=sentence.text,
                    index=sentence.index,
                    score=final_score,
                    base_score=base_score,
                    position_bonus=float(position_bonus),
                    numeric_bonus=float(numeric_bonus),
                    keyword_bonus=float(keyword_bonus),
                    word_count=sentence.word_count,
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def _cosine_similarity_matrix(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        normalized = matrix / safe_norms
        return normalized @ normalized.T

    def _pagerank(self, similarity: np.ndarray) -> np.ndarray:
        n = similarity.shape[0]
        if n == 0:
            return np.array([], dtype=float)

        row_sums = similarity.sum(axis=1, keepdims=True)
        transition = np.divide(
            similarity,
            row_sums,
            out=np.zeros_like(similarity, dtype=float),
            where=row_sums != 0,
        )

        scores = np.ones(n, dtype=float) / n
        teleport = (1.0 - self.damping_factor) / n

        for _ in range(self.max_iter):
            new_scores = teleport + self.damping_factor * transition.T.dot(scores)
            if np.linalg.norm(new_scores - scores, ord=1) < self.tol:
                scores = new_scores
                break
            scores = new_scores

        return scores

    def _select_non_redundant(self, ranked: list[RankedSentence], k: int) -> list[RankedSentence]:
        if not ranked:
            return []
        if len(ranked) == 1 or k <= 1:
            return ranked[:1]

        selected: list[RankedSentence] = []

        for candidate in ranked:
            if len(selected) >= k:
                break

            if not selected:
                selected.append(candidate)
                continue

            similarity = self._max_similarity(candidate.text, [s.text for s in selected])
            if similarity < self.redundancy_threshold:
                selected.append(candidate)

        if len(selected) < min(k, len(ranked)):
            for candidate in ranked:
                if len(selected) >= k:
                    break
                if candidate not in selected:
                    selected.append(candidate)

        return selected

    def _max_similarity(self, text: str, selected_texts: list[str]) -> float:
        if not selected_texts:
            return 0.0

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )
        matrix = vectorizer.fit_transform([text] + selected_texts).toarray().astype(float)
        target = matrix[0]
        others = matrix[1:]

        target_norm = np.linalg.norm(target)
        if target_norm == 0.0:
            return 0.0

        sims = []
        for vec in others:
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0.0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(target, vec) / (target_norm * vec_norm)))
        return max(sims) if sims else 0.0

    @staticmethod
    def _contains_numeric(text: str) -> bool:
        return any(char.isdigit() for char in text)

    @staticmethod
    def _keyword_density(text: str, keywords: list[str]) -> float:
        if not keywords:
            return 0.0
        lowered = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in lowered)
        return matches / len(keywords)

    @staticmethod
    def _position_score(index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        return 1.0 - (index / (total - 1))