from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text, safe_truncate_words
from .sentence_ranker import RankedSentence, rank_sentences
from .sentence_splitter import split_into_sentences


@dataclass
class SummaryResult:
    summary: str
    selected_sentences: list[str]
    selected_indices: list[int]
    ranked_sentences: list[dict]
    original_sentence_count: int
    summary_word_count: int


class TopKExtractiveSummarizer:
    def __init__(
        self,
        top_k_sentences: int = 3,
        min_sentence_words: int = 5,
        max_sentence_words: int = 80,
        max_summary_words: int = 90,
        redundancy_threshold: float = 0.85,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        use_position_bonus: bool = True,
        position_bonus_weight: float = 0.15,
        use_numeric_bonus: bool = True,
        numeric_bonus_weight: float = 0.10,
        use_keyword_bonus: bool = True,
        keyword_bonus_weight: float = 0.15,
        domain_keywords: list[str] | None = None,
    ) -> None:
        self.top_k_sentences = top_k_sentences
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words
        self.max_summary_words = max_summary_words
        self.redundancy_threshold = redundancy_threshold
        self.stop_words = stop_words
        self.ngram_range = ngram_range
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
            return SummaryResult(
                summary="",
                selected_sentences=[],
                selected_indices=[],
                ranked_sentences=[],
                original_sentence_count=0,
                summary_word_count=0,
            )

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
            return SummaryResult(
                summary=single,
                selected_sentences=[single],
                selected_indices=[0],
                ranked_sentences=[],
                original_sentence_count=1,
                summary_word_count=len(single.split()),
            )

        ranked = rank_sentences(
            sentences=sentences,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_sentence_words=self.min_sentence_words,
            max_sentence_words=self.max_sentence_words,
            use_position_bonus=self.use_position_bonus,
            position_bonus_weight=self.position_bonus_weight,
            use_numeric_bonus=self.use_numeric_bonus,
            numeric_bonus_weight=self.numeric_bonus_weight,
            use_keyword_bonus=self.use_keyword_bonus,
            keyword_bonus_weight=self.keyword_bonus_weight,
            domain_keywords=self.domain_keywords,
        )

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

        summary = " ".join(summary_sentences).strip()

        return SummaryResult(
            summary=summary,
            selected_sentences=summary_sentences,
            selected_indices=[s.index for s in selected[: len(summary_sentences)]],
            ranked_sentences=[asdict(s) for s in ranked],
            original_sentence_count=len(sentences),
            summary_word_count=current_words,
        )

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

        texts = [text] + selected_texts
        vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
        matrix = vectorizer.fit_transform(texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).reshape(-1)
        return float(sims.max()) if sims.size > 0 else 0.0