from __future__ import annotations

import re
from typing import List


def clean_text(t: str | None) -> str:
    if not t:
        return ""
    # nettoyage léger (important de ne pas détruire RWY/FL/ATC etc.)
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_count(s: str) -> int:
    s = s.strip()
    return 0 if not s else len(s.split())


def select_by_word_budget(
    sentences: List[str],
    ranked_indices: List[int],
    max_words: int,
    ensure_one: bool = True,
) -> List[int]:
    chosen: List[int] = []
    wc = 0
    for idx in ranked_indices:
        w = word_count(sentences[idx])
        # prend au moins une phrase même si elle dépasse le budget
        if wc + w <= max_words or (ensure_one and len(chosen) == 0):
            chosen.append(idx)
            wc += w
        if wc >= max_words:
            break
    chosen.sort()  # conserver l'ordre d'apparition
    return chosen