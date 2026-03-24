from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from rouge_score import rouge_scorer


RUNWAY_RE = re.compile(r"\b(?:RWY|RUNWAY)\s*\d+[LRC]?\b", re.IGNORECASE)
ALTITUDE_RE = re.compile(r"\b(?:FL\d{2,3}|\d{3,5}\s*(?:FT|FEET))\b", re.IGNORECASE)
AIRPORT_RE = re.compile(r"\b[A-Z]{3,4}\b")


def compute_rouge(reference: str, prediction: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", prediction or "")
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rouge2_f1": float(scores["rouge2"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
    }


def compression_ratio(source: str, summary: str) -> float:
    source_words = len((source or "").split())
    summary_words = len((summary or "").split())
    if source_words == 0:
        return 0.0
    return summary_words / source_words


def _extract_entities(pattern: re.Pattern[str], text: str) -> set[str]:
    return {m.group(0).upper().strip() for m in pattern.finditer(text or "")}


def critical_entity_coverage(reference: str, prediction: str) -> dict[str, float]:
    ref_runways = _extract_entities(RUNWAY_RE, reference)
    pred_runways = _extract_entities(RUNWAY_RE, prediction)

    ref_altitudes = _extract_entities(ALTITUDE_RE, reference)
    pred_altitudes = _extract_entities(ALTITUDE_RE, prediction)

    ref_airports = _extract_entities(AIRPORT_RE, reference)
    pred_airports = _extract_entities(AIRPORT_RE, prediction)

    def cov(ref_set: set[str], pred_set: set[str]) -> float:
        if not ref_set:
            return 1.0
        return len(ref_set & pred_set) / len(ref_set)

    return {
        "runway_coverage": cov(ref_runways, pred_runways),
        "altitude_coverage": cov(ref_altitudes, pred_altitudes),
        "airport_code_coverage": cov(ref_airports, pred_airports),
    }


def aggregate_metric_dicts(items: Iterable[dict[str, float]]) -> dict[str, float]:
    items = list(items)
    if not items:
        return {}

    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}