from __future__ import annotations

import argparse
import json
import time

from asrs_sum.core.topk_summarizer import TopKExtractiveSummarizer
from asrs_sum.core.textrank_summarizer import TextRankExtractiveSummarizer
from asrs_sum.utils.io import load_yaml
from asrs_sum.utils.logger import setup_logger


def build_summarizer(config_path: str, method: str = "topk"):
    config = load_yaml(config_path)

    if method == "topk":
        cfg = config["summarization"]
        return TopKExtractiveSummarizer(
            top_k_sentences=cfg["top_k_sentences"],
            min_sentence_words=cfg["min_sentence_words"],
            max_sentence_words=cfg["max_sentence_words"],
            max_summary_words=cfg["max_summary_words"],
            redundancy_threshold=cfg["redundancy_threshold"],
            stop_words=cfg["stop_words"],
            ngram_range=tuple(cfg["ngram_range"]),
            use_position_bonus=cfg["use_position_bonus"],
            position_bonus_weight=cfg["position_bonus_weight"],
            use_numeric_bonus=cfg["use_numeric_bonus"],
            numeric_bonus_weight=cfg["numeric_bonus_weight"],
            use_keyword_bonus=cfg["use_keyword_bonus"],
            keyword_bonus_weight=cfg["keyword_bonus_weight"],
            domain_keywords=cfg["domain_keywords"],
        )

    if method == "textrank":
        cfg = config["textrank"]
        return TextRankExtractiveSummarizer(
            top_k_sentences=cfg["top_k_sentences"],
            min_sentence_words=cfg["min_sentence_words"],
            max_sentence_words=cfg["max_sentence_words"],
            max_summary_words=cfg["max_summary_words"],
            redundancy_threshold=cfg["redundancy_threshold"],
            stop_words=cfg["stop_words"],
            ngram_range=tuple(cfg["ngram_range"]),
            damping_factor=cfg["damping_factor"],
            similarity_threshold=cfg["similarity_threshold"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            use_position_bonus=cfg["use_position_bonus"],
            position_bonus_weight=cfg["position_bonus_weight"],
            use_numeric_bonus=cfg["use_numeric_bonus"],
            numeric_bonus_weight=cfg["numeric_bonus_weight"],
            use_keyword_bonus=cfg["use_keyword_bonus"],
            keyword_bonus_weight=cfg["keyword_bonus_weight"],
            domain_keywords=cfg["domain_keywords"],
        )

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict summary for a single ASRS narrative.")
    parser.add_argument("--text", type=str, required=True, help="Narrative text")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--method", type=str, default="topk", choices=["topk", "textrank"])
    args = parser.parse_args()

    logger = setup_logger("predict")
    summarizer = build_summarizer(args.config, args.method)

    start = time.perf_counter()
    result = summarizer.summarize(args.text)
    elapsed = time.perf_counter() - start

    payload = {
        "method": args.method,
        "summary": result.summary,
        "selected_sentences": result.selected_sentences,
        "selected_indices": result.selected_indices,
        "summary_word_count": result.summary_word_count,
        "latency_seconds": round(elapsed, 6),
    }

    logger.info("Prediction complete with method=%s.", args.method)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()