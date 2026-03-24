from __future__ import annotations

import argparse

from src.asrs_sum.evaluation.metrics import (
    aggregate_metric_dicts,
    compression_ratio,
    compute_rouge,
    critical_entity_coverage,
)
from src.asrs_sum.utils.io import load_yaml, read_csv, save_json
from src.asrs_sum.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--input", type=str, required=True, help="CSV with source, reference, prediction")
    parser.add_argument("--output", type=str, required=True, help="Output JSON report")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--prediction-column", type=str, default="prediction")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_cfg = config["data"]

    logger = setup_logger(
        "evaluate",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    df = read_csv(args.input)

    text_col = data_cfg["text_column"]
    summary_col = data_cfg["summary_column"]
    prediction_col = args.prediction_column

    required_cols = [text_col, summary_col, prediction_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rouge_items = []
    entity_items = []
    compression_values = []
    latency_values = df["latency_seconds"].tolist() if "latency_seconds" in df.columns else []

    for _, row in df.iterrows():
        source = str(row[text_col]) if row[text_col] == row[text_col] else ""
        reference = str(row[summary_col]) if row[summary_col] == row[summary_col] else ""
        prediction = str(row[prediction_col]) if row[prediction_col] == row[prediction_col] else ""

        rouge_items.append(compute_rouge(reference, prediction))
        entity_items.append(critical_entity_coverage(reference, prediction))
        compression_values.append(compression_ratio(source, prediction))

    report = {
        "row_count": int(len(df)),
        "rouge": aggregate_metric_dicts(rouge_items),
        "entity_coverage": aggregate_metric_dicts(entity_items),
        "compression_ratio_mean": float(sum(compression_values) / len(compression_values)) if compression_values else 0.0,
        "latency_seconds_mean": float(sum(latency_values) / len(latency_values)) if latency_values else None,
    }

    save_json(report, args.output)

    logger.info("Evaluation complete.")
    logger.info("ROUGE: %s", report["rouge"])
    logger.info("Entity coverage: %s", report["entity_coverage"])


if __name__ == "__main__":
    main()