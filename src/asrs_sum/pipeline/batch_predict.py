from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from asrs_sum.pipeline.predict import build_summarizer
from asrs_sum.utils.io import load_yaml, read_csv, write_csv
from asrs_sum.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch summary prediction for ASRS narratives.")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier CSV d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin vers le fichier CSV de sortie")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Chemin vers le fichier YAML")
    parser.add_argument("--method", type=str, default="topk", choices=["topk", "textrank"])
    return parser.parse_args()


def validate_input_dataframe(df: pd.DataFrame, text_column: str) -> None:
    if df.empty:
        raise ValueError("Le fichier CSV d'entrée est vide.")

    if text_column not in df.columns:
        raise ValueError(
            f"Colonne texte '{text_column}' introuvable dans le CSV. "
            f"Colonnes disponibles: {list(df.columns)}"
        )


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    if not config:
        raise ValueError(
            f"Impossible de charger la configuration depuis '{args.config}'. "
            "Vérifie que le fichier existe et contient un YAML valide."
        )

    data_cfg = config["data"]
    logging_cfg = config["logging"]

    logger = setup_logger(
        name="batch_predict",
        level=logging_cfg["level"],
        log_file=logging_cfg["log_file"],
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Démarrage du batch prediction avec method=%s.", args.method)
    logger.info("Fichier d'entrée : %s", input_path.resolve())
    logger.info("Fichier de sortie : %s", output_path.resolve())

    if not input_path.exists():
        raise FileNotFoundError(f"Le fichier d'entrée n'existe pas : {input_path}")

    df = read_csv(input_path)
    text_col = data_cfg["text_column"]

    logger.info("CSV chargé avec succès.")
    logger.info("Nombre de lignes : %d", len(df))
    logger.info("Colonnes détectées : %s", list(df.columns))

    validate_input_dataframe(df, text_col)

    summarizer = build_summarizer(args.config, args.method)
    logger.info("Summarizer initialisé avec succès.")

    predictions: list[str] = []
    prediction_word_counts: list[int] = []
    selected_sentence_counts: list[int] = []
    latencies: list[float] = []
    status_list: list[str] = []
    error_messages: list[str] = []

    total_start = time.perf_counter()

    for idx, raw_text in enumerate(df[text_col].fillna("").astype(str).tolist(), start=1):
        try:
            start = time.perf_counter()
            result = summarizer.summarize(raw_text)
            elapsed = time.perf_counter() - start

            predictions.append(result.summary)
            prediction_word_counts.append(len(result.summary.split()) if result.summary else 0)
            selected_sentence_counts.append(len(result.selected_sentences))
            latencies.append(elapsed)
            status_list.append("ok")
            error_messages.append("")

        except Exception as exc:
            logger.exception("Erreur sur la ligne %d", idx)
            predictions.append("")
            prediction_word_counts.append(0)
            selected_sentence_counts.append(0)
            latencies.append(0.0)
            status_list.append("error")
            error_messages.append(str(exc))

        if idx % 10 == 0 or idx == len(df):
            logger.info("Progression : %d / %d lignes traitées.", idx, len(df))

    total_elapsed = time.perf_counter() - total_start

    output_df = df.copy()
    output_df["method"] = args.method
    output_df["prediction"] = predictions
    output_df["prediction_word_count"] = prediction_word_counts
    output_df["selected_sentence_count"] = selected_sentence_counts
    output_df["latency_seconds"] = latencies
    output_df["status"] = status_list
    output_df["error_message"] = error_messages

    write_csv(output_df, output_path)

    ok_count = sum(1 for s in status_list if s == "ok")
    error_count = sum(1 for s in status_list if s == "error")
    mean_latency = float(pd.Series(latencies).mean()) if latencies else 0.0

    logger.info("Batch prediction terminé.")
    logger.info("Lignes OK : %d", ok_count)
    logger.info("Lignes en erreur : %d", error_count)
    logger.info("Latence moyenne : %.6f sec", mean_latency)
    logger.info("Temps total : %.6f sec", total_elapsed)
    logger.info("Fichier sauvegardé : %s", output_path.resolve())


if __name__ == "__main__":
    main()