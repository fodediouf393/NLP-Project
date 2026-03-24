from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = Path("data/raw")
DATASET_NAME = "elihoole/asrs-aviation-reports"


def normalize_split(ds_split) -> pd.DataFrame:
    df = ds_split.to_pandas()

    rename_candidates = {
        "Report 1_Narrative": "narrative",
        "Report 1.2_Synopsis": "synopsis",
        "acn_num_ACN": "acn",
    }

    for old_name, new_name in rename_candidates.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    required = ["narrative", "synopsis"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes requises introuvables après normalisation: {missing}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    df["narrative"] = df["narrative"].fillna("").astype(str)
    df["synopsis"] = df["synopsis"].fillna("").astype(str)

    if "acn" not in df.columns:
        df["acn"] = range(len(df))

    keep_cols = ["acn", "narrative", "synopsis"]
    extra_cols = [c for c in df.columns if c not in keep_cols]
    return df[keep_cols + extra_cols]


def save_split(df: pd.DataFrame, split_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{split_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"[OK] {split_name}: {len(df)} lignes -> {output_path}")


def main() -> None:
    dataset = load_dataset(DATASET_NAME)

    print("Splits disponibles :", list(dataset.keys()))

    for split_name in dataset.keys():
        df = normalize_split(dataset[split_name])
        save_split(df, split_name)


if __name__ == "__main__":
    main()