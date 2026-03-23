from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from datasets import load_dataset  # type: ignore

from .utils import clean_text


@dataclass
class Task1Columns:
    text_col: str
    target_col: str
    id_col: Optional[str] = None  # ex: "acn_num_ACN" si dispo


def load_asrs_dataset(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def iter_examples(ds, cols: Task1Columns, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    n = 0
    for ex in ds:
        narrative = clean_text(ex.get(cols.text_col))
        reference = clean_text(ex.get(cols.target_col))
        if not narrative or not reference:
            continue

        ex_id = None
        if cols.id_col:
            ex_id = ex.get(cols.id_col)
        if ex_id is None:
            # fallback: certaines versions HF ont un champ "id" ou "acn"
            ex_id = ex.get("id") or ex.get("acn") or ex.get("acn_num_ACN")

        yield {
            "id": ex_id,
            "narrative": narrative,
            "reference": reference,
        }

        n += 1
        if limit is not None and n >= limit:
            break