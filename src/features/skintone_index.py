from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


def extract_monk_value(concept_name: str) -> int | None:
    match = re.search(r"MONKSKINTONESCALE\s*:?\s*(\d+)", str(concept_name), re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    if value < 1 or value > 10:
        return None
    return value


def bin_monk_value(value: int | None) -> str | None:
    if value is None:
        return None
    if 1 <= value <= 3:
        return "light"
    if 4 <= value <= 6:
        return "medium"
    return "dark"


def _coerce_monk_value(row: pd.Series) -> int | None:
    value = row.get("value_as_number")
    if pd.notna(value):
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and 1 <= numeric <= 10:
            return numeric
    concept_name = row.get("concept_name")
    return extract_monk_value(concept_name)


def build_skintone_index(skintone_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = skintone_df.copy()
    measure = df.get("measure", pd.Series("", index=df.index)).astype(str)
    monk_mask = measure.str.contains("MONKSKINTONESCALE", case=False, na=False)
    df = df[monk_mask].copy()
    if df.empty:
        empty = pd.DataFrame(columns=["person_id", "skintone_monk", "skintone_bin"])
        return df, empty
    df["monk_value"] = df.apply(_coerce_monk_value, axis=1)
    per_person = (
        df.dropna(subset=["monk_value"])
        .groupby("person_id")["monk_value"]
        .median()
        .reset_index()
    )
    per_person["skintone_monk"] = per_person["monk_value"].astype(int)
    per_person["skintone_bin"] = per_person["skintone_monk"].apply(bin_monk_value)
    return df, per_person[["person_id", "skintone_monk", "skintone_bin"]]
