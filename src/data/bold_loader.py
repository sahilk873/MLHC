from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


def _find_column(columns: list[str], pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for col in columns:
        if regex.search(col):
            return col
    return None


def infer_sao2_spo2_columns(columns: list[str]) -> Tuple[str | None, str | None]:
    sao2 = _find_column(columns, r"\bsao2\b|arterial.*oxygen.*saturation")
    spo2 = _find_column(columns, r"\bspo2\b|pulse.*ox")
    return sao2, spo2


def load_bold_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def validate_bold_schema(df: pd.DataFrame) -> Tuple[str, str]:
    columns = list(df.columns)
    sao2_col, spo2_col = infer_sao2_spo2_columns(columns)
    if sao2_col is None or spo2_col is None:
        raise ValueError("Could not identify SaO2/SpO2 columns in BOLD table")
    return sao2_col, spo2_col
