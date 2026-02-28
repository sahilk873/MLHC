from __future__ import annotations

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def add_error_columns(df: pd.DataFrame, sao2_col: str, spo2_col: str) -> pd.DataFrame:
    df = df.copy()
    df["sao2"] = pd.to_numeric(df[sao2_col], errors="coerce")
    df["spo2"] = pd.to_numeric(df[spo2_col], errors="coerce")
    df["error"] = df["sao2"] - df["spo2"]
    df["abs_error"] = df["error"].abs()
    return df


def add_hidden_hypoxemia(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    df = df.copy()
    for t in thresholds:
        col = f"hidden_hypoxemia_T{t}"
        df[col] = (df["sao2"] < 88) & (df["spo2"] >= t)
    return df
