from __future__ import annotations

import pandas as pd


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["spo2"]
