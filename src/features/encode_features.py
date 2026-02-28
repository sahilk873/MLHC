from __future__ import annotations

import pandas as pd


def select_covariates(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    return numeric
