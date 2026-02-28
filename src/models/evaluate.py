from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def grouped_mae(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group, sub in df.groupby(group_col):
        out[str(group)] = mae(sub["sao2"].values, sub["sao2_hat"].values)
    return out


def bootstrap_ci(values: List[float], seed: int = 1337) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(np.mean(sample))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci_low": float(lower), "ci_high": float(upper)}


def evaluate_dataset(df: pd.DataFrame, output_path: str, group_cols: List[str]) -> None:
    metrics = {
        "mae": mae(df["sao2"].values, df["sao2_hat"].values),
        "rmse": rmse(df["sao2"].values, df["sao2_hat"].values),
    }
    for col in group_cols:
        if col in df.columns:
            metrics[f"grouped_mae_{col}"] = grouped_mae(df, col)
    Path(output_path).write_text(json.dumps(metrics, indent=2))
