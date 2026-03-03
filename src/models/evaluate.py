from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import os
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ece_regression(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df.empty:
        return float("nan")
    df["bin"] = pd.qcut(df["y_pred"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True)
    total = len(df)
    ece = 0.0
    for _, sub in grouped:
        weight = len(sub) / total
        ece += weight * float(np.abs(sub["y_true"].mean() - sub["y_pred"].mean()))
    return float(ece)


def grouped_mae(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group, sub in df.groupby(group_col):
        out[str(group)] = mae(sub["sao2"].values, sub["sao2_hat"].values)
    return out


def _rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def hidden_hypoxemia_rate(df: pd.DataFrame, threshold: int) -> float:
    col = f"hidden_hypoxemia_T{threshold}"
    if col not in df.columns:
        return float("nan")
    return _rate(df[col].astype(float))


def hypoxemia_rate(df: pd.DataFrame) -> float:
    return _rate((df["sao2"] < 88).astype(float))


def confusion_rates(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["sao2"] < 88
    y_pred = df["sao2_hat"] < 88
    if y_true.sum() == 0:
        fnr = float("nan")
    else:
        fnr = float(((y_true) & (~y_pred)).sum() / y_true.sum())
    if (~y_true).sum() == 0:
        fpr = float("nan")
    else:
        fpr = float(((~y_true) & (y_pred)).sum() / (~y_true).sum())
    missed = float(((y_true) & (~y_pred)).sum() / len(df)) if len(df) else float("nan")
    missed_conditional = fnr
    return {"fnr": fnr, "fpr": fpr, "missed_rate": missed, "missed_rate_conditional": missed_conditional}


def bootstrap_ci(values: List[float], seed: int = 1337) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(np.mean(sample))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci_low": float(lower), "ci_high": float(upper)}


def _bootstrap_params() -> Tuple[int, int]:
    n_boot = int(os.environ.get("BOOTSTRAP_N", "1000"))
    sample_size = int(os.environ.get("BOOTSTRAP_SAMPLE_SIZE", "0"))
    return n_boot, sample_size


def _cluster_col(df: pd.DataFrame) -> str | None:
    id_cols = [
        "person_id",
        "unique_subject_id",
        "subject_id",
        "patient_id",
        "unique_hospital_admission_id",
        "hospital_admission_id",
    ]
    for col in id_cols:
        if col in df.columns:
            return col
    return None


def _bootstrap_sample(df: pd.DataFrame, rng: np.random.Generator, draw: int) -> pd.DataFrame:
    cluster_col = _cluster_col(df)
    if cluster_col is None:
        idx = np.arange(len(df))
        sample_idx = rng.choice(idx, size=draw, replace=True)
        return df.iloc[sample_idx]
    clusters = df[cluster_col].dropna().unique()
    if len(clusters) == 0:
        idx = np.arange(len(df))
        sample_idx = rng.choice(idx, size=draw, replace=True)
        return df.iloc[sample_idx]
    cluster_draw = len(clusters) if draw <= 0 else min(len(clusters), draw)
    sampled = rng.choice(clusters, size=cluster_draw, replace=True)
    return df[df[cluster_col].isin(sampled)]


def bootstrap_metric(
    df: pd.DataFrame, metric_fn: Callable[[pd.DataFrame], float], seed: int = 1337
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n_boot, sample_size = _bootstrap_params()
    draw = len(df) if sample_size <= 0 else min(len(df), sample_size)
    for _ in range(n_boot):
        sample = _bootstrap_sample(df, rng, draw)
        values.append(metric_fn(sample))
    values_arr = np.array(values, dtype=float)
    values_arr = values_arr[np.isfinite(values_arr)]
    if len(values_arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    lower, upper = np.percentile(values_arr, [2.5, 97.5])
    return {"mean": float(np.mean(values_arr)), "ci_low": float(lower), "ci_high": float(upper)}


def grouped_metric(
    df: pd.DataFrame, group_col: str, metric_fn: Callable[[pd.DataFrame], float], min_group_n: int = 0
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group, sub in df.groupby(group_col):
        if min_group_n and len(sub) < min_group_n:
            continue
        out[str(group)] = metric_fn(sub)
    return out


def grouped_gap(values: Dict[str, float]) -> float:
    vals = [v for v in values.values() if pd.notna(v)]
    if not vals:
        return float("nan")
    return float(max(vals) - min(vals))


def worst_group(values: Dict[str, float]) -> Tuple[str, float]:
    if not values:
        return "NA", float("nan")
    group, value = max(values.items(), key=lambda item: item[1])
    return str(group), float(value)


def bootstrap_group_gap(
    df: pd.DataFrame,
    group_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
    seed: int = 1337,
    min_group_n: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n_boot, sample_size = _bootstrap_params()
    draw = len(df) if sample_size <= 0 else min(len(df), sample_size)
    for _ in range(n_boot):
        sample = _bootstrap_sample(df, rng, draw)
        grouped = grouped_metric(sample, group_col, metric_fn, min_group_n=min_group_n)
        values.append(grouped_gap(grouped))
    values_arr = np.array(values, dtype=float)
    values_arr = values_arr[np.isfinite(values_arr)]
    if len(values_arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    lower, upper = np.percentile(values_arr, [2.5, 97.5])
    return {"mean": float(np.mean(values_arr)), "ci_low": float(lower), "ci_high": float(upper)}


def bootstrap_worst_group(
    df: pd.DataFrame,
    group_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
    seed: int = 1337,
    min_group_n: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n_boot, sample_size = _bootstrap_params()
    draw = len(df) if sample_size <= 0 else min(len(df), sample_size)
    for _ in range(n_boot):
        sample = _bootstrap_sample(df, rng, draw)
        grouped = grouped_metric(sample, group_col, metric_fn, min_group_n=min_group_n)
        _, value = worst_group(grouped)
        values.append(value)
    values_arr = np.array(values, dtype=float)
    values_arr = values_arr[np.isfinite(values_arr)]
    if len(values_arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    lower, upper = np.percentile(values_arr, [2.5, 97.5])
    return {"mean": float(np.mean(values_arr)), "ci_low": float(lower), "ci_high": float(upper)}


def evaluate_dataset(
    df: pd.DataFrame,
    output_path: str,
    group_cols: List[str],
    thresholds: List[int],
    min_group_n: int = 0,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "mae": mae(df["sao2"].values, df["sao2_hat"].values),
        "rmse": rmse(df["sao2"].values, df["sao2_hat"].values),
        "mae_ci": bootstrap_metric(df, lambda d: mae(d["sao2"].values, d["sao2_hat"].values)),
        "rmse_ci": bootstrap_metric(df, lambda d: rmse(d["sao2"].values, d["sao2_hat"].values)),
        "ece": ece_regression(df["sao2"].values, df["sao2_hat"].values),
        "ece_ci": bootstrap_metric(
            df, lambda d: ece_regression(d["sao2"].values, d["sao2_hat"].values)
        ),
        "hypoxemia_rate": hypoxemia_rate(df),
        "hypoxemia_rate_ci": bootstrap_metric(df, hypoxemia_rate),
    }

    for threshold in thresholds:
        key = f"hidden_hypoxemia_T{threshold}"
        metrics[f"{key}_rate"] = hidden_hypoxemia_rate(df, threshold)
        metrics[f"{key}_rate_ci"] = bootstrap_metric(
            df, lambda d, t=threshold: hidden_hypoxemia_rate(d, t)
        )

    conf = confusion_rates(df)
    metrics.update(
        {
            "fnr": conf["fnr"],
            "fpr": conf["fpr"],
            "missed_hypoxemia_rate": conf["missed_rate"],
            "missed_hypoxemia_rate_conditional": conf["missed_rate_conditional"],
            "fnr_ci": bootstrap_metric(df, lambda d: confusion_rates(d)["fnr"]),
            "fpr_ci": bootstrap_metric(df, lambda d: confusion_rates(d)["fpr"]),
            "missed_hypoxemia_rate_ci": bootstrap_metric(
                df, lambda d: confusion_rates(d)["missed_rate"]
            ),
            "missed_hypoxemia_rate_conditional_ci": bootstrap_metric(
                df, lambda d: confusion_rates(d)["missed_rate_conditional"]
            ),
        }
    )

    for col in group_cols:
        if col not in df.columns:
            continue
        grouped_mae_vals = grouped_metric(
            df, col, lambda d: mae(d["sao2"].values, d["sao2_hat"].values), min_group_n=min_group_n
        )
        metrics[f"grouped_mae_{col}"] = grouped_mae_vals
        metrics[f"grouped_mae_{col}_gap"] = grouped_gap(grouped_mae_vals)
        metrics[f"grouped_mae_{col}_gap_ci"] = bootstrap_group_gap(
            df, col, lambda d: mae(d["sao2"].values, d["sao2_hat"].values), min_group_n=min_group_n
        )
        wg_group, wg_value = worst_group(grouped_mae_vals)
        metrics[f"worst_group_mae_{col}"] = {"group": wg_group, "value": wg_value}
        metrics[f"worst_group_mae_{col}_ci"] = bootstrap_worst_group(
            df, col, lambda d: mae(d["sao2"].values, d["sao2_hat"].values), min_group_n=min_group_n
        )

        for threshold in thresholds:
            key = f"hidden_hypoxemia_T{threshold}"
            grouped_hh = grouped_metric(
                df, col, lambda d, t=threshold: hidden_hypoxemia_rate(d, t), min_group_n=min_group_n
            )
            metrics[f"grouped_{key}_{col}"] = grouped_hh
            metrics[f"grouped_{key}_{col}_gap"] = grouped_gap(grouped_hh)
            metrics[f"grouped_{key}_{col}_gap_ci"] = bootstrap_group_gap(
                df, col, lambda d, t=threshold: hidden_hypoxemia_rate(d, t), min_group_n=min_group_n
            )
            wg_group, wg_value = worst_group(grouped_hh)
            metrics[f"worst_group_{key}_{col}"] = {"group": wg_group, "value": wg_value}
            metrics[f"worst_group_{key}_{col}_ci"] = bootstrap_worst_group(
                df, col, lambda d, t=threshold: hidden_hypoxemia_rate(d, t), min_group_n=min_group_n
            )

        grouped_fnr = grouped_metric(df, col, lambda d: confusion_rates(d)["fnr"], min_group_n=min_group_n)
        grouped_fpr = grouped_metric(df, col, lambda d: confusion_rates(d)["fpr"], min_group_n=min_group_n)
        grouped_missed = grouped_metric(
            df, col, lambda d: confusion_rates(d)["missed_rate"], min_group_n=min_group_n
        )
        metrics[f"grouped_fnr_{col}"] = grouped_fnr
        metrics[f"grouped_fnr_{col}_gap"] = grouped_gap(grouped_fnr)
        metrics[f"grouped_fnr_{col}_gap_ci"] = bootstrap_group_gap(
            df, col, lambda d: confusion_rates(d)["fnr"], min_group_n=min_group_n
        )
        metrics[f"grouped_fpr_{col}"] = grouped_fpr
        metrics[f"grouped_fpr_{col}_gap"] = grouped_gap(grouped_fpr)
        metrics[f"grouped_fpr_{col}_gap_ci"] = bootstrap_group_gap(
            df, col, lambda d: confusion_rates(d)["fpr"], min_group_n=min_group_n
        )
        metrics[f"grouped_missed_hypoxemia_rate_{col}"] = grouped_missed
        metrics[f"grouped_missed_hypoxemia_rate_{col}_gap"] = grouped_gap(grouped_missed)
        metrics[f"grouped_missed_hypoxemia_rate_{col}_gap_ci"] = bootstrap_group_gap(
            df, col, lambda d: confusion_rates(d)["missed_rate"], min_group_n=min_group_n
        )

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    metrics = _clean(metrics)
    Path(output_path).write_text(json.dumps(metrics, indent=2))
    return metrics
