from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogram(series: pd.Series, title: str, path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna(), bins=20, color="#4C72B0", edgecolor="black")
    plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def bar_plot(labels: List[str], values: List[float], title: str, ylabel: str, path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#55A868", edgecolor="black")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def bar_plot_ci(
    labels: List[str],
    values: List[float],
    ci_low: List[float],
    ci_high: List[float],
    title: str,
    ylabel: str,
    path: str,
) -> None:
    plt.figure(figsize=(7, 4))
    safe_low = []
    safe_high = []
    for v, lo, hi in zip(values, ci_low, ci_high):
        if v is None or lo is None or hi is None:
            safe_low.append(0.0)
            safe_high.append(0.0)
            continue
        safe_low.append(max(v - lo, 0.0))
        safe_high.append(max(hi - v, 0.0))
    yerr = [np.array(safe_low), np.array(safe_high)]
    plt.bar(labels, values, yerr=yerr, capsize=4, color="#C44E52", edgecolor="black")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def line_plot(
    x: List[float], ys: List[List[float]], labels: List[str], title: str, ylabel: str, path: str
) -> None:
    plt.figure(figsize=(6, 4))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def boxplot_by_group(
    df: pd.DataFrame, value_col: str, group_col: str, title: str, path: str
) -> None:
    data = []
    labels = []
    for group, sub in df.groupby(group_col):
        data.append(sub[value_col].dropna().values)
        labels.append(str(group))
    if not data:
        return
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def scatter_plot(
    x: List[float],
    y: List[float],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="#4C72B0")
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi, yi, label, fontsize=8, ha="left", va="bottom")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def calibration_curve(
    y_true: pd.Series, y_pred: pd.Series, bins: int = 10
) -> Tuple[List[float], List[float]]:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df.empty:
        return [], []
    df["bin"] = pd.qcut(df["y_pred"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True)
    pred_means = grouped["y_pred"].mean().tolist()
    true_means = grouped["y_true"].mean().tolist()
    return pred_means, true_means


def calibration_plot(
    curves: Iterable[Tuple[List[float], List[float], str]], title: str, path: str
) -> None:
    plt.figure(figsize=(5, 5))
    for x, y, label in curves:
        if not x or not y:
            continue
        plt.plot(x, y, marker="o", label=label)
    plt.plot([80, 100], [80, 100], "--", color="gray", label="Ideal")
    plt.title(title)
    plt.xlabel("Predicted SaO2")
    plt.ylabel("Observed SaO2")
    plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
