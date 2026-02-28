from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def histogram(series: pd.Series, title: str, path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna(), bins=20, color="#4C72B0", edgecolor="black")
    plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
