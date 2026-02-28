from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.viz.figures import histogram
from src.viz.tables import write_table


def main() -> None:
    results_fig = Path("results/figures")
    results_tab = Path("results/tables")
    results_fig.mkdir(parents=True, exist_ok=True)
    results_tab.mkdir(parents=True, exist_ok=True)

    bold_path = Path("results/metrics/bold_analysis.parquet")
    encode_path = Path("results/metrics/encode_analysis.parquet")

    if bold_path.exists():
        df = pd.read_parquet(bold_path)
        histogram(df["error"], "BOLD Error Distribution", str(results_fig / "bold_error.png"))
        write_table(df.head(50), str(results_tab / "bold_preview.csv"))

    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        if not df.empty and "error" in df.columns:
            histogram(
                df["error"],
                "ENCoDE Error Distribution",
                str(results_fig / "encode_error.png"),
            )
        if not df.empty:
            write_table(df.head(50), str(results_tab / "encode_preview.csv"))

    # Model metrics tables
    models_bold = Path("results/metrics/models_bold.json")
    models_encode = Path("results/metrics/models_encode.json")

    if models_bold.exists():
        data = json.loads(models_bold.read_text())
        rows = []
        for name, metrics in data.items():
            race = metrics.get("grouped_mae_race_ethnicity", {})
            gap = None
            if race:
                vals = list(race.values())
                gap = max(vals) - min(vals)
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "race_mae_gap": gap,
                }
            )
        if rows:
            write_table(pd.DataFrame(rows), str(results_tab / "model_metrics_bold.csv"))

    if models_encode.exists():
        data = json.loads(models_encode.read_text())
        rows = []
        for name, metrics in data.items():
            skin = metrics.get("grouped_mae_skintone_bin", {})
            gap = None
            if skin:
                vals = list(skin.values())
                gap = max(vals) - min(vals)
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "skintone_mae_gap": gap,
                }
            )
        if rows:
            write_table(pd.DataFrame(rows), str(results_tab / "model_metrics_encode.csv"))


if __name__ == "__main__":
    main()
