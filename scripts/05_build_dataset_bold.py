from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.bold_loader import load_bold_table, validate_bold_schema
from src.data.harmonize import add_error_columns, add_hidden_hypoxemia, normalize_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    candidates = list(root.rglob("*.csv")) + list(root.rglob("*.csv.gz"))
    if not candidates:
        raise FileNotFoundError("No BOLD CSV files found")
    main_table = max(candidates, key=lambda p: p.stat().st_size)
    df = load_bold_table(str(main_table))
    df = normalize_columns(df)
    sao2_col, spo2_col = validate_bold_schema(df)
    # Basic cleaning: drop missing/implausible oxygen saturation values
    df[sao2_col] = pd.to_numeric(df[sao2_col], errors="coerce")
    df[spo2_col] = pd.to_numeric(df[spo2_col], errors="coerce")
    df = df.dropna(subset=[sao2_col, spo2_col])
    df = df[(df[sao2_col] >= 0) & (df[sao2_col] <= 100)]
    df = df[(df[spo2_col] >= 0) & (df[spo2_col] <= 100)]

    df = add_error_columns(df, sao2_col, spo2_col)
    df = add_hidden_hypoxemia(df, [90, 92, 94])
    df["dataset"] = "bold"

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "bold_analysis.parquet", index=False)


if __name__ == "__main__":
    main()
