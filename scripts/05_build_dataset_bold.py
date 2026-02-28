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
    df = add_error_columns(df, sao2_col, spo2_col)
    df = add_hidden_hypoxemia(df, [90, 92, 94])
    df["dataset"] = "bold"

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "bold_analysis.parquet", index=False)


if __name__ == "__main__":
    main()
