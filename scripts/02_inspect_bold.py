from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.bold_loader import infer_sao2_spo2_columns, load_bold_table


def find_main_table(root: Path) -> Path:
    candidates = list(root.rglob("*.csv")) + list(root.rglob("*.csv.gz"))
    if not candidates:
        raise FileNotFoundError("No CSV files found in BOLD directory")
    return max(candidates, key=lambda p: p.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    table_path = find_main_table(root)
    df = load_bold_table(str(table_path))

    sao2_col, spo2_col = infer_sao2_spo2_columns(list(df.columns))
    summary = {
        "file": str(table_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sao2_col": sao2_col,
        "spo2_col": spo2_col,
    }

    if sao2_col and spo2_col:
        summary["sao2_min"] = float(pd.to_numeric(df[sao2_col], errors="coerce").min())
        summary["sao2_max"] = float(pd.to_numeric(df[sao2_col], errors="coerce").max())
        summary["spo2_min"] = float(pd.to_numeric(df[spo2_col], errors="coerce").min())
        summary["spo2_max"] = float(pd.to_numeric(df[spo2_col], errors="coerce").max())

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "bold_summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# BOLD Schema Report",
        f"- file: {table_path}",
        f"- rows: {summary['rows']}",
        f"- columns: {summary['columns']}",
        f"- sao2_col: {summary['sao2_col']}",
        f"- spo2_col: {summary['spo2_col']}",
    ]
    if sao2_col and spo2_col:
        md_lines.extend(
            [
                f"- sao2_min: {summary['sao2_min']}",
                f"- sao2_max: {summary['sao2_max']}",
                f"- spo2_min: {summary['spo2_min']}",
                f"- spo2_max: {summary['spo2_max']}",
            ]
        )
    (log_dir / "bold_schema.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
