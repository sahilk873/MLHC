from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYWORDS = ["spo2", "sao2", "oxim", "pulse", "abg", "arterial", "sat"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    meas_path = Path(args.root) / "MEASUREMENT.csv"
    if not meas_path.exists():
        raise FileNotFoundError(meas_path)

    df = pd.read_csv(meas_path, low_memory=False)
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    key_cols = [
        "person_id",
        "visit_occurrence_id",
        "visit_occurrences_id",
        "measurement_datetime",
        "measurement_date",
        "measurement_time",
        "measurement_source_value",
        "value_as_number",
        "unit_source_value",
    ]
    missing = {}
    for col in key_cols:
        if col in df.columns:
            missing[col] = float(df[col].isna().mean())
    summary["missing_fraction"] = missing

    if "measurement_source_value" in df.columns:
        top = (
            df["measurement_source_value"]
            .astype(str)
            .value_counts()
            .head(50)
            .to_dict()
        )
    else:
        top = {}
    summary["top_measurement_source_value"] = top

    keyword_hits = {}
    if "measurement_source_value" in df.columns:
        source = df["measurement_source_value"].astype(str)
        for key in KEYWORDS:
            keyword_hits[key] = int(source.str.contains(key, case=False, na=False).sum())
    summary["keyword_hits"] = keyword_hits

    if "value_as_number" in df.columns:
        values = pd.to_numeric(df["value_as_number"], errors="coerce")
        summary["value_as_number"] = {
            "count": int(values.count()),
            "min": float(values.min()) if values.count() else None,
            "max": float(values.max()) if values.count() else None,
            "mean": float(values.mean()) if values.count() else None,
        }

    if "unit_source_value" in df.columns:
        summary["unit_source_value_top"] = (
            df["unit_source_value"].astype(str).value_counts().head(20).to_dict()
        )

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "encode_measurement_inspect.json").write_text(
        json.dumps(summary, indent=2)
    )

    md_lines = ["# ENCoDE MEASUREMENT Inspection", f"- rows: {summary['rows']}"]
    for col, frac in missing.items():
        md_lines.append(f"- missing {col}: {frac:.3f}")
    md_lines.append("## Top measurement_source_value")
    for k, v in top.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("## Keyword hits")
    for k, v in keyword_hits.items():
        md_lines.append(f"- {k}: {v}")
    (reports / "encode_measurement_inspect.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
