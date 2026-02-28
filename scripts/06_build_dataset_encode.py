from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.encode_loader import (
    build_concept_map,
    extract_skintone_measurements,
    load_encode_tables,
    validate_encode_schema,
)
from src.data.harmonize import add_error_columns, add_hidden_hypoxemia
from src.features.skintone_index import build_skintone_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    tables = load_encode_tables(args.root)
    validate_encode_schema(tables)
    pairs_path = Path("artifacts/encode_pairs.parquet")
    if not pairs_path.exists():
        raise FileNotFoundError("Run scripts/04_build_pairs_encode.py first")
    pairs = pd.read_parquet(pairs_path)
    if pairs.empty or "person_id" not in pairs.columns:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        warning = ["# ENCoDE Pairing Warning", "- No matched SaO2/SpO2 pairs found."]
        rules_path = reports_dir / "pairing_rules.json"
        if rules_path.exists():
            warning.append(f"- pairing_rules.json: {rules_path}")
        inspect_path = reports_dir / "encode_measurement_inspect.md"
        if inspect_path.exists():
            warning.append(f"- measurement inspection: {inspect_path}")
        (reports_dir / "encode_pairing_warning.md").write_text("\n".join(warning))
        (reports_dir / "encode_pairing_status.json").write_text(
            '{"ENCODE_PAIRING_AVAILABLE": false}'
        )
        out_dir = Path("results/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(
            columns=[
                "person_id",
                "visit_occurrence_id",
                "sao2_time",
                "sao2",
                "spo2_time",
                "spo2",
                "delta_minutes",
                "skintone_monk",
                "skintone_bin",
                "dataset",
                "error",
                "abs_error",
                "hidden_hypoxemia_T90",
                "hidden_hypoxemia_T92",
                "hidden_hypoxemia_T94",
            ]
        )
        empty.to_parquet(out_dir / "encode_analysis.parquet", index=False)
        print("WARN: encode_pairs.parquet empty; wrote empty encode_analysis.parquet")
        return

    concept_map = build_concept_map(tables["CONCEPT"])
    measurement_df = tables["MEASUREMENT"]
    skintone_measurements = extract_skintone_measurements(measurement_df, concept_map)
    skintone_raw, skintone_index = build_skintone_index(skintone_measurements)

    df = pairs.merge(skintone_index, on="person_id", how="left")
    df["dataset"] = "encode"
    df = add_error_columns(df, "sao2", "spo2")
    df = add_hidden_hypoxemia(df, [90, 92, 94])

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "encode_analysis.parquet", index=False)

    skintone_dir = Path("results/tables")
    skintone_dir.mkdir(parents=True, exist_ok=True)
    skintone_index.to_csv(skintone_dir / "skintone_distribution.csv", index=False)


if __name__ == "__main__":
    main()
