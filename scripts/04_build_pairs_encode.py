from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.data.encode_loader import load_encode_tables, validate_encode_schema


SAO2_CONCEPT_IDS = {
    3016502,  # Oxygen saturation in Arterial blood (LOINC)
    3039426,  # Oxygen saturation calculated from PaO2 (LOINC)
    3013502,  # Oxygen saturation in Blood (LOINC)
}
SPO2_CONCEPT_IDS = {
    4196147,   # Peripheral oxygen saturation (SNOMED)
    40762499, # Oxygen saturation in Arterial blood by Pulse oximetry (LOINC)
    40762508, # Oxygen saturation in Arterial blood by Pulse oximetry -- resting
    40762509, # Oxygen saturation in Blood Postductal by Pulse oximetry
    3024385,  # Deprecated Oxygen saturation in Capillary blood by Oximetry
}
TIME_WINDOW_MINUTES = 5


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_archive_legacy(artifacts_dir: Path) -> None:
    legacy_path = artifacts_dir / "encode_pairs_legacy.parquet"
    if legacy_path.exists():
        return
    candidates = [
        artifacts_dir / "encode_pairs.parquet",
        Path("results/metrics/encode_pairs.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            _ensure_dir(artifacts_dir)
            shutil.copy2(candidate, legacy_path)
            return


def _build_pairs(measurement_df: pd.DataFrame) -> pd.DataFrame:
    df = measurement_df.copy()
    required = {"person_id", "measurement_concept_id", "measurement_datetime", "value_as_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in MEASUREMENT: {sorted(missing)}")

    df = df[
        df["measurement_concept_id"].isin(SAO2_CONCEPT_IDS | SPO2_CONCEPT_IDS)
    ].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "visit_occurrence_id",
                "sao2_time",
                "sao2",
                "spo2_time",
                "spo2",
                "delta_minutes",
            ]
        )

    df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"], errors="coerce")
    df = df.dropna(subset=["measurement_datetime", "value_as_number"]).copy()
    if "visit_occurrence_id" not in df.columns:
        df["visit_occurrence_id"] = pd.NA

    sao2 = df[df["measurement_concept_id"].isin(SAO2_CONCEPT_IDS)].copy()
    spo2 = df[df["measurement_concept_id"].isin(SPO2_CONCEPT_IDS)].copy()

    if sao2.empty or spo2.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "visit_occurrence_id",
                "sao2_time",
                "sao2",
                "spo2_time",
                "spo2",
                "delta_minutes",
            ]
        )

    sao2 = sao2.rename(
        columns={
            "measurement_datetime": "sao2_time",
            "value_as_number": "sao2",
        }
    )
    spo2 = spo2.rename(
        columns={
            "measurement_datetime": "spo2_time",
            "value_as_number": "spo2",
        }
    )

    sao2["visit_key"] = pd.to_numeric(sao2["visit_occurrence_id"], errors="coerce").fillna(-1)
    spo2["visit_key"] = pd.to_numeric(spo2["visit_occurrence_id"], errors="coerce").fillna(-1)

    sao2 = sao2.sort_values(["sao2_time", "person_id", "visit_key"]).reset_index(drop=True)
    spo2 = spo2.sort_values(["spo2_time", "person_id", "visit_key"]).reset_index(drop=True)

    pairs = pd.merge_asof(
        sao2,
        spo2,
        by=["person_id", "visit_key"],
        left_on="sao2_time",
        right_on="spo2_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=TIME_WINDOW_MINUTES),
        suffixes=("_sao2", "_spo2"),
    )

    pairs = pairs.dropna(subset=["spo2_time", "spo2"]).copy()
    pairs["delta_minutes"] = (pairs["sao2_time"] - pairs["spo2_time"]).dt.total_seconds() / 60.0
    pairs = pairs[(pairs["delta_minutes"] >= 0) & (pairs["delta_minutes"] <= TIME_WINDOW_MINUTES)].copy()

    output = pairs[
        [
            "person_id",
            "visit_occurrence_id_sao2",
            "sao2_time",
            "sao2",
            "spo2_time",
            "spo2",
            "delta_minutes",
        ]
    ].rename(columns={"visit_occurrence_id_sao2": "visit_occurrence_id"})

    return output.reset_index(drop=True)


def _one_per_visit(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    if pairs["visit_occurrence_id"].isna().all():
        return (
            pairs.sort_values(["person_id", "sao2_time"])
            .groupby("person_id", as_index=False)
            .first()
        )
    return (
        pairs.sort_values(["person_id", "visit_occurrence_id", "sao2_time"])
        .groupby(["person_id", "visit_occurrence_id"], as_index=False)
        .first()
    )


def _write_report(report_path: Path, pairs_all: pd.DataFrame, pairs_first: pd.DataFrame) -> None:
    lines = []
    lines.append("# ENCoDE Pairing (Concept IDs)")
    lines.append("")
    lines.append(f"- SaO2 concept_ids: `{sorted(SAO2_CONCEPT_IDS)}`")
    lines.append(f"- SpO2 concept_ids: `{sorted(SPO2_CONCEPT_IDS)}`")
    lines.append(f"- Time window: 0–{TIME_WINDOW_MINUTES} minutes (SpO2 preceding SaO2)")
    lines.append("")

    if pairs_all.empty:
        lines.append("**WARNING:** Zero pairs found with the specified concept IDs.")
        lines.append("")
    lines.append(f"- Total pairs: {len(pairs_all)}")
    lines.append(f"- One per visit (or person if visit missing): {len(pairs_first)}")
    lines.append("")

    if not pairs_all.empty:
        delta = pairs_all["delta_minutes"].describe()
        lines.append("**Delta minutes (SpO2 -> SaO2)**")
        lines.append("")
        lines.append(delta.to_string())
        lines.append("")

        missing_visit = int(pairs_all["visit_occurrence_id"].isna().sum())
        lines.append("**Missingness**")
        lines.append("")
        lines.append(f"- Missing visit_occurrence_id: {missing_visit} / {len(pairs_all)}")
        lines.append("")

        lines.append("**Uniqueness / density**")
        lines.append("")
        per_person = pairs_all.groupby("person_id").size()
        lines.append(
            f"- Pairs per person: mean={per_person.mean():.2f}, "
            f"median={per_person.median():.2f}, max={per_person.max():.0f}"
        )
        if pairs_all["visit_occurrence_id"].isna().all():
            lines.append("- Visit IDs missing; per-visit stats not computed.")
        else:
            per_visit = pairs_all.dropna(subset=["visit_occurrence_id"]).groupby(
                ["person_id", "visit_occurrence_id"]
            ).size()
            lines.append(
                f"- Pairs per visit: mean={per_visit.mean():.2f}, "
                f"median={per_visit.median():.2f}, max={per_visit.max():.0f}"
            )

        lines.append("")

    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    tables = load_encode_tables(args.root)
    validate_encode_schema(tables)
    measurement_df = tables["MEASUREMENT"].copy()

    pairs_all = _build_pairs(measurement_df)
    pairs_first = _one_per_visit(pairs_all)

    artifacts_dir = Path("artifacts")
    _ensure_dir(artifacts_dir)
    _safe_archive_legacy(artifacts_dir)

    pairs_all.to_parquet(artifacts_dir / "encode_pairs_concept.parquet", index=False)
    pairs_first.to_parquet(
        artifacts_dir / "encode_pairs_one_per_visit_concept.parquet", index=False
    )

    pairs_all.to_parquet(artifacts_dir / "encode_pairs.parquet", index=False)
    pairs_first.to_parquet(artifacts_dir / "encode_pairs_one_per_visit.parquet", index=False)

    reports_dir = Path("reports")
    _ensure_dir(reports_dir)
    _write_report(reports_dir / "04_encode_pairing.md", pairs_all, pairs_first)


if __name__ == "__main__":
    main()
