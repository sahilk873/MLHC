from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd


SPO2_PATTERNS = [
    r"\bSPO2\b",
    r"PULSE\s*OX",
    r"OXIMETRY",
    r"O2\s*SAT.*(?:PULSE|OX)",
]

SAO2_PATTERNS = [
    r"\bSAO2\b",
    r"ARTERIAL.*O2",
    r"ABG.*(?:O2|SAT)",
    r"CO[- ]?OX",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "visit_occurrences_id" in df.columns and "visit_occurrence_id" not in df.columns:
        df = df.rename(columns={"visit_occurrences_id": "visit_occurrence_id"})
    return df


def build_measurement_datetime(df: pd.DataFrame) -> pd.Series:
    if "measurement_datetime" in df.columns:
        return pd.to_datetime(df["measurement_datetime"], errors="coerce")
    if "measurement_date" in df.columns:
        date = pd.to_datetime(df["measurement_date"], errors="coerce").dt.date
        if "measurement_time" in df.columns:
            time = pd.to_datetime(df["measurement_time"], errors="coerce").dt.time
            return pd.to_datetime(
                date.astype(str) + " " + time.astype(str), errors="coerce"
            )
        return pd.to_datetime(date.astype(str), errors="coerce")
    return pd.to_datetime(pd.NaT)


def _regex_any(patterns: list[str], series: pd.Series) -> pd.Series:
    combined = "|".join(patterns)
    return series.astype(str).str.contains(combined, case=False, na=False, regex=True)


def classify_measurements(
    df: pd.DataFrame, fallback: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    stats = {}
    source = df.get("measurement_source_value", pd.Series("", index=df.index)).astype(str)
    value_source = df.get("value_source_value", pd.Series("", index=df.index)).astype(str)
    combined_source = (source + "|" + value_source).astype(str)
    spo2_mask = _regex_any(SPO2_PATTERNS, combined_source)
    sao2_mask = _regex_any(SAO2_PATTERNS, combined_source)
    stats["spo2_regex_hits"] = int(spo2_mask.sum())
    stats["sao2_regex_hits"] = int(sao2_mask.sum())

    spo2 = df[spo2_mask].copy()
    sao2 = df[sao2_mask].copy()

    if (spo2.empty and sao2.empty) and fallback:
        values = pd.to_numeric(df.get("value_as_number"), errors="coerce")
        unit = df.get("unit_source_value", pd.Series("", index=df.index))
        unit_mask = unit.astype(str).str.contains(r"%|percent|percentage", case=False, na=False)
        range_mask = values.between(50, 105, inclusive="both")
        candidate = df[unit_mask & range_mask].copy()
        hints = candidate.get("measurement_source_value", pd.Series("", index=candidate.index)).astype(str)
        hint_values = candidate.get("value_source_value", pd.Series("", index=candidate.index)).astype(str)
        combined_hints = (hints + "|" + hint_values).astype(str)
        art_mask = combined_hints.str.contains(r"ART|ABG|CO[- ]?OX", case=False, na=False)
        pulse_mask = combined_hints.str.contains(r"PULSE|OX", case=False, na=False)
        sao2 = candidate[art_mask].copy()
        spo2 = candidate[pulse_mask & ~art_mask].copy()
        stats["fallback_candidates"] = int(len(candidate))
        stats["fallback_sao2"] = int(len(sao2))
        stats["fallback_spo2"] = int(len(spo2))

    return sao2, spo2, stats


def build_pairs(
    measurement_df: pd.DataFrame, restrict_range: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    df = normalize_columns(measurement_df)
    df["measurement_datetime"] = build_measurement_datetime(df)

    required = ["person_id", "measurement_datetime"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "visit_occurrence_id" not in df.columns:
        df["visit_occurrence_id"] = pd.NA

    sao2, spo2, stats = classify_measurements(df, fallback=True)

    if restrict_range:
        sao2 = sao2[
            pd.to_numeric(sao2.get("value_as_number"), errors="coerce").between(70, 100)
        ].copy()
        spo2 = spo2[
            pd.to_numeric(spo2.get("value_as_number"), errors="coerce").between(70, 100)
        ].copy()

    sao2 = sao2.dropna(subset=["measurement_datetime", "value_as_number"]).copy()
    spo2 = spo2.dropna(subset=["measurement_datetime", "value_as_number"]).copy()

    sao2 = sao2.rename(columns={"measurement_datetime": "sao2_time"})
    spo2 = spo2.rename(columns={"measurement_datetime": "spo2_time"})

    sao2 = sao2.sort_values(["person_id", "visit_occurrence_id", "sao2_time"])
    spo2 = spo2.sort_values(["person_id", "visit_occurrence_id", "spo2_time"])

    pairs = pd.merge_asof(
        sao2,
        spo2,
        by=["person_id", "visit_occurrence_id"],
        left_on="sao2_time",
        right_on="spo2_time",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
        suffixes=("_sao2", "_spo2"),
    )

    pairs = pairs.dropna(subset=["spo2_time", "value_as_number_spo2"]).copy()
    pairs["delta_minutes"] = (
        pairs["sao2_time"] - pairs["spo2_time"]
    ).dt.total_seconds() / 60.0
    pairs = pairs[(pairs["delta_minutes"] > 0) & (pairs["delta_minutes"] <= 5)].copy()

    output = pd.DataFrame(
        {
            "person_id": pairs["person_id"],
            "visit_occurrence_id": pairs["visit_occurrence_id"],
            "sao2_time": pairs["sao2_time"],
            "sao2": pairs["value_as_number_sao2"],
            "spo2_time": pairs["spo2_time"],
            "spo2": pairs["value_as_number_spo2"],
            "delta_minutes": pairs["delta_minutes"],
        }
    )

    output = output.sort_values(["person_id", "visit_occurrence_id", "sao2_time"])
    if output["visit_occurrence_id"].isna().all():
        first = output.groupby("person_id", as_index=False).first()
    else:
        first = output.groupby(
            ["person_id", "visit_occurrence_id"], as_index=False
        ).first()

    stats["pairs_all"] = int(len(output))
    stats["pairs_first_per_visit"] = int(len(first))
    return output, first, stats
