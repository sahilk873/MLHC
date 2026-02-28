from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from .schemas import ENCODE_TABLES, OMOP_KEY_COLUMNS


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_encode_tables(root: str) -> Dict[str, pd.DataFrame]:
    root_path = Path(root)
    tables: Dict[str, pd.DataFrame] = {}
    for name in ENCODE_TABLES:
        candidate = root_path / f"{name}.csv"
        if candidate.exists():
            tables[name] = _load_csv(candidate)
    return tables


def validate_encode_schema(tables: Dict[str, pd.DataFrame]) -> None:
    missing = [name for name in ENCODE_TABLES if name not in tables]
    if missing:
        raise FileNotFoundError(f"Missing ENCoDE tables: {missing}")
    for name, df in tables.items():
        required = OMOP_KEY_COLUMNS.get(name, [])
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {name}")


def build_concept_map(concept_df: pd.DataFrame) -> Dict[int, str]:
    return dict(zip(concept_df["concept_id"], concept_df["concept_name"]))


def _parse_concept_name(concept_name: str) -> Dict[str, str]:
    # Expected pattern: SKINTONE@<LOCATION>__<DEVICE>.<MEASURE>
    match = re.search(
        r"SKINTONE@(?P<location>[^_]+)__(?P<device>[^.]+)\.(?P<measure>.+)",
        concept_name or "",
    )
    if not match:
        return {"location": None, "device": None, "measure": None}
    return match.groupdict()


def extract_skintone_measurements(
    measurement_df: pd.DataFrame, concept_map: Dict[int, str]
) -> pd.DataFrame:
    df = measurement_df.copy()
    df["concept_name"] = df["measurement_concept_id"].map(concept_map)
    df = df[df["concept_name"].notna()].copy()
    parsed = df["concept_name"].apply(_parse_concept_name)
    parsed_df = pd.DataFrame(parsed.tolist()).reindex(columns=["location", "device", "measure"])
    df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    df = df[df["measure"].notna()].copy()
    keep_cols = [
        "person_id",
        "visit_occurrence_id",
        "measurement_datetime",
        "concept_name",
        "value_as_number",
        "location",
        "device",
        "measure",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep_cols]
