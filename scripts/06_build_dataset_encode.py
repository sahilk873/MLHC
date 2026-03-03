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

    # Demographics from PERSON
    person = tables["PERSON"].copy()
    for col in [
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "gender_source_value",
        "race_source_value",
        "ethnicity_source_value",
        "year_of_birth",
        "month_of_birth",
        "day_of_birth",
    ]:
        if col not in person.columns:
            person[col] = pd.NA

    def _load_global_concepts(ids: set[int]) -> dict[int, str]:
        global_path = Path("CONCEPT.csv")
        if not global_path.exists() or not ids:
            return {}
        found: dict[int, str] = {}
        usecols = ["concept_id", "concept_name"]
        for chunk in pd.read_csv(global_path, sep="\t", usecols=usecols, chunksize=200000):
            match = chunk[chunk["concept_id"].isin(ids)]
            if not match.empty:
                found.update(dict(zip(match["concept_id"], match["concept_name"])))
            if len(found) >= len(ids):
                break
        return found

    def _concept_name(series: pd.Series) -> pd.Series:
        ids = set(series.dropna().astype(int).tolist())
        local_map = {k: v for k, v in concept_map.items() if k in ids}
        if len(local_map) < len(ids):
            local_map.update(_load_global_concepts(ids - set(local_map.keys())))
        return series.map(local_map).astype("object")

    person["gender_name"] = _concept_name(person["gender_concept_id"])
    person["race_name"] = _concept_name(person["race_concept_id"])
    person["ethnicity_name"] = _concept_name(person["ethnicity_concept_id"])

    def _pick_value(source: pd.Series, fallback: pd.Series) -> pd.Series:
        src = source.astype("object")
        fb = fallback.astype("object")
        return src.where(src.notna() & (src.astype(str).str.strip() != ""), fb)

    person["sex"] = _pick_value(person["gender_source_value"], person["gender_name"])
    person["race_ethnicity"] = _pick_value(person["race_source_value"], person["race_name"])
    ethnicity = _pick_value(person["ethnicity_source_value"], person["ethnicity_name"])
    person.loc[
        person["race_ethnicity"].isna() & ethnicity.notna(), "race_ethnicity"
    ] = ethnicity
    person.loc[
        person["race_ethnicity"].notna()
        & ethnicity.notna()
        & (ethnicity.astype(str).str.strip() != "")
        & (person["race_ethnicity"].astype(str) != ethnicity.astype(str)),
        "race_ethnicity",
    ] = person["race_ethnicity"].astype(str) + " / " + ethnicity.astype(str)

    person_demo = person[
        ["person_id", "sex", "race_ethnicity", "year_of_birth", "month_of_birth", "day_of_birth"]
    ].copy()

    df = df.merge(person_demo, on="person_id", how="left")

    # Age at SaO2 time (years)
    if "sao2_time" in df.columns and "year_of_birth" in df.columns:
        sao2_time = pd.to_datetime(df["sao2_time"], errors="coerce")
        birth_year = pd.to_numeric(df["year_of_birth"], errors="coerce")
        age = sao2_time.dt.year - birth_year
        if "month_of_birth" in df.columns and "day_of_birth" in df.columns:
            birth_month = pd.to_numeric(df["month_of_birth"], errors="coerce")
            birth_day = pd.to_numeric(df["day_of_birth"], errors="coerce")
            birthday = pd.to_datetime(
                {
                    "year": sao2_time.dt.year,
                    "month": birth_month.where(birth_month.notna(), 7),
                    "day": birth_day.where(birth_day.notna(), 1),
                },
                errors="coerce",
            )
            before_bday = sao2_time < birthday
            age = age - before_bday.astype(float)
        df["age"] = age
    image_index_path = Path("results/logs/encode_image_index.parquet")
    if image_index_path.exists():
        image_index = pd.read_parquet(image_index_path)
        if not image_index.empty and "person_id" in image_index.columns:
            image_index["person_id"] = image_index["person_id"].astype(pairs["person_id"].dtype)
            device_mode = (
                image_index.groupby("person_id")["device_folder"]
                .agg(lambda x: x.value_counts().idxmax())
                .rename("device_folder")
            )
            location_mode = (
                image_index.groupby("person_id")["location_id"]
                .agg(lambda x: x.value_counts().idxmax())
                .rename("location_id")
            )
            device_count = image_index.groupby("person_id")["device_folder"].nunique().rename(
                "device_folder_count"
            )
            location_count = image_index.groupby("person_id")["location_id"].nunique().rename(
                "location_id_count"
            )
            img_summary = pd.concat([device_mode, location_mode, device_count, location_count], axis=1)
            df = df.merge(img_summary, on="person_id", how="left")
    # Basic cleaning: drop missing/implausible oxygen saturation values
    df["sao2"] = pd.to_numeric(df["sao2"], errors="coerce")
    df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    df = df.dropna(subset=["sao2", "spo2"])
    df = df[(df["sao2"] >= 0) & (df["sao2"] <= 100)]
    df = df[(df["spo2"] >= 0) & (df["spo2"] <= 100)]

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
