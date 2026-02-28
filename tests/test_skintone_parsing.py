from __future__ import annotations

import pandas as pd

from src.data.encode_loader import extract_skintone_measurements
from src.features.skintone_index import build_skintone_index, extract_monk_value, bin_monk_value


def test_extract_skintone_measurements() -> None:
    concept_map = {1: "SKINTONE@FOREARM__IPHONE.MONKSKINTONESCALE 5"}
    measurement_df = pd.DataFrame(
        {
            "person_id": [1],
            "visit_occurrence_id": [2],
            "measurement_datetime": ["2020-01-01"],
            "measurement_concept_id": [1],
            "value_as_number": [5],
        }
    )
    out = extract_skintone_measurements(measurement_df, concept_map)
    assert out.loc[0, "location"] == "FOREARM"
    assert out.loc[0, "device"] == "IPHONE"
    assert out.loc[0, "measure"].startswith("MONKSKINTONESCALE")


def test_monk_parsing_and_binning() -> None:
    value = extract_monk_value("MONKSKINTONESCALE 7")
    assert value == 7
    assert bin_monk_value(value) == "dark"


def test_build_skintone_index_uses_value_as_number() -> None:
    df = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "concept_name": ["X", "Y", "Z"],
            "measure": ["MONKSKINTONESCALE", "MONKSKINTONESCALE", "MONKSKINTONESCALE"],
            "value_as_number": [5, 6, 2],
        }
    )
    _, index = build_skintone_index(df)
    assert index.loc[index["person_id"] == 1, "skintone_monk"].iloc[0] == 6
    assert index.loc[index["person_id"] == 2, "skintone_bin"].iloc[0] == "light"
