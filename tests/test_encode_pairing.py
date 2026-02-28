from __future__ import annotations

import pandas as pd

from src.data.encode_pairing import build_pairs


def test_pairing_logic() -> None:
    df = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "visit_occurrence_id": [10, 10, 10],
            "measurement_datetime": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:03:00",
                "2020-01-01 00:04:00",
            ],
            "measurement_source_value": ["SpO2", "SaO2", "SpO2"],
            "value_as_number": [95, 90, 96],
        }
    )
    pairs, first, stats = build_pairs(df, restrict_range=True)
    assert len(pairs) == 1
    assert abs(pairs.loc[0, "delta_minutes"] - 3.0) < 1e-6
    assert len(first) == 1
