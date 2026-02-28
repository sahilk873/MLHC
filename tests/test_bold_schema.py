from __future__ import annotations

import pandas as pd

from src.data.bold_loader import validate_bold_schema


def test_bold_schema_detection() -> None:
    df = pd.DataFrame({"SaO2": [95], "SpO2": [97]})
    sao2_col, spo2_col = validate_bold_schema(df)
    assert sao2_col.lower().startswith("sao2")
    assert spo2_col.lower().startswith("spo2")
