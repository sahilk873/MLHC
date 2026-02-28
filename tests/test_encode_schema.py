from __future__ import annotations

import pandas as pd

from src.data.encode_loader import validate_encode_schema


def test_encode_schema_validation() -> None:
    tables = {
        "PERSON": pd.DataFrame({"person_id": [1]}),
        "VISIT_OCCURRENCE": pd.DataFrame({"visit_occurrence_id": [1], "person_id": [1]}),
        "MEASUREMENT": pd.DataFrame({"measurement_id": [1], "person_id": [1]}),
        "CONCEPT": pd.DataFrame({"concept_id": [1], "concept_name": ["SKINTONE@X__Y.M"]}),
    }
    validate_encode_schema(tables)
