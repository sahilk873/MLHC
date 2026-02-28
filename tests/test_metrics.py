from __future__ import annotations

import pandas as pd

from src.data.harmonize import add_hidden_hypoxemia


def test_hidden_hypoxemia_thresholds() -> None:
    df = pd.DataFrame({"sao2": [87, 89], "spo2": [92, 91]})
    out = add_hidden_hypoxemia(df, [90, 92])
    assert bool(out.loc[0, "hidden_hypoxemia_T92"]) is True
    assert bool(out.loc[1, "hidden_hypoxemia_T92"]) is False
