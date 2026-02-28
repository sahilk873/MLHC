from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha, random_state=1337)
    model.fit(X, y)
    return model
