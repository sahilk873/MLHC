from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha, random_state=1337)
    model.fit(X, y)
    return model


def fit_ridge_reweighted(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, alpha: float = 1.0
) -> Ridge:
    weights = groups.copy()
    weights = weights.fillna("Unknown").astype(str)
    counts = weights.value_counts()
    sample_weight = weights.map(lambda g: 1.0 / counts.get(g, 1.0)).astype(float).values
    model = Ridge(alpha=alpha, random_state=1337)
    model.fit(X, y, sample_weight=sample_weight)
    return model
