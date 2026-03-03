from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression

    def predict(self, x: pd.Series) -> np.ndarray:
        return self.model.predict(x)


@dataclass
class OffsetCalibrator:
    base_model: object
    offset: float

    def predict(self, x: pd.Series) -> np.ndarray:
        if hasattr(self.base_model, "predict"):
            preds = self.base_model.predict(x)
        else:
            raise ValueError("Base model must implement predict")
        return np.asarray(preds) - float(self.offset)


def fit_isotonic(x: pd.Series, y: pd.Series) -> IsotonicCalibrator:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x.values, y.values)
    return IsotonicCalibrator(model=model)
