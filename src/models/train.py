from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import os
import numpy as np
import pandas as pd


@dataclass
class TrainArtifacts:
    isotonic_path: Path
    ridge_path: Path
    reweighted_ridge_path: Path
    isotonic_safe_path: Path
    ridge_safe_path: Path
    split_path: Path


def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hyp = y_true < 88
    if hyp.sum() == 0:
        return float("nan")
    missed = (hyp) & (y_pred >= 88)
    return float(missed.sum() / hyp.sum())


def _find_offset(
    y_true: np.ndarray, y_pred: np.ndarray, target_fnr: float = 0.5, max_offset: float = 5.0
) -> float:
    best_offset = 0.0
    last_offset = 0.0
    for offset in np.arange(0.0, max_offset + 0.01, 0.1):
        fnr = _fnr(y_true, y_pred - offset)
        if np.isnan(fnr):
            return 0.0
        if fnr <= target_fnr:
            best_offset = float(offset)
            break
        last_offset = float(offset)
    return best_offset if best_offset > 0 else last_offset


def build_safe_model(base_model: object, train_df: pd.DataFrame, feature_cols: list[str]) -> object:
    from .calibrators import OffsetCalibrator

    if feature_cols == ["spo2"]:
        preds = base_model.predict(train_df[feature_cols])
    else:
        preds = base_model.predict(train_df[feature_cols])
    offset = _find_offset(train_df["sao2"].to_numpy(), np.asarray(preds))
    return OffsetCalibrator(base_model=base_model, offset=offset)


def fit_models_in_memory(train_df: pd.DataFrame) -> Dict[str, object]:
    from .calibrators import fit_isotonic
    from .debiasing import fit_ridge, fit_ridge_reweighted

    models: Dict[str, object] = {}
    iso = fit_isotonic(train_df["spo2"], train_df["sao2"])
    ridge = fit_ridge(train_df[["spo2"]], train_df["sao2"])
    if "race_ethnicity" in train_df.columns:
        ridge_rew = fit_ridge_reweighted(
            train_df[["spo2"]], train_df["sao2"], train_df["race_ethnicity"]
        )
    else:
        ridge_rew = ridge
    models["isotonic"] = iso
    models["ridge"] = ridge
    models["ridge_reweighted"] = ridge_rew
    models["isotonic_safe"] = build_safe_model(iso, train_df, ["spo2"])
    models["ridge_safe"] = build_safe_model(ridge, train_df, ["spo2"])
    return models


def _split_dataframe(df: pd.DataFrame, seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    id_cols = [
        "person_id",
        "subject_id",
        "unique_subject_id",
        "patient_id",
        "unique_hospital_admission_id",
        "hospital_admission_id",
    ]
    id_col = next((col for col in id_cols if col in df.columns), None)
    if id_col is not None:
        ids = df[id_col].dropna().unique()
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_ids = set(ids[:cut])
        train = df[df[id_col].isin(train_ids)].copy()
        test = df[~df[id_col].isin(train_ids)].copy()
        train.attrs["split_id_col"] = id_col
        test.attrs["split_id_col"] = id_col
    else:
        # Fallback: row-level split (leakage risk). Prefer adding IDs in preprocessing.
        mask = rng.random(len(df)) < 0.8
        train = df[mask].copy()
        test = df[~mask].copy()
        train.attrs["split_id_col"] = "row_random"
        test.attrs["split_id_col"] = "row_random"
    return train, test


def split_dataframe(df: pd.DataFrame, seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _split_dataframe(df, seed=seed)


def train_models(df: pd.DataFrame, output_dir: str) -> TrainArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _split_dataframe(df)
    split_info = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "id_col": train_df.attrs.get("split_id_col", "unknown"),
    }

    isotonic_path = output_path / "isotonic.json"
    ridge_path = output_path / "ridge.json"
    reweighted_ridge_path = output_path / "ridge_reweighted.json"
    isotonic_safe_path = output_path / "isotonic_safe.json"
    ridge_safe_path = output_path / "ridge_safe.json"
    split_path = output_path / "split.json"

    if os.environ.get("SKIP_SKLEARN", "0") == "1":
        # Simple linear fit as a safe fallback when OpenMP is unavailable.
        x = train_df["spo2"].to_numpy()
        y = train_df["sao2"].to_numpy()
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 1.0, 0.0
        isotonic_path.write_text(json.dumps({"type": "identity"}, indent=2))
        ridge_path.write_text(json.dumps({"type": "linear", "slope": slope, "intercept": intercept}, indent=2))
        reweighted_ridge_path.write_text(
            json.dumps({"type": "linear", "slope": slope, "intercept": intercept}, indent=2)
        )
        isotonic_safe_path.write_text(json.dumps({"type": "linear", "slope": slope, "intercept": intercept}, indent=2))
        ridge_safe_path.write_text(json.dumps({"type": "linear", "slope": slope, "intercept": intercept}, indent=2))
        split_path.write_text(json.dumps(split_info, indent=2))
    else:
        models = fit_models_in_memory(train_df)
        iso = models["isotonic"]
        ridge = models["ridge"]
        ridge_rew = models["ridge_reweighted"]
        isotonic_safe = models["isotonic_safe"]
        ridge_safe = models["ridge_safe"]
        isotonic_path.write_bytes(pickle.dumps(iso))
        ridge_path.write_bytes(pickle.dumps(ridge))
        reweighted_ridge_path.write_bytes(pickle.dumps(ridge_rew))
        isotonic_safe_path.write_bytes(pickle.dumps(isotonic_safe))
        ridge_safe_path.write_bytes(pickle.dumps(ridge_safe))
        split_path.write_text(json.dumps(split_info, indent=2))

    return TrainArtifacts(
        isotonic_path=isotonic_path,
        ridge_path=ridge_path,
        reweighted_ridge_path=reweighted_ridge_path,
        isotonic_safe_path=isotonic_safe_path,
        ridge_safe_path=ridge_safe_path,
        split_path=split_path,
    )
