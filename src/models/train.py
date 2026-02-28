from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import os
import numpy as np
import pandas as pd


@dataclass
class TrainArtifacts:
    isotonic_path: Path
    ridge_path: Path
    split_path: Path


def _split_dataframe(df: pd.DataFrame, seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if "person_id" in df.columns:
        ids = df["person_id"].dropna().unique()
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_ids = set(ids[:cut])
        train = df[df["person_id"].isin(train_ids)].copy()
        test = df[~df["person_id"].isin(train_ids)].copy()
    else:
        mask = rng.random(len(df)) < 0.8
        train = df[mask].copy()
        test = df[~mask].copy()
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
    }

    isotonic_path = output_path / "isotonic.json"
    ridge_path = output_path / "ridge.json"
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
        split_path.write_text(json.dumps(split_info, indent=2))
    else:
        from .calibrators import fit_isotonic
        from .debiasing import fit_ridge

        iso = fit_isotonic(train_df["spo2"], train_df["sao2"])
        ridge = fit_ridge(train_df[["spo2"]], train_df["sao2"])
        isotonic_path.write_bytes(pickle.dumps(iso))
        ridge_path.write_bytes(pickle.dumps(ridge))
        split_path.write_text(json.dumps(split_info, indent=2))

    return TrainArtifacts(
        isotonic_path=isotonic_path,
        ridge_path=ridge_path,
        split_path=split_path,
    )
