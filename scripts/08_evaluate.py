from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.models.baselines import baseline_predict
from src.models.evaluate import grouped_mae, mae, rmse
from src.models.train import split_dataframe


def _load_model(path: Path) -> Any:
    data = path.read_bytes()
    try:
        return pickle.loads(data)
    except Exception:
        return json.loads(data.decode("utf-8"))


def _predict(model: Any, df: pd.DataFrame) -> pd.Series:
    if isinstance(model, dict):
        if model.get("type") == "identity":
            return df["spo2"]
        if model.get("type") == "linear":
            return model.get("slope", 1.0) * df["spo2"] + model.get("intercept", 0.0)
        raise ValueError(f"Unknown model dict type: {model}")
    # sklearn-style
    if hasattr(model, "predict"):
        if df.shape[1] == 1 and "spo2" in df.columns:
            return pd.Series(model.predict(df["spo2"]), index=df.index)
        return pd.Series(model.predict(df[["spo2"]]), index=df.index)
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        return pd.Series(model.model.predict(df["spo2"]), index=df.index)
    raise ValueError("Unsupported model type")


def _metrics(df: pd.DataFrame, group_cols: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mae": mae(df["sao2"].values, df["sao2_hat"].values),
        "rmse": rmse(df["sao2"].values, df["sao2_hat"].values),
    }
    for col in group_cols:
        if col in df.columns:
            out[f"grouped_mae_{col}"] = grouped_mae(df, col)
    return out


def evaluate_file(path: Path, output_name: str) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    if df.empty:
        print(f"WARN: {path} is empty; skipping evaluation")
        return {}
    df["sao2_hat"] = baseline_predict(df)
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(df, group_cols=["race_ethnicity", "skintone_bin"])
    (out_dir / output_name).write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    bold_path = Path("results/metrics/bold_analysis.parquet")
    encode_path = Path("results/metrics/encode_analysis.parquet")
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_bold = {}
    baseline_encode = {}

    models_dir = Path("results/models")
    if not models_dir.exists():
        print("WARN: results/models not found; skipping model evaluation")
        return

    model_paths = {
        "isotonic": models_dir / "isotonic.json",
        "ridge": models_dir / "ridge.json",
    }
    models = {name: _load_model(path) for name, path in model_paths.items() if path.exists()}

    results_bold: Dict[str, Any] = {}
    results_encode: Dict[str, Any] = {}

    if bold_path.exists():
        bold_df = pd.read_parquet(bold_path)
        train_df, test_df = split_dataframe(bold_df, seed=1337)
        if not test_df.empty:
            test_df = test_df.copy()
            test_df["sao2_hat"] = baseline_predict(test_df)
            baseline_bold = _metrics(test_df, group_cols=["race_ethnicity"])
            (out_dir / "baseline_bold.json").write_text(json.dumps(baseline_bold, indent=2))

        if models:
            for name, model in models.items():
                eval_df = test_df.copy()
                eval_df["sao2_hat"] = _predict(model, eval_df)
                results_bold[name] = _metrics(eval_df, group_cols=["race_ethnicity"])

    if encode_path.exists():
        encode_df = pd.read_parquet(encode_path)
        if not encode_df.empty:
            eval_base = encode_df.copy()
            eval_base["sao2_hat"] = baseline_predict(eval_base)
            baseline_encode = _metrics(eval_base, group_cols=["skintone_bin"])
            (out_dir / "baseline_encode.json").write_text(json.dumps(baseline_encode, indent=2))

            if models:
                for name, model in models.items():
                    eval_df = encode_df.copy()
                    eval_df["sao2_hat"] = _predict(model, eval_df)
                    results_encode[name] = _metrics(eval_df, group_cols=["skintone_bin"])

    if results_bold:
        (out_dir / "models_bold.json").write_text(json.dumps(results_bold, indent=2))
    if results_encode:
        (out_dir / "models_encode.json").write_text(json.dumps(results_encode, indent=2))

    # Final summary for quick inspection
    summary: Dict[str, Any] = {
        "bold": {"baseline": baseline_bold, "models": results_bold},
        "encode": {"baseline": baseline_encode, "models": results_encode},
    }
    (out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
