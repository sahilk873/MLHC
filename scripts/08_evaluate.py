from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.models.baselines import baseline_predict
from src.models.evaluate import evaluate_dataset
from src.models.train import fit_models_in_memory, split_dataframe


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


def evaluate_file(path: Path, output_name: str, group_cols: list[str]) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    if df.empty:
        print(f"WARN: {path} is empty; skipping evaluation")
        return {}
    df = df.copy()
    df["sao2_hat"] = baseline_predict(df)
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_dataset(df, str(out_dir / output_name), group_cols, thresholds=[90, 92, 94])
    return metrics


def main() -> None:
    bold_path = Path("results/metrics/bold_analysis.parquet")
    encode_path = Path("results/metrics/encode_analysis.parquet")
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_bold: Dict[str, Any] = {}
    baseline_encode: Dict[str, Any] = {}

    models_dir = Path("results/models")
    if not models_dir.exists():
        print("WARN: results/models not found; skipping model evaluation")
        return

    model_paths = {
        "isotonic": models_dir / "isotonic.json",
        "ridge": models_dir / "ridge.json",
        "ridge_reweighted": models_dir / "ridge_reweighted.json",
        "isotonic_safe": models_dir / "isotonic_safe.json",
        "ridge_safe": models_dir / "ridge_safe.json",
    }
    models = {name: _load_model(path) for name, path in model_paths.items() if path.exists()}

    results_bold: Dict[str, Any] = {}
    results_encode: Dict[str, Any] = {}

    min_group_n = 30
    if bold_path.exists():
        bold_df = pd.read_parquet(bold_path)
        train_df, test_df = split_dataframe(bold_df, seed=1337)
        if not test_df.empty:
            test_df = test_df.copy()
            test_df["sao2_hat"] = baseline_predict(test_df)
            baseline_bold = evaluate_dataset(
                test_df,
                str(out_dir / "baseline_bold.json"),
                group_cols=["race_ethnicity"],
                thresholds=[90, 92, 94],
                min_group_n=min_group_n,
            )

        if models:
            for name, model in models.items():
                eval_df = test_df.copy()
                eval_df["sao2_hat"] = _predict(model, eval_df)
                results_bold[name] = evaluate_dataset(
                    eval_df,
                    str(out_dir / f"model_{name}_bold.json"),
                    group_cols=["race_ethnicity"],
                    thresholds=[90, 92, 94],
                    min_group_n=min_group_n,
                )

    if encode_path.exists():
        encode_df = pd.read_parquet(encode_path)
        if not encode_df.empty:
            eval_base = encode_df.copy()
            eval_base["sao2_hat"] = baseline_predict(eval_base)
            baseline_encode = evaluate_dataset(
                eval_base,
                str(out_dir / "baseline_encode.json"),
                group_cols=["skintone_bin", "race_ethnicity", "device_folder", "location_id"],
                thresholds=[90, 92, 94],
                min_group_n=min_group_n,
            )

            if models:
                for name, model in models.items():
                    eval_df = encode_df.copy()
                    eval_df["sao2_hat"] = _predict(model, eval_df)
                    results_encode[name] = evaluate_dataset(
                        eval_df,
                        str(out_dir / f"model_{name}_encode.json"),
                        group_cols=["skintone_bin", "race_ethnicity", "device_folder", "location_id"],
                        thresholds=[90, 92, 94],
                        min_group_n=min_group_n,
                    )

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

    # Repeated splits summary (robustness)
    if os.environ.get("SKIP_REPEATED_SPLITS", "0") != "1":
        seeds = [7, 11, 19, 29, 43]
        if bold_path.exists():
            bold_df = pd.read_parquet(bold_path)
            repeated = []
            for seed in seeds:
                train_df, test_df = split_dataframe(bold_df, seed=seed)
                if test_df.empty:
                    continue
                models_in_mem = fit_models_in_memory(train_df)
                # baseline
                test_df = test_df.copy()
                test_df["sao2_hat"] = baseline_predict(test_df)
                base_metrics = evaluate_dataset(
                    test_df,
                    str(out_dir / f"repeated_baseline_bold_{seed}.json"),
                    ["race_ethnicity"],
                    [90, 92, 94],
                    min_group_n=min_group_n,
                )
                base_metrics["seed"] = seed
                base_metrics["model"] = "baseline"
                repeated.append(base_metrics)
                for name, model in models_in_mem.items():
                    eval_df = test_df.copy()
                    eval_df["sao2_hat"] = _predict(model, eval_df)
                    metrics = evaluate_dataset(
                        eval_df,
                        str(out_dir / f"repeated_{name}_bold_{seed}.json"),
                        ["race_ethnicity"],
                        [90, 92, 94],
                        min_group_n=min_group_n,
                    )
                    # flag pathological FNR
                    metrics["pathological_fnr"] = bool(metrics.get("fnr", 0.0) >= 0.99)
                    metrics["seed"] = seed
                    metrics["model"] = name
                    repeated.append(metrics)
            if repeated:
                (out_dir / "repeated_splits_bold.json").write_text(json.dumps(repeated, indent=2))


if __name__ == "__main__":
    main()
