from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from src.viz.figures import (
    bar_plot,
    bar_plot_ci,
    boxplot_by_group,
    calibration_curve,
    calibration_plot,
    histogram,
    line_plot,
    scatter_plot,
)
from src.viz.tables import write_table


def _load_model(path: Path):
    if not path.exists():
        return None
    data = path.read_bytes()
    try:
        return pickle.loads(data)
    except Exception:
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None


def _predict(model, df: pd.DataFrame) -> pd.Series:
    if model is None:
        return df["spo2"]
    if isinstance(model, dict):
        if model.get("type") == "identity":
            return df["spo2"]
        if model.get("type") == "linear":
            return model.get("slope", 1.0) * df["spo2"] + model.get("intercept", 0.0)
        raise ValueError(f"Unknown model dict type: {model}")
    if hasattr(model, "predict"):
        if df.shape[1] == 1 and "spo2" in df.columns:
            return pd.Series(model.predict(df["spo2"]), index=df.index)
        return pd.Series(model.predict(df[["spo2"]]), index=df.index)
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        return pd.Series(model.model.predict(df["spo2"]), index=df.index)
    raise ValueError("Unsupported model type")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _cohort_summary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    age_col = "age" if "age" in df.columns else "admission_age"
    sex_col = "sex_female" if "sex_female" in df.columns else "sex"
    age_mean = df[age_col].dropna().mean() if age_col in df.columns else float("nan")
    age_std = df[age_col].dropna().std() if age_col in df.columns else float("nan")
    if sex_col in df.columns:
        series = df[sex_col].dropna()
        if series.dtype == bool:
            female_rate = series.mean()
        elif pd.api.types.is_numeric_dtype(series):
            # Common encoding: 1 = female, 0 = male
            female_rate = (series.astype(float) > 0.5).mean()
        else:
            female_rate = (series.astype(str).str.lower() == "female").mean()
    else:
        female_rate = float("nan")
    race_counts = df["race_ethnicity"].value_counts(dropna=False).to_dict() if "race_ethnicity" in df.columns else {}
    skintone_counts = df["skintone_bin"].value_counts(dropna=False).to_dict() if "skintone_bin" in df.columns else {}

    def _format_counts(counts: dict, top_n: int = 5) -> str:
        if not counts:
            return ""
        items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        top = items[:top_n]
        rest = items[top_n:]
        parts = [f"{label}: {int(val)}" for label, val in top]
        if rest:
            other = int(sum(val for _, val in rest))
            parts.append(f"Other: {other}")
        return "; ".join(parts)
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n": int(len(df)),
                "age_mean": age_mean,
                "age_std": age_std,
                "female_rate": female_rate,
                "race_distribution": _format_counts(race_counts, top_n=6),
                "skintone_bin_distribution": _format_counts(skintone_counts, top_n=6),
            }
        ]
    )


def _group_counts(df: pd.DataFrame, group_col: str, dataset: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group, sub in df.groupby(group_col):
        n = int(len(sub))
        hypox = int((sub["sao2"] < 88).sum()) if "sao2" in sub.columns else 0
        hidden = int(sub["hidden_hypoxemia_T92"].sum()) if "hidden_hypoxemia_T92" in sub.columns else 0
        rows.append(
            {
                "dataset": dataset,
                "group_col": group_col,
                "group": str(group),
                "n": n,
                "hypoxemia_events": hypox,
                "hidden_hypoxemia_T92_events": hidden,
                "low_n": n < 30,
                "low_events": hypox < 10,
            }
        )
    return pd.DataFrame(rows)


def _extract_ci(metric: dict | None) -> tuple[float | None, float | None, float | None]:
    if not metric:
        return None, None, None
    return metric.get("mean"), metric.get("ci_low"), metric.get("ci_high")


def _table_row(name: str, metrics: dict, group_col: str, threshold: int) -> dict:
    mae_ci = metrics.get("mae_ci", {})
    rmse_ci = metrics.get("rmse_ci", {})
    gap_ci = metrics.get(f"grouped_mae_{group_col}_gap_ci", {})
    fnr_gap_ci = metrics.get(f"grouped_fnr_{group_col}_gap_ci", {})
    hh_gap_ci = metrics.get(f"grouped_hidden_hypoxemia_T{threshold}_{group_col}_gap_ci", {})
    hh_rate_ci = metrics.get(f"hidden_hypoxemia_T{threshold}_rate_ci", {})
    row = {
        "model": name,
        "mae": metrics.get("mae"),
        "mae_ci_low": mae_ci.get("ci_low"),
        "mae_ci_high": mae_ci.get("ci_high"),
        "rmse": metrics.get("rmse"),
        "rmse_ci_low": rmse_ci.get("ci_low"),
        "rmse_ci_high": rmse_ci.get("ci_high"),
        "ece": metrics.get("ece"),
        "ece_ci_low": metrics.get("ece_ci", {}).get("ci_low"),
        "ece_ci_high": metrics.get("ece_ci", {}).get("ci_high"),
        "mae_gap": gap_ci.get("mean"),
        "mae_gap_ci_low": gap_ci.get("ci_low"),
        "mae_gap_ci_high": gap_ci.get("ci_high"),
        "worst_group_mae": metrics.get(f"worst_group_mae_{group_col}", {}).get("value"),
        "hidden_hypoxemia_prevalence": metrics.get(f"hidden_hypoxemia_T{threshold}_rate"),
        "hidden_hypoxemia_prevalence_ci_low": hh_rate_ci.get("ci_low"),
        "hidden_hypoxemia_prevalence_ci_high": hh_rate_ci.get("ci_high"),
        "missed_hypoxemia_rate": metrics.get("missed_hypoxemia_rate"),
        "missed_hypoxemia_rate_ci_low": metrics.get("missed_hypoxemia_rate_ci", {}).get("ci_low"),
        "missed_hypoxemia_rate_ci_high": metrics.get("missed_hypoxemia_rate_ci", {}).get("ci_high"),
        "fnr_gap": fnr_gap_ci.get("mean"),
        "fnr_gap_ci_low": fnr_gap_ci.get("ci_low"),
        "fnr_gap_ci_high": fnr_gap_ci.get("ci_high"),
        f"hidden_hypoxemia_T{threshold}_gap": hh_gap_ci.get("mean"),
        f"hidden_hypoxemia_T{threshold}_gap_ci_low": hh_gap_ci.get("ci_low"),
        f"hidden_hypoxemia_T{threshold}_gap_ci_high": hh_gap_ci.get("ci_high"),
    }
    return row


def main() -> None:
    results_fig = Path("results/figures")
    results_tab = Path("results/tables")
    results_fig.mkdir(parents=True, exist_ok=True)
    results_tab.mkdir(parents=True, exist_ok=True)

    bold_path = Path("results/metrics/bold_analysis.parquet")
    encode_path = Path("results/metrics/encode_analysis.parquet")

    if bold_path.exists():
        df = pd.read_parquet(bold_path)
        histogram(df["error"], "BOLD Error Distribution", str(results_fig / "bold_error.png"))
        write_table(df.head(50), str(results_tab / "bold_preview.csv"))
        if "race_ethnicity" in df.columns:
            boxplot_by_group(
                df, "error", "race_ethnicity", "BOLD Error by Race/Ethnicity", str(results_fig / "bold_error_by_race.png")
            )
            race_counts = df["race_ethnicity"].value_counts()
            bar_plot(
                race_counts.index.astype(str).tolist(),
                race_counts.values.tolist(),
                "BOLD Race/Ethnicity Distribution",
                "Count",
                str(results_fig / "bold_race_distribution.png"),
            )

    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        if not df.empty and "error" in df.columns:
            histogram(
                df["error"],
                "ENCoDE Error Distribution",
                str(results_fig / "encode_error.png"),
            )
        if not df.empty:
            write_table(df.head(50), str(results_tab / "encode_preview.csv"))
        if "skintone_bin" in df.columns:
            boxplot_by_group(
                df,
                "error",
                "skintone_bin",
                "ENCoDE Error by Skin Tone Bin",
                str(results_fig / "encode_error_by_skintone.png"),
            )
            skin_counts = df["skintone_bin"].value_counts()
            bar_plot(
                skin_counts.index.astype(str).tolist(),
                skin_counts.values.tolist(),
                "ENCoDE Skin Tone Bin Distribution",
                "Count",
                str(results_fig / "encode_skintone_distribution.png"),
            )

    # Model metrics tables (legacy + overview)
    models_bold = Path("results/metrics/models_bold.json")
    models_encode = Path("results/metrics/models_encode.json")

    if models_bold.exists():
        data = json.loads(models_bold.read_text())
        rows = []
        baseline = _load_json(Path("results/metrics/baseline_bold.json"))
        if baseline:
            rows.append(
                {
                    "model": "baseline",
                    "mae": baseline.get("mae"),
                    "rmse": baseline.get("rmse"),
                    "race_mae_gap": baseline.get("grouped_mae_race_ethnicity_gap"),
                }
            )
        for name, metrics in data.items():
            race = metrics.get("grouped_mae_race_ethnicity", {})
            gap = None
            if race:
                vals = list(race.values())
                gap = max(vals) - min(vals)
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "race_mae_gap": gap,
                }
            )
        if rows:
            write_table(pd.DataFrame(rows), str(results_tab / "model_metrics_bold.csv"))

    if models_encode.exists():
        data = json.loads(models_encode.read_text())
        rows = []
        baseline = _load_json(Path("results/metrics/baseline_encode.json"))
        if baseline:
            rows.append(
                {
                    "model": "baseline",
                    "mae": baseline.get("mae"),
                    "rmse": baseline.get("rmse"),
                    "skintone_mae_gap": baseline.get("grouped_mae_skintone_bin_gap"),
                }
            )
        for name, metrics in data.items():
            skin = metrics.get("grouped_mae_skintone_bin", {})
            gap = None
            if skin:
                vals = list(skin.values())
                gap = max(vals) - min(vals)
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "skintone_mae_gap": gap,
                }
            )
        if rows:
            write_table(pd.DataFrame(rows), str(results_tab / "model_metrics_encode.csv"))

    # Submission-grade tables with CIs
    bold_rows = []
    encode_rows = []
    threshold = 92
    baseline_bold = _load_json(Path("results/metrics/baseline_bold.json"))
    baseline_encode = _load_json(Path("results/metrics/baseline_encode.json"))
    if baseline_bold:
        bold_rows.append(_table_row("baseline", baseline_bold, "race_ethnicity", threshold))
    if baseline_encode:
        encode_rows.append(_table_row("baseline", baseline_encode, "skintone_bin", threshold))

    for name in ["ridge", "ridge_reweighted", "isotonic_safe", "ridge_safe"]:
        bold_metrics = _load_json(Path(f"results/metrics/model_{name}_bold.json"))
        encode_metrics = _load_json(Path(f"results/metrics/model_{name}_encode.json"))
        if bold_metrics:
            bold_rows.append(_table_row(name, bold_metrics, "race_ethnicity", threshold))
        if encode_metrics:
            encode_rows.append(_table_row(name, encode_metrics, "skintone_bin", threshold))

    if bold_rows:
        bold_df = pd.DataFrame(bold_rows)
        rename_cols = {
            "model": "Model",
            "mae": "MAE",
            "mae_ci_low": "MAE CI Low",
            "mae_ci_high": "MAE CI High",
            "rmse": "RMSE",
            "rmse_ci_low": "RMSE CI Low",
            "rmse_ci_high": "RMSE CI High",
            "ece": "ECE",
            "ece_ci_low": "ECE CI Low",
            "ece_ci_high": "ECE CI High",
            "mae_gap": "MAE Gap (max--min)",
            "mae_gap_ci_low": "MAE Gap CI Low",
            "mae_gap_ci_high": "MAE Gap CI High",
            "worst_group_mae": "Worst-group MAE",
            "hidden_hypoxemia_prevalence": "Hidden Hypoxemia Prevalence (T92)",
            "hidden_hypoxemia_prevalence_ci_low": "HH Prev CI Low",
            "hidden_hypoxemia_prevalence_ci_high": "HH Prev CI High",
            "missed_hypoxemia_rate": "Missed Hypoxemia Rate",
            "missed_hypoxemia_rate_ci_low": "Missed Rate CI Low",
            "missed_hypoxemia_rate_ci_high": "Missed Rate CI High",
            "fnr_gap": "FNR Gap (max--min)",
            "fnr_gap_ci_low": "FNR Gap CI Low",
            "fnr_gap_ci_high": "FNR Gap CI High",
            "hidden_hypoxemia_T92_gap": "HH T92 Gap (max--min)",
            "hidden_hypoxemia_T92_gap_ci_low": "HH T92 Gap CI Low",
            "hidden_hypoxemia_T92_gap_ci_high": "HH T92 Gap CI High",
        }
        bold_df = bold_df.rename(columns=rename_cols)
        write_table(bold_df, str(results_tab / "table2_bold_models.csv"))
    if encode_rows:
        encode_df = pd.DataFrame(encode_rows)
        encode_df = encode_df.rename(columns=rename_cols)
        write_table(encode_df, str(results_tab / "table3_encode_models.csv"))

    # Tradeoff summary tables
    if bold_rows:
        tradeoff_bold = pd.DataFrame(
            [
                {
                    "model": row["model"],
                    "mae": row["mae"],
                    "mae_gap": row["mae_gap"],
                    "missed_hypoxemia_rate": row["missed_hypoxemia_rate"],
                }
                for row in bold_rows
            ]
        )
        write_table(tradeoff_bold, str(results_tab / "tradeoff_summary_bold.csv"))
    if encode_rows:
        tradeoff_encode = pd.DataFrame(
            [
                {
                    "model": row["model"],
                    "mae": row["mae"],
                    "mae_gap": row["mae_gap"],
                    "missed_hypoxemia_rate": row["missed_hypoxemia_rate"],
                }
                for row in encode_rows
            ]
        )
        write_table(tradeoff_encode, str(results_tab / "tradeoff_summary_encode.csv"))

    # Cohort summary table
    cohort_rows = []
    if bold_path.exists():
        cohort_rows.append(_cohort_summary(pd.read_parquet(bold_path), "bold"))
    if encode_path.exists():
        cohort_rows.append(_cohort_summary(pd.read_parquet(encode_path), "encode"))
    if cohort_rows:
        write_table(pd.concat(cohort_rows, ignore_index=True), str(results_tab / "table1_cohort.csv"))

    # Group counts / event counts (flag low-n)
    count_rows = []
    if bold_path.exists():
        df = pd.read_parquet(bold_path)
        count_rows.append(_group_counts(df, "race_ethnicity", "bold"))
    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        count_rows.append(_group_counts(df, "skintone_bin", "encode"))
    count_rows = [r for r in count_rows if not r.empty]
    if count_rows:
        write_table(pd.concat(count_rows, ignore_index=True), str(results_tab / "group_counts.csv"))

    # Repeated splits summary table (mean + 95% CI across seeds)
    repeated_path = Path("results/metrics/repeated_splits_bold.json")
    if repeated_path.exists():
        data = json.loads(repeated_path.read_text())
        if data:
            rows = []
            df_rep = pd.DataFrame(data)
            for model, sub in df_rep.groupby("model"):
                def _mean_ci(series):
                    vals = series.dropna().astype(float)
                    if len(vals) == 0:
                        return (None, None, None)
                    mean = vals.mean()
                    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    ci = 1.96 * std / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
                    return (mean, mean - ci, mean + ci)

                mae_m, mae_lo, mae_hi = _mean_ci(sub["mae"])
                gap_m, gap_lo, gap_hi = _mean_ci(sub["grouped_mae_race_ethnicity_gap"])
                miss_m, miss_lo, miss_hi = _mean_ci(sub["missed_hypoxemia_rate"])
                fnr_gap_m, fnr_gap_lo, fnr_gap_hi = _mean_ci(sub["grouped_fnr_race_ethnicity_gap"])
                ece_m, ece_lo, ece_hi = _mean_ci(sub["ece"])
                rows.append(
                    {
                        "model": model,
                        "mae_mean": mae_m,
                        "mae_ci_low": mae_lo,
                        "mae_ci_high": mae_hi,
                        "mae_gap_mean": gap_m,
                        "mae_gap_ci_low": gap_lo,
                        "mae_gap_ci_high": gap_hi,
                        "missed_hypoxemia_rate_mean": miss_m,
                        "missed_hypoxemia_rate_ci_low": miss_lo,
                        "missed_hypoxemia_rate_ci_high": miss_hi,
                        "fnr_gap_mean": fnr_gap_m,
                        "fnr_gap_ci_low": fnr_gap_lo,
                        "fnr_gap_ci_high": fnr_gap_hi,
                        "ece_mean": ece_m,
                        "ece_ci_low": ece_lo,
                        "ece_ci_high": ece_hi,
                        "pathological_fnr_any": bool(sub.get("pathological_fnr", False).any()),
                    }
                )
            write_table(pd.DataFrame(rows), str(results_tab / "repeated_splits_summary_bold.csv"))

    # Calibration curves (baseline vs isotonic if available)
    iso_model = _load_model(Path("results/models/isotonic.json"))

    if bold_path.exists():
        df = pd.read_parquet(bold_path)
        if not df.empty and "sao2" in df.columns and "spo2" in df.columns:
            curves = []
            pred_base, true_base = calibration_curve(df["sao2"], df["spo2"])
            curves.append((pred_base, true_base, "Baseline (SpO2)"))
            if iso_model is not None:
                pred = _predict(iso_model, df)
                pred_iso, true_iso = calibration_curve(df["sao2"], pred)
                curves.append((pred_iso, true_iso, "Isotonic"))
            calibration_plot(curves, "BOLD Calibration", str(results_fig / "bold_calibration.png"))

    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        if not df.empty and "sao2" in df.columns and "spo2" in df.columns:
            curves = []
            pred_base, true_base = calibration_curve(df["sao2"], df["spo2"])
            curves.append((pred_base, true_base, "Baseline (SpO2)"))
            if iso_model is not None:
                pred = _predict(iso_model, df)
                pred_iso, true_iso = calibration_curve(df["sao2"], pred)
                curves.append((pred_iso, true_iso, "Isotonic"))
            calibration_plot(curves, "ENCoDE Calibration", str(results_fig / "encode_calibration.png"))

    # Sensitivity: hidden hypoxemia rate vs threshold (overall)
    thresholds = [90, 92, 94]
    if bold_path.exists():
        df = pd.read_parquet(bold_path)
        rates = [float(df[f"hidden_hypoxemia_T{t}"].mean()) for t in thresholds if f"hidden_hypoxemia_T{t}" in df.columns]
        if rates:
            line_plot(thresholds[: len(rates)], [rates], ["BOLD"], "Hidden Hypoxemia Rate vs Threshold", "Rate", str(results_fig / "bold_hidden_hypoxemia_sweep.png"))

    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        rates = [float(df[f"hidden_hypoxemia_T{t}"].mean()) for t in thresholds if f"hidden_hypoxemia_T{t}" in df.columns]
        if rates:
            line_plot(thresholds[: len(rates)], [rates], ["ENCoDE"], "Hidden Hypoxemia Rate vs Threshold", "Rate", str(results_fig / "encode_hidden_hypoxemia_sweep.png"))

    # External validation gap summary (MAE gap by skin tone with CIs)
    if encode_rows:
        labels = [row["model"] for row in encode_rows]
        values = [row["mae_gap"] for row in encode_rows]
        ci_low = [row["mae_gap_ci_low"] for row in encode_rows]
        ci_high = [row["mae_gap_ci_high"] for row in encode_rows]
        if all(v is not None for v in values):
            bar_plot_ci(
                labels,
                values,
                ci_low,
                ci_high,
                "ENCoDE Skin Tone MAE Gap (T92)",
                "MAE Gap",
                str(results_fig / "encode_mae_gap_ci.png"),
            )

    if bold_rows:
        labels = [row["model"] for row in bold_rows]
        values = [row["fnr_gap"] for row in bold_rows]
        ci_low = [row["fnr_gap_ci_low"] for row in bold_rows]
        ci_high = [row["fnr_gap_ci_high"] for row in bold_rows]
        if all(v is not None for v in values):
            bar_plot_ci(
                labels,
                values,
                ci_low,
                ci_high,
                "BOLD FNR Gap by Race/Ethnicity",
                "FNR Gap",
                str(results_fig / "bold_fnr_gap_ci.png"),
            )

    if encode_rows:
        labels = [row["model"] for row in encode_rows]
        values = [row["fnr_gap"] for row in encode_rows]
        ci_low = [row["fnr_gap_ci_low"] for row in encode_rows]
        ci_high = [row["fnr_gap_ci_high"] for row in encode_rows]
        if all(v is not None for v in values):
            bar_plot_ci(
                labels,
                values,
                ci_low,
                ci_high,
                "ENCoDE FNR Gap by Skin Tone",
                "FNR Gap",
                str(results_fig / "encode_fnr_gap_ci.png"),
            )

    # Device / location sensitivity (ENCoDE)
    if encode_path.exists():
        df = pd.read_parquet(encode_path)
        if "device_folder" in df.columns:
            device_counts = df["device_folder"].value_counts().head(10)
            if not device_counts.empty:
                bar_plot(
                    device_counts.index.astype(str).tolist(),
                    device_counts.values.tolist(),
                    "ENCoDE Device Distribution",
                    "Count",
                    str(results_fig / "encode_device_distribution.png"),
                )
        if "device_folder" in df.columns and "error" in df.columns:
            device_mae = df.groupby("device_folder")["error"].apply(lambda x: x.abs().mean())
            if not device_mae.empty:
                bar_plot(
                    device_mae.index.astype(str).tolist(),
                    device_mae.values.tolist(),
                    "ENCoDE MAE by Device",
                    "MAE",
                    str(results_fig / "encode_mae_by_device.png"),
                )
        if "location_id" in df.columns and "error" in df.columns:
            top_locations = df["location_id"].value_counts().head(10).index
            subset = df[df["location_id"].isin(top_locations)]
            location_mae = subset.groupby("location_id")["error"].apply(lambda x: x.abs().mean())
            if not location_mae.empty:
                bar_plot(
                    location_mae.index.astype(str).tolist(),
                    location_mae.values.tolist(),
                    "ENCoDE MAE by Location (Top 10)",
                    "MAE",
                    str(results_fig / "encode_mae_by_location.png"),
                )

    # Tradeoff plots: MAE vs FNR
    def _tradeoff_plot(dataset: str, output_name: str, group_col: str) -> None:
        models = ["baseline", "ridge", "ridge_reweighted", "isotonic_safe", "ridge_safe"]
        xs = []
        ys = []
        labels = []
        for model in models:
            if model == "baseline":
                metrics = _load_json(Path(f"results/metrics/baseline_{dataset}.json"))
            else:
                metrics = _load_json(Path(f"results/metrics/model_{model}_{dataset}.json"))
            if not metrics:
                continue
            xs.append(metrics.get("mae"))
            ys.append(metrics.get("fnr"))
            labels.append(model)
        if xs and ys:
            scatter_plot(
                xs,
                ys,
                labels,
                f"{dataset.upper()} MAE vs FNR Tradeoff",
                "MAE",
                "FNR",
                str(results_fig / output_name),
            )

    _tradeoff_plot("bold", "bold_mae_fnr_tradeoff.png", "race_ethnicity")
    _tradeoff_plot("encode", "encode_mae_fnr_tradeoff.png", "skintone_bin")

if __name__ == "__main__":
    main()
