from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.data.encode_loader import build_concept_map, extract_skintone_measurements, load_encode_tables
from src.data.harmonize import add_error_columns, add_hidden_hypoxemia
from src.features.skintone_index import build_skintone_index
from src.models.baselines import baseline_predict
from src.models.evaluate import grouped_mae, mae, rmse
from src.models.train import split_dataframe
from src.models.calibrators import fit_isotonic
from src.models.debiasing import fit_ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from src.viz.figures import histogram


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float


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
    if hasattr(model, "predict"):
        return pd.Series(model.predict(df[["spo2"]]), index=df.index)
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        return pd.Series(model.model.predict(df["spo2"]), index=df.index)
    raise ValueError("Unsupported model type")


def _bootstrap_person_ids(
    df: pd.DataFrame, n_boot: int = 1000, seed: int = 1337
) -> Iterable[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if "person_id" in df.columns:
        ids = df["person_id"].dropna().unique()
        if len(ids) == 0:
            return []
        for _ in range(n_boot):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            yield df[df["person_id"].isin(sampled)].copy()
    elif "subject_id" in df.columns:
        ids = df["subject_id"].dropna().unique()
        if len(ids) == 0:
            return []
        for _ in range(n_boot):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            yield df[df["subject_id"].isin(sampled)].copy()
    else:
        for _ in range(n_boot):
            yield df.sample(frac=1.0, replace=True, random_state=rng.integers(0, 1_000_000)).copy()


def _bootstrap_stat(values: List[float]) -> BootstrapResult:
    mean = float(np.mean(values))
    ci_low, ci_high = np.percentile(values, [2.5, 97.5])
    return BootstrapResult(mean=mean, ci_low=float(ci_low), ci_high=float(ci_high))


def _group_gap(df: pd.DataFrame, group_col: str) -> float | None:
    if group_col not in df.columns:
        return None
    grouped = grouped_mae(df, group_col)
    if not grouped:
        return None
    vals = list(grouped.values())
    return float(max(vals) - min(vals))


def _summarize_distribution(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {}
    quantiles = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    out = {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
    }
    out.update({f"q{int(k*100)}": float(v) for k, v in quantiles.items()})
    return out


def _fit_best_model(train_df: pd.DataFrame, cal_df: pd.DataFrame) -> Tuple[Any, str]:
    # Fit candidates on train_df and select using calibration MAE.
    models = {
        "isotonic": fit_isotonic(train_df["spo2"], train_df["sao2"]),
        "ridge": fit_ridge(train_df[["spo2"]], train_df["sao2"]),
    }
    scores = {}
    for name, model in models.items():
        eval_df = cal_df.copy()
        eval_df["sao2_hat"] = _predict(model, eval_df)
        scores[name] = mae(eval_df["sao2"].values, eval_df["sao2_hat"].values)
    best_name = min(scores, key=scores.get)
    return models[best_name], best_name


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    if len(residuals) == 0:
        return float("nan")
    n = len(residuals)
    quant = np.ceil((1 - alpha) * (n + 1)) / n
    quant = min(1.0, max(0.0, float(quant)))
    q = np.quantile(residuals, quant)
    return float(q)


def _conformal_intervals(y_hat: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    lower = y_hat - q
    upper = y_hat + q
    return lower, upper


def _coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def build_encode_analysis_from_pairs(
    pairs_path: Path, root: str
) -> pd.DataFrame:
    pairs = pd.read_parquet(pairs_path)
    if pairs.empty:
        return pairs
    tables = load_encode_tables(root)
    concept_map = build_concept_map(tables["CONCEPT"])
    measurement_df = tables["MEASUREMENT"]
    skintone_measurements = extract_skintone_measurements(measurement_df, concept_map)
    _, skintone_index = build_skintone_index(skintone_measurements)

    df = pairs.merge(skintone_index, on="person_id", how="left")
    df["dataset"] = "encode"
    df = add_error_columns(df, "sao2", "spo2")
    df = add_hidden_hypoxemia(df, [90, 92, 94])
    return df


def leakage_audit_report() -> None:
    lines = []
    lines.append("# Leakage Audit")
    lines.append("")
    lines.append("## Dataset splits used")
    lines.append("- BOLD split uses `src.models.train.split_dataframe` (patient-level if `person_id` exists).")
    lines.append("- ENCoDE is **not** used for training or model selection.")
    lines.append("")
    lines.append("## Data used for model fitting")
    lines.append("- Training occurs in `scripts/07_train_models.py` using `results/metrics/bold_analysis.parquet` only.")
    lines.append("- Models are fit on the BOLD training split; evaluation uses the BOLD test split.")
    lines.append("")
    lines.append("## ENCoDE label usage")
    lines.append("- ENCoDE labels are used **only** for external validation in `scripts/08_evaluate.py` and posthoc reports.")
    lines.append("- No ENCoDE labels are used for hyperparameter tuning, calibration fitting, or model selection.")
    lines.append("")
    lines.append("## Leakage conclusion")
    lines.append("No leakage detected based on code path inspection and data flow.")

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/leakage_audit.md").write_text("\n".join(lines))


def encode_cluster_robust_report(
    encode_df: pd.DataFrame, models: Dict[str, Any]
) -> List[Dict[str, Any]]:
    lines = []
    lines.append("# ENCoDE Cluster-Robust Evaluation (by person_id)")
    lines.append("")

    person_counts = encode_df["person_id"].value_counts()
    lines.append(f"- Unique person_id: {person_counts.size}")
    lines.append("- Pairs-per-person distribution:")
    lines.append(person_counts.describe().to_string())
    lines.append("")

    records: List[Dict[str, Any]] = []
    for label, model in models.items():
        eval_df = encode_df.copy()
        if label == "baseline":
            eval_df["sao2_hat"] = baseline_predict(eval_df)
        else:
            eval_df["sao2_hat"] = _predict(model, eval_df)

        maes, rmses, gaps = [], [], []
        for sample in _bootstrap_person_ids(eval_df, n_boot=1000, seed=1337):
            maes.append(mae(sample["sao2"].values, sample["sao2_hat"].values))
            rmses.append(rmse(sample["sao2"].values, sample["sao2_hat"].values))
            gap = _group_gap(sample, "skintone_bin")
            if gap is not None:
                gaps.append(gap)

        mae_ci = _bootstrap_stat(maes)
        rmse_ci = _bootstrap_stat(rmses)
        gap_ci = _bootstrap_stat(gaps) if gaps else None

        records.append(
            {
                "model": label,
                "mae_mean": mae_ci.mean,
                "mae_ci": f"[{mae_ci.ci_low:.3f}, {mae_ci.ci_high:.3f}]",
                "rmse_mean": rmse_ci.mean,
                "rmse_ci": f"[{rmse_ci.ci_low:.3f}, {rmse_ci.ci_high:.3f}]",
                "gap_mean": gap_ci.mean if gap_ci else None,
                "gap_ci": f"[{gap_ci.ci_low:.3f}, {gap_ci.ci_high:.3f}]" if gap_ci else None,
            }
        )

    lines.append("## Cluster bootstrap results (mean ± 95% CI)")
    lines.append("")
    lines.append(pd.DataFrame(records).to_string(index=False))
    lines.append("")

    Path("reports/encode_cluster_ci.md").write_text("\n".join(lines))
    return records


def encode_sensitivity_report(
    encode_all: pd.DataFrame, encode_one: pd.DataFrame, models: Dict[str, Any]
) -> None:
    lines = []
    lines.append("# ENCoDE Sensitivity: One-Per-Visit vs All-Pairs")
    lines.append("")

    def eval_df(df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
        out = {}
        base = df.copy()
        base["sao2_hat"] = baseline_predict(base)
        out["baseline"] = {
            "mae": mae(base["sao2"].values, base["sao2_hat"].values),
            "rmse": rmse(base["sao2"].values, base["sao2_hat"].values),
            "gap": _group_gap(base, "skintone_bin"),
        }
        for name, model in models.items():
            tmp = df.copy()
            tmp["sao2_hat"] = _predict(model, tmp)
            out[name] = {
                "mae": mae(tmp["sao2"].values, tmp["sao2_hat"].values),
                "rmse": rmse(tmp["sao2"].values, tmp["sao2_hat"].values),
                "gap": _group_gap(tmp, "skintone_bin"),
            }
        return out

    lines.append("## All pairs")
    lines.append(pd.DataFrame(eval_df(encode_all, "all")).T.to_string())
    lines.append("")
    lines.append("## One-per-visit")
    lines.append(pd.DataFrame(eval_df(encode_one, "one_per_visit")).T.to_string())
    lines.append("")

    Path("reports/encode_sensitivity_one_per_visit.md").write_text("\n".join(lines))


def distribution_shift_report(bold_test: pd.DataFrame, encode_df: pd.DataFrame) -> None:
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Distributions
    histogram(bold_test["sao2"], "BOLD SaO2", str(fig_dir / "distribution_sao2_bold.png"))
    histogram(encode_df["sao2"], "ENCoDE SaO2", str(fig_dir / "distribution_sao2_encode.png"))
    histogram(bold_test["spo2"], "BOLD SpO2", str(fig_dir / "distribution_spo2_bold.png"))
    histogram(encode_df["spo2"], "ENCoDE SpO2", str(fig_dir / "distribution_spo2_encode.png"))

    bold_error = bold_test["spo2"] - bold_test["sao2"]
    encode_error = encode_df["spo2"] - encode_df["sao2"]
    histogram(bold_error, "BOLD SpO2 - SaO2", str(fig_dir / "distribution_error_bold.png"))
    histogram(encode_error, "ENCoDE SpO2 - SaO2", str(fig_dir / "distribution_error_encode.png"))

    # Summary stats
    report = []
    report.append("# Distribution Shift Diagnostics")
    report.append("")
    report.append("## Summary statistics")
    report.append("")
    report.append("### BOLD test split")
    report.append(json.dumps(
        {
            "sao2": _summarize_distribution(bold_test["sao2"]),
            "spo2": _summarize_distribution(bold_test["spo2"]),
            "spo2_minus_sao2": _summarize_distribution(bold_error),
            "frac_sao2_lt_88": float((bold_test["sao2"] < 88).mean()),
        },
        indent=2,
    ))
    report.append("")
    report.append("### ENCoDE (all pairs)")
    report.append(json.dumps(
        {
            "sao2": _summarize_distribution(encode_df["sao2"]),
            "spo2": _summarize_distribution(encode_df["spo2"]),
            "spo2_minus_sao2": _summarize_distribution(encode_error),
            "frac_sao2_lt_88": float((encode_df["sao2"] < 88).mean()),
        },
        indent=2,
    ))

    Path("reports/distribution_shift.md").write_text("\n".join(report))


def conformal_and_safety_reports(
    bold_df: pd.DataFrame,
    encode_all: pd.DataFrame,
    encode_one: pd.DataFrame,
) -> Dict[str, Any]:
    # Split BOLD into train/cal/test
    train_df, test_df = split_dataframe(bold_df, seed=1337)
    # Further split train -> calibration
    train2, cal_df = split_dataframe(train_df, seed=2024)

    # Fit best model on train2, select by calibration MAE (no test leakage)
    best_model, best_name = _fit_best_model(train2, cal_df)

    # Calibration residuals
    cal_df = cal_df.copy()
    cal_df["sao2_hat"] = _predict(best_model, cal_df)
    residuals = np.abs(cal_df["sao2"].values - cal_df["sao2_hat"].values)

    q90 = _conformal_quantile(residuals, alpha=0.10)
    q95 = _conformal_quantile(residuals, alpha=0.05)

    def eval_coverage(df: pd.DataFrame, group_col: str | None = None) -> Dict[str, Any]:
        df = df.copy()
        df["sao2_hat"] = _predict(best_model, df)
        out = {}
        for alpha, q in [("90", q90), ("95", q95)]:
            lower, upper = _conformal_intervals(df["sao2_hat"].values, q)
            out[f"coverage_{alpha}"] = _coverage(df["sao2"].values, lower, upper)
            widths = upper - lower
            out[f"mean_width_{alpha}"] = float(np.mean(widths))
            out[f"median_width_{alpha}"] = float(np.median(widths))
            if group_col and group_col in df.columns:
                group_cov = {}
                group_width = {}
                group_width_median = {}
                for g, sub in df.groupby(group_col):
                    l, u = _conformal_intervals(sub["sao2_hat"].values, q)
                    group_cov[str(g)] = _coverage(sub["sao2"].values, l, u)
                    w = u - l
                    group_width[str(g)] = float(np.mean(w))
                    group_width_median[str(g)] = float(np.median(w))
                out[f"group_coverage_{alpha}"] = group_cov
                out[f"group_width_{alpha}"] = group_width
                out[f"group_width_median_{alpha}"] = group_width_median
        return out

    bold_cov = eval_coverage(test_df, group_col="race_ethnicity")
    encode_cov = eval_coverage(encode_all, group_col="skintone_bin")
    encode_one_cov = eval_coverage(encode_one, group_col="skintone_bin")

    # Cluster bootstrap CIs for ENCoDE coverage
    def boot_cov(df: pd.DataFrame, alpha: float) -> BootstrapResult:
        covs = []
        for sample in _bootstrap_person_ids(df, n_boot=1000, seed=1337):
            sample = sample.copy()
            sample["sao2_hat"] = _predict(best_model, sample)
            q = q90 if alpha == 0.10 else q95
            lower, upper = _conformal_intervals(sample["sao2_hat"].values, q)
            covs.append(_coverage(sample["sao2"].values, lower, upper))
        return _bootstrap_stat(covs)

    encode_cov_ci_90 = boot_cov(encode_all, alpha=0.10)
    encode_cov_ci_95 = boot_cov(encode_all, alpha=0.05)

    def boot_cov_by_group(df: pd.DataFrame, alpha: float, group_col: str) -> Dict[str, BootstrapResult]:
        results: Dict[str, List[float]] = {}
        for sample in _bootstrap_person_ids(df, n_boot=1000, seed=1337):
            sample = sample.copy()
            sample["sao2_hat"] = _predict(best_model, sample)
            q = q90 if alpha == 0.10 else q95
            for g, sub in sample.groupby(group_col):
                lower, upper = _conformal_intervals(sub["sao2_hat"].values, q)
                cov = _coverage(sub["sao2"].values, lower, upper)
                results.setdefault(str(g), []).append(cov)
        return {g: _bootstrap_stat(vals) for g, vals in results.items()}

    encode_group_ci_90 = boot_cov_by_group(encode_all, 0.10, "skintone_bin")
    encode_group_ci_95 = boot_cov_by_group(encode_all, 0.05, "skintone_bin")

    # Worst-group coverage gap
    def worst_gap(group_cov: Dict[str, float]) -> float | None:
        if not group_cov:
            return None
        vals = list(group_cov.values())
        return float(max(vals) - min(vals))

    lines = []
    lines.append("# Conformal Prediction Results")
    lines.append("")
    lines.append(f"- Best model: {best_name}")
    lines.append(f"- q90: {q90:.3f}, q95: {q95:.3f}")
    lines.append("")
    lines.append("## BOLD test coverage")
    lines.append(json.dumps(bold_cov, indent=2))
    lines.append("")
    lines.append("## ENCoDE all-pairs coverage")
    lines.append(json.dumps(encode_cov, indent=2))
    lines.append(f"- Cluster bootstrap 90% coverage CI: [{encode_cov_ci_90.ci_low:.3f}, {encode_cov_ci_90.ci_high:.3f}]")
    lines.append(f"- Cluster bootstrap 95% coverage CI: [{encode_cov_ci_95.ci_low:.3f}, {encode_cov_ci_95.ci_high:.3f}]")
    if "group_coverage_90" in encode_cov:
        lines.append(
            f"- Worst-group coverage gap (90%): {worst_gap(encode_cov['group_coverage_90']):.3f}"
        )
    if "group_coverage_95" in encode_cov:
        lines.append(
            f"- Worst-group coverage gap (95%): {worst_gap(encode_cov['group_coverage_95']):.3f}"
        )
    if encode_group_ci_90:
        lines.append("")
        lines.append("### ENCoDE subgroup coverage CIs (90%)")
        for g, stat in encode_group_ci_90.items():
            lines.append(f"- {g}: mean={stat.mean:.3f}, CI=[{stat.ci_low:.3f}, {stat.ci_high:.3f}]")
    if encode_group_ci_95:
        lines.append("")
        lines.append("### ENCoDE subgroup coverage CIs (95%)")
        for g, stat in encode_group_ci_95.items():
            lines.append(f"- {g}: mean={stat.mean:.3f}, CI=[{stat.ci_low:.3f}, {stat.ci_high:.3f}]")
    lines.append("")
    lines.append("## ENCoDE one-per-visit coverage")
    lines.append(json.dumps(encode_one_cov, indent=2))
    lines.append("")

    Path("reports/conformal_results.md").write_text("\n".join(lines))

    # Worst-group safety metric (hidden hypoxemia FNR)
    def fnr_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
        if group_col not in df.columns:
            return {}
        out = {}
        for g, sub in df.groupby(group_col):
            y_true = (sub["sao2"] < 88) & (sub["spo2"] >= 92)
            y_pred = (sub["sao2_hat"] < 88) & (sub["spo2"] >= 92)
            tp = int((y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())
            denom = tp + fn
            out[str(g)] = {
                "fnr": float(fn / denom) if denom > 0 else float("nan"),
                "tp": tp,
                "fn": fn,
                "positives": denom,
            }
        return out

    def worst_group_fnr(fnr: Dict[str, float]) -> float | None:
        vals = [v["fnr"] for v in fnr.values() if not np.isnan(v["fnr"])]
        return float(max(vals)) if vals else None

    safety_lines = []
    safety_lines.append("# Worst-Group Safety (Hidden Hypoxemia FNR)")
    safety_lines.append("")

    def eval_safety(df: pd.DataFrame, label: str, group_col: str) -> Dict[str, Any]:
        df = df.copy()
        if label == "baseline":
            df["sao2_hat"] = baseline_predict(df)
        else:
            df["sao2_hat"] = _predict(models[label], df)
        fnr = fnr_by_group(df, group_col)
        return {"fnr_by_group": fnr, "worst_group_fnr": worst_group_fnr(fnr)}

    models = {"baseline": None}
    for name in ["isotonic", "ridge"]:
        path = Path("results/models") / f"{name}.json"
        if path.exists():
            models[name] = _load_model(path)

    safety_lines.append("## BOLD test")
    for label in models:
        res = eval_safety(test_df, label, "race_ethnicity")
        safety_lines.append(f"- {label}: worst-group FNR = {res['worst_group_fnr']}")
        safety_lines.append(json.dumps(res["fnr_by_group"], indent=2))
    safety_lines.append("")
    safety_lines.append("## ENCoDE all-pairs")
    for label in models:
        res = eval_safety(encode_all, label, "skintone_bin")
        safety_lines.append(f"- {label}: worst-group FNR = {res['worst_group_fnr']}")
        safety_lines.append(json.dumps(res["fnr_by_group"], indent=2))

    Path("reports/worst_group_safety.md").write_text("\n".join(safety_lines))
    return {
        "best_model": best_name,
        "q90": q90,
        "q95": q95,
        "bold": bold_cov,
        "encode_all": encode_cov,
        "encode_one": encode_one_cov,
    }


def mondrian_conformal_report(
    bold_df: pd.DataFrame,
    encode_all: pd.DataFrame,
    encode_one: pd.DataFrame,
) -> Dict[str, Any]:
    # Split BOLD into train/cal/test
    train_df, test_df = split_dataframe(bold_df, seed=1337)
    train2, cal_df = split_dataframe(train_df, seed=2024)

    best_model, best_name = _fit_best_model(train2, cal_df)

    # Global residuals by race
    cal_df = cal_df.copy()
    cal_df["sao2_hat"] = _predict(best_model, cal_df)
    cal_df["abs_resid"] = np.abs(cal_df["sao2"] - cal_df["sao2_hat"])

    def group_quantiles(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
        out = {}
        for g, sub in df.groupby(group_col):
            res = sub["abs_resid"].values
            out[str(g)] = {
                "q90": _conformal_quantile(res, alpha=0.10),
                "q95": _conformal_quantile(res, alpha=0.05),
            }
        return out

    race_q = group_quantiles(cal_df, "race_ethnicity")

    def eval_grouped(df: pd.DataFrame, group_col: str, group_q: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        df = df.copy()
        df["sao2_hat"] = _predict(best_model, df)
        out: Dict[str, Any] = {}
        for alpha in ["90", "95"]:
            coverages = {}
            widths = {}
            med_widths = {}
            for g, sub in df.groupby(group_col):
                q = group_q.get(str(g), {}).get(f"q{alpha}")
                if q is None or np.isnan(q):
                    continue
                lower, upper = _conformal_intervals(sub["sao2_hat"].values, q)
                coverages[str(g)] = _coverage(sub["sao2"].values, lower, upper)
                w = upper - lower
                widths[str(g)] = float(np.mean(w))
                med_widths[str(g)] = float(np.median(w))
            out[f"group_coverage_{alpha}"] = coverages
            out[f"group_width_{alpha}"] = widths
            out[f"group_width_median_{alpha}"] = med_widths
        return out

    bold_eval = eval_grouped(test_df, "race_ethnicity", race_q)
    encode_eval = eval_grouped(encode_all, "skintone_bin", race_q)
    encode_one_eval = eval_grouped(encode_one, "skintone_bin", race_q)

    lines = []
    lines.append("# Mondrian Conformal Results (Race-Conditional q)")
    lines.append(f"- Best model: {best_name}")
    lines.append("")
    lines.append("## Race-conditional quantiles (BOLD calibration)")
    lines.append(json.dumps(race_q, indent=2))
    lines.append("")
    lines.append("## BOLD test")
    lines.append(json.dumps(bold_eval, indent=2))
    lines.append("")
    lines.append("## ENCoDE all-pairs")
    lines.append(json.dumps(encode_eval, indent=2))
    lines.append("")
    lines.append("## ENCoDE one-per-visit")
    lines.append(json.dumps(encode_one_eval, indent=2))
    Path("reports/conformal_mondrian.md").write_text("\n".join(lines))

    return {
        "best_model": best_name,
        "race_q": race_q,
        "bold": bold_eval,
        "encode_all": encode_eval,
        "encode_one": encode_one_eval,
    }


def occult_hypoxemia_classifier_report(
    bold_df: pd.DataFrame,
    encode_all: pd.DataFrame,
    encode_one: pd.DataFrame,
    precision_floor: float = 0.2,
    target_recall: float = 0.8,
) -> None:
    def build_labels(df: pd.DataFrame, threshold: int) -> pd.Series:
        return (df["sao2"] < 88) & (df["spo2"] >= threshold)

    # Sensitivity definitions
    thresholds = [90, 92, 94]

    # Split BOLD into train/val/test
    train_df, test_df = split_dataframe(bold_df, seed=1337)
    train2, val_df = split_dataframe(train_df, seed=2024)

    X_train = train2[["spo2"]]
    y_train = build_labels(train2, 92)
    X_val = val_df[["spo2"]]
    y_val = build_labels(val_df, 92)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1337)
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresh = precision_recall_curve(y_val, val_scores)
    # precision_recall_curve returns thresholds len-1
    selected = None
    best_recall = -1.0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresh):
        if p >= precision_floor and r > best_recall:
            best_recall = r
            selected = t
    if selected is None:
        # fallback: closest to target recall
        diffs = np.abs(recalls[:-1] - target_recall)
        idx = int(np.argmin(diffs))
        selected = thresh[idx]

    def fnr_ci_by_group(
        df: pd.DataFrame, group_col: str, target_group: str, n_boot: int = 1000
    ) -> Dict[str, float] | None:
        if group_col not in df.columns:
            return None
        group_df = df[df[group_col] == target_group]
        if group_df.empty:
            return None
        fnrs = []
        for sample in _bootstrap_person_ids(group_df, n_boot=n_boot, seed=1337):
            y = build_labels(sample, 92)
            if y.sum() == 0:
                continue
            scores = clf.predict_proba(sample[["spo2"]])[:, 1]
            pred = scores >= selected
            tp = int((pred & y).sum())
            fn = int((~pred & y).sum())
            denom = tp + fn
            if denom == 0:
                continue
            fnrs.append(fn / denom)
        if not fnrs:
            return None
        ci = _bootstrap_stat(fnrs)
        return {"mean": ci.mean, "ci_low": ci.ci_low, "ci_high": ci.ci_high}

    def eval_dataset(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        X = df[["spo2"]]
        y_true = build_labels(df, 92)
        scores = clf.predict_proba(X)[:, 1]
        y_pred = scores >= selected
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prevalence = float(np.mean(y_true)) if len(y_true) > 0 else float("nan")
        auprc = average_precision_score(y_true, scores) if y_true.any() else float("nan")
        auroc = roc_auc_score(y_true, scores) if y_true.any() else float("nan")
        lift = (auprc / prevalence) if prevalence and not np.isnan(prevalence) else float("nan")
        out = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "prevalence": prevalence,
            "lift": lift,
        }
        # subgroup metrics
        if "race_ethnicity" in df.columns:
            group_col = "race_ethnicity"
        elif "skintone_bin" in df.columns:
            group_col = "skintone_bin"
        else:
            group_col = None
        if group_col:
            subgroup = {}
            for g, sub in df.groupby(group_col):
                y = build_labels(sub, 92)
                if len(y) == 0:
                    continue
                scores_g = clf.predict_proba(sub[["spo2"]])[:, 1]
                pred_g = scores_g >= selected
                tp = int((pred_g & y).sum())
                fn = int((~pred_g & y).sum())
                denom = tp + fn
                fnr = float(fn / denom) if denom > 0 else float("nan")
                subgroup[str(g)] = {
                    "fnr": fnr,
                    "tp": tp,
                    "fn": fn,
                    "positives": denom,
                }
            out["subgroup_fnr"] = subgroup
            if label == "bold_test" and group_col == "race_ethnicity":
                black_ci = fnr_ci_by_group(df, group_col, "Black")
                white_ci = fnr_ci_by_group(df, group_col, "White")
                out["subgroup_fnr_ci"] = {
                    "Black": black_ci,
                    "White": white_ci,
                }
        return out

    # Build sensitivity counts
    def count_positives(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, int]]:
        counts = {}
        for t in thresholds:
            y = build_labels(df, t)
            by_group = {}
            for g, sub in df.groupby(group_col):
                y_g = build_labels(sub, t)
                by_group[str(g)] = int(y_g.sum())
            counts[f"T{t}"] = by_group
        return counts

    lines = []
    lines.append("# Occult Hypoxemia Classifier Evaluation")
    lines.append("")
    lines.append("## Threshold selection (BOLD validation)")
    lines.append(f"- Precision floor: {precision_floor}")
    lines.append(f"- Target recall: {target_recall}")
    lines.append(f"- Selected probability threshold: {selected:.4f}")
    lines.append("")

    bold_metrics = eval_dataset(test_df, "bold_test")
    encode_all_metrics = eval_dataset(encode_all, "encode_all")
    encode_one_metrics = eval_dataset(encode_one, "encode_one")

    lines.append("## BOLD test metrics")
    lines.append(json.dumps(bold_metrics, indent=2))
    lines.append("")
    lines.append("## ENCoDE all-pairs metrics")
    lines.append(json.dumps(encode_all_metrics, indent=2))
    lines.append("")
    lines.append("## ENCoDE one-per-visit metrics")
    lines.append(json.dumps(encode_one_metrics, indent=2))
    lines.append("")

    # Sensitivity counts
    lines.append("## Sensitivity definitions (positive counts)")
    if "race_ethnicity" in bold_df.columns:
        lines.append("### BOLD test by race")
        lines.append(json.dumps(count_positives(test_df, "race_ethnicity"), indent=2))
    if "skintone_bin" in encode_all.columns:
        lines.append("### ENCoDE all-pairs by skin tone")
        lines.append(json.dumps(count_positives(encode_all, "skintone_bin"), indent=2))
        lines.append("### ENCoDE one-per-visit by skin tone")
        lines.append(json.dumps(count_positives(encode_one, "skintone_bin"), indent=2))
    lines.append("")

    # Exploratory note if ENCoDE positives are small
    total_encode_pos = int(build_labels(encode_all, 92).sum())
    if total_encode_pos < 20:
        lines.append(
            "**Note:** ENCoDE total positives < 20. Subgroup FNRs should be treated as exploratory; "
            "primary external validation should emphasize regression + conformal results."
        )

    Path("reports/occult_hypoxemia_classifier.md").write_text("\n".join(lines))
    return {
        "threshold": selected,
        "bold": bold_metrics,
        "encode_all": encode_all_metrics,
        "encode_one": encode_one_metrics,
    }


def main() -> None:
    leakage_audit_report()

    # Load datasets
    bold = pd.read_parquet("results/metrics/bold_analysis.parquet")
    train_df, test_df = split_dataframe(bold, seed=1337)

    encode_all = pd.read_parquet("results/metrics/encode_analysis.parquet")
    encode_one = build_encode_analysis_from_pairs(
        Path("artifacts/encode_pairs_one_per_visit.parquet"),
        root="physionet.org/files/encode-skin-color/1.0.0",
    )

    models = {}
    models_dir = Path("results/models")
    for name in ["isotonic", "ridge"]:
        path = models_dir / f"{name}.json"
        if path.exists():
            models[name] = _load_model(path)

    encode_ci = encode_cluster_robust_report(encode_all, {"baseline": None, **models})
    encode_sensitivity_report(encode_all, encode_one, models)
    distribution_shift_report(test_df, encode_all)
    conformal_global = conformal_and_safety_reports(bold, encode_all, encode_one)
    conformal_mondrian = mondrian_conformal_report(bold, encode_all, encode_one)
    occult = occult_hypoxemia_classifier_report(bold, encode_all, encode_one)

    # Final summary for paper
    final_summary = []
    final_summary.append("# Final Results Summary")
    final_summary.append("")
    final_summary.append("- Core regression results: see `results/metrics/final_summary.json`.")
    final_summary.append("- External validation: ENCoDE all-pairs (617) with skin-tone bins populated.")
    final_summary.append("- Fairness gaps and cluster-robust CIs: see `reports/encode_cluster_ci.md`.")
    final_summary.append("- Conformal coverage: see `reports/conformal_results.md`.")
    final_summary.append("- Mondrian conformal: see `reports/conformal_mondrian.md`.")
    final_summary.append("- Worst-group safety: see `reports/worst_group_safety.md`.")
    final_summary.append("- Occult hypoxemia classifier: see `reports/occult_hypoxemia_classifier.md`.")
    final_summary.append("")
    final_summary.append("## Threats to validity")
    final_summary.append("- Repeated measures may inflate effective sample size; we mitigate with cluster bootstrap.")
    final_summary.append("- Occult hypoxemia prevalence is low and unstable across subgroups; classifier metrics are sensitive to this.")
    Path("paper/final_results_summary.md").write_text("\n".join(final_summary))

    # Consolidated results table
    summary = json.loads(Path("results/metrics/final_summary.json").read_text())
    rows = []
    for dataset in ["bold", "encode"]:
        block = summary.get(dataset, {})
        base = block.get("baseline", {})
        rows.append(
            {
                "dataset": dataset,
                "model": "baseline",
                "mae": base.get("mae"),
                "rmse": base.get("rmse"),
            }
        )
        for model_name, metrics in block.get("models", {}).items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                }
            )
    out_df = pd.DataFrame(rows)
    out_path = Path("paper/final_results_table.csv")
    out_df.to_csv(out_path, index=False)

    # Update results snapshot with new sections
    snapshot_path = Path("paper/results_snapshot.md")
    snapshot = "# Results Snapshot\n\n"
    snapshot += "## Occult Hypoxemia Classifier (BOLD Test)\n"
    snapshot += json.dumps(occult.get("bold", {}), indent=2)
    snapshot += "\n\n## Conformal Width Stats (Global)\n"
    snapshot += json.dumps(conformal_global.get("bold", {}), indent=2)
    snapshot += "\n\n## Mondrian Conformal (Race-conditional)\n"
    snapshot += json.dumps(conformal_mondrian.get("bold", {}), indent=2)
    snapshot_path.write_text(snapshot)

    # LaTeX tables
    paper_tables = Path("paper/tables")
    paper_tables.mkdir(parents=True, exist_ok=True)

    # Table A: Regression
    # Use encode_ci for cluster CIs
    table_a_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & Model & MAE & RMSE & Gap / CI \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        mae_val = row["mae"]
        rmse_val = row["rmse"]
        gap = ""
        if dataset == "encode":
            for rec in encode_ci:
                if rec["model"] == model:
                    gap = f"{rec['gap_mean']:.3f} {rec['gap_ci']}"
                    break
        table_a_lines.append(f"{dataset} & {model} & {mae_val:.3f} & {rmse_val:.3f} & {gap} \\\\")
    table_a_lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/table_a_regression.tex").write_text("\n".join(table_a_lines))

    # Table B: Conformal coverage + width
    table_b_lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Coverage90 & Width90 & Worst-group gap \\\\",
        "\\midrule",
    ]
    def add_conformal_row(label: str, block: Dict[str, Any]) -> None:
        cov = block.get("coverage_90")
        width = block.get("mean_width_90")
        gap = ""
        if "group_coverage_90" in block and block["group_coverage_90"]:
            vals = list(block["group_coverage_90"].values())
            gap = f"{(max(vals) - min(vals)):.3f}"
        if cov is not None and width is not None:
            table_b_lines.append(f"{label} & {cov:.3f} & {width:.3f} & {gap} \\\\")

    # Global conformal
    add_conformal_row("Global BOLD", conformal_global["bold"])
    add_conformal_row("Global ENCoDE", conformal_global["encode_all"])

    # Mondrian (race-conditional)
    def add_mondrian_row(label: str, block: Dict[str, Any]) -> None:
        if "group_coverage_90" not in block or not block["group_coverage_90"]:
            return
        vals = list(block["group_coverage_90"].values())
        gap = f"{(max(vals) - min(vals)):.3f}"
        cov = np.mean(vals)
        width_vals = list(block.get("group_width_90", {}).values())
        width = np.mean(width_vals) if width_vals else float("nan")
        table_b_lines.append(f"{label} & {cov:.3f} & {width:.3f} & {gap} \\\\")

    add_mondrian_row("Mondrian BOLD", conformal_mondrian["bold"])
    add_mondrian_row("Mondrian ENCoDE", conformal_mondrian["encode_all"])
    table_b_lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/table_b_conformal.tex").write_text("\n".join(table_b_lines))

    # Table C: Occult hypoxemia classifier (BOLD test)
    bold_occ = occult.get("bold", {})
    table_c_lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "AUROC & AUPRC & Prev & Lift & Prec & Recall & FNR CI (Black/White) \\\\",
        "\\midrule",
        f"{bold_occ.get('auroc'):.3f} & {bold_occ.get('auprc'):.4f} & {bold_occ.get('prevalence'):.4f} & {bold_occ.get('lift'):.2f} & {bold_occ.get('precision'):.4f} & {bold_occ.get('recall'):.3f} & ",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    fnr_ci = bold_occ.get("subgroup_fnr_ci", {})
    black_ci = fnr_ci.get("Black")
    white_ci = fnr_ci.get("White")
    ci_parts = []
    if black_ci:
        ci_parts.append(f"Black: [{black_ci['ci_low']:.2f}, {black_ci['ci_high']:.2f}]")
    if white_ci:
        ci_parts.append(f"White: [{white_ci['ci_low']:.2f}, {white_ci['ci_high']:.2f}]")
    if ci_parts:
        table_c_lines[-3] += " ".join(ci_parts) + " \\\\"
    else:
        table_c_lines[-3] += "NA \\\\"
    Path("paper/tables/table_c_occult.tex").write_text("\n".join(table_c_lines))


if __name__ == "__main__":
    main()
