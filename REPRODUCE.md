# Reproducibility Guide

## Setup
```
uv sync --all-extras
```

## Full Pipeline
```
bash scripts/run_all.sh --root ./physionet.org/files
```

If you hit OpenMP shared memory errors, the script already sets:
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `KMP_AFFINITY=disabled`, `KMP_INIT_AT_FORK=FALSE`.

## Expected Outputs
- `results/manifests/data_manifest.json`
- `results/metrics/bold_analysis.parquet`
- `results/metrics/encode_analysis.parquet`
- `artifacts/encode_pairs.parquet`
- `results/metrics/baseline_bold.json`
- `results/metrics/baseline_encode.json`
- `results/metrics/models_bold.json`
- `results/metrics/models_encode.json`
- `results/metrics/final_summary.json`
- `reports/leakage_audit.md`
- `reports/encode_cluster_ci.md`
- `reports/encode_sensitivity_one_per_visit.md`
- `reports/distribution_shift.md`
- `reports/conformal_results.md`
- `reports/worst_group_safety.md`
- `reports/figures/distribution_*.png`
- `paper/final_results_summary.md`
- `results/figures/*.png`
- `results/tables/*.csv`
- `paper/assets_manifest.json`
