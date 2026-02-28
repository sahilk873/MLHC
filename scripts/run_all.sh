#!/usr/bin/env bash
set -euo pipefail

# Avoid OpenMP shared memory permission issues on some clusters.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled
export KMP_INIT_AT_FORK=FALSE
export KMP_USE_SHM=0
export KMP_SHM_DISABLE=1
export OMP_DYNAMIC=FALSE
export PYTHONPATH="$(pwd)"
: "${SKIP_SKLEARN:=0}"
export SKIP_SKLEARN

ENV_VARS="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL NUMEXPR_NUM_THREADS=1 KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE KMP_USE_SHM=0 KMP_SHM_DISABLE=1 OMP_DYNAMIC=FALSE SKIP_SKLEARN=${SKIP_SKLEARN}"

ROOT="./physionet.org/files"
if [[ "${1:-}" == "--root" ]]; then
  ROOT="${2:-$ROOT}"
elif [[ "${1:-}" != "" ]]; then
  ROOT="$1"
fi

env ${ENV_VARS} python scripts/00_env_check.py
env ${ENV_VARS} python scripts/01_manifest.py --root "${ROOT}"
env ${ENV_VARS} python scripts/02_inspect_bold.py --root "${ROOT}/blood-gas-oximetry/1.0"
env ${ENV_VARS} python scripts/03_inspect_encode.py --root "${ROOT}/encode-skin-color/1.0.0"
env ${ENV_VARS} python scripts/04a_inspect_encode_measurements.py --root "${ROOT}/encode-skin-color/1.0.0"
env ${ENV_VARS} python scripts/04_build_pairs_encode.py --root "${ROOT}/encode-skin-color/1.0.0"
env ${ENV_VARS} python scripts/05_build_dataset_bold.py --root "${ROOT}/blood-gas-oximetry/1.0"
env ${ENV_VARS} python scripts/06_build_dataset_encode.py --root "${ROOT}/encode-skin-color/1.0.0"
if [[ "${SKIP_SKLEARN:-0}" == "1" ]]; then
  echo "WARN: SKIP_SKLEARN=1 set; skipping scripts/07_train_models.py"
else
  env ${ENV_VARS} python scripts/07_train_models.py
fi
env ${ENV_VARS} python scripts/08_evaluate.py
env ${ENV_VARS} python scripts/09_make_figures_tables.py
env ${ENV_VARS} python scripts/10_make_paper_assets.py
env ${ENV_VARS} python scripts/11_posthoc_reports.py
