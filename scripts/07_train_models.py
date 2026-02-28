from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

from pathlib import Path

import pandas as pd

from src.models.train import train_models


def main() -> None:
    data_path = Path("results/metrics/bold_analysis.parquet")
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/05_build_dataset_bold.py first")
    df = pd.read_parquet(data_path)
    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_models(df, str(output_dir))


if __name__ == "__main__":
    main()
