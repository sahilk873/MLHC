from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def csv_to_latex(csv_path: Path, tex_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.fillna("NA")
    tex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
    tex_path.write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--paper-build", default="paper_build")
    args = parser.parse_args()

    results_root = Path(args.results)
    paper_build = Path(args.paper_build)
    figs_dir = paper_build / "figs"
    tables_dir = paper_build / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Copy figures
    for fig in (results_root / "figures").glob("*.png"):
        (figs_dir / fig.name).write_bytes(fig.read_bytes())

    # Convert tables
    for table in (results_root / "tables").glob("*.csv"):
        tex_path = tables_dir / f"{table.stem}.tex"
        csv_to_latex(table, tex_path)


if __name__ == "__main__":
    main()
