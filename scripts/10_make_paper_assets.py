from __future__ import annotations

import json
import shutil
from pathlib import Path


def main() -> None:
    paper_figs = Path("paper/figs")
    paper_tables = Path("paper/tables")
    paper_figs.mkdir(parents=True, exist_ok=True)
    paper_tables.mkdir(parents=True, exist_ok=True)

    assets = []
    for fig in Path("results/figures").glob("*.png"):
        dest = paper_figs / fig.name
        shutil.copy2(fig, dest)
        assets.append({"type": "figure", "src": str(fig), "dest": str(dest)})
    for table in Path("results/tables").glob("*.csv"):
        dest = paper_tables / table.name
        shutil.copy2(table, dest)
        assets.append({"type": "table", "src": str(table), "dest": str(dest)})

    Path("paper/assets_manifest.json").write_text(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
