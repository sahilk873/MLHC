from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.encode_loader import build_concept_map, load_encode_tables, validate_encode_schema


def index_images(root: Path) -> pd.DataFrame:
    image_paths = []
    for folder in ["Android_processed", "iPhone_processed"]:
        path = root / "Image_records" / folder
        if not path.exists():
            continue
        for file in path.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                name = file.stem
                parts = name.split("_")
                person_id = parts[0] if len(parts) > 0 else None
                location_id = parts[1] if len(parts) > 1 else None
                image_paths.append(
                    {
                        "person_id": person_id,
                        "location_id": location_id,
                        "path": str(file),
                        "device_folder": folder,
                    }
                )
    return pd.DataFrame(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    tables = load_encode_tables(str(root))
    validate_encode_schema(tables)

    concept_map = build_concept_map(tables["CONCEPT"])
    concept_names = pd.Series(concept_map.values())
    skintone_concepts = concept_names[concept_names.str.contains("SKINTONE@", case=False, na=False)]

    summary = {
        "people": int(len(tables["PERSON"])),
        "measurements": int(len(tables["MEASUREMENT"])),
        "skintone_concepts": int(len(skintone_concepts)),
    }

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "encode_summary.json").write_text(json.dumps(summary, indent=2))

    image_index = index_images(root)
    image_path = log_dir / "encode_image_index.parquet"
    if not image_index.empty:
        image_index.to_parquet(image_path, index=False)
    else:
        image_index.to_csv(log_dir / "encode_image_index.csv", index=False)


if __name__ == "__main__":
    main()
