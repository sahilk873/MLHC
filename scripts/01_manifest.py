from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_type(path: Path) -> str:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return "csv"
    if path.suffix.lower() == ".gz":
        return "csv.gz"
    if path.suffix.lower() == ".parquet":
        return "parquet"
    if path.suffix.lower() in {".pdf"}:
        return "pdf"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    targets = [
        root / "blood-gas-oximetry" / "1.0",
        root / "encode-skin-color" / "1.0.0",
    ]
    entries = []
    for target in targets:
        if not target.exists():
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file():
                entries.append(
                    {
                        "path": str(path.relative_to(root)),
                        "size": path.stat().st_size,
                        "sha256": sha256_path(path),
                        "type": infer_type(path),
                    }
                )

    out_dir = Path("results/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_manifest.json"
    out_path.write_text(json.dumps(entries, indent=2))

    encode_root = root / "encode-skin-color" / "1.0.0"
    required = ["PERSON.csv", "VISIT_OCCURRENCE.csv", "MEASUREMENT.csv", "CONCEPT.csv"]
    if encode_root.exists():
        missing = [name for name in required if not (encode_root / name).exists()]
        if missing:
            print(f"Missing ENCoDE tables: {missing}")
            sys.exit(1)


if __name__ == "__main__":
    main()
