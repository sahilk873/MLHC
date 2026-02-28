from __future__ import annotations

import json
import sys
from pathlib import Path

import runpy


def test_manifest_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "files"
    bold = root / "blood-gas-oximetry" / "1.0"
    encode = root / "encode-skin-color" / "1.0.0"
    bold.mkdir(parents=True)
    encode.mkdir(parents=True)
    (bold / "a.csv").write_text("a,b\\n1,2\\n")
    (encode / "PERSON.csv").write_text("person_id\\n1\\n")
    (encode / "VISIT_OCCURRENCE.csv").write_text("visit_occurrence_id,person_id\\n1,1\\n")
    (encode / "MEASUREMENT.csv").write_text("measurement_id,person_id\\n1,1\\n")
    (encode / "CONCEPT.csv").write_text("concept_id,concept_name\\n1,TEST\\n")

    sys.argv = ["01_manifest.py", "--root", str(root)]
    runpy.run_path(str(Path("scripts/01_manifest.py")), run_name="__main__")

    manifest = Path("results/manifests/data_manifest.json")
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    paths = [entry["path"] for entry in data]
    assert "blood-gas-oximetry/1.0/a.csv" in paths
    assert "encode-skin-color/1.0.0/PERSON.csv" in paths
