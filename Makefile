.PHONY: setup test manifest build train eval paper-assets

setup:
	uv sync --all-extras

test:
	python -m pytest -q

manifest:
	python scripts/01_manifest.py --root ./physionet.org/files

build:
	python scripts/05_build_dataset_bold.py --root ./physionet.org/files/blood-gas-oximetry/1.0
	python scripts/06_build_dataset_encode.py --root ./physionet.org/files/encode-skin-color/1.0.0

train:
	python scripts/07_train_models.py

eval:
	python scripts/08_evaluate.py

paper-assets:
	python scripts/09_make_figures_tables.py
	python scripts/10_make_paper_assets.py
