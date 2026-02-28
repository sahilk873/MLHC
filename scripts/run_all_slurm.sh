#!/usr/bin/env bash
#SBATCH --job-name=mlhc_pipeline
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err
#SBATCH --chdir=/nas/longleaf/home/sahilk/MLHC

set -euo pipefail

# If your cluster requires a partition, add:
#SBATCH --partition=YOUR_PARTITION

# If you use modules, load them here. Example:
# module load python/3.11

# Keep threading conservative to avoid OpenMP issues.
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
export SKIP_SKLEARN=0

ROOT="./physionet.org/files"

mkdir -p results/logs

bash scripts/run_all.sh --root "${ROOT}"
