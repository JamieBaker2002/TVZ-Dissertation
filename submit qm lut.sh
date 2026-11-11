#!/usr/bin/env bash
#SBATCH --job-name=tvz_qmlut
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project
python compute_qm_lut.py "$@"
