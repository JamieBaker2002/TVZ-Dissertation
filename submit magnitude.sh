#!/usr/bin/env bash
#SBATCH --job-name=tvz_mag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project
python magnitude_estimate.py "$@"
