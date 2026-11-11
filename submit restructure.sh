#!/usr/bin/env bash
#SBATCH --job-name=tvz_restruct
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Array range set per-submission with --array on the command line, e.g.:
#   sbatch --array=0-59 --nodelist=<node> submit_restructure.sh

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project

STATIONS=(data/waveforms/NZ.*)
STATION_DIR="${STATIONS[$SLURM_ARRAY_TASK_ID]}"
sta_code=$(basename "$STATION_DIR" | cut -d. -f2)

echo "Array task $SLURM_ARRAY_TASK_ID: station $sta_code"
python restructure_archive.py --station "$sta_code"
