#!/usr/bin/env bash
#SBATCH --job-name=tvz_pn
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Array range set per-submission with --array on the command line, e.g.:
#   Threshold sweep test (5 stations):  sbatch --array=0-4 --nodelist=<node> submit_phasenet.sh <threshold>
#   Full run (60 stations):             sbatch --array=0-59 --nodelist=<node> submit_phasenet.sh <threshold>
# GPU access confirmed unavailable on this account (Week 4) - CPU only.

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project

THRESHOLD="${1:?Usage: sbatch --array=... submit_phasenet.sh <threshold>}"

STATIONS=(data/waveforms/NZ.*)
STATION_DIR="${STATIONS[$SLURM_ARRAY_TASK_ID]}"
sta_code=$(basename "$STATION_DIR" | cut -d. -f2)

echo "Array task $SLURM_ARRAY_TASK_ID: station $sta_code, threshold $THRESHOLD"
python phasenet_pick.py --station "$sta_code" --threshold "$THRESHOLD"
