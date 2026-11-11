#!/usr/bin/env bash
#SBATCH --job-name=tvz_qmloc
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Reads pre-computed, event-count-balanced chunk boundaries from
# quakemigrate/locate_chunks.json (built by compute_balanced_chunks.py),
# rather than naive equal-CALENDAR-TIME chunking - Locate's cost scales
# with real event count, which is heavily concentrated around the 4
# September mainshock, not evenly spread across the 60 days.
#
# Submit with: sbatch --array=0-N --nodelist=<node1>,... submit_qm_locate_balanced.sh
# (N = number of chunks actually printed by compute_balanced_chunks.py - 1)

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project

CHUNK_START=$(python3 -c "
import json
with open('quakemigrate/locate_chunks.json') as f:
    chunks = json.load(f)
print(chunks[$SLURM_ARRAY_TASK_ID]['start'])
")
CHUNK_END=$(python3 -c "
import json
with open('quakemigrate/locate_chunks.json') as f:
    chunks = json.load(f)
print(chunks[$SLURM_ARRAY_TASK_ID]['end'])
")

echo "Array task $SLURM_ARRAY_TASK_ID: $CHUNK_START to $CHUNK_END"
python qm_locate.py --starttime "$CHUNK_START" --endtime "$CHUNK_END" \
    --run-name tvz_full --threads 8
