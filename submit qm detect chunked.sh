#!/usr/bin/env bash
#SBATCH --job-name=tvz_qmdet
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Splits 1 Aug - 30 Sep 2019 into N_CHUNKS roughly-equal time windows, run in
# parallel as separate array tasks. All tasks share the same run_name
# ("tvz_full") since detect output is organised per-day (scanmseed files) -
# each day only gets written once by whichever chunk covers it, so outputs
# coexist without collision and Trigger (Week 14) can scan the combined set.
#
# Submit with: sbatch --array=0-7 --nodelist=<node1>,<node2>,... submit_qm_detect_chunked.sh
# (or submit each array index to a different --nodelist individually if a
# single sbatch call cannot span multiple explicitly-named nodes)

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project

N_CHUNKS=8
python3 -c "
from obspy import UTCDateTime
start = UTCDateTime('2019-08-01T00:00:00.0')
end = UTCDateTime('2019-09-30T00:00:00.0')
n_chunks = $N_CHUNKS
task_id = $SLURM_ARRAY_TASK_ID
total_days = (end - start) / 86400.0
chunk_days = total_days / n_chunks
chunk_start = start + task_id * chunk_days * 86400
chunk_end = start + (task_id + 1) * chunk_days * 86400
print(chunk_start.isoformat())
print(chunk_end.isoformat())
" > /tmp/chunk_times_${SLURM_ARRAY_TASK_ID}.txt

CHUNK_START=$(sed -n '1p' /tmp/chunk_times_${SLURM_ARRAY_TASK_ID}.txt)
CHUNK_END=$(sed -n '2p' /tmp/chunk_times_${SLURM_ARRAY_TASK_ID}.txt)

echo "Array task $SLURM_ARRAY_TASK_ID: $CHUNK_START to $CHUNK_END"
python qm_detect.py --starttime "$CHUNK_START" --endtime "$CHUNK_END" \
    --run-name tvz_full --threads 8 --decimate 4
