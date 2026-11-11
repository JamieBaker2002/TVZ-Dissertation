#!/usr/bin/env bash
#SBATCH --job-name=tvz_eqt
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-59%8
# ARRAY JOB: one task per station (0-59, since we have 60 stations - verify
# this count with `ls -d data/waveforms/NZ.* | wc -l` before submitting, and
# adjust the range if it differs). The single-station sequential test took
# ~48 min, which would be ~48 HOURS run one-by-one across all 60 - completely
# incompatible with the deadline. Running as a throttled array instead:
# %8 caps it at 8 stations running simultaneously, so wall-clock time should
# be roughly (60/8) x 48 min =~ 6 hours instead of 48. Check node availability
# first with `sinfo -p compute -N` - if there are fewer than ~8 free nodes,
# lower the %8 throttle to avoid the whole batch queueing behind itself.
#
# GPU access was checked and is NOT currently available on this account
# (gpu/gpu-reserved QOS both exist on the cluster but aren't granted to
# either of this user's accounts) - running on CPU per the proposal's own
# stated contingency plan.

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic

# The anaconda module's own libstdc++.so.6 is older than the one bundled in
# our conda environment, and was shadowing it - causing "GLIBCXX_3.4.30 not
# found" on lxml's import chain. Prepending the environment's own lib dir
# fixes the search order.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/tvz_project

STATIONS=(data/waveforms/NZ.*)
STATION_DIR="${STATIONS[$SLURM_ARRAY_TASK_ID]}"
sta_code=$(basename "$STATION_DIR" | cut -d. -f2)

echo "Array task $SLURM_ARRAY_TASK_ID: station $sta_code"
python eqtransformer_pick.py --station "$sta_code"
