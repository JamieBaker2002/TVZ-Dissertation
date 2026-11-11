#!/usr/bin/env bash
#SBATCH --job-name=tvz_download
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NOTE: no --partition or --account set — this cluster's naming is unknown
# to me. Check `sinfo` for partition names and `sacctmgr show associations
# user=$USER` for your account string, then add:
#   #SBATCH --partition=<name>
#   #SBATCH --account=<name>
# if your cluster requires them (some don't; try submitting without first).

set -euo pipefail

module purge
module load python/anaconda/2024.10/3.12.7
source activate tvz-seismic

cd ~/tvz_project
python download_waveforms.py
