#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Week 1: HPC environment setup for TVZ earthquake catalogue project
# Run this ON THE HPC (via VS Code Remote-SSH terminal or a direct SSH session)
# ---------------------------------------------------------------------------
# NOTE: module names/commands below are typical HPC conventions — adjust to
# whatever `module avail` shows on your specific cluster (ADA/HALI at UEA).
set -euo pipefail

PROJECT_DIR="${HOME}/tvz_project"
mkdir -p "${PROJECT_DIR}/software"
cd "${PROJECT_DIR}/software"

echo "=== 1. Load base modules (adjust to your cluster's module names) ==="
module purge
module load python/anaconda/2024.10/3.12.7
module load cmake 2>/dev/null || true
module load gcc 2>/dev/null || true   # if this fails, your cluster likely uses 'intel' instead — check `module avail 2>&1 | grep -i gcc`

echo "=== 2. Create conda environment from environment.yml ==="
conda env create -f "${PROJECT_DIR}/environment.yml" || \
  conda env update -f "${PROJECT_DIR}/environment.yml" --prune
source activate tvz-seismic

echo "=== 3. Build NonLinLoc from source ==="
if [ ! -d NonLinLoc ]; then
  git clone https://github.com/alomax/NonLinLoc.git
fi
cd NonLinLoc/src
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
echo "NonLinLoc binaries built in: ${PROJECT_DIR}/software/NonLinLoc/src/build"
cd "${PROJECT_DIR}/software"

echo "=== 4. Load Julia module ==="
module load julia/1.12.6
julia --version

echo "=== 5. Install GrowClust3D.jl (unregistered — install from GitHub HEAD) ==="
julia -e 'using Pkg; Pkg.add(url="https://github.com/dttrugman/GrowClust3D.jl")'

echo "=== 6. Sanity check: import everything in Python ==="
python - <<'PYEOF'
import obspy, seisbench, pyocto, eqcorrscan, pygmt
print("obspy:", obspy.__version__)
print("seisbench:", seisbench.__version__)
print("pyocto:", pyocto.__version__)
print("eqcorrscan:", eqcorrscan.__version__)
print("pygmt:", pygmt.__version__)
PYEOF

echo "=== Setup complete. NonLinLoc + Julia/GrowClust3D.jl installed outside conda. ==="
echo "Next: run week1_toy_test.py to verify the full stack end-to-end."
