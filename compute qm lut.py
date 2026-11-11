#!/usr/bin/env python
"""
Week 12: Compute QuakeMigrate's travel-time lookup table (LUT).

Confirmed directly from QuakeMigrate's own documentation: only a 1D
velocity model utility is currently implemented for LUT generation (not a
limitation we are working around - this is the tool's actual supported
usage pattern). Uses the "1dnlloc" method, which calls NonLinLoc's
Grid2Time Eikonal solver internally to compute a 2D travel-time slice per
station, then sweeps this through the required azimuth range to populate
the full 3D grid via bilinear interpolation - this is why it is expected
to be considerably lighter than a genuine full 3D computation would be.

Reuses the real Chow et al. (2022) 1D profile and station set already
built for NonLinLoc/GrowClust3D, and the existing compiled NonLinLoc
binaries from Week 6 (via nlloc_path) - no new installation needed.

UNCERTAIN DETAIL, flagged clearly: the exact keyword name for pointing at
an existing NonLinLoc installation (nlloc_path) is mentioned in
QuakeMigrate's installation documentation but was not shown in a concrete
code example. Using it as documented; if wrong, the resulting error should
name the correct kwarg directly.

Use --test for a coarse, fast run (larger node spacing) to validate the
mechanics before committing to the full-resolution grid.
"""
import os
import argparse
import pickle
import time

from quakemigrate.lut import compute_traveltimes
from quakemigrate.io import read_stations, read_vmodel

PROJECT_DIR = os.path.expanduser("~/tvz_project")
QM_LUT_DIR = os.path.join(PROJECT_DIR, "quakemigrate", "lut")
NLLOC_BIN_DIR = os.path.join(PROJECT_DIR, "software", "NonLinLoc", "src", "bin")


def main(test_mode=False):
    with open(os.path.join(QM_LUT_DIR, "grid_spec.pkl"), "rb") as f:
        grid_spec = pickle.load(f)

    if test_mode:
        # Coarsen node spacing for a fast mechanics check
        grid_spec.node_spacing = [8.0, 8.0, 8.0]
        save_file = os.path.join(QM_LUT_DIR, "tvz_lut_test")
        print("TEST MODE: node spacing coarsened to 8 km", flush=True)
    else:
        save_file = os.path.join(QM_LUT_DIR, "tvz_lut")

    stations = read_stations(os.path.join(QM_LUT_DIR, "stations.csv"), delimiter=",")
    vmod = read_vmodel(os.path.join(QM_LUT_DIR, "vmodel.csv"), delimiter=",")
    print(f"{len(stations)} stations, {len(vmod)} velocity depth levels loaded", flush=True)

    start = time.time()
    lut = compute_traveltimes(
        grid_spec,
        stations,
        method="1dnlloc",
        phases=["P", "S"],
        vmod=vmod,
        block_model=False,
        log=True,
        save_file=save_file,
        nlloc_path=NLLOC_BIN_DIR,
    )
    elapsed = time.time() - start
    print(f"LUT computed in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Saved: {save_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Coarse, fast test run")
    args = parser.parse_args()
    main(test_mode=args.test)
