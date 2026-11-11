#!/usr/bin/env python
"""
Week 13: QuakeMigrate Detect stage - migrates and stacks onset functions
across the network to generate a continuous coalescence function, from
which candidate events will later be identified (Trigger stage, Week 14).

Built directly against QuakeMigrate's own bundled real examples (cloned
from GitHub): examples/template_scripts/detect.py (generic template) and
examples/Volcanotectonic_Iceland/dike_intrusion_detect.py (the closest
real-world analogue to our own volcano-tectonic TVZ setting among
QuakeMigrate's examples - closer than the icequake examples).

STA/LTA onset parameters (bandpass filters, window lengths) are taken
directly from the Volcanotectonic_Iceland example as a documented starting
point, not independently tuned for the TVZ - per the proposal's own
guidance, these should be checked against a short test window before
committing to the full 61-day run ("test on 1 month first").

Archive uses the "STATION.YEAR.JULIANDAY" format, matching the daily files
restructured in this same week (Week 13) from the original 15-day
download chunks.
"""
import os
import argparse

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

from quakemigrate import QuakeScan
from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal.onsets import STALTAOnset

PROJECT_DIR = os.path.expanduser("~/tvz_project")
QM_DIR = os.path.join(PROJECT_DIR, "quakemigrate")
QM_ARCHIVE_DIR = os.path.join(PROJECT_DIR, "data", "waveforms_qm")


def main(starttime, endtime, run_name, threads, decimate_factor):
    station_file = os.path.join(QM_DIR, "lut", "stations.csv")
    lut_file = os.path.join(QM_DIR, "lut", "tvz_lut")
    run_path = os.path.join(QM_DIR, "runs")

    stations = read_stations(station_file, delimiter=",")

    archive = Archive(
        archive_path=QM_ARCHIVE_DIR, stations=stations,
        archive_format="STATION.YEAR.JULIANDAY",
    )

    lut = read_lut(lut_file=lut_file)
    if decimate_factor > 1:
        lut = lut.decimate([decimate_factor, decimate_factor, decimate_factor])
        print(f"LUT decimated by factor {decimate_factor} for the detect stage "
              f"(finer resolution reused later for locate)", flush=True)

    onset = STALTAOnset(position="classic", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 16, 2]}
    onset.sta_lta_windows = {"P": [0.2, 1.0], "S": [0.2, 1.0]}

    scan = QuakeScan(
        archive, lut, onset=onset, run_path=run_path, run_name=run_name,
        log=True, loglevel="info",
    )

    scan.timestep = 120.0
    scan.threads = threads

    print(f"Running detect: {starttime} to {endtime}, run_name={run_name}, "
          f"threads={threads}", flush=True)
    scan.detect(starttime, endtime)
    print("Detect complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--starttime", default="2019-08-01T00:00:00.0")
    parser.add_argument("--endtime", default="2019-09-30T00:00:00.0")
    parser.add_argument("--run-name", default="tvz_full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--decimate", type=int, default=2,
                         help="LUT decimation factor for detect (default 2x, "
                              "i.e. 4km from the 2km base LUT)")
    args = parser.parse_args()
    main(starttime=args.starttime, endtime=args.endtime, run_name=args.run_name,
         threads=args.threads, decimate_factor=args.decimate)
