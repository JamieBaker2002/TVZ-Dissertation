#!/usr/bin/env python
"""
Week 14: QuakeMigrate Trigger stage - identifies discrete candidate events
from the continuous coalescence function computed in Week 13 (Detect).

Built directly against QuakeMigrate's own confirmed real templates:
examples/template_scripts/trigger.py and examples/Volcanotectonic_Iceland/
dike_intrusion_trigger.py.

normalise_coalescence=True is used (matching both real examples) - this
divides the raw coalescence by a noise-level reference before comparing
against the threshold, which is why a small, dataset-independent-ish
static_threshold (order 1-2) is sensible despite our own raw coalescence
values being on a completely different scale (~100,000) - confirmed by
direct inspection before writing this script, not assumed.

Threshold is a CLI argument to support a small sweep (mirroring the
PhaseNet picking-threshold approach in Week 8), rather than borrowing
either example's specific value, which were tuned for different datasets
(icequakes, a different Iceland volcano).
"""
import os
import argparse
import glob
import shutil
import pandas as pd

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

from quakemigrate.io import read_lut
from quakemigrate.signal import Trigger

PROJECT_DIR = os.path.expanduser("~/tvz_project")
QM_DIR = os.path.join(PROJECT_DIR, "quakemigrate")


def run_one(starttime, endtime, run_name, threshold, lut):
    # Wipe any existing trigger output BEFORE this run, not just back it up
    # after - guarantees this threshold's results cannot inherit stale
    # files left over from a previous (more permissive) threshold's run,
    # which is what happened in the original sweep: QuakeMigrate appears to
    # only overwrite a day's CSV when it finds NEW events for that specific
    # threshold, so a stricter threshold finding zero new events for a day
    # would silently leave an earlier, looser threshold's file in place.
    trigger_dir = os.path.join(QM_DIR, "runs", run_name, "trigger")
    if os.path.exists(trigger_dir):
        shutil.rmtree(trigger_dir)

    trig = Trigger(lut, run_path=os.path.join(QM_DIR, "runs"), run_name=run_name,
                    log=True, loglevel="info")
    trig.marginal_window = 1.0
    trig.min_event_interval = 2.0
    trig.normalise_coalescence = True
    trig.threshold_method = "static"
    trig.static_threshold = threshold
    trig.smooth_coa = True
    trig.smoothing_kernel_sigma = 0.2
    trig.smoothing_kernel_width = 2
    trig.plot_trigger_summary = True  # re-enabled for the final chosen run

    print(f"Running trigger: threshold={threshold}", flush=True)
    trig.trigger(starttime, endtime, interactive_plot=False)

    trigger_dir = os.path.join(QM_DIR, "runs", run_name, "trigger")
    # Real output path includes a "subname" directory level
    # (trigger/{subname}/events/...) confirmed from source but not
    # predictable in advance - searching recursively rather than guessing it
    csv_files = glob.glob(os.path.join(trigger_dir, "**", "*TriggeredEvents.csv"), recursive=True)
    total_events = 0
    for f in csv_files:
        total_events += len(pd.read_csv(f))
    print(f"  -> threshold {threshold}: {total_events} total triggered events "
          f"across {len(csv_files)} day-files", flush=True)

    # Preserve this threshold's output before the next sweep value overwrites it
    backup_dir = os.path.join(QM_DIR, "runs", f"{run_name}_thresh{threshold}_trigger_backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(trigger_dir, backup_dir)

    return total_events


def main(starttime, endtime, run_name, thresholds):
    lut_file = os.path.join(QM_DIR, "lut", "tvz_lut")
    lut = read_lut(lut_file=lut_file)

    results = {}
    for threshold in thresholds:
        results[threshold] = run_one(starttime, endtime, run_name, threshold, lut)

    print("\n=== Sweep summary ===", flush=True)
    for threshold, count in results.items():
        print(f"  threshold {threshold}: {count} events", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--starttime", default="2019-08-01T00:00:00.0")
    parser.add_argument("--endtime", default="2019-09-30T00:00:00.0")
    parser.add_argument("--run-name", default="tvz_full")
    parser.add_argument("--thresholds", type=float, nargs="+",
                         default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0],
                         help="Static normalised-coalescence thresholds to sweep")
    args = parser.parse_args()
    main(starttime=args.starttime, endtime=args.endtime, run_name=args.run_name,
         thresholds=args.thresholds)
