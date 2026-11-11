#!/usr/bin/env python
"""
Week 8: PhaseNet picking pass across the full downloaded dataset.

Adapted directly from Week 4's eqtransformer_pick.py, reusing every
memory/chunking lesson learned there (processing per-station, per-chunk-file,
one station per process invocation - see that file's header for the full
reasoning). Key differences from EQTransformer, confirmed against SeisBench's
own source rather than assumed:
  - PhaseNet has no detection_threshold - only P_threshold/S_threshold, since
    its architecture has no dedicated event-detection output channel
    (P/S/noise classes only)
  - No proposal-specified overlap override for PhaseNet (unlike EQTransformer's
    stated 55 s) - left at SeisBench's own sensible default (1500 samples/15s
    for PhaseNet's 30 s window)
  - Threshold is a CLI argument here, to support the proposal's stated
    threshold sweep (0.05, 0.1, 0.2, 0.3, 0.5) on a small station subset
    before committing to a full 60-station run at the selected value

detections may not be meaningfully populated for PhaseNet given the lack of
a real detection channel - saved defensively (skipped with a log message,
not a crash, if empty/absent) rather than assumed to exist.

Resumable: if this station's pick CSV (for the given threshold) already
exists, exits immediately without reloading the model.
"""
import os
import argparse
import glob
import time
import obspy
import torch
import seisbench
import seisbench.models as sbm
import pandas as pd

PROJECT_DIR = os.path.expanduser("~/tvz_project")
WAVEFORM_DIR = os.path.join(PROJECT_DIR, "data", "waveforms")
PICKS_DIR = os.path.join(PROJECT_DIR, "picks", "phasenet")
os.makedirs(PICKS_DIR, exist_ok=True)

BATCH_SIZE = 256


def main(station_code, threshold):
    picks_csv = os.path.join(PICKS_DIR, f"NZ.{station_code}_thresh{threshold}_picks.csv")
    detections_csv = os.path.join(PICKS_DIR, f"NZ.{station_code}_thresh{threshold}_detections.csv")

    if os.path.exists(picks_csv):
        print(f"{station_code} @ threshold {threshold}: already processed, skipping", flush=True)
        return

    station_dir = os.path.join(WAVEFORM_DIR, f"NZ.{station_code}")
    mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
    if not mseed_files:
        print(f"{station_code}: no waveform files found - skipping", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{station_code}: using device {device}", flush=True)
    if device == "cpu":
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        torch.set_num_threads(n_threads)
        print(f"{station_code}: no GPU detected - using {n_threads} CPU threads "
              "(no GPU access on this account, same as Week 4)", flush=True)

    seisbench.use_backup_repository()
    model = sbm.PhaseNet.from_pretrained("original")
    model.eval()
    if device == "cuda":
        model.cuda()

    run_start = time.time()
    all_picks, all_detections = [], []
    any_chunk_failed = False

    for chunk_file in mseed_files:
        st = obspy.read(chunk_file)
        st.merge(method=0)
        st = st.split()

        try:
            output = model.classify(
                st,
                batch_size=BATCH_SIZE,
                P_threshold=threshold,
                S_threshold=threshold,
            )
        except Exception as e:
            print(f"{station_code} [{os.path.basename(chunk_file)}]: FAILED — {e}", flush=True)
            any_chunk_failed = True
            continue

        all_picks.append(output.picks.to_dataframe())
        try:
            if len(output.detections) > 0:
                all_detections.append(output.detections.to_dataframe())
        except Exception:
            pass  # PhaseNet has no real detection channel - expected, not an error

        del st, output

    if not all_picks:
        print(f"{station_code}: no chunks succeeded, skipping station entirely", flush=True)
        return

    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df.to_csv(picks_csv, index=False)

    if all_detections:
        pd.concat(all_detections, ignore_index=True).to_csv(detections_csv, index=False)
    else:
        print(f"{station_code}: no detections saved (expected for PhaseNet - no detection channel)", flush=True)

    flag = " (one or more chunks failed - see log above)" if any_chunk_failed else ""
    elapsed = time.time() - run_start
    print(f"{station_code} @ threshold {threshold}: {len(picks_df)} picks "
          f"across {len(mseed_files)} chunks{flag} [{elapsed:.0f}s]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True,
                         help="Station code to process, e.g. ALRZ (one station per invocation)")
    parser.add_argument("--threshold", type=float, required=True,
                         help="P and S phase probability threshold (proposal sweep: 0.05, 0.1, 0.2, 0.3, 0.5)")
    args = parser.parse_args()
    main(station_code=args.station, threshold=args.threshold)
