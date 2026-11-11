#!/usr/bin/env python
"""
Week 4: EQTransformer picking pass across the full downloaded dataset.

Runs SeisBench's EQTransformer (conservative pre-trained model) over ONE
station's 1 Aug - 30 Sep 2019 waveform data per invocation, using the
proposal's stated parameters:
  - 55-second overlapping window: EQTransformer's window is 60 s (6000
    samples at 100 Hz); SeisBench's 'overlap' argument accepts either a
    fraction of window length or a raw sample count, so overlap=5500
    samples gives exactly 55 s of overlap between windows.
  - event (detection) probability threshold >= 0.2  -> detection_threshold
  - phase pick probability threshold >= 0.01        -> P_threshold/S_threshold
(Same threshold convention already verified working in the Week 1 toy test,
where the reverse mix-up caused 0 picks.)

Saves two CSVs for the station under picks/eqtransformer/: one for discrete
phase picks, one for event detections, using SeisBench's own built-in
to_dataframe() export. Deliberately does NOT save the full continuous
probability curves (the annotate() output) - only the discrete picks and
detections that phase association actually needs, since the raw curves
would be roughly as large as the original 192 GB waveform archive. This is
a scope decision, documented here and in the progress log, not an oversight.

Processes each of the station's ~15-day chunk files separately rather than
merging the full 61-day span into one classify() call, since a full 61-day
station's worth of windows (at 5 s step, 55 s overlap) numbers over a
million and caused an OOM at 32 GB when done in one call.

Deliberately one station per Python process invocation (see
submit_eqtransformer.sh, which loops over stations calling this once each),
rather than one process looping over all stations: peak memory on a single
station+chunked run still reached ~56 GB out of a 64 GB request, almost
certainly from PyTorch's CPU allocator retaining freed memory rather than
returning it to the OS between chunks. Restarting the Python process per
station guarantees the OS reclaims that memory regardless of the exact
cause, rather than risking accumulation across 60 stations in one process.

Resumable: if this station's pick CSV already exists, exits immediately
without reloading the model.
"""
import os
import sys
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
PICKS_DIR = os.path.join(PROJECT_DIR, "picks", "eqtransformer")
os.makedirs(PICKS_DIR, exist_ok=True)

OVERLAP_SAMPLES = 5500       # 55 s overlap out of EQTransformer's 60 s (6000-sample) window
DETECTION_THRESHOLD = 0.2
PHASE_THRESHOLD = 0.01
BATCH_SIZE = 256             # SeisBench's own default; reduce if memory is still tight


def main(station_code):
    picks_csv = os.path.join(PICKS_DIR, f"NZ.{station_code}_picks.csv")
    detections_csv = os.path.join(PICKS_DIR, f"NZ.{station_code}_detections.csv")

    if os.path.exists(picks_csv):
        print(f"{station_code}: already processed, skipping (no model load needed)", flush=True)
        return

    station_dir = os.path.join(WAVEFORM_DIR, f"NZ.{station_code}")
    mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
    if not mseed_files:
        print(f"{station_code}: no waveform files found at {station_dir} — skipping", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{station_code}: using device {device}", flush=True)
    if device == "cpu":
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        torch.set_num_threads(n_threads)
        print(f"{station_code}: no GPU detected - using {n_threads} CPU threads", flush=True)

    seisbench.use_backup_repository()  # same mitigation as Week 1, in case the primary repo is flaky
    model = sbm.EQTransformer.from_pretrained("original")
    model.eval()
    if device == "cuda":
        model.cuda()

    run_start = time.time()
    all_picks, all_detections = [], []
    any_chunk_failed = False

    for chunk_file in mseed_files:
        st = obspy.read(chunk_file)
        st.merge(method=0)
        st = st.split()  # convert masked gap regions back into separate real traces

        try:
            output = model.classify(
                st,
                batch_size=BATCH_SIZE,
                overlap=OVERLAP_SAMPLES,
                detection_threshold=DETECTION_THRESHOLD,
                P_threshold=PHASE_THRESHOLD,
                S_threshold=PHASE_THRESHOLD,
            )
        except Exception as e:
            print(f"{station_code} [{os.path.basename(chunk_file)}]: FAILED — {e}", flush=True)
            any_chunk_failed = True
            continue

        all_picks.append(output.picks.to_dataframe())
        all_detections.append(output.detections.to_dataframe())
        # explicit cleanup between chunks to help release memory before the next one
        del st, output

    if not all_picks:
        print(f"{station_code}: no chunks succeeded, skipping station entirely", flush=True)
        return

    picks_df = pd.concat(all_picks, ignore_index=True)
    detections_df = pd.concat(all_detections, ignore_index=True)
    picks_df.to_csv(picks_csv, index=False)
    detections_df.to_csv(detections_csv, index=False)
    flag = " (one or more chunks failed - see log above)" if any_chunk_failed else ""

    elapsed = time.time() - run_start
    print(f"{station_code}: {len(detections_df)} detections, {len(picks_df)} picks "
          f"across {len(mseed_files)} chunks{flag} [{elapsed:.0f}s]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True,
                         help="Station code to process, e.g. ALRZ (processes one station per invocation)")
    args = parser.parse_args()
    main(station_code=args.station)
