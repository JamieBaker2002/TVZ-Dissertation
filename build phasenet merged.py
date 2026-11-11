#!/usr/bin/env python
"""
Week 10: Build the event->picks mapping for the PhaseNet/GaMMA catalogue,
matching the column structure the cross-correlation script expects
(event_idx, station, phase, time) - the same shape as PyOcto's assignments
used for the EQTransformer catalogue, so eqcorrscan_dtcc.py can be reused
with minimal changes.
"""
import os
import pandas as pd

PROJECT_DIR = os.path.expanduser("~/tvz_project")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")

assignments = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_gamma_assignments.csv"))
declustered = pd.read_csv(os.path.join(PROJECT_DIR, "picks", "phasenet", "phasenet_declustered.csv")).reset_index(drop=True)

merged = assignments.merge(declustered, left_on="pick_index", right_index=True, how="left")
merged = merged.rename(columns={"event_index": "event_idx"})[["event_idx", "station", "phase", "time"]]

out_path = os.path.join(CATALOG_DIR, "phasenet_gamma_merged.csv")
merged.to_csv(out_path, index=False)
print(f"Saved: {out_path} ({len(merged)} rows, {merged['event_idx'].nunique()} events)")
