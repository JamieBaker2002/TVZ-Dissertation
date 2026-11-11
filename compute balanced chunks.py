#!/usr/bin/env python
"""
Week 14: Compute event-count-balanced time chunks for the Locate stage.

The original chunking scheme (Week 13, reused for Locate) split the 60-day
period into equal CALENDAR-TIME windows. This works well for Detect (whose
cost is roughly constant per unit time, independent of how many real
events occurred), but Locate's cost scales with the number of triggered
events in each window - and real events are heavily concentrated around
the 4 September mainshock and its aftershock sequence, not evenly spread
across the 60 days. A 7.5-day equal-time chunk containing the mainshock
timed out at 12 hours (1,448 events located, incomplete) while a 3-day
quiet-period test finished in under 16 minutes (18 events) - confirming
the imbalance directly rather than assuming it.

This script reads the REAL per-day triggered-event counts (already
computed in Week 14's Trigger stage) and splits the 60 days into N chunks
with approximately equal TOTAL event count, guaranteeing no single chunk
is overloaded regardless of temporal clustering - a more robust fix than
guessing a smaller fixed time window and hoping it's small enough.
"""
import os
import glob
import json
import pandas as pd
from obspy import UTCDateTime

PROJECT_DIR = os.path.expanduser("~/tvz_project")
TRIGGER_EVENTS_DIR = os.path.join(PROJECT_DIR, "quakemigrate", "runs", "tvz_full", "trigger", "events")
N_CHUNKS = 20

# Real per-EVENT times, not per-day counts - a first attempt at day-level
# balancing failed on the real data: 6,910 of 10,583 events (65%) turned
# out to fall within a single contiguous 16-day stretch, so day-level
# grouping could not split that stretch any further and produced one
# enormous chunk. Loading every individual event timestamp allows the
# chunk boundaries to fall *within* a single busy day if needed.
all_times = []
for f in sorted(glob.glob(os.path.join(TRIGGER_EVENTS_DIR, "*_TriggeredEvents.csv"))):
    df = pd.read_csv(f)
    if len(df) == 0:
        continue
    all_times.extend(UTCDateTime(t) for t in df["CoaTime"])

all_times.sort()
total_events = len(all_times)
events_per_chunk = -(-total_events // N_CHUNKS)  # ceiling division

print(f"{total_events} total triggered events", flush=True)
print(f"Events/chunk (target): {events_per_chunk}", flush=True)

overall_start = UTCDateTime("2019-08-01T00:00:00.0")
overall_end = UTCDateTime("2019-09-30T00:00:00.0")

chunk_bounds = []
prev_end = overall_start
for i in range(0, total_events, events_per_chunk):
    chunk_times = all_times[i:i + events_per_chunk]
    is_last = (i + events_per_chunk) >= total_events
    chunk_start = prev_end
    chunk_end = overall_end if is_last else chunk_times[-1] + 60
    chunk_bounds.append({
        "start": chunk_start.isoformat(),
        "end": chunk_end.isoformat(),
        "n_events": len(chunk_times),
    })
    print(f"Chunk {len(chunk_bounds) - 1}: {chunk_start.isoformat()} to "
          f"{chunk_end.isoformat()} ({len(chunk_times)} events)", flush=True)
    prev_end = chunk_end

out_path = os.path.join(PROJECT_DIR, "quakemigrate", "locate_chunks.json")
with open(out_path, "w") as f:
    json.dump(chunk_bounds, f, indent=2)
print(f"\nSaved: {out_path} ({len(chunk_bounds)} chunks)", flush=True)
