#!/usr/bin/env python
"""
Week 6 (proposal task): EQcorrscan cross-correlation, producing the dt.cc
differential-time file GrowClust3D needs as input.

Uses EQcorrscan's write_correlations() with the proposal's stated parameters
(1-15 Hz bandpass, minimum correlation 0.65), operating on the quality-
filtered NonLinLoc catalogue (904 events).

Real complications handled here, not guessed at:
  - Our picks only record network.station.location (e.g. "NZ.ALRZ.10"), not
    the exact channel - so actual channel codes are looked up from the real
    downloaded files first, since stations use inconsistent component
    naming (some N/E, some 1/2, confirmed back in Week 1-2).
  - write_correlations expects ObsPy Event/Origin/Pick objects, not a plain
    dataframe - built explicitly below.
  - Per-event waveform snippets are read directly from whichever ~15-day
    chunk file covers that event's time, using ObsPy's starttime/endtime
    trimmed read (avoids loading each full chunk file into memory).

UNCERTAIN DETAIL, flagged clearly: write_correlations' docstring shows no
explicit output-path argument, only that it writes "dt.cc" - inferred to
mean the current working directory (matching the related dt.ct-writing code
seen in EQcorrscan's source). Script chdir's into the intended output folder
first to handle this; if the real output lands elsewhere, the fix should be
a one-line correction once we see it.

Use --limit N to test on the first N events before committing to the full
904-event run - strongly recommended given how much is inferred here.
"""
import os
import glob
import argparse
import pandas as pd
import obspy
from obspy import UTCDateTime, Catalog
from obspy.core.event import Event, Origin, Pick, WaveformStreamID
from eqcorrscan.utils.catalog_to_dd import write_correlations

PROJECT_DIR = os.path.expanduser("~/tvz_project")
WAVEFORM_DIR = os.path.join(PROJECT_DIR, "data", "waveforms")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "growclust")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOWCUT = 1.0        # proposal: 1-15 Hz bandpass
HIGHCUT = 15.0
MIN_CC = 0.65       # proposal: correlation > 0.65
EXTRACT_LEN = 4.0   # seconds of waveform extracted around each pick
PRE_PICK = 1.0      # seconds before the pick where the window starts
SHIFT_LEN = 0.5     # seconds the pick is allowed to shift for max correlation
MAX_SEP = 2.0        # km - maximum hypocentral separation to attempt correlating a pair
MIN_LINK = 6         # minimum shared picks between two events to attempt correlation


def get_channel_codes():
    """For each station, find real vertical/horizontal channel codes from
    the actual downloaded files, rather than assume a naming convention."""
    lookup = {}
    station_dirs = sorted(glob.glob(os.path.join(WAVEFORM_DIR, "NZ.*")))
    for station_dir in station_dirs:
        sta_code = os.path.basename(station_dir).split(".")[1]
        mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
        if not mseed_files:
            continue
        st = obspy.read(mseed_files[0], headonly=True)
        channels = sorted({tr.stats.channel for tr in st})
        vertical = next((c for c in channels if c.endswith("Z")), None)
        horizontal = next((c for c in channels if not c.endswith("Z")), None)
        if vertical:
            lookup[sta_code] = {"Z": vertical, "H": horizontal or vertical}
    return lookup


def load_event_stream(origin_time, stations, buffer=30.0):
    """Load a short window around the event's origin time per station.
    Determines the correct chunk file via its filename (which already
    encodes its date range, e.g. "...2019-08-01_2019-08-16.mseed") rather
    than opening every chunk file's header to check - avoids hammering the
    filesystem with unnecessary reads, which was causing severe I/O-wait
    slowdowns on a busy shared node during testing."""
    t0 = origin_time - buffer
    t1 = origin_time + buffer
    st = obspy.Stream()
    for sta_code in stations:
        station_dir = os.path.join(WAVEFORM_DIR, f"NZ.{sta_code}")
        mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
        for f in mseed_files:
            # filename ends in ..._YYYY-MM-DD_YYYY-MM-DD.mseed
            date_part = os.path.basename(f).rsplit(".", 1)[0].split(".")[-1]
            try:
                start_str, end_str = date_part.split("_")
                chunk_start = UTCDateTime(start_str)
                chunk_end = UTCDateTime(end_str) + 86400  # end date is inclusive
            except (ValueError, IndexError):
                continue
            if chunk_start <= t1 and chunk_end >= t0:
                try:
                    st += obspy.read(f, starttime=t0, endtime=t1)
                except Exception:
                    continue
    return st


def main(limit=None, chunk_size=100):
    filtered = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    assignments = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_pyocto_assignments.csv"))

    # Map .hyp filename -> PyOcto event_idx (E00000000 -> 0, E00000001 -> 1, ...)
    filtered["event_idx"] = filtered["file"].str.extract(r"_E(\d+)\.")[0].astype(int)

    # Sort by latitude before chunking: the TVZ is a roughly NNE-trending
    # linear rift, so this keeps geographically close events more likely to
    # land in the same or an adjacent chunk than an arbitrary/chronological
    # ordering would. Not a perfect spatial clustering, but a cheap and
    # reasonable improvement given the time available - some cross-chunk
    # event pairs within max_sep will still be missed at chunk boundaries;
    # documented limitation, not a silent gap.
    filtered = filtered.sort_values("lat").reset_index(drop=True)

    events_to_process = filtered
    if limit:
        events_to_process = events_to_process.head(limit)
    print(f"Processing {len(events_to_process)} events in chunks of {chunk_size}", flush=True)

    channel_lookup = get_channel_codes()
    print(f"Channel lookup built for {len(channel_lookup)} stations", flush=True)

    # Consistent event ID mapping across ALL chunks - critical so dt.cc
    # references stay valid once chunk outputs are concatenated. Without
    # this, each chunk's write_correlations() call could assign the same
    # PyOcto event a different integer ID, corrupting the combined file.
    event_id_mapper = {f"event_{int(row['event_idx'])}": int(row["event_idx"]) + 1
                        for _, row in events_to_process.iterrows()}

    os.chdir(OUTPUT_DIR)
    n_chunks = -(-len(events_to_process) // chunk_size)  # ceiling division
    chunk_files = []

    for chunk_i in range(n_chunks):
        chunk_dtcc = f"dt_chunk{chunk_i}.cc"
        chunk_files.append(chunk_dtcc)

        if os.path.exists(chunk_dtcc):
            print(f"=== Chunk {chunk_i + 1}/{n_chunks}: already done, skipping ===", flush=True)
            continue

        chunk = events_to_process.iloc[chunk_i * chunk_size:(chunk_i + 1) * chunk_size]
        print(f"=== Chunk {chunk_i + 1}/{n_chunks} ({len(chunk)} events) ===", flush=True)

        catalog = Catalog()
        stream_dict = {}

        for i, (_, row) in enumerate(chunk.iterrows()):
            event_idx = row["event_idx"]
            event_picks = assignments[assignments["event_idx"] == event_idx]
            if event_picks.empty:
                continue

            resource_id = f"event_{event_idx}"
            origin_time = UTCDateTime(row["origin_time"])
            event = Event(resource_id=resource_id)
            origin = Origin(time=origin_time, latitude=row["lat"], longitude=row["lon"],
                             depth=row["depth_km"] * 1000.0)
            event.origins.append(origin)

            stations_used = set()
            for _, pick_row in event_picks.iterrows():
                parts = pick_row["station"].split(".")
                sta_code = parts[1]
                if sta_code not in channel_lookup:
                    continue
                chan = channel_lookup[sta_code]["Z"] if pick_row["phase"] == "P" else channel_lookup[sta_code]["H"]
                wid = WaveformStreamID(network_code="NZ", station_code=sta_code,
                                        location_code=parts[2] if len(parts) > 2 else "",
                                        channel_code=chan)
                pick = Pick(time=UTCDateTime(pick_row["time"]), waveform_id=wid,
                            phase_hint=pick_row["phase"])
                event.picks.append(pick)
                stations_used.add(sta_code)

            if len(event.picks) < 2:
                continue

            catalog.events.append(event)
            stream_dict[resource_id] = load_event_stream(origin_time, stations_used)

            if (i + 1) % 25 == 0 or (i + 1) == len(chunk):
                print(f"  Loaded {i + 1}/{len(chunk)} events in this chunk", flush=True)

        write_correlations(
            catalog=catalog,
            stream_dict=stream_dict,
            extract_len=EXTRACT_LEN,
            pre_pick=PRE_PICK,
            shift_len=SHIFT_LEN,
            event_id_mapper=event_id_mapper,
            lowcut=LOWCUT,
            highcut=HIGHCUT,
            max_sep=MAX_SEP,
            min_link=MIN_LINK,
            min_cc=MIN_CC,
            all_horiz=True,
        )
        if os.path.exists("dt.cc"):
            os.rename("dt.cc", chunk_dtcc)
        else:
            open(chunk_dtcc, "w").close()  # no correlations found in this chunk - still mark it done

        del catalog, stream_dict
        print(f"  Chunk {chunk_i + 1} complete -> {chunk_dtcc}", flush=True)

    # Concatenate all chunk outputs into the final dt.cc GrowClust3D expects
    with open("dt.cc", "w") as outfile:
        for cf in chunk_files:
            if os.path.exists(cf):
                with open(cf) as infile:
                    outfile.write(infile.read())
    print(f"Final dt.cc assembled from {len(chunk_files)} chunks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N events (for testing)")
    parser.add_argument("--chunk-size", type=int, default=100,
                         help="Number of events per chunk (bounds memory use)")
    args = parser.parse_args()
    main(limit=args.limit, chunk_size=args.chunk_size)
