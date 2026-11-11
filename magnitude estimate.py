#!/usr/bin/env python
"""
Week 7 (extra task): Local magnitude (ML) estimation for the EQTransformer/
PyOcto/NonLinLoc/GrowClust3D catalogue.

Uses the real, published Taupo-region local magnitude formula from
Illsley-Kemp et al. (2021), "Volcanic Unrest at Taupo Volcano in 2019",
Geochemistry Geophysics Geosystems, confirmed directly from the paper
(not the generic Hutton & Boore 1987 base formula, which was the fallback
plan before the actual paper text was obtained):

    ML = log10(A) - log10(A0) + C                              ... Eq 1
    -log10(A0) = 1.5337*log10(r/17) + 0.002343*(r-17) + 2       ... Eq 2

where A = Wood-Anderson peak-to-trough displacement (mm), measured on the
VERTICAL component (per the paper's stated method), r = hypocentral
distance (km), C = per-station correction (set to 0.0 here - the paper's
per-station correction table was not available to us, so this is a
documented simplification, not the full published method).

Response removal uses the real instrument response downloaded in Week 2
(level="response"), followed by ObsPy's standard Wood-Anderson simulation.

Aggregates per-station ML to a single event magnitude via the median.

Run via sbatch (submit_magnitude.sh). Resumable: events already in the
output CSV are skipped. Use --limit N to test on a subset first.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.signal.invsim import WOODANDERSON
from obspy.geodetics import gps2dist_azimuth

PROJECT_DIR = os.path.expanduser("~/tvz_project")
WAVEFORM_DIR = os.path.join(PROJECT_DIR, "data", "waveforms")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
INVENTORY_FILE = os.path.join(PROJECT_DIR, "data", "inventory", "NZ_2019-08-01_2019-09-30.xml")

STATION_CORRECTION = 0.0  # documented simplification - no per-station correction table available


def get_vertical_channel_codes():
    """Real vertical-channel code per station, from the actual downloaded files."""
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
        if vertical:
            lookup[sta_code] = vertical
    return lookup


def load_trace(sta_code, channel, t0, t1):
    station_dir = os.path.join(WAVEFORM_DIR, f"NZ.{sta_code}")
    mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
    for f in mseed_files:
        date_part = os.path.basename(f).rsplit(".", 1)[0].split(".")[-1]
        try:
            start_str, end_str = date_part.split("_")
            chunk_start = UTCDateTime(start_str)
            chunk_end = UTCDateTime(end_str) + 86400
        except (ValueError, IndexError):
            continue
        if chunk_start <= t1 and chunk_end >= t0:
            try:
                st = obspy.read(f, starttime=t0, endtime=t1)
                st = st.select(channel=channel)
                if len(st) > 0:
                    return st[0]
            except Exception:
                continue
    return None


def compute_station_ml(trace, inventory, sta_code, channel, distance_km):
    try:
        tr = trace.copy()
        tr.detrend("demean")
        tr.remove_response(inventory=inventory, output="DISP", water_level=60)
        tr.simulate(paz_simulate=WOODANDERSON)
    except Exception:
        return None

    if len(tr.data) == 0:
        return None

    amplitude_mm = (tr.data.max() - tr.data.min()) * 1000.0  # m -> mm
    if amplitude_mm <= 0:
        return None

    r = distance_km
    # Eq 2 gives -log10(A0) directly; Eq 1 is ML = log10(A) - log10(A0) + C
    # => ML = log10(A) + [-log10(A0)] + C
    neg_log_a0 = 1.5337 * np.log10(r / 17.0) + 0.002343 * (r - 17.0) + 2.0
    ml = np.log10(amplitude_mm) + neg_log_a0 + STATION_CORRECTION
    return ml


def main(limit=None):
    filtered = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    assignments = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_pyocto_assignments.csv"))
    filtered["event_idx"] = filtered["file"].str.extract(r"_E(\d+)\.")[0].astype(int)

    inventory = obspy.read_inventory(INVENTORY_FILE)
    station_coords = {}
    for net in inventory:
        for sta in net:
            station_coords[sta.code] = (sta.latitude, sta.longitude, sta.elevation)

    vertical_channels = get_vertical_channel_codes()
    print(f"{len(vertical_channels)} stations with vertical-channel data", flush=True)

    out_path = os.path.join(CATALOG_DIR, "eqtransformer_magnitudes.csv")
    done_events = set()
    if os.path.exists(out_path):
        done_events = set(pd.read_csv(out_path)["event_idx"])
        print(f"Resuming: {len(done_events)} events already done", flush=True)

    events_to_process = filtered[~filtered["event_idx"].isin(done_events)]
    if limit:
        events_to_process = events_to_process.head(limit)
    print(f"Processing {len(events_to_process)} events", flush=True)

    rows = []
    for i, (_, row) in enumerate(events_to_process.iterrows()):
        event_idx = row["event_idx"]
        event_picks = assignments[assignments["event_idx"] == event_idx]
        stations_for_event = event_picks["station"].str.split(".").str[1].unique()

        origin_time = UTCDateTime(row["origin_time"])
        station_mls = []

        for sta_code in stations_for_event:
            if sta_code not in vertical_channels or sta_code not in station_coords:
                continue
            sta_lat, sta_lon, sta_elev = station_coords[sta_code]
            dist_m, _, _ = gps2dist_azimuth(row["lat"], row["lon"], sta_lat, sta_lon)
            dist_km = dist_m / 1000.0
            depth_km = row["depth_km"] + sta_elev / 1000.0  # approx hypocentral adjustment
            hypo_dist_km = np.sqrt(dist_km ** 2 + depth_km ** 2)

            channel = vertical_channels[sta_code]
            trace = load_trace(sta_code, channel, origin_time - 2, origin_time + 15)
            if trace is None:
                continue

            ml = compute_station_ml(trace, inventory, sta_code, channel, hypo_dist_km)
            if ml is not None and -2 < ml < 8:  # sanity bound, discard nonsensical outliers
                station_mls.append(ml)

        if station_mls:
            rows.append({
                "event_idx": event_idx,
                "n_stations": len(station_mls),
                "ml_median": float(np.median(station_mls)),
                "ml_std": float(np.std(station_mls)) if len(station_mls) > 1 else 0.0,
            })
        else:
            rows.append({"event_idx": event_idx, "n_stations": 0, "ml_median": None, "ml_std": None})

        if (i + 1) % 50 == 0 or (i + 1) == len(events_to_process):
            print(f"  Processed {i + 1}/{len(events_to_process)} events", flush=True)

    new_df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(new_df)} total events)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N remaining events (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)
