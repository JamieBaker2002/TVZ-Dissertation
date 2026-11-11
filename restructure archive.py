#!/usr/bin/env python
"""
Week 13: Restructure the downloaded waveform archive into daily files,
matching QuakeMigrate's "STATION.YEAR.JULIANDAY" built-in archive format
(*{station}.*.{year}.{jday:03d}), confirmed directly from QuakeMigrate's
own Archive.path_structure() source code.

QuakeMigrate's Archive class only recognises a handful of built-in
directory/file-naming conventions, all keyed to one file per station per
calendar day - our own download structure (15-day chunk files, a
deliberate choice for the download step itself, given GeoNet's rate
limits) doesn't match any of them. This is a genuine, mechanically simple
preprocessing step (splitting, not reprocessing), not a data quality
issue.

One station's data is processed per invocation (matching the same
per-station-per-process memory-safety pattern used throughout this
project since Week 4), intended for use with a SLURM job array.
"""
import os
import argparse
import glob
import obspy

PROJECT_DIR = os.path.expanduser("~/tvz_project")
WAVEFORM_DIR = os.path.join(PROJECT_DIR, "data", "waveforms")
QM_ARCHIVE_DIR = os.path.join(PROJECT_DIR, "data", "waveforms_qm")


def main(station_code):
    os.makedirs(QM_ARCHIVE_DIR, exist_ok=True)

    station_dir = os.path.join(WAVEFORM_DIR, f"NZ.{station_code}")
    mseed_files = sorted(glob.glob(os.path.join(station_dir, "*.mseed")))
    if not mseed_files:
        print(f"{station_code}: no waveform files found - skipping", flush=True)
        return

    n_written = 0
    for chunk_file in mseed_files:
        st = obspy.read(chunk_file)
        st.merge(method=0)
        st = st.split()  # break at gaps into clean contiguous traces, avoiding
                          # masked arrays that MiniSEED cannot write directly

        chunk_start = min(tr.stats.starttime for tr in st)
        chunk_end = max(tr.stats.endtime for tr in st)

        day = obspy.UTCDateTime(chunk_start.date)
        while day < chunk_end:
            day_end = day + 86400
            day_st = st.slice(starttime=day, endtime=day_end, nearest_sample=False)
            if len(day_st) == 0 or all(len(tr.data) == 0 for tr in day_st):
                day += 86400
                continue

            year = day.year
            doy = day.julday
            # QuakeMigrate's "STATION.YEAR.JULIANDAY" format matches the glob
            # pattern *{station}.*.{year}.{jday:03d} - this requires a middle
            # dot-separated segment between station and year (a single dot,
            # as in our first attempt, does not match). Using the band code
            # of the first trace as that segment.
            band = day_st[0].stats.channel[:2] if len(day_st[0].stats.channel) >= 2 else "XX"
            out_name = f"NZ.{station_code}.{band}.{year}.{doy:03d}"
            out_path = os.path.join(QM_ARCHIVE_DIR, out_name)

            if not os.path.exists(out_path):
                day_st.write(out_path, format="MSEED")
                n_written += 1
            day += 86400

        del st

    print(f"{station_code}: {n_written} daily files written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True,
                         help="Station code to process, e.g. ALRZ (one station per invocation)")
    args = parser.parse_args()
    main(station_code=args.station)
