#!/usr/bin/env python
"""
Week 3: Continuous waveform download for the TVZ project.

Window: 1 Aug - 30 Sep 2019 (2019 Taupo unrest + M5.2 mainshock).
Network: NZ (GeoNet permanent network) only - BARNZ (YA) excluded per the
Week 2 finding that it covers the Bay of Plenty, not the Taupo caldera.

NOTE: This replaces an earlier version that used ObsPy's MassDownloader.
MassDownloader's internal "availability" check crashed against GeoNet's
response (ValueError: could not convert string to float: ''), most likely
triggered by infrasound/pressure channels (HD/LD bands, e.g. at COVZ) which
lack standard orientation metadata. Rather than debug that library internal
under time pressure, this downloads directly via Client.get_waveforms() -
the same call already proven to work in the Week 1 toy test - driven by the
station inventory already downloaded in Week 2, one request per station
covering the full window.

NOTE 2: A second issue showed up on the first real run — GeoNet's dataselect
service caps each request's result set at 60 files (documented at
https://service.geonet.org.nz/fdsnws/dataselect/1/ : "The result set is
limited to 60 files OR 30 minutes"). Our full 61-day, 3-component wildcard
request per station was generating more files internally than that cap
allows. Fixed by chunking each station's request into ~15-day windows
(15 days x 3 components = ~45 files, comfortably under 60).

Safe to resubmit if interrupted: chunk-files that already exist are skipped.
"""
import os
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client

PROJECT_DIR = os.path.expanduser("~/tvz_project")
WAVEFORM_DIR = os.path.join(PROJECT_DIR, "data", "waveforms")
INVENTORY_FILE = os.path.join(PROJECT_DIR, "data", "inventory", "NZ_2019-08-01_2019-09-30.xml")
os.makedirs(WAVEFORM_DIR, exist_ok=True)

STARTTIME = UTCDateTime("2019-08-01T00:00:00")
ENDTIME = UTCDateTime("2019-09-30T23:59:59")

# Broadband first, then short period, then strong-motion as a last resort.
# Deliberately excludes HD/LD (infrasound/pressure) and other non-seismometer bands.
PREFERRED_BANDS = ["HH", "EH", "BH"]

CHUNK_DAYS = 15  # keeps each request's file count well under GeoNet's 60-file cap


def date_chunks(start, end, max_days):
    chunks = []
    cur = start
    step = max_days * 86400
    while cur < end:
        chunk_end = min(cur + step, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end
    return chunks


def select_channel_band(station):
    bands = {cha.code[:2] for cha in station.channels}
    for band in PREFERRED_BANDS:
        if band in bands:
            return band
    return None


if __name__ == "__main__":
    inventory = read_inventory(INVENTORY_FILE)
    client = Client("GEONET")
    chunks = date_chunks(STARTTIME, ENDTIME, CHUNK_DAYS)
    print(f"Window split into {len(chunks)} chunks of up to {CHUNK_DAYS} days each")

    targets = []
    for net in inventory:
        for sta in net:
            band = select_channel_band(sta)
            if band:
                targets.append((net.code, sta.code, band))
            else:
                print(f"Skipping {net.code}.{sta.code}: no HH/EH/BH channel available")

    print(f"{len(targets)} stations selected for download (of {sum(len(n) for n in inventory)} total)")

    n_ok, n_skipped, n_failed = 0, 0, 0
    for net_code, sta_code, band in targets:
        station_dir = os.path.join(WAVEFORM_DIR, f"{net_code}.{sta_code}")
        os.makedirs(station_dir, exist_ok=True)

        for chunk_start, chunk_end in chunks:
            tag = f"{chunk_start.strftime('%Y-%m-%d')}_{chunk_end.strftime('%Y-%m-%d')}"
            out_path = os.path.join(station_dir, f"{net_code}.{sta_code}.{band}.{tag}.mseed")

            if os.path.exists(out_path):
                n_skipped += 1
                continue

            try:
                st = client.get_waveforms(
                    network=net_code, station=sta_code, location="*",
                    channel=f"{band}?", starttime=chunk_start, endtime=chunk_end,
                )
                if len(st) == 0:
                    print(f"{net_code}.{sta_code} [{tag}]: no data returned")
                    n_failed += 1
                    continue
                st.write(out_path, format="MSEED")
                n_ok += 1
            except Exception as e:
                print(f"{net_code}.{sta_code} [{tag}]: FAILED — {e}")
                n_failed += 1

        print(f"{net_code}.{sta_code}: done ({len(chunks)} chunks processed)")

    print(f"\nDone. {n_ok} chunk-files downloaded, {n_skipped} already present, {n_failed} failed "
          f"(out of {len(targets) * len(chunks)} total chunk-requests attempted).")
