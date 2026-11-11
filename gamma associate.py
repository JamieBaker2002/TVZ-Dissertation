#!/usr/bin/env python
"""
Week 9: GaMMA phase association for the PhaseNet catalogue (the proposal's
specified associator for Phase 2, distinct from PyOcto used for Phase 1).

API confirmed directly against AI4EPS's own official examples and source
(app.py) rather than assumed:
  - association(picks, stations, config, event_idx0=0, method="BGMM")
    returns (events, assignments)
  - picks dataframe: columns "id" (station label), "timestamp" (datetime),
    "prob" (pick probability), "type" (phase, LOWERCASE "p"/"s")
  - stations dataframe: "id", plus "x(km)"/"y(km)" (from a projection) and
    "z(km)" (= -elevation_m / 1000, i.e. depth-positive-down convention)
  - assignments: list of (pick_index, event_index, gamma_score) tuples

Projection: aeqd (Azimuthal Equidistant) centred on the same reference point
(Lake Taupo, -38.8/176.0) used throughout this project (same choice already
made for GrowClust3D's trace1D mode) - not a new, inconsistent choice.

Velocity: same real, citable TVZ values used for PyOcto's homogeneous model
in Week 5 (Vp=5.7 km/s, Stratford et al. 2002; Vp/Vs=1.7).

Association quality parameters (min 10 picks, 5 P + 2 S) match PyOcto's
settings from Week 5, for a fair Phase 4 comparison between the two
association methods, not because GaMMA requires these specific values.

Station labels use the same full "NZ.STATION.LOCATION" convention as
throughout the project, matching our existing NonLinLoc GTSRCE labels
directly - no renaming step needed when building the .obs file later.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import obspy
from pyproj import Proj
from gamma.utils import association

PROJECT_DIR = os.path.expanduser("~/tvz_project")
PICKS_DIR = os.path.join(PROJECT_DIR, "picks", "phasenet")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
INVENTORY_FILE = os.path.join(PROJECT_DIR, "data", "inventory", "NZ_2019-08-01_2019-09-30.xml")

TARGET_LAT, TARGET_LON = -38.8, 176.0
VP = 5.7
VS = 5.7 / 1.7


def decluster(df, window=5.0):
    """Same declustering rule used for EQTransformer in Week 5."""
    kept = []
    for (station, phase), group in df.groupby(["station", "phase"]):
        group = group.sort_values("time_unix").reset_index(drop=True)
        cluster_start = 0
        for i in range(1, len(group) + 1):
            if i == len(group) or (group.loc[i, "time_unix"] - group.loc[i - 1, "time_unix"]) > window:
                cluster = group.loc[cluster_start:i - 1]
                kept.append(cluster.loc[cluster["probability"].idxmax()])
                cluster_start = i
    return pd.DataFrame(kept).reset_index(drop=True)


def main(limit_picks=None):
    picks_files = sorted(glob.glob(os.path.join(PICKS_DIR, "*_thresh0.3_picks.csv")))
    print(f"{len(picks_files)} station pick files found", flush=True)

    all_picks = pd.concat([pd.read_csv(f) for f in picks_files], ignore_index=True)
    all_picks["time_dt"] = pd.to_datetime(all_picks["time"])
    all_picks["time_unix"] = all_picks["time_dt"].astype("int64") / 1e9
    print(f"Raw picks loaded: {len(all_picks)}", flush=True)

    declustered = decluster(all_picks)
    print(f"After declustering: {len(declustered)} "
          f"({100 * len(declustered) / len(all_picks):.1f}% retained)", flush=True)
    declustered.to_csv(os.path.join(PICKS_DIR, "phasenet_declustered.csv"), index=False)

    if limit_picks:
        declustered = declustered.sort_values("time_unix").head(limit_picks)
        print(f"--limit-picks set: using first {len(declustered)} picks only (testing)", flush=True)

    # Pick dataframe in GaMMA's required format
    pick_df = pd.DataFrame({
        "id": declustered["station"],
        "timestamp": declustered["time_dt"],
        "prob": declustered["probability"],
        "type": declustered["phase"].str.lower(),
    })

    # Station dataframe: same real inventory used throughout the project
    inventory = obspy.read_inventory(INVENTORY_FILE)
    proj = Proj(proj="aeqd", lon_0=TARGET_LON, lat_0=TARGET_LAT, ellps="WGS84", units="km")

    # Lookup: bare station code -> (lat, lon, elevation_m)
    station_coords = {}
    for net in inventory:
        for sta in net:
            station_coords[sta.code] = (sta.latitude, sta.longitude, sta.elevation)

    used_stations = sorted(declustered["station"].unique())  # full "NZ.STA.LOC" labels
    station_rows = []
    for label in used_stations:
        bare_code = label.split(".")[1]
        if bare_code not in station_coords:
            continue
        lat, lon, elev = station_coords[bare_code]
        x_km, y_km = proj(lon, lat)
        station_rows.append({
            "id": label, "longitude": lon, "latitude": lat,
            "x(km)": x_km, "y(km)": y_km, "z(km)": -elev / 1000.0,
        })
    stations_df = pd.DataFrame(station_rows)
    print(f"{len(stations_df)} stations prepared for association", flush=True)

    config = {
        "dims": ["x(km)", "y(km)", "z(km)"],
        "vel": {"p": VP, "s": VS},  # GaMMA wants this dict form, not vp/vs directly -
                                     # confirmed from source (utils.py line 163); our
                                     # earlier vp/vs keys were being silently ignored
                                     # in favour of GaMMA's own generic default (6.0)
        "use_dbscan": True,
        "dbscan_eps": 10.0,
        "dbscan_min_samples": 3,
        "dbscan_min_cluster_size": 10,  # default (500) would eliminate every cluster
                                         # at our data scale - overridden to match
        "use_amplitude": False,
        "method": "BGMM",
        "oversample_factor": 10,
        "min_picks_per_eq": 10,
        "min_p_picks_per_eq": 5,
        "min_s_picks_per_eq": 2,
        "min_stations": 3,       # lower confidence - reasonable estimate, revisit if
                                  # results look off rather than erroring
        "covariance_prior": (5.0, 5.0),  # lower confidence - reasonable estimate
        "max_sigma11": 2.0,              # lower confidence - reasonable estimate
        "x(km)": (stations_df["x(km)"].min() - 10, stations_df["x(km)"].max() + 10),
        "y(km)": (stations_df["y(km)"].min() - 10, stations_df["y(km)"].max() + 10),
        "z(km)": (-3, 40),
        "bfgs_bounds": (
            (stations_df["x(km)"].min() - 10, stations_df["x(km)"].max() + 10),
            (stations_df["y(km)"].min() - 10, stations_df["y(km)"].max() + 10),
            (-3, 40),
            (None, None),
        ),
    }

    print("Running GaMMA association...", flush=True)
    events, assignments = association(pick_df, stations_df, config, method=config["method"])

    if events is None or len(events) == 0:
        print("No events associated - check config/parameters", flush=True)
        return

    events_df = pd.DataFrame(events)
    events_df[["longitude", "latitude"]] = events_df.apply(
        lambda row: pd.Series(proj(row["x(km)"], row["y(km)"], inverse=True)), axis=1
    )
    events_df["depth_km"] = events_df["z(km)"]

    assignments_df = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])

    events_out = os.path.join(CATALOG_DIR, "phasenet_gamma_events.csv")
    assignments_out = os.path.join(CATALOG_DIR, "phasenet_gamma_assignments.csv")
    events_df.to_csv(events_out, index=False)
    assignments_df.to_csv(assignments_out, index=False)

    print(f"Candidate events found: {len(events_df)}", flush=True)
    print(f"Picks assigned to events: {len(assignments_df)} of {len(pick_df)} declustered picks", flush=True)
    print(f"Saved: {events_out}", flush=True)
    print(f"Saved: {assignments_out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-picks", type=int, default=None,
                         help="Use only the first N (time-sorted) declustered picks, for testing")
    args = parser.parse_args()
    main(limit_picks=args.limit_picks)
