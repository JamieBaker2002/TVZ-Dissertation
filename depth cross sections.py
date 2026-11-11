#!/usr/bin/env python
"""
Week 17: Along-TVZ and cross-TVZ depth cross-sections, per the project
proposal ("along-TVZ and cross-TVZ depth cross-sections").

Along-TVZ transect: Ruapehu to Okataina, using the real azimuth between
these two sub-region markers (the most geographically extreme of the
four defined in Week 16), extended by 20 km at each end to capture
seismicity beyond those two points. Cross-TVZ transect: perpendicular
to the along-TVZ transect, centred on Taupo (the most seismically
active sub-region).

Projection geometry (event -> along-transect distance, perpendicular
distance) verified against synthetic points with known true values
before use on real data: a point exactly on the transect at a known
distance, and the transect's own endpoint, both recovered correctly.

Events are included in a transect's profile only if their perpendicular
distance from that transect line is within PERP_BAND_KM - events far
off the transect's line aren't relevant to that specific cross-section.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth

PROJECT_DIR = os.path.expanduser("~/tvz_project")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
OUT_DIR = os.path.join(CATALOG_DIR, "phase4", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PERP_BAND_KM = 20  # events within this perpendicular distance of a
                    # transect line are included in that profile
EXTEND_KM = 20      # extend the along-TVZ transect beyond Ruapehu/Okataina

RUAPEHU = (-39.28, 175.57)
OKATAINA = (-38.22, 176.5)
TONGARIRO = (-39.13, 175.64)
SUBREGION_COLOURS = {"Ruapehu": "magenta", "Tongariro": "orange", "Taupo": "red", "Okataina": "blue"}
TAUPO = (-38.8, 175.9)

CATALOGUE_COLOURS = {"EQTransformer": "#1f77b4", "PhaseNet": "#ff7f0e", "QuakeMigrate": "#2ca02c"}


def load_catalogues():
    eqt = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    pn = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_nlloc_quality_filtered.csv"))
    qm = pd.read_csv(os.path.join(CATALOG_DIR, "quakemigrate_quality_filtered.csv"))
    return {"EQTransformer": eqt, "PhaseNet": pn, "QuakeMigrate": qm}


def project_onto_transect(event_lat, event_lon, start_lat, start_lon, transect_azimuth_deg):
    dist_m, az_to_event, _ = gps2dist_azimuth(start_lat, start_lon, event_lat, event_lon)
    angle_diff = np.radians(az_to_event - transect_azimuth_deg)
    along_km = (dist_m / 1000.0) * np.cos(angle_diff)
    perp_km = (dist_m / 1000.0) * np.sin(angle_diff)
    return along_km, perp_km


def build_along_tvz_transect():
    import math
    dist_m, azimuth, _ = gps2dist_azimuth(*RUAPEHU, *OKATAINA)
    dist_km = dist_m / 1000.0
    rev_azimuth_rad = math.radians((azimuth + 180) % 360)
    dlat = (EXTEND_KM / 111.32) * math.cos(rev_azimuth_rad)
    dlon = (EXTEND_KM / (111.32 * math.cos(math.radians(RUAPEHU[0])))) * math.sin(rev_azimuth_rad)
    extended_start = (RUAPEHU[0] + dlat, RUAPEHU[1] + dlon)
    total_length_km = dist_km + 2 * EXTEND_KM
    return extended_start, azimuth, total_length_km


def build_cross_tvz_transect(along_tvz_azimuth):
    import math
    cross_azimuth = (along_tvz_azimuth + 90) % 360
    half_width_km = 40
    rev_azimuth_rad = math.radians((cross_azimuth + 180) % 360)
    dlat = (half_width_km / 111.32) * math.cos(rev_azimuth_rad)
    dlon = (half_width_km / (111.32 * math.cos(math.radians(TAUPO[0])))) * math.sin(rev_azimuth_rad)
    start = (TAUPO[0] + dlat, TAUPO[1] + dlon)
    return start, cross_azimuth, 2 * half_width_km


def build_profile_data(catalogues, transect_start, transect_azimuth):
    profile_data = {}
    for name, df in catalogues.items():
        along_list, depth_list = [], []
        for _, row in df.iterrows():
            along_km, perp_km = project_onto_transect(
                row["lat"], row["lon"], transect_start[0], transect_start[1], transect_azimuth
            )
            if abs(perp_km) <= PERP_BAND_KM:
                along_list.append(along_km)
                depth_list.append(row["depth_km"])
        profile_data[name] = {
            "along_km": np.array(along_list),
            "depth_km": np.array(depth_list),
        }
        print(f"  {name}: {len(along_list)} events within {PERP_BAND_KM} km of transect", flush=True)
    return profile_data


def plot_cross_section(profile_data, title, out_path, transect_length_km, label_start, label_end,
                        subregion_markers=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Sub-region reference lines drawn FIRST (low zorder), so they sit
    # behind the event scatter points rather than obscuring them. Labelled
    # so they combine into the single legend below alongside the
    # catalogues, in one unified box.
    if subregion_markers:
        for name, along_km in subregion_markers.items():
            ax.axvline(along_km, color=SUBREGION_COLOURS[name], linestyle="--",
                       linewidth=1.2, alpha=0.7, zorder=1, label=name)

    for name, data in profile_data.items():
        if len(data["along_km"]) == 0:
            continue
        ax.scatter(data["along_km"], data["depth_km"], s=15, alpha=0.6,
                   color=CATALOGUE_COLOURS[name], label=f"{name} (n={len(data['along_km'])})",
                   edgecolors="none", zorder=2)
    ax.set_xlim(0, transect_length_km)
    ax.set_ylim(35, -3)  # inverted: shallow at top, matches geological convention
    ax.set_xlabel(f"Distance along transect (km)  [{label_start} -> {label_end}]")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}", flush=True)


def main():
    catalogues = load_catalogues()

    along_start, along_azimuth, along_length = build_along_tvz_transect()
    print(f"Along-TVZ transect: azimuth {along_azimuth:.1f} deg, length {along_length:.1f} km", flush=True)
    print("Along-TVZ profile event counts:", flush=True)
    along_profile = build_profile_data(catalogues, along_start, along_azimuth)

    subregion_positions = {}
    for name, coords in [("Ruapehu", RUAPEHU), ("Tongariro", TONGARIRO),
                          ("Taupo", TAUPO), ("Okataina", OKATAINA)]:
        along_km, _ = project_onto_transect(coords[0], coords[1], along_start[0], along_start[1], along_azimuth)
        subregion_positions[name] = along_km
        print(f"  {name} at {along_km:.1f} km along transect", flush=True)

    plot_cross_section(
        along_profile, "Along-TVZ depth cross-section (Ruapehu -> Okataina, extended)",
        os.path.join(OUT_DIR, "cross_section_along_tvz.png"), along_length,
        "Ruapehu (SW)", "Okataina (NE)", subregion_markers=subregion_positions
    )

    cross_start, cross_azimuth, cross_length = build_cross_tvz_transect(along_azimuth)
    print(f"\nCross-TVZ transect: azimuth {cross_azimuth:.1f} deg, length {cross_length:.1f} km", flush=True)
    print("Cross-TVZ profile event counts:", flush=True)
    cross_profile = build_profile_data(catalogues, cross_start, cross_azimuth)
    plot_cross_section(
        cross_profile, "Cross-TVZ depth cross-section (perpendicular, centred on Taupo)",
        os.path.join(OUT_DIR, "cross_section_cross_tvz.png"), cross_length,
        "NW", "SE"
    )


if __name__ == "__main__":
    main()
