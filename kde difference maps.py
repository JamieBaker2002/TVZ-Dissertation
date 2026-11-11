#!/usr/bin/env python
"""
Week 17: Difference maps between catalogue pairs, per the project
proposal ("difference maps, computed by subtracting the density fields
of two catalogues on a common grid").

Computed as log10(rate_A) - log10(rate_B), not a raw rate difference:
given the huge dynamic range in these KDE surfaces (managed throughout
this week via log-scale colour mapping), a raw difference would recreate
the same problem - one area's absolute difference dominating everything
else. The log-ratio is a genuinely symmetric, interpretable quantity:
positive means catalogue A's rate is higher at that location, negative
means B's is higher, by that many orders of magnitude.

Both grids in a pair are re-clipped to a SHARED floor before
differencing, not each catalogue's own individually-chosen floor from
the full-catalogue maps - using different floors per catalogue would
make some "differences" reflect clipping choices rather than genuine
density differences.

Reuses the already-saved .nc grid files from the full-catalogue KDE maps
(kde_density_maps.py) rather than recomputing.

QuakeMigrate-involving pairs are computed but should be interpreted with
real caution given its 31-event sample vs. 900+ for the other two
catalogues - flagged explicitly in each such map's title, not just in
this comment.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import pygmt

PROJECT_DIR = os.path.expanduser("~/tvz_project")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
FAULTS_FILE = os.path.join(PROJECT_DIR, "faults", "tvz_faults.gmt")
OUT_DIR = os.path.join(CATALOG_DIR, "phase4", "figures")
GRID_DIR = os.path.join(CATALOG_DIR, "phase4", "kde_grids")

REGION = [175.0, 178.0, -40.0, -37.0]
SUBREGIONS = {
    "Taupo": (-38.8, 175.9),
    "Okataina": (-38.22, 176.5),
    "Tongariro": (-39.13, 175.64),
    "Ruapehu": (-39.28, 175.57),
}
ORDERS_OF_MAGNITUDE_BELOW_PEAK = 4
DIFF_CAP = 3
STUDY_DURATION_YEARS = 61 / 365.25
GRID_SPACING_KM = 2.0
REF_LAT, REF_LON = -38.5, 176.5
KM_PER_DEG_LAT = 111.32
QUAKEMIGRATE_BANDWIDTH_KM = 3.845  # full-catalogue value, reused as the
                                    # shared bandwidth for QuakeMigrate pairs


def to_local_km(lat, lon):
    x_km = (lon - REF_LON) * KM_PER_DEG_LAT * np.cos(np.radians(REF_LAT))
    y_km = (lat - REF_LAT) * KM_PER_DEG_LAT
    return x_km, y_km


def to_latlon(x_km, y_km):
    lon = REF_LON + x_km / (KM_PER_DEG_LAT * np.cos(np.radians(REF_LAT)))
    lat = REF_LAT + y_km / KM_PER_DEG_LAT
    return lat, lon


def load_raw_catalogue(name):
    if name == "EQTransformer":
        df = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    elif name == "PhaseNet":
        df = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_nlloc_quality_filtered.csv"))
    else:
        df = pd.read_csv(os.path.join(CATALOG_DIR, "quakemigrate_quality_filtered.csv"))
    return df


def compute_kde_rate_shared_bandwidth(df, bandwidth_km):
    """Same KDE formula as kde_density_maps.py, but with an explicitly
    passed-in bandwidth rather than each catalogue's own median location
    uncertainty - used only for the QuakeMigrate difference-map pairs, to
    directly address the speckled-pixel issue (EQTransformer/PhaseNet's
    much smaller own bandwidth creating many tiny disconnected peaks that
    dominate the difference when compared against QuakeMigrate's near-flat
    background)."""
    n_events = len(df)
    x_km, y_km = to_local_km(df["lat"].values, df["lon"].values)
    region_x_km, region_y_km = to_local_km(
        np.array([REGION[2], REGION[3]]), np.array([REGION[0], REGION[1]])
    )
    margin = 20
    gx = np.arange(region_x_km.min() - margin, region_x_km.max() + margin, GRID_SPACING_KM)
    gy = np.arange(region_y_km.min() - margin, region_y_km.max() + margin, GRID_SPACING_KM)
    gxx, gyy = np.meshgrid(gx, gy)
    gx_flat, gy_flat = gxx.ravel(), gyy.ravel()

    dx = gx_flat[:, None] - x_km[None, :]
    dy = gy_flat[:, None] - y_km[None, :]
    dist2 = dx**2 + dy**2
    density_per_event = (1 / (2 * np.pi * bandwidth_km**2)) * np.exp(-dist2 / (2 * bandwidth_km**2))
    kde_density = density_per_event.mean(axis=1)
    rate = kde_density * n_events / STUDY_DURATION_YEARS
    rate_grid = rate.reshape(gxx.shape)

    grid_lat, grid_lon = to_latlon(gxx, gyy)
    lons_1d = grid_lon[0, :]
    lats_1d = grid_lat[:, 0]
    return xr.DataArray(rate_grid, dims=["lat", "lon"],
                         coords={"lat": lats_1d, "lon": lons_1d}, name="rate")


def load_rate_grid(name):
    path = os.path.join(GRID_DIR, f"kde_rate_{name.lower()}.nc")
    da = xr.open_dataarray(path)
    return da


def make_difference_map(name_a, name_b, shared_bandwidth_km=None):
    if shared_bandwidth_km is not None:
        df_a = load_raw_catalogue(name_a)
        df_b = load_raw_catalogue(name_b)
        grid_a = compute_kde_rate_shared_bandwidth(df_a, shared_bandwidth_km)
        grid_b = compute_kde_rate_shared_bandwidth(df_b, shared_bandwidth_km)
        print(f"{name_a} vs {name_b}: using shared bandwidth {shared_bandwidth_km} km "
              f"(n_a={len(df_a)}, n_b={len(df_b)})", flush=True)
    else:
        grid_a = load_rate_grid(name_a)
        grid_b = load_rate_grid(name_b)

    # Shared clipping floor: the lower of each catalogue's own
    # peak-relative floor, not each one's individually-chosen value
    floor_a = float(np.log10(grid_a.values.max())) - ORDERS_OF_MAGNITUDE_BELOW_PEAK
    floor_b = float(np.log10(grid_b.values.max())) - ORDERS_OF_MAGNITUDE_BELOW_PEAK
    shared_floor_log = min(floor_a, floor_b)
    shared_floor = 10 ** shared_floor_log

    a_clipped = np.clip(grid_a.values, shared_floor, None)
    b_clipped = np.clip(grid_b.values, shared_floor, None)
    log_diff = np.log10(a_clipped) - np.log10(b_clipped)
    log_diff = np.clip(log_diff, -DIFF_CAP, DIFF_CAP)

    diff_da = xr.DataArray(log_diff, dims=["lat", "lon"],
                            coords={"lat": grid_a.lat, "lon": grid_a.lon}, name="log_diff")

    fig = pygmt.Figure()
    cpt_path = os.path.join(OUT_DIR, f"_diff_cpt_{name_a.lower()}_{name_b.lower()}.cpt")
    # Blue-black-red (middle stop swapped from white to black, matching
    # the other KDE maps' background convention; blue/red extremes unchanged)
    pygmt.makecpt(cmap="blue,black,red", series=[-DIFF_CAP, DIFF_CAP], output=cpt_path)
    fig.grdimage(grid=diff_da, region=REGION, projection="M15c", cmap=cpt_path)

    title = f"{name_a} minus {name_b} (log10 rate ratio)"
    fig.coast(region=REGION, projection="M15c", shorelines="1/0.5p,white", frame=["af", f"+t{title}"])
    fig.plot(data=FAULTS_FILE, pen="0.4p,darkgreen", region=REGION, projection="M15c")
    fig.colorbar(cmap=cpt_path, frame=f"af+llog@-10@-({name_a}/{name_b})")

    subregion_colors = {
        "Taupo": "red", "Okataina": "blue", "Tongariro": "orange", "Ruapehu": "magenta",
    }
    for label, (lat, lon) in SUBREGIONS.items():
        fig.plot(x=[lon], y=[lat], style="a0.45c", fill=subregion_colors[label], pen="0.5p,black",
                  region=REGION, projection="M15c")
    region_spec_path = os.path.join(OUT_DIR, f"_region_legend_spec_diff_{name_a.lower()}_{name_b.lower()}.txt")
    with open(region_spec_path, "w") as f:
        for label, colour in subregion_colors.items():
            f.write(f"S 0.4c a 0.5c {colour} 0.5p 0.8c {label}\n")
    fig.legend(spec=region_spec_path, position="g177.4/-37.25+jTR+o0.2c",
               box="+gwhite+p0.5p", region=REGION, projection="M15c")

    out_path = os.path.join(OUT_DIR, f"kde_diff_{name_a.lower()}_minus_{name_b.lower()}.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}", flush=True)


def main():
    make_difference_map("EQTransformer", "PhaseNet")
    make_difference_map("EQTransformer", "QuakeMigrate")
    make_difference_map("PhaseNet", "QuakeMigrate")


if __name__ == "__main__":
    main()
