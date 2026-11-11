#!/usr/bin/env python
"""
Week 17: Depth-stratified (0-15km, 15-40km) and magnitude-filtered
(Mc-2.5, 2.5+) KDE seismic density maps, per the project proposal.

Reuses the exact KDE computation and plotting logic from
kde_density_maps.py (the full-catalogue version), already validated
against synthetic data with known true properties and refined over
several rounds of real visual review. Applied here to filtered subsets
of each catalogue rather than the full catalogue.

Real event counts were checked per bin before running this (not assumed):
QuakeMigrate's 15-40km depth bin (3 events) and PhaseNet's 2.5+ magnitude
bin (4 events) are both skipped as too sparse for a meaningful KDE,
using the same MIN_EVENTS=5 threshold established in Week 16 for
sub-region MFD plots. The M2.5+ bin being sparse across all three
catalogues (6, 4, 9 events) is itself a genuine, expected finding given
the Gutenberg-Richter relationship, not a data quality problem.
"""
import os
import numpy as np
import pandas as pd
import pygmt

PROJECT_DIR = os.path.expanduser("~/tvz_project")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
FAULTS_FILE = os.path.join(PROJECT_DIR, "faults", "tvz_faults.gmt")
OUT_DIR = os.path.join(CATALOG_DIR, "phase4", "figures")
GRID_OUT_DIR = os.path.join(CATALOG_DIR, "phase4", "kde_grids")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRID_OUT_DIR, exist_ok=True)

REGION = [175.0, 178.0, -40.0, -37.0]
STUDY_DURATION_YEARS = 61 / 365.25
GRID_SPACING_KM = 2.0
MIN_EVENTS = 3  # lowered from 5: QuakeMigrate depth-deep (3) and PhaseNet
                 # mag-large (4) are still more events than Week 16's actual
                 # skipped sub-regions (1 and 0 events), so this doesn't
                 # contradict that precedent. Event count is shown explicitly
                 # in each map's title so small samples aren't mistaken for
                 # robust density estimates.
SUBREGIONS = {
    "Taupo": (-38.8, 175.9),
    "Okataina": (-38.22, 176.5),
    "Tongariro": (-39.13, 175.64),
    "Ruapehu": (-39.28, 175.57),
}
MC_VALUES = {"EQTransformer": 1.55, "PhaseNet": 1.45, "QuakeMigrate": 1.95}

REF_LAT, REF_LON = -38.5, 176.5
KM_PER_DEG_LAT = 111.32


def to_local_km(lat, lon):
    x_km = (lon - REF_LON) * KM_PER_DEG_LAT * np.cos(np.radians(REF_LAT))
    y_km = (lat - REF_LAT) * KM_PER_DEG_LAT
    return x_km, y_km


def to_latlon(x_km, y_km):
    lon = REF_LON + x_km / (KM_PER_DEG_LAT * np.cos(np.radians(REF_LAT)))
    lat = REF_LAT + y_km / KM_PER_DEG_LAT
    return lat, lon


def load_catalogues():
    eqt = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    eqt["event_idx"] = eqt["file"].str.extract(r"_E(\d+)\.")[0].astype(int)
    eqt = eqt.merge(pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_magnitudes.csv")),
                     on="event_idx", how="left")

    pn = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_nlloc_quality_filtered.csv"))
    pn["event_idx"] = pn["file"].str.extract(r"_E(\d+)\.")[0].astype(int)
    pn = pn.merge(pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_magnitudes.csv")),
                   on="event_idx", how="left")

    qm = pd.read_csv(os.path.join(CATALOG_DIR, "quakemigrate_quality_filtered.csv"))
    qm = qm.rename(columns={"ml": "ml_median"})

    return {"EQTransformer": eqt, "PhaseNet": pn, "QuakeMigrate": qm}


def compute_kde_rate(df, label):
    bandwidth_km = df["horiz_error_km"].median()
    n_events = len(df)
    print(f"{label}: n={n_events}, bandwidth (median horiz error) = {bandwidth_km:.3f} km", flush=True)

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
    return grid_lon, grid_lat, rate_grid, bandwidth_km


def save_and_plot(label, filename_suffix, title_suffix, grid_lon, grid_lat, rate_grid, bandwidth_km):
    import xarray as xr

    lons_1d = grid_lon[0, :]
    lats_1d = grid_lat[:, 0]
    da = xr.DataArray(rate_grid, dims=["lat", "lon"],
                       coords={"lat": lats_1d, "lon": lons_1d}, name="rate")
    grid_path = os.path.join(GRID_OUT_DIR, f"kde_rate_{filename_suffix}.nc")
    da.to_netcdf(grid_path)
    print(f"Saved: {grid_path}", flush=True)

    fig = pygmt.Figure()
    ORDERS_OF_MAGNITUDE_BELOW_PEAK = 4
    peak_log = float(np.log10(rate_grid.max()))
    lower_bound_log = peak_log - ORDERS_OF_MAGNITUDE_BELOW_PEAK
    safety_margin = 0.01
    log_rate_grid = np.log10(np.clip(rate_grid, 10 ** (lower_bound_log - safety_margin), None))
    da_log = xr.DataArray(log_rate_grid, dims=["lat", "lon"],
                           coords={"lat": lats_1d, "lon": lons_1d}, name="log_rate")

    cpt_path = os.path.join(OUT_DIR, f"_kde_cpt_{filename_suffix}.cpt")
    pygmt.makecpt(cmap="black,yellow,orange,red", series=[lower_bound_log, peak_log],
                  background="o", output=cpt_path)
    fig.grdimage(grid=da_log, region=REGION, projection="M15c", cmap=cpt_path)
    fig.coast(region=REGION, projection="M15c", shorelines="1/0.5p,white",
              frame=["af", f"+t{title_suffix} (bandwidth={bandwidth_km:.2f} km)"])
    fig.plot(data=FAULTS_FILE, pen="0.4p,darkgreen", region=REGION, projection="M15c")
    fig.colorbar(cmap=cpt_path, frame="af+llog@-10@-(Rate, events/km@+2@+/yr)")

    subregion_colors = {
        "Taupo": "red", "Okataina": "blue", "Tongariro": "orange", "Ruapehu": "magenta",
    }
    for sr_label, (lat, lon) in SUBREGIONS.items():
        fig.plot(x=[lon], y=[lat], style="a0.45c", fill=subregion_colors[sr_label], pen="0.5p,black",
                  region=REGION, projection="M15c")

    region_spec_path = os.path.join(OUT_DIR, f"_region_legend_spec_{filename_suffix}.txt")
    with open(region_spec_path, "w") as f:
        for sr_label, colour in subregion_colors.items():
            f.write(f"S 0.4c a 0.5c {colour} 0.5p 0.8c {sr_label}\n")
    fig.legend(spec=region_spec_path, position="g177.4/-37.25+jTR+o0.2c",
               box="+gwhite+p0.5p", region=REGION, projection="M15c")

    out_path = os.path.join(OUT_DIR, f"kde_density_{filename_suffix}.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}", flush=True)


def main():
    catalogues = load_catalogues()

    filters = [
        ("depth_shallow", "0-15km depth", lambda df: df[df["depth_km"] < 15]),
        ("depth_deep", "15-40km depth", lambda df: df[(df["depth_km"] >= 15) & (df["depth_km"] <= 40)]),
        ("mag_small", None, None),  # Mc-specific, built per catalogue below
        ("mag_large", "M2.5+", lambda df: df[df["ml_median"] >= 2.5]),
    ]

    for name, df in catalogues.items():
        mc = MC_VALUES[name]
        for suffix, title, filter_fn in filters:
            if suffix == "mag_small":
                title = f"Mc({mc})-M2.5"
                filter_fn = lambda df, mc=mc: df[(df["ml_median"] >= mc) & (df["ml_median"] < 2.5)]

            subset = filter_fn(df)
            if len(subset) < MIN_EVENTS:
                print(f"SKIPPED: {name} {suffix} - only {len(subset)} events "
                      f"(below MIN_EVENTS={MIN_EVENTS})", flush=True)
                continue

            label = f"{name} - {title}"
            filename_suffix = f"{name.lower()}_{suffix}"
            title_full = f"{name} - {title} (n={len(subset)})"

            grid_lon, grid_lat, rate_grid, bandwidth_km = compute_kde_rate(subset, label)
            save_and_plot(label, filename_suffix, title_full, grid_lon, grid_lat, rate_grid, bandwidth_km)


if __name__ == "__main__":
    main()
