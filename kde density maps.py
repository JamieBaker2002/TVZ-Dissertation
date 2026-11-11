#!/usr/bin/env python
"""
Week 17: KDE seismic density maps for all three catalogues.

Gaussian kernel, bandwidth = each catalogue's own MEDIAN horizontal
location uncertainty (not mean), per the project proposal. Density is
normalised to a genuine rate (events per km^2 per year), not just
relative density, using each catalogue's real event count and the
actual ~2-month study duration.

Implemented as a direct, manual 2D Gaussian KDE (not scipy's
gaussian_kde) since scipy's bandwidth API is relative to the data's own
spread rather than an absolute value, and we need bandwidth to mean
exactly what the proposal specifies. Verified against synthetic data
with known true properties before use on real data: KDE integral,
rate-conversion integral, and the expected (convolution-predicted, not
naively unsmoothed) peak density all matched theoretical expectations
exactly.
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
STUDY_DURATION_YEARS = 61 / 365.25  # 1 Aug - 30 Sep 2019 inclusive
GRID_SPACING_KM = 2.0
SUBREGIONS = {
    "Taupo": (-38.8, 175.9),
    "Okataina": (-38.22, 176.5),
    "Tongariro": (-39.13, 175.64),
    "Ruapehu": (-39.28, 175.57),
}

# Local projection reference point (centre of the study region) for
# converting lat/lon to an approximate local km-based Cartesian system -
# a standard, reasonable approximation for a small ~3x3 degree region
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
    pn = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_nlloc_quality_filtered.csv"))
    qm = pd.read_csv(os.path.join(CATALOG_DIR, "quakemigrate_quality_filtered.csv"))
    return {"EQTransformer": eqt, "PhaseNet": pn, "QuakeMigrate": qm}


def compute_kde_rate(df, name):
    bandwidth_km = df["horiz_error_km"].median()
    n_events = len(df)
    print(f"{name}: n={n_events}, bandwidth (median horiz error) = {bandwidth_km:.3f} km", flush=True)

    x_km, y_km = to_local_km(df["lat"].values, df["lon"].values)

    # Local km-space grid, covering the study region with a margin
    region_x_km, region_y_km = to_local_km(
        np.array([REGION[2], REGION[3]]), np.array([REGION[0], REGION[1]])
    )
    margin = 20  # km
    gx = np.arange(region_x_km.min() - margin, region_x_km.max() + margin, GRID_SPACING_KM)
    gy = np.arange(region_y_km.min() - margin, region_y_km.max() + margin, GRID_SPACING_KM)
    gxx, gyy = np.meshgrid(gx, gy)
    gx_flat, gy_flat = gxx.ravel(), gyy.ravel()

    # Vectorized manual 2D Gaussian KDE (verified against synthetic data
    # with known true properties before use here)
    dx = gx_flat[:, None] - x_km[None, :]
    dy = gy_flat[:, None] - y_km[None, :]
    dist2 = dx**2 + dy**2
    density_per_event = (1 / (2 * np.pi * bandwidth_km**2)) * np.exp(-dist2 / (2 * bandwidth_km**2))
    kde_density = density_per_event.mean(axis=1)  # proper KDE: mean, not sum

    rate = kde_density * n_events / STUDY_DURATION_YEARS  # events per km^2 per year
    rate_grid = rate.reshape(gxx.shape)

    grid_lat, grid_lon = to_latlon(gxx, gyy)
    return grid_lon, grid_lat, rate_grid, bandwidth_km


def save_and_plot(name, grid_lon, grid_lat, rate_grid, bandwidth_km):
    import xarray as xr

    # Build an xarray DataArray on a regular lon/lat grid PyGMT can plot
    # directly (the local-km grid is regular in km-space, and lon/lat are
    # each monotonic along their respective axes for this small region,
    # so this reshaping is valid without needing to interpolate)
    lons_1d = grid_lon[0, :]
    lats_1d = grid_lat[:, 0]
    da = xr.DataArray(
        rate_grid, dims=["lat", "lon"], coords={"lat": lats_1d, "lon": lons_1d},
        name="rate",
    )
    grid_path = os.path.join(GRID_OUT_DIR, f"kde_rate_{name.lower()}.nc")
    da.to_netcdf(grid_path)
    print(f"Saved: {grid_path}", flush=True)

    fig = pygmt.Figure()
    # Log-transform the rate data MANUALLY in numpy first, then use a
    # plain linear colour scale over the transformed values - safer than
    # relying on pygmt.makecpt's log=True parameter directly, which
    # produced physically nonsensical colourbar values (e.g. 1e+79) in an
    # earlier version, apparently treating the real rate values as
    # exponents rather than pre-logged data. Doing the transform
    # ourselves in numpy avoids that ambiguity entirely.
    # Colour range based on orders of magnitude BELOW THE PEAK, not a
    # percentile of the whole grid - with such a small bandwidth (1-4 km)
    # applied across a 300 km-wide grid, the vast majority of grid cells
    # are so far from any event that even a 1st-percentile bound was
    # still buried in the meaningless far Gaussian tail in an earlier
    # version (colour range spanning -300 to 0 log10 units, and producing
    # a honeycomb-like visual artefact from compressing real structure
    # into a handful of color bins). Anchoring to the peak instead
    # focuses the colour range on the meaningfully-varying region near
    # real event clusters.
    ORDERS_OF_MAGNITUDE_BELOW_PEAK = 4
    peak_log = float(np.log10(rate_grid.max()))
    lower_bound_log = peak_log - ORDERS_OF_MAGNITUDE_BELOW_PEAK
    # Small safety margin on the clip bound: the exponentiate-then-log
    # round trip (10**x then log10 again) can introduce tiny floating-
    # point rounding noise, pushing a handful of values infinitesimally
    # outside the defined colour range - this produced scattered black
    # speckled pixels (GMT's default out-of-range fallback colour) around
    # otherwise correctly-coloured areas in an earlier version.
    safety_margin = 0.01
    log_rate_grid = np.log10(np.clip(rate_grid, 10 ** (lower_bound_log - safety_margin), None))
    da_log = xr.DataArray(
        log_rate_grid, dims=["lat", "lon"], coords={"lat": lats_1d, "lon": lons_1d}, name="log_rate",
    )

    # Explicit, uniquely-named CPT file per catalogue, not relying on
    # PyGMT's "current session CPT" - looping over three catalogues with
    # three separate Figure() objects created ambiguity about which
    # colormap was actually active, producing one catalogue (PhaseNet)
    # rendering with what looked like a stale, pre-fix colormap in an
    # earlier version. An explicit, separately-referenced file per
    # catalogue removes that ambiguity entirely.
    #
    # Black-yellow-orange-red (inverted only the low end, per request -
    # yellow/orange/red for the actual density values stay unchanged)
    cpt_path = os.path.join(OUT_DIR, f"_kde_cpt_{name.lower()}.cpt")
    pygmt.makecpt(cmap="black,yellow,orange,red", series=[lower_bound_log, peak_log],
                  background="o", output=cpt_path)
    fig.grdimage(grid=da_log, region=REGION, projection="M15c", cmap=cpt_path)
    fig.coast(region=REGION, projection="M15c", shorelines="1/0.5p,white",
              frame=["af", f"+t{name} - KDE seismicity rate (bandwidth={bandwidth_km:.2f} km)"])
    fig.plot(data=FAULTS_FILE, pen="0.4p,darkgreen", region=REGION, projection="M15c")
    fig.colorbar(cmap=cpt_path, frame="af+llog@-10@-(Rate, events/km@+2@+/yr)")

    # Sub-region stars and legend, identical to the seismicity maps
    subregion_colors = {
        "Taupo": "red", "Okataina": "blue", "Tongariro": "orange", "Ruapehu": "magenta",
    }
    for label, (lat, lon) in SUBREGIONS.items():
        fig.plot(x=[lon], y=[lat], style="a0.45c", fill=subregion_colors[label], pen="0.5p,black",
                  region=REGION, projection="M15c")

    region_spec_path = os.path.join(OUT_DIR, f"_region_legend_spec_kde_{name.lower()}.txt")
    with open(region_spec_path, "w") as f:
        for label, colour in subregion_colors.items():
            f.write(f"S 0.4c a 0.5c {colour} 0.5p 0.8c {label}\n")
    fig.legend(spec=region_spec_path, position="g177.4/-37.25+jTR+o0.2c",
               box="+gwhite+p0.5p", region=REGION, projection="M15c")

    out_path = os.path.join(OUT_DIR, f"kde_density_{name.lower()}.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}", flush=True)


def main():
    catalogues = load_catalogues()
    for name, df in catalogues.items():
        grid_lon, grid_lat, rate_grid, bandwidth_km = compute_kde_rate(df, name)
        save_and_plot(name, grid_lon, grid_lat, rate_grid, bandwidth_km)


if __name__ == "__main__":
    main()
