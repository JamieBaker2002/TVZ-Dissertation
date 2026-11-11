#!/usr/bin/env python
"""
Week 17: PyGMT seismicity maps for all three catalogues, with real
topography, coastline, and TVZ fault traces (New Zealand Active Faults
Database, GNS Science, Langridge et al. 2016) overlaid.

Events are coloured by depth and sized by magnitude, following standard
seismicity map convention. The four sub-region centres (Week 16) are
marked for reference.

Fault trace data source: GNS Science New Zealand Active Faults Database,
downloaded by the student directly from data.gns.cri.nz and preprocessed
into GMT multi-segment format (tvz_faults.gmt) - real, publicly available
data, not fabricated. Must be acknowledged as "GNS Science" per the
dataset's own terms of use.
"""
import os
import argparse
import pandas as pd
import pygmt

PROJECT_DIR = os.path.expanduser("~/tvz_project")
CATALOG_DIR = os.path.join(PROJECT_DIR, "catalogs")
FAULTS_FILE = os.path.join(PROJECT_DIR, "faults", "tvz_faults.gmt")
OUT_DIR = os.path.join(CATALOG_DIR, "phase4", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

REGION = [175.0, 178.0, -40.0, -37.0]
# "01m" (~1.8km resolution) is a standard, proportionate choice for a
# regional map at this scale - "03s" (~90m) was considered but would be a
# much larger, slower download for detail invisible at this map scale
TOPO_RESOLUTION = "01m"
SUBREGIONS = {
    "Taupo": (-38.8, 175.9),
    "Okataina": (-38.22, 176.5),
    "Tongariro": (-39.13, 175.64),
    "Ruapehu": (-39.28, 175.57),
}


def load_catalogues():
    eqt = pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_nlloc_quality_filtered.csv"))
    eqt = eqt.merge(pd.read_csv(os.path.join(CATALOG_DIR, "eqtransformer_magnitudes.csv")),
                     left_on=eqt["file"].str.extract(r"_E(\d+)\.")[0].astype(int),
                     right_on="event_idx", how="left")

    pn = pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_nlloc_quality_filtered.csv"))
    pn = pn.merge(pd.read_csv(os.path.join(CATALOG_DIR, "phasenet_magnitudes.csv")),
                   left_on=pn["file"].str.extract(r"_E(\d+)\.")[0].astype(int),
                   right_on="event_idx", how="left")

    qm = pd.read_csv(os.path.join(CATALOG_DIR, "quakemigrate_quality_filtered.csv"))
    qm = qm.rename(columns={"ml": "ml_median"})

    return {"EQTransformer": eqt, "PhaseNet": pn, "QuakeMigrate": qm}


def make_map(name, df, out_path):
    fig = pygmt.Figure()

    # Real topography/bathymetry (downloaded from GMT remote server,
    # confirmed accessible from compute nodes)
    grid = pygmt.datasets.load_earth_relief(resolution=TOPO_RESOLUTION, region=REGION)
    fig.grdimage(grid=grid, region=REGION, projection="M15c", cmap="geo", shading=True)
    fig.coast(region=REGION, projection="M15c", shorelines="1/0.5p,black", frame=["af", f"+t{name}"])

    # Real TVZ fault traces (GNS Science NZAFD)
    fig.plot(data=FAULTS_FILE, pen="0.5p,darkred", region=REGION, projection="M15c")

    # Depth colour scale, fixed across all three catalogues for fair
    # comparison. Values are clipped to BOTH ends of this range before
    # plotting - GMT's default behaviour for values outside a defined
    # colour range is to fall back to an undefined colour (black above
    # the range, white below it) rather than clip to the nearest colour.
    # An earlier version only clipped the upper bound, which fixed one
    # genuine 33.2 km QuakeMigrate event but missed that EQTransformer
    # and PhaseNet both contain genuinely shallow, slightly negative-depth
    # events (down to about -3 km, consistent with this project's own
    # NonLinLoc grid convention) - these rendered as unexplained white
    # dots until checked directly and confirmed as real, valid data.
    pygmt.makecpt(cmap="viridis", series=[0, 30], reverse=True)
    depth_plot = df["depth_km"].clip(lower=0, upper=30)

    # Events: sized by magnitude, coloured by (clipped) depth
    mag_col = "ml_median"
    sizes = 0.06 + 0.05 * (df[mag_col].clip(lower=0))
    fig.plot(
        x=df["lon"], y=df["lat"], size=sizes, fill=depth_plot,
        cmap=True, style="cc", pen="0.3p,black", region=REGION, projection="M15c",
    )
    fig.colorbar(frame="af+lDepth (km)")

    # Magnitude size legend, enlarged and relocated to a distinct ocean
    # patch in the map's south-east (separate from the sub-region legend's
    # Bay of Plenty location, avoiding a cramped vertical stack)
    legend_mags = [2, 3, 4]
    mag_spec_path = os.path.join(OUT_DIR, f"_mag_legend_spec_{name.lower()}.txt")
    with open(mag_spec_path, "w") as f:
        for m in legend_mags:
            legend_size = 0.08 + 0.06 * m
            f.write(f"S 0.4c c {legend_size}c gray 0.3p 0.8c ML {m}\n")
    fig.legend(spec=mag_spec_path, position="g177.5/-39.5+jCM+o0c",
               box="+gwhite+p0.5p", region=REGION, projection="M15c")

    # Sub-region centres: distinct colour per region, own spec-file legend,
    # enlarged (symbols and gaps scaled up from an earlier smaller version)
    subregion_colors = {
        "Taupo": "red", "Okataina": "blue", "Tongariro": "orange", "Ruapehu": "magenta",
    }
    for label, (lat, lon) in SUBREGIONS.items():
        fig.plot(x=[lon], y=[lat], style="a0.45c", fill=subregion_colors[label], pen="0.5p,black",
                  region=REGION, projection="M15c")

    region_spec_path = os.path.join(OUT_DIR, f"_region_legend_spec_{name.lower()}.txt")
    with open(region_spec_path, "w") as f:
        for label, colour in subregion_colors.items():
            f.write(f"S 0.4c a 0.5c {colour} 0.5p 0.8c {label}\n")
    fig.legend(spec=region_spec_path, position="g177.4/-37.25+jTR+o0.2c",
               box="+gwhite+p0.5p", region=REGION, projection="M15c")

    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", choices=["EQTransformer", "PhaseNet", "QuakeMigrate"],
                         default=None, help="Build only this one catalogue's map (for testing)")
    args = parser.parse_args()

    catalogues = load_catalogues()
    if args.catalogue:
        catalogues = {args.catalogue: catalogues[args.catalogue]}

    for name, df in catalogues.items():
        out_path = os.path.join(OUT_DIR, f"seismicity_map_{name.lower()}.png")
        print(f"Building map for {name} ({len(df)} events)...", flush=True)
        make_map(name, df, out_path)


if __name__ == "__main__":
    main()
