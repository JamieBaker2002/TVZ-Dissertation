#!/usr/bin/env python
"""
Reference/locator map, not part of the original proposal's figure list -
requested to give readers geographic context for the study region.

Two-scale structure: main map shows all of New Zealand with a rectangle
highlighting the TVZ study region (175-178E, 40-37S); an inset in one
corner shows a global view with a rectangle highlighting New Zealand's
location, using PyGMT's own documented inset feature (fig.inset()).
"""
import os
import pygmt

PROJECT_DIR = os.path.expanduser("~/tvz_project")
OUT_DIR = os.path.join(PROJECT_DIR, "catalogs", "phase4", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Full New Zealand extent (both islands, with margin)
NZ_REGION = [165, 180, -48, -34]
STUDY_REGION = [175.0, 178.0, -40.0, -37.0]  # our TVZ study region

fig = pygmt.Figure()

# Main map: all of New Zealand, real topography for context
grid = pygmt.datasets.load_earth_relief(resolution="02m", region=NZ_REGION)
fig.grdimage(grid=grid, region=NZ_REGION, projection="M15c", cmap="geo", shading=True)
fig.coast(region=NZ_REGION, projection="M15c", shorelines="1/0.5p,black",
          frame=["af", "+tNew Zealand - TVZ Study Region"])

# Rectangle highlighting the TVZ study region
fig.plot(
    data=[[STUDY_REGION[0], STUDY_REGION[2], STUDY_REGION[1], STUDY_REGION[3]]],
    style="r+s", pen="2p,red", region=NZ_REGION, projection="M15c",
)

# Global inset, bottom-right corner, showing New Zealand's location on
# a flat world map (Robinson projection, centred at 180 deg so New
# Zealand sits comfortably within the map rather than at an edge/seam)
with fig.inset(position="jBR+w8c/4c+o0.2c", box="+gwhite+p1p"):
    fig.coast(
        region="d", projection="N180/8c", land="gray", water="lightblue",
        frame="g",
    )
    fig.plot(
        data=[[NZ_REGION[0], NZ_REGION[2], NZ_REGION[1], NZ_REGION[3]]],
        style="r+s", pen="1.5p,red", region="d", projection="N180/8c",
    )

out_path = os.path.join(OUT_DIR, "reference_map.png")
fig.savefig(out_path, dpi=200)
print(f"Saved: {out_path}", flush=True)
