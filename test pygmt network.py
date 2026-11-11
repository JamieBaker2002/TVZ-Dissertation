#!/usr/bin/env python
"""Quick test: can PyGMT actually download remote topography/coastline
data on this HPC's compute nodes? This is a genuinely different network
question from the pip/npm/GitHub access already confirmed elsewhere in
this project."""
import pygmt

print("Testing PyGMT remote data access...")
fig = pygmt.Figure()
fig.basemap(region=[175, 178, -40, -37], projection="M10c", frame=True)
fig.coast(shorelines=True, land="lightgray", water="lightblue")
fig.savefig("pygmt_network_test.png")
print("SUCCESS: coastline data downloaded and figure saved")
