import pandas as pd

qm = pd.read_csv("catalogs/quakemigrate_quality_filtered.csv")
print(qm[["lat", "lon", "depth_km", "ml"]].describe())
print()
print("Any NaN depths:", qm["depth_km"].isna().sum())
print("Any NaN lat/lon:", qm[["lat", "lon"]].isna().sum().sum())
print()

candidates = qm[(qm["lon"] > 176.0) & (qm["lat"] < -39.0)]
print("Events south of Taupo, east of 176.0E:")
print(candidates[["lat", "lon", "depth_km", "ml"]])
