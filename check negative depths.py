import pandas as pd

for name, path in [
    ("EQTransformer", "catalogs/eqtransformer_nlloc_quality_filtered.csv"),
    ("PhaseNet", "catalogs/phasenet_nlloc_quality_filtered.csv"),
]:
    df = pd.read_csv(path)
    n_neg = (df["depth_km"] < 0).sum()
    print(f"{name}: {len(df)} events, {n_neg} with negative depth_km")
    if n_neg > 0:
        print(df[df["depth_km"] < 0][["lat", "lon", "depth_km"]].sort_values("depth_km"))
    print()
