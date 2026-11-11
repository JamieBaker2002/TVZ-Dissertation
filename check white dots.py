import pandas as pd

for name, path in [
    ("EQTransformer", "catalogs/eqtransformer_nlloc_quality_filtered.csv"),
    ("PhaseNet", "catalogs/phasenet_nlloc_quality_filtered.csv"),
]:
    df = pd.read_csv(path)
    n_nan = df["depth_km"].isna().sum()
    print(f"{name}: {len(df)} events, {n_nan} with NaN depth_km")
    if n_nan > 0:
        print(df[df["depth_km"].isna()][["lat", "lon", "depth_km"]].head(10))
    print()
