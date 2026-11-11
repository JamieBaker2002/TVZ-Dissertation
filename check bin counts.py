import pandas as pd

MC_VALUES = {"EQTransformer": 1.55, "PhaseNet": 1.45, "QuakeMigrate": 1.95}
CATALOG_FILES = {
    "EQTransformer": "catalogs/eqtransformer_nlloc_quality_filtered.csv",
    "PhaseNet": "catalogs/phasenet_nlloc_quality_filtered.csv",
    "QuakeMigrate": "catalogs/quakemigrate_quality_filtered.csv",
}
MAG_COL = {"EQTransformer": None, "PhaseNet": None, "QuakeMigrate": "ml"}

for name, path in CATALOG_FILES.items():
    df = pd.read_csv(path)
    mag_col = MAG_COL[name]
    if mag_col is None:
        # EQT/PhaseNet magnitudes are in a separate file, merged by event_idx
        mag_path = path.replace("_nlloc_quality_filtered.csv", "_magnitudes.csv")
        mags = pd.read_csv(mag_path)
        df["event_idx"] = df["file"].str.extract(r"_E(\d+)\.")[0].astype(int)
        df = df.merge(mags, on="event_idx", how="left")
        mag_col = "ml_median"

    n_shallow = (df["depth_km"] < 15).sum()
    n_deep = ((df["depth_km"] >= 15) & (df["depth_km"] <= 40)).sum()
    mc = MC_VALUES[name]
    n_small = ((df[mag_col] >= mc) & (df[mag_col] < 2.5)).sum()
    n_large = (df[mag_col] >= 2.5).sum()

    print(f"{name} (n={len(df)}, Mc={mc}):")
    print(f"  Depth 0-15km:  {n_shallow}")
    print(f"  Depth 15-40km: {n_deep}")
    print(f"  Mag Mc-2.5:    {n_small}")
    print(f"  Mag 2.5+:      {n_large}")
    print()
