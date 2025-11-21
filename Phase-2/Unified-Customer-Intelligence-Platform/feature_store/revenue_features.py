import pandas as pd
import pathlib as Path

def build_revenue_features(silver_path: Path):
    df = pd.read_parquet(silver_path / "revenue.parquet")

    df["revenue_mean"] = df.iloc[:,1:].mean(axis=1)
    df["revenue_std"] = df.iloc[:,1:].std(axis=1)
    df["revenue_min"] = df.iloc[:,1:].min(axis=1)
    df["revenue_max"] = df.iloc[:,1:].max(axis=1)

    cols = ["customer_id", "revenue_mean", "revenue_std", "revenue_min", "revenue_max"]

    return df[cols]
