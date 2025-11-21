import pandas as pd
from pathlib import Path

def build_usage_features(silver_path: Path):
    df = pd.read_parquet(silver_path / "usage.parquet")

    agg = df.groupby("customer_id").agg({
        "data_used_gb": ["mean", "max"],
        "voice_minutes": ["mean"],
        "support_calls": ["sum"],
        "downtime_minutes": ["mean", "max"]
    })

    agg.columns = ["usage_" + "_".join(col) for col in agg.columns]

    return agg.reset_index()
