import pandas as pd
from pathlib import Path

def build_marketing_features(silver_path: Path):
    df = pd.read_parquet(silver_path / "marketing.parquet")

    agg = df.groupby("customer_id").agg({
        "touch_count": "sum",
        "clicks": "sum",
        "conversions": "sum"
    })

    agg = agg.add_prefix("mkt_")

    return agg.reset_index()
