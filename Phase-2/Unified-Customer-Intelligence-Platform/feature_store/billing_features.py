import pandas as pd
from pathlib import Path

def build_billing_features(silver_path: Path):
    df = pd.read_parquet(silver_path / "billing.parquet")

    agg = df.groupby("customer_id").agg({
        "billed_amount": ["mean", "std", "min", "max"],
        "paid_amount": ["mean", "std"],
        "payment_delay_days": ["mean", "max"]
    })

    agg.columns = ["billing_" + "_".join(col) for col in agg.columns]

    agg = agg.reset_index()
    return agg
