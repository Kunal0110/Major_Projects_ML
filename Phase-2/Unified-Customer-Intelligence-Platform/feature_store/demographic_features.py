import pandas as pd
from pathlib import Path

def build_demographic_features(silver_path: Path):
    df = pd.read_parquet(silver_path / "customers.parquet")

    # encode contract type
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }
    df["contract_encoded"] = df["contract_type"].map(contract_map)

    df["is_senior"] = df["senior_citizen"]
    df["has_partner"] = df["partner"]
    df["has_dependents"] = df["dependents"]

    df["avg_monthly_charge"] = df["monthly_charges"]

    features = df[[
        "customer_id",
        "contract_encoded",
        "is_senior",
        "has_partner",
        "has_dependents",
        "tenure_months",
        "avg_monthly_charge"
    ]]

    return features
