import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def load_customers():
    return pd.read_csv(RAW_DIR / "telco_customers.csv")

def load_billing():
    return pd.read_csv(RAW_DIR / "monthly_billing_history.csv")

def load_usage():
    return pd.read_csv(RAW_DIR / "usage_events.csv")

def load_marketing():
    return pd.read_csv(RAW_DIR / "marketing_touches.csv")

def load_revenue():
    return pd.read_csv(RAW_DIR / "revenue_sequences.csv")

def load_churn_labels():
    return pd.read_csv(RAW_DIR / "churn_labels.csv")


if __name__ == "__main__":
    # Test run
    print(load_customers().head())