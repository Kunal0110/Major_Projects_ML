import pandas as pd
import numpy as np
import json
from great_expectations.validator.validator import Validator
from great_expectations.core.batch import BatchRequest

# Bronze Transformation

def bronze_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning:
    - strip column names
    - normalize null markers
    - remove leading/trailing whitespace
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([" ", "", "?", "NA", "None"], np.nan)

    #strip string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

# Silver Transformation

def enforce_types(df: pd.DataFrame, schema_path: str) -> pd.DataFrame:
    df = df.copy()

    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    for col, dtype in schema["dtypes"].items():
        if col in df.columns:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif dtype == "str":
                df[col] = df[col].astype(str)
            elif dtype == "bool":
                df[col] = df[col].astype("Int64")
    return df

def fill_missing(df: pd.DataFrame, fill_rules: dict) -> pd.DataFrame:
    """
    Fill missing values based on user-specified rules.
    """
    df = df.copy()

    for col, rule in fill_rules.items():
        if rule == "median":
            df[col] = df[col].fillna(df[col].median())
        elif rule == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif isinstance(rule, (int, float, str)):
            df[col] = df[col].fillna(rule)

    return df

# Gold Transformation

def merge_gold_tables(customers, billing, usage, marketing, revenue, churn):
    ''' Merge all tables into single gold dataset '''
    billing_agg = billing.groupby("customer_id").agg({
        "billed_amount": ["mean", "std", "min", "max"],
        "paid_amount": ["mean", "std"],
        "payment_delay_days": ["mean", "max"]
    })
    billing_agg.columns = ["billing_" + "_".join(col) for col in billing_agg.columns]

    # usage aggregates
    usage_agg = usage.groupby("customer_id").agg({
        "data_used_gb": ["mean", "max"],
        "voice_minutes": ["mean"],
        "support_calls": ["sum"],
        "downtime_minutes": ["mean", "max"]
    })
    usage_agg.columns = ["usage_" + "_".join(col) for col in usage_agg.columns]

    # marketing aggregates
    marketing_agg = marketing.groupby("customer_id").agg({
        "touch_count": "sum",
        "clicks": "sum",
        "conversions": "sum"
    }).add_prefix("mkt_")

    # revenue aggregates
    revenue["avg_monthly_revenue"] = revenue.iloc[:, 1:].mean(axis=1)
    revenue["revenue_volatility"] = revenue.iloc[:, 1:].std(axis=1)
    revenue_agg = revenue[["customer_id", "avg_monthly_revenue", "revenue_volatility"]]

    gold = customers.merge(billing_agg, on="customer_id", how="left")
    gold = gold.merge(usage_agg, on="customer_id", how="left")
    gold = gold.merge(marketing_agg, on="customer_id", how="left")
    gold = gold.merge(revenue_agg, on="customer_id", how="left")
    gold = gold.merge(churn, on="customer_id", how="left")

    return gold

