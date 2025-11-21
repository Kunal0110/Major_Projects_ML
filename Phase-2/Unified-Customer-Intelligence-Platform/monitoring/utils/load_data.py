import pandas as pd
from pathlib import Path

def load_reference_data():
    """Load reference (training) data"""
    gold_path = Path("data/gold/customer_gold_master.parquet")
    df = pd.read_parquet(gold_path)
    
    # Use first 80% as reference
    split_idx = int(len(df) * 0.8)
    reference = df.iloc[:split_idx].copy()
    
    # Drop customer_id if present
    if 'customer_id' in reference.columns:
        reference = reference.drop(columns=['customer_id'])
    
    return reference

def load_current_data():
    """Load current (recent) data for comparison"""
    gold_path = Path("data/gold/customer_gold_master.parquet")
    df = pd.read_parquet(gold_path)
    
    # Use last 20% as current
    split_idx = int(len(df) * 0.8)
    current = df.iloc[split_idx:].copy()
    
    # Drop customer_id if present
    if 'customer_id' in current.columns:
        current = current.drop(columns=['customer_id'])
    
    return current
