import pandas as pd
import numpy as np

# Expected features for the model (29 features excluding customer_id and churn)
EXPECTED_FEATURES = [
    'gender', 'senior_citizen', 'partner', 'dependents', 'tenure_months', 
    'contract_type', 'internet_service', 'monthly_charges', 'total_charges', 
    'region', 'billing_billed_amount_mean', 'billing_billed_amount_std', 
    'billing_billed_amount_min', 'billing_billed_amount_max', 
    'billing_paid_amount_mean', 'billing_paid_amount_std', 
    'billing_payment_delay_days_mean', 'billing_payment_delay_days_max', 
    'usage_data_used_gb_mean', 'usage_data_used_gb_max', 
    'usage_voice_minutes_mean', 'usage_support_calls_sum', 
    'usage_downtime_minutes_mean', 'usage_downtime_minutes_max', 
    'mkt_touch_count', 'mkt_clicks', 'mkt_conversions', 
    'avg_monthly_revenue', 'revenue_volatility'
]

# Default values for missing features
FEATURE_DEFAULTS = {
    # Categorical defaults
    'gender': 'Male',
    'partner': 'No',
    'dependents': 'No',
    'contract_type': 'Month-to-month',
    'internet_service': 'DSL',
    'region': 'North',
    
    # Numeric defaults (use median/mean values)
    'senior_citizen': 0,
    'tenure_months': 12,
    'monthly_charges': 65.0,
    'total_charges': 1000.0,
    'billing_billed_amount_mean': 65.0,
    'billing_billed_amount_std': 10.0,
    'billing_billed_amount_min': 50.0,
    'billing_billed_amount_max': 80.0,
    'billing_paid_amount_mean': 65.0,
    'billing_paid_amount_std': 10.0,
    'billing_payment_delay_days_mean': 2.0,
    'billing_payment_delay_days_max': 5.0,
    'usage_data_used_gb_mean': 15.0,
    'usage_data_used_gb_max': 25.0,
    'usage_voice_minutes_mean': 100.0,
    'usage_support_calls_sum': 2,
    'usage_downtime_minutes_mean': 5.0,
    'usage_downtime_minutes_max': 15.0,
    'mkt_touch_count': 3,
    'mkt_clicks': 5,
    'mkt_conversions': 1,
    'avg_monthly_revenue': 65.0,
    'revenue_volatility': 8.0
}

def prepare_features(customer_data: dict) -> pd.DataFrame:
    """
    Prepare customer data with all expected features, filling missing ones with defaults
    """
    # Create a copy of the input data
    data = customer_data.copy()
    
    # Add missing features with default values
    for feature in EXPECTED_FEATURES:
        if feature not in data:
            data[feature] = FEATURE_DEFAULTS[feature]
            print(f"Added missing feature '{feature}' with default value: {FEATURE_DEFAULTS[feature]}")
    
    # Create DataFrame with expected column order
    df = pd.DataFrame([data])[EXPECTED_FEATURES]
    
    # Ensure proper data types
    categorical_features = ['gender', 'partner', 'dependents', 'contract_type', 'internet_service', 'region']
    numeric_features = [f for f in EXPECTED_FEATURES if f not in categorical_features]
    
    # Convert to proper types
    for col in categorical_features:
        df[col] = df[col].astype(str)
    
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Prepared features: {len(df.columns)} columns")
    print(f"Categorical: {categorical_features}")
    print(f"Numeric: {len(numeric_features)} features")
    
    return df