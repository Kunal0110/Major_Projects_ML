import pandas as pd
import numpy as np
from faker import Faker
import random
from pathlib import Path

fake = Faker()
NUM_CUSTOMERS = 5000

# Customer profile table
def generate_customer_profile():
    customers = []

    for i in range(NUM_CUSTOMERS):
        customer_id = f"CUST_{1000+i}"
        tenure = np.random.randint(1,72) # 1 to 72 months
        senior = np.random.binomial(1, 0.18)
        partner = np.random.binomial(1,0.45)
        dependents = np.random.binomial(1,0.35)

        contract_type = np.random.choice(
            ["Month-to-month", "One year", "Two years"],
            p=[0.55, 0.25, 0.20]
        )

        internet_service = np.random.choice(
            ["DSL", "fibre optic", " None"],
            p=[0.3,0.6,0.1]
        )
        monthly_charges = np.round(np.random.uniform(30,120), 2)
        total_charges = np.round(monthly_charges * tenure + np.random.uniform(-20,50),2)

        customers.append([
            customer_id,
            fake.random_element(["Male", "Female"]),
            senior,
            partner,
            dependents,
            tenure,
            contract_type,
            internet_service,
            monthly_charges,
            total_charges,
            fake.state()
        ])

    df = pd.DataFrame(customers, columns=[
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure_months", "contract_type", "internet_service",
        "monthly_charges", "total_charges", "region"
    ])

    return df

# Billing History Table (12 months)

def generate_billing_history(customer_df):
    billing_rows = []

    for _, row in customer_df.iterrows():
        for m in range(1,13):
            billed = row["monthly_charges"] + random.uniform(-5,10)
            paid = billed - random.uniform(0,20) if random.random() < 0.2 else billed
            delay = max(0, int(random.gauss(3,2))) if paid < billed else 0

            billing_rows.append([
                row["customer_id"], m, round(billed, 2), round(paid, 2), delay
            ])

    df = pd.DataFrame(billing_rows, columns=["customer_id", "month", "billed_amount", "paid_amount", "payment_delay_days"])

    return df

# Usage events table

def generate_usage_events(customer_df):
    usage_rows = []

    for _, row in customer_df.iterrows():
        for m in range(1,13):
            usage_rows.append([
                row["customer_id"],
                m,
                round(np.random.exponential(30),2),  # data GB
                round(np.random.uniform(100,800), 1), #voice minutes
                np.random.poisson(0.3),  # support calls
                np.random.randint(20, 120),  # app logins
                np.random.randint(0,40)  # downtime in minutes
            ])

    df = pd.DataFrame(usage_rows, columns=[
        "customer_id", "month", "data_used_gb", "voice_minutes",
        "support_calls", "app_logins", "downtime_minutes"
    ])
    return df

# Marketing touches

def generate_marketing_touches(customer_df):
    touches = []

    for _, row in customer_df.iterrows():
        for _ in range(np.random.randint(1,5)):
            touches.append([
                row["customer_id"],
                random.choice(["Email", "SMS", "Push Notification", "Promotion banner"]),
                np.random.randint(1,6),  # touch count
                np.random.randint(0,4),  # clicks
                np.random.randint(0,2)  #conversions
            ])

    df = pd.DataFrame(touches, columns=[
        "customer_id", "campaign_type", "touch_count", "clicks", "conversions"
    ])

    return df

# Revenue Sequences

def generate_revenue_sequences(customer_df):
    rows = []

    for _, row in customer_df.iterrows():
        revenue = []
        for m in range(12):
            base = row["monthly_charges"]
            noise = np.random.uniform(-10,20)
            revenue.append(round(base + noise, 2))

        rows.append([row["customer_id"]] + revenue)
    
    cols = ["customer_id"] + [f"m{i}" for i in range(1,13)]
    df = pd.DataFrame(rows, columns=cols)
    return df

# Churn Labels

def generate_churn_labels(customer_df):
    churn_prob = (
        0.15 +
        0.002 * customer_df["tenure_months"].apply(lambda x: (72 - x)) +
        0.1 * customer_df["senior_citizen"] +
        0.25 * (customer_df["contract_type"] == "Month-to-month").astype(int)
    )

    churn = np.random.binomial(1, np.clip(churn_prob, 0, 1))

    df = pd.DataFrame({
        "customer_id": customer_df["customer_id"],
        "churn": churn
    })

    return df

# Main Execution

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_customers = generate_customer_profile()
    df_billing = generate_billing_history(df_customers)
    df_usage = generate_usage_events(df_customers)
    df_marketing = generate_marketing_touches(df_customers)
    df_revenue = generate_revenue_sequences(df_customers)
    df_churn = generate_churn_labels(df_customers)

    df_customers.to_csv(output_dir / "telco_customers.csv", index=False)
    df_billing.to_csv(output_dir / "monthly_billing_history.csv", index=False)
    df_usage.to_csv(output_dir / "usage_events.csv", index=False)
    df_marketing.to_csv(output_dir / "marketing_touches.csv", index=False)
    df_revenue.to_csv(output_dir / "revenue_sequences.csv", index=False)
    df_churn.to_csv(output_dir / "churn_labels.csv", index=False)

    print("Hybrid dataset created successfully!")


if __name__ == "__main__":
    main()