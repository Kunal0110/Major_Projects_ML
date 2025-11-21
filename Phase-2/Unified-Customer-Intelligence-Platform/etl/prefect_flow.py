from prefect import flow, task

from extract import (
    load_customers, load_billing, load_usage,
    load_marketing, load_revenue, load_churn_labels
)

from transform import bronze_clean, enforce_types, fill_missing, merge_gold_tables
from load import save_to_stage


@task
def bronze_stage():
    customers = bronze_clean(load_customers())
    billing = bronze_clean(load_billing())
    usage = bronze_clean(load_usage())
    marketing = bronze_clean(load_marketing())
    revenue = bronze_clean(load_revenue())
    churn = bronze_clean(load_churn_labels())

    save_to_stage(customers, "bronze", "customers")
    save_to_stage(billing, "bronze", "billing")
    save_to_stage(usage, "bronze", "usage")
    save_to_stage(marketing, "bronze", "marketing")
    save_to_stage(revenue, "bronze", "revenue")
    save_to_stage(churn, "bronze", "churn")

    return customers, billing, usage, marketing, revenue, churn


@task
def silver_stage(customers, billing, usage, marketing, revenue, churn):
    customers_s = enforce_types(customers, "etl/expectations/customers_schema.json")
    billing_s = enforce_types(billing, "etl/expectations/billing_schema.json")
    usage_s = enforce_types(usage, "etl/expectations/usage_schema.json")
    marketing_s = enforce_types(marketing, "etl/expectations/marketing_schema.json")
    revenue_s = enforce_types(revenue, "etl/expectations/revenue_schema.json")
    churn_s = churn.copy()

    save_to_stage(customers_s, "silver", "customers")
    save_to_stage(billing_s, "silver", "billing")
    save_to_stage(usage_s, "silver", "usage")
    save_to_stage(marketing_s, "silver", "marketing")
    save_to_stage(revenue_s, "silver", "revenue")
    save_to_stage(churn_s, "silver", "churn")

    return customers_s, billing_s, usage_s, marketing_s, revenue_s, churn_s


@task
def gold_stage(customers, billing, usage, marketing, revenue, churn):
    gold = merge_gold_tables(customers, billing, usage, marketing, revenue, churn)
    save_to_stage(gold, "gold", "customer_gold_master")
    return gold


@flow(name="Unified-Customer-Intelligence-ETL")
def etl_flow():
    bronze_outputs = bronze_stage()
    silver_outputs = silver_stage(*bronze_outputs)
    gold_output = gold_stage(*silver_outputs)

    return gold_output


if __name__ == "__main__":
    etl_flow()