import streamlit as st
import pandas as pd
from streamlit_app.utils.api_client import predict_churn, batch_churn, explain_churn
from streamlit_app.utils.charts import gauge_plot
from streamlit_app.utils.auth import is_authenticated, get_current_user
import plotly.graph_objects as go

if not is_authenticated():
    st.error("🔒 Please login from the Home page to access this feature")
    st.stop()

user = get_current_user()
st.title("🧠 Churn Intelligence Workspace")
st.info(f"👤 Logged in as: {user['name']} | 📧 {user['email']}")

st.subheader("Single Customer Prediction")

with st.form("single_customer_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0,1])
        partner = st.selectbox("Partner", [0,1])
        dependents = st.selectbox("Dependents", [0,1])
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        
        st.markdown("**Service**")
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        st.markdown("**Billing**")
        monthly = st.number_input("Monthly Charges", 10, 200, 70)
        total = st.number_input("Total Charges", 0, 10000, 840)
        bill_mean = st.number_input("Avg Billed Amount", 10, 200, 70)
        bill_min = st.number_input("Min Billed Amount", 10, 200, 60)
        bill_max = st.number_input("Max Billed Amount", 10, 200, 80)
        bill_std = st.number_input("Billing Std Dev", 0, 50, 5)
        paid_mean = st.number_input("Avg Paid Amount", 10, 200, 65)
        paid_std = st.number_input("Payment Std Dev", 0, 50, 3)
        delay_mean = st.number_input("Avg Payment Delay (days)", 0, 30, 2)
        delay_max = st.number_input("Max Payment Delay (days)", 0, 60, 5)
    
    with col3:
        st.markdown("**Usage**")
        data_mean = st.number_input("Avg Data Used (GB)", 0, 500, 50)
        data_max = st.number_input("Max Data Used (GB)", 0, 1000, 75)
        voice_mean = st.number_input("Avg Voice Minutes", 0, 2000, 300)
        downtime_mean = st.number_input("Avg Downtime (min)", 0, 100, 10)
        downtime_max = st.number_input("Max Downtime (min)", 0, 200, 30)
        support_calls = st.number_input("Support Calls", 0, 20, 2)
        
        st.markdown("**Marketing & Revenue**")
        mkt_touch = st.number_input("Marketing Touches", 0, 50, 3)
        mkt_clicks = st.number_input("Marketing Clicks", 0, 20, 1)
        mkt_conv = st.number_input("Marketing Conversions", 0, 10, 0)
        avg_revenue = st.number_input("Avg Monthly Revenue", 10, 200, 70)
        revenue_vol = st.number_input("Revenue Volatility", 0, 100, 10)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "region": region,
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "monthly_charges": monthly,
        "total_charges": total,
        "billing_billed_amount_mean": bill_mean,
        "billing_billed_amount_min": bill_min,
        "billing_billed_amount_max": bill_max,
        "billing_billed_amount_std": bill_std,
        "billing_paid_amount_mean": paid_mean,
        "billing_paid_amount_std": paid_std,
        "billing_payment_delay_days_mean": delay_mean,
        "billing_payment_delay_days_max": delay_max,
        "usage_data_used_gb_mean": data_mean,
        "usage_data_used_gb_max": data_max,
        "usage_voice_minutes_mean": voice_mean,
        "usage_downtime_minutes_mean": downtime_mean,
        "usage_downtime_minutes_max": downtime_max,
        "usage_support_calls_sum": support_calls,
        "mkt_touch_count": mkt_touch,
        "mkt_clicks": mkt_clicks,
        "mkt_conversions": mkt_conv,
        "avg_monthly_revenue": avg_revenue,
        "revenue_volatility": revenue_vol
    }

    result = predict_churn(payload)
    
    if "churn_probability" in result:
        st.success("Prediction Complete")
        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
        st.metric("Prediction", "Will Churn" if result['churn_prediction'] == 1 else "Will Stay")

        gauge = gauge_plot(result["churn_probability"])
        st.plotly_chart(gauge)

        st.write("SHAP Explanation")
        shap_result = explain_churn(payload)
        if "error" in shap_result:
            st.warning(f"SHAP explanation unavailable: {shap_result.get('detail', 'Unknown error')}")
        else:
            st.json(shap_result)
    else:
        st.error("Prediction Failed")
        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
        st.error(f"**Details:** {result.get('detail', 'No details available')}")
        st.info("💡 Check the terminal running FastAPI to see the full error and expected columns.")

st.markdown("---")
st.subheader("Batch Churn Prediction (Upload CSV)")

template_df = pd.DataFrame([{
    "gender": "Male",
    "senior_citizen": 0,
    "partner": 1,
    "dependents": 0,
    "region": "North",
    "tenure_months": 12,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "monthly_charges": 70,
    "total_charges": 840,
    "billing_billed_amount_mean": 70,
    "billing_billed_amount_min": 60,
    "billing_billed_amount_max": 80,
    "billing_billed_amount_std": 5,
    "billing_paid_amount_mean": 65,
    "billing_paid_amount_std": 3,
    "billing_payment_delay_days_mean": 2,
    "billing_payment_delay_days_max": 5,
    "usage_data_used_gb_mean": 50,
    "usage_data_used_gb_max": 75,
    "usage_voice_minutes_mean": 300,
    "usage_downtime_minutes_mean": 10,
    "usage_downtime_minutes_max": 30,
    "usage_support_calls_sum": 2,
    "mkt_touch_count": 3,
    "mkt_clicks": 1,
    "mkt_conversions": 0,
    "avg_monthly_revenue": 70,
    "revenue_volatility": 10
}])

st.download_button(
    "📥 Download CSV Template",
    template_df.to_csv(index=False),
    "churn_batch_template.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload CSV for Batch Churn Scoring", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    result = batch_churn(df)

    if "predictions" in result:
        st.write("Batch Results:")
        st.dataframe(result["predictions"])

        st.download_button(
            "Download Results",
            pd.DataFrame(result["predictions"]).to_csv(index=False),
            "batch_churn_predictions.csv"
        )
    else:
        st.error("Batch Prediction Failed")
        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
        st.error(f"**Details:** {result.get('detail', 'No details available')}")
    