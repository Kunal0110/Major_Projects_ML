import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from streamlit_app.utils.charts import pca_plot
from streamlit_app.utils.auth import is_authenticated, get_current_user

if not is_authenticated():
    st.error("🔒 Please login from the Home page to access this feature")
    st.stop()

user = get_current_user()
st.title("🎯 Customer Segmentation Studio")
st.info(f"👤 Logged in as: {user['name']} | 📧 {user['email']}")

model_dir = Path("models/segmentation")

if (model_dir / "kmeans_model.pkl").exists():
    kmeans = joblib.load(model_dir / "kmeans_model.pkl")
    
    if (model_dir / "cluster_profiles.csv").exists():
        profiles = pd.read_csv(model_dir / "cluster_profiles.csv")
        st.write("Cluster Profiles")
        st.dataframe(profiles)
    else:
        st.warning("Cluster profiles not found")
    
    st.markdown("---")
    st.write("Cluster Visualization")
    
    pca_file = model_dir / "pca_components.csv"
    if pca_file.exists():
        pcs = pd.read_csv(pca_file)
        if "cluster" in pcs.columns:
            fig = pca_plot(pcs[["pc1", "pc2"]].values, pcs["cluster"].values)
            st.plotly_chart(fig)
        else:
            st.warning("Cluster labels not found in PCA file")
    else:
        st.warning("PCA components not found. Run segmentation training again")
    
    st.markdown("---")
    st.subheader("Batch Segmentation Prediction")
    
    template_df = pd.DataFrame([{
        "usage_data_used_gb_mean": 50,
        "usage_voice_minutes_mean": 300,
        "billing_billed_amount_mean": 70,
        "billing_payment_delay_days_max": 5,
        "mkt_touch_count": 3,
        "avg_monthly_revenue": 70
    }])
    
    st.download_button(
        "📥 Download CSV Template",
        template_df.to_csv(index=False),
        "segmentation_batch_template.csv",
        "text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV for Batch Segmentation", type=["csv"])
    
    if uploaded_file:
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv(uploaded_file)
        X = df[["usage_data_used_gb_mean", "usage_voice_minutes_mean", "billing_billed_amount_mean", 
                "billing_payment_delay_days_max", "mkt_touch_count", "avg_monthly_revenue"]]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        segments = kmeans.predict(X_scaled)
        
        df["segment"] = segments
        st.write("Segmentation Results:")
        st.dataframe(df)
        
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "segmentation_results.csv"
        )
else:
    st.error("Segmentation model not found. Run training first.")
    st.code("python -m models.segmentation.train_segmentation")
