"""Streamlit dashboard for 30-day readmission risk
Run with
    streamlit run app/dashboard.py
The dashboard uploads either CSV / Parquet or raw JSON, calls the FastAPI
service at *API_URL* and shows KPIs, a
preview table, a bar chart, and an optional SHAP waterfall explanation.
"""
from __future__ import annotations

from pathlib import Path
import json, time, io, requests

import pandas as pd
import altair as alt
import streamlit as st
import shap, joblib, matplotlib.pyplot as plt  # noqa: F401 (plt needed for shap)
import sys, pathlib
import scipy.sparse as sp 

root_dir = pathlib.Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Config 
API_URL   = st.secrets.get("API_URL", "http://127.0.0.1:8000")
THEME_RED = "#dc2626"   # tailwind-red-600

st.set_page_config(
    page_title="30-Day Readmission Risk",
    page_icon="🏥",
    layout="wide",
)

# Sidebar – mode selector & info
with st.sidebar:
    st.title("🏥 Readmission Demo")
    st.markdown(f"**API**: {API_URL}")
    mode = st.radio("Input type …", ["Upload file", "Paste JSON"], index=0)

# Helper → call API
def predict_json(df: pd.DataFrame) -> list[float]:
    payload = {"patients": df.to_dict("records")}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["risk"]


def predict_file(upload: st.runtime.uploaded_file_manager.UploadedFile) -> list[float]:
    files = {"file": (upload.name, upload.getvalue(), upload.type)}
    r = requests.post(f"{API_URL}/predict_csv", files=files, timeout=60)
    r.raise_for_status()
    return r.json()["risk"]

# Input – either upload or paste JSON

df_in: pd.DataFrame | None = None
uploaded = None  # type: st.runtime.uploaded_file_manager.UploadedFile | None

if mode == "Upload file":
    uploaded = st.file_uploader("CSV or Parquet (≤ 5 MB)", type=["csv", "parquet"])
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1].lower()
        df_in = pd.read_csv(uploaded) if ext == "csv" else pd.read_parquet(uploaded)
else:
    json_text = st.text_area("Paste raw JSON", height=200,
                             placeholder='{"patients":[ … ]}')
    if json_text:
        try:
            records = json.loads(json_text)["patients"]
            df_in = pd.DataFrame(records)
        except Exception as exc:
            st.error(f" Invalid JSON - {exc}")

# Prediction block
if df_in is not None:
    st.success(f"Loaded **{len(df_in):,}** rows · **{df_in.shape[1]}** features")

    with st.spinner("Scoring …"):
        t0 = time.perf_counter()
        probs = predict_file(uploaded) if uploaded else predict_json(df_in)
        dt   = time.perf_counter() - t0

    res: pd.DataFrame = (
        df_in.assign(Readmit_Prob=probs)
              .assign(Risk=lambda d: pd.cut(
                  d.Readmit_Prob,
                  bins=[0, .20, .50, 1],
                  labels=["Low", "Medium", "High"],
                  include_lowest=True))
    )

    # ── KPI cards 
    counts = res.Risk.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Low risk",     f"{counts['Low']:,}")
    c2.metric("Medium risk",  f"{counts['Medium']:,}")
    c3.metric("High risk",    f"{counts['High']:,}")

    # ── Overview bar chart 
    chart_data = pd.DataFrame({"Risk": counts.index, "Count": counts.values})
    bar = (alt.Chart(chart_data)
             .mark_bar()
             .encode(x="Risk", y="Count", color=alt.value(THEME_RED)))
    st.altair_chart(bar, use_container_width=True)

    # ── Preview table (first N rows) 
    N_PREVIEW = 500
    preview = res.head(N_PREVIEW)

    st.subheader(f"📊 Predictions - first {N_PREVIEW} rows")
    st.dataframe(
        preview[["Readmit_Prob", "Risk"]]
              .style.background_gradient(cmap="Reds", subset=["Readmit_Prob"])
              .format({"Readmit_Prob": "{:.2%}"}),
        use_container_width=True,
    )

    # optional download of full results 
    csv_bytes = res.to_csv(index=False).encode()
    st.download_button(" Download full CSV", csv_bytes, "readmit_preds.csv")

    # ── SHAP explanation (expander) 
    with st.expander("Model explanation (SHAP)"):

        @st.cache_resource(show_spinner=False)
        def _load_explainer():
            pipe = joblib.load(root_dir / "models" / "best_model.pkl")
            eng, pre, clf = pipe.named_steps["engineer"], pipe.named_steps["pre"], pipe.named_steps["clf"]

            def to_numeric(df: pd.DataFrame):
                return pre.transform(eng.transform(df))

            bg_num = to_numeric(res.sample(min(200, len(res)), random_state=42))
            if sp.issparse(bg_num):
                bg_num = bg_num.toarray()

            expl = shap.TreeExplainer(clf, bg_num)
            return expl, to_numeric, pre.get_feature_names_out()

        explainer, to_numeric, feat_names = _load_explainer()

        idx = st.slider("Row to explain", 0, len(res) - 1, 0)
        row_num = to_numeric(res.iloc[[idx]])
        if sp.issparse(row_num):
            row_num = row_num.toarray()

        sv = explainer(row_num)
        shap.plots.waterfall(sv[0], max_display=14, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)


    st.caption(f"Scored in {dt:.2f}s · {len(res):,} predictions")
else:
    st.info(" Upload a file *or* paste JSON to begin.")
