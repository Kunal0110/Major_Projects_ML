import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def gauge_plot(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%'},
        title={'text': "Churn Probability"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if prob > 0.5 else "green"}}
    ))
    return fig

def pca_plot(pcs, clusters):
    df = pd.DataFrame({
        "PC1": pcs[:,0],
        "PC2": pcs[:,1],
        "cluster": clusters
    })

    fig = px.scatter(
        df, x = "PC1", y = "PC2",
        color="cluster",
        title="Customer Clusters (PCS Projection)",
        color_continuous_scale="Turbo"
    )

    return fig
