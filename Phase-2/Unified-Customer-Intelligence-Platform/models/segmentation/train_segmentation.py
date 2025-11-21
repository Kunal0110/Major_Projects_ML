import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import VarianceThreshold

from pathlib import Path

# Auto-select features based on variance
def select_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Remove customer_id and target columns
    feature_cols = [col for col in numeric_cols if col not in ['customer_id', 'churn']]
    selector = VarianceThreshold(threshold=0.1)
    selected = selector.fit_transform(df[feature_cols])
    selected_features = np.array(feature_cols)[selector.get_support()]
    return selected_features.tolist()

Gold_Path = Path("data/gold/customer_gold_master.parquet")
Out_Dir = Path("models/segmentation")
Out_Dir.mkdir(parents=True, exist_ok=True)

def train_segmentation():
    df = pd.read_parquet(Gold_Path)
    print(df.columns.to_list())
    
    # Auto-select features
    features = select_features(df)
    print(f"Selected {len(features)} features: {features[:10]}...")
    
    df_seg = df[["customer_id"] + features].dropna()
    X = df_seg[features]

    # Use RobustScaler for better outlier handling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Plot
    inertias = []
    k_range = range(2,10)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig(Out_Dir / "elbow_plot.png")
    plt.close()

    # Multiple clustering metrics
    sil_scores = []
    ch_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_range, sil_scores, marker="o")
    ax1.set_title("Silhouette Scores")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Score")
    
    ax2.plot(k_range, ch_scores, marker="s", color='red')
    ax2.set_title("Calinski-Harabasz Scores")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score")
    
    plt.tight_layout()
    plt.savefig(Out_Dir / "clustering_metrics.png")
    plt.close()


    # Choose best k using combined metrics
    sil_normalized = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores))
    ch_normalized = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores))
    combined_scores = 0.6 * sil_normalized + 0.4 * ch_normalized
    
    best_k = k_range[int(np.argmax(combined_scores))]
    print(f"Best K = {best_k} (Silhouette: {sil_scores[best_k-2]:.3f}, CH: {ch_scores[best_k-2]:.3f})")


    # Final Model with better initialization
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, init='k-means++')
    df_seg["cluster"] = final_kmeans.fit_predict(X_scaled)

    # Saving models
    joblib.dump(final_kmeans, Out_Dir / "kmeans_model.pkl")
    joblib.dump(scaler, Out_Dir / "segmentation_scaler.pkl")
    joblib.dump(features, Out_Dir / "selected_features.pkl")

    # PCA Visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pcs_df = pd.DataFrame({
        "pc1": pcs[:,0],
        "pc2": pcs[:,1],
        "cluster": df_seg["cluster"]
    })

    pcs_df.to_csv(Out_Dir / "pca_components.csv", index=False)

    plt.scatter(pcs[:,0], pcs[:,1], c=df_seg["cluster"], cmap="tab10")
    plt.title("Customer Segmemts")
    plt.savefig(Out_Dir / "pca_clusters.png")
    plt.close()

    # Enhanced Cluster Profiling
    profile_mean = df_seg.groupby("cluster")[features].mean()
    profile_std = df_seg.groupby("cluster")[features].std()
    profile_count = df_seg.groupby("cluster").size()
    
    profile_mean.to_csv(Out_Dir / "cluster_profiles_mean.csv")
    profile_std.to_csv(Out_Dir / "cluster_profiles_std.csv")
    profile_count.to_csv(Out_Dir / "cluster_sizes.csv")
    
    print("Cluster sizes:", profile_count.values)

    print("Segmentation complete")

if __name__ == "__main__":
    train_segmentation()