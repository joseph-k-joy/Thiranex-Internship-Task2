"""
segmentation.py
---------------
Preprocessing, dimensionality reduction, and clustering logic.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ── Feature selection ────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "spending_score",
    "purchase_frequency",
    "avg_order_value",
    "total_annual_spend",
    "days_since_last_visit",
    "num_categories_bought",
    "loyalty_years",
    "discount_usage_pct",
    "online_purchase_pct",
    "return_rate",
    "support_tickets",
]


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Scale numeric features and return (X_scaled, scaler).
    """
    X = df[NUMERIC_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ── Optimal K ────────────────────────────────────────────────────────────────

def compute_elbow_data(X_scaled: np.ndarray, k_range=range(2, 11)):
    """
    Return inertia, silhouette, davies-bouldin, and calinski-harabasz scores
    for each k so we can pick the best number of clusters.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        records.append({
            "k"                  : k,
            "inertia"            : km.inertia_,
            "silhouette"         : silhouette_score(X_scaled, labels),
            "davies_bouldin"     : davies_bouldin_score(X_scaled, labels),
            "calinski_harabasz"  : calinski_harabasz_score(X_scaled, labels),
        })
    return pd.DataFrame(records)


def pick_optimal_k(metrics_df: pd.DataFrame) -> int:
    """
    Simple heuristic: highest silhouette score.
    """
    return int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])


# ── Clustering ───────────────────────────────────────────────────────────────

def kmeans_clustering(X_scaled: np.ndarray, k: int):
    """Fit K-Means and return (model, labels)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def hierarchical_clustering(X_scaled: np.ndarray, k: int):
    """Fit Agglomerative Clustering and return labels."""
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    return hc, labels


def compute_linkage_matrix(X_scaled: np.ndarray):
    """Compute Ward linkage matrix for dendrogram (uses a sample for speed)."""
    sample = X_scaled[:200] if len(X_scaled) > 200 else X_scaled
    return linkage(sample, method="ward")


# ── PCA ──────────────────────────────────────────────────────────────────────

def apply_pca(X_scaled: np.ndarray, n_components: int = 2):
    """Reduce to n_components principal components."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


# ── Segment profiles ─────────────────────────────────────────────────────────

def build_segment_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Attach cluster labels to df and compute per-segment mean/mode statistics.
    """
    df = df.copy()
    df["Segment"] = labels

    numeric_means = (
        df.groupby("Segment")[NUMERIC_FEATURES]
        .mean()
        .round(2)
    )

    # Most common categorical values per segment
    cat_modes = df.groupby("Segment")[["gender", "location",
                                        "income_level", "preferred_category"]].agg(
        lambda x: x.value_counts().index[0]
    )

    segment_size = df.groupby("Segment").size().rename("segment_size")

    profile = pd.concat([segment_size, numeric_means, cat_modes], axis=1)
    return profile, df
