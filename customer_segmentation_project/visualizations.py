"""
visualizations.py
-----------------
All plotting functions for the customer segmentation project.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from segmentation import NUMERIC_FEATURES

# ── Palette & style ──────────────────────────────────────────────────────────
PALETTE  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
            "#FF9800", "#00BCD4", "#E91E63", "#607D8B"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, path: str):
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# ── 1. EDA ───────────────────────────────────────────────────────────────────

def plot_eda_overview(df: pd.DataFrame, out_dir: str):
    """Distribution plots for key numeric features."""
    cols = ["age", "annual_income", "spending_score",
            "purchase_frequency", "avg_order_value", "total_annual_spend"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Customer Data – Feature Distributions", fontsize=16, fontweight="bold")
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col], bins=30, kde=True, color=PALETTE[0], ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, f"{out_dir}/01_eda_distributions.png")


def plot_categorical_breakdown(df: pd.DataFrame, out_dir: str):
    """Bar charts for categorical variables."""
    cats = ["gender", "location", "income_level", "preferred_category"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Customer Demographics – Categorical Breakdown", fontsize=15, fontweight="bold")
    for ax, cat in zip(axes.flat, cats):
        vc = df[cat].value_counts()
        vc_df = vc.reset_index()
        vc_df.columns = [cat, "count"]
        sns.barplot(data=vc_df, x="count", y=cat,
                    hue=cat, palette=PALETTE[:len(vc)], ax=ax, legend=False)
        ax.set_title(cat.replace("_", " ").title())
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        for i, v in enumerate(vc.values):
            ax.text(v + 2, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, f"{out_dir}/02_eda_categorical.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    """Correlation heatmap for numeric features."""
    corr = df[NUMERIC_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"{out_dir}/03_correlation_heatmap.png")


# ── 2. Cluster selection ─────────────────────────────────────────────────────

def plot_elbow_metrics(metrics_df: pd.DataFrame, optimal_k: int, out_dir: str):
    """Elbow + silhouette + DB + CH scores side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Cluster Selection Metrics", fontsize=15, fontweight="bold")

    specs = [
        ("inertia",             "Inertia (WCSS)",              "#2196F3"),
        ("silhouette",          "Silhouette Score (↑ better)", "#4CAF50"),
        ("davies_bouldin",      "Davies-Bouldin (↓ better)",   "#FF5722"),
        ("calinski_harabasz",   "Calinski-Harabasz (↑ better)","#9C27B0"),
    ]
    for ax, (col, title, color) in zip(axes.flat, specs):
        ax.plot(metrics_df["k"], metrics_df[col], "o-", color=color, lw=2)
        ax.axvline(optimal_k, color="crimson", ls="--", lw=1.5,
                   label=f"Optimal k={optimal_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of Clusters (k)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, f"{out_dir}/04_cluster_selection.png")


# ── 3. Dendrogram ────────────────────────────────────────────────────────────

def plot_dendrogram(Z, optimal_k: int, out_dir: str):
    """Hierarchical clustering dendrogram (first 200 samples)."""
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               color_threshold=None, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (sample of 200 customers)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    _save(fig, f"{out_dir}/05_dendrogram.png")


# ── 4. Cluster scatter (PCA) ─────────────────────────────────────────────────

def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray,
                      pca, title: str, fname: str, out_dir: str):
    """2-D PCA scatter coloured by cluster."""
    fig, ax = plt.subplots(figsize=(11, 8))
    k = len(np.unique(labels))
    for seg in sorted(np.unique(labels)):
        mask = labels == seg
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=PALETTE[seg % len(PALETTE)], label=f"Segment {seg}",
                   alpha=0.65, s=25, edgecolors="none")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Segment", markerscale=1.5)
    plt.tight_layout()
    _save(fig, f"{out_dir}/{fname}")


# ── 5. Segment profiles ──────────────────────────────────────────────────────

def plot_segment_radar(profile: pd.DataFrame, out_dir: str):
    """Radar chart comparing segments on key metrics (normalised 0-1)."""
    metrics = ["age", "annual_income", "spending_score",
               "purchase_frequency", "avg_order_value",
               "loyalty_years", "online_purchase_pct"]

    data = profile[metrics].copy()
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], size=10)
    ax.set_yticklabels([])

    for idx, row in data_norm.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, color=PALETTE[idx % len(PALETTE)], lw=2)
        ax.fill(angles, values, color=PALETTE[idx % len(PALETTE)], alpha=0.15)

    patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)],
                               label=f"Segment {i}")
               for i in data_norm.index]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Segment Profile Radar Chart\n(Normalised Metrics)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, f"{out_dir}/07_segment_radar.png")


def plot_segment_bar_profiles(df_labeled: pd.DataFrame, out_dir: str):
    """Box plots for top metrics split by segment."""
    metrics = ["annual_income", "spending_score",
               "purchase_frequency", "total_annual_spend",
               "loyalty_years", "avg_order_value"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Segment Comparison – Key Metrics", fontsize=15, fontweight="bold")

    for ax, metric in zip(axes.flat, metrics):
        order = sorted(df_labeled["Segment"].unique())
        palette = {str(s): PALETTE[s % len(PALETTE)] for s in order}
        tmp = df_labeled.copy()
        tmp["Segment"] = tmp["Segment"].astype(str)
        sns.boxplot(data=tmp, x="Segment", y=metric,
                    hue="Segment", palette=palette, ax=ax,
                    order=[str(o) for o in order],
                    width=0.55, flierprops={"markersize": 3}, legend=False)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Segment")
    plt.tight_layout()
    _save(fig, f"{out_dir}/08_segment_boxplots.png")


def plot_category_preferences(df_labeled: pd.DataFrame, out_dir: str):
    """Stacked bar of preferred categories by segment."""
    ct = (df_labeled.groupby(["Segment", "preferred_category"])
          .size().unstack(fill_value=0))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    ct_pct.plot(kind="bar", stacked=True, ax=ax,
                colormap="tab10", edgecolor="white", width=0.65)
    ax.set_title("Preferred Product Category by Segment (%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    _save(fig, f"{out_dir}/09_category_preferences.png")


def plot_rfm_scatter(df_labeled: pd.DataFrame, out_dir: str):
    """
    RFM-style scatter: Recency (days_since_last_visit) vs Frequency
    (purchase_frequency), sized by Monetary (total_annual_spend).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for seg in sorted(df_labeled["Segment"].unique()):
        sub = df_labeled[df_labeled["Segment"] == seg]
        sizes = (sub["total_annual_spend"] / sub["total_annual_spend"].max()) * 300 + 20
        ax.scatter(sub["days_since_last_visit"], sub["purchase_frequency"],
                   s=sizes, c=PALETTE[seg % len(PALETTE)],
                   alpha=0.55, label=f"Segment {seg}", edgecolors="none")

    ax.set_xlabel("Days Since Last Visit (Recency ↓ better)")
    ax.set_ylabel("Purchase Frequency (per year)")
    ax.set_title("RFM Analysis — Recency vs Frequency\n(Bubble size = Total Annual Spend)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Segment", markerscale=0.7)
    plt.tight_layout()
    _save(fig, f"{out_dir}/10_rfm_scatter.png")
