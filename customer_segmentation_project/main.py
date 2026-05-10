"""
main.py
-------
Customer Segmentation Project — Main Entry Point
Thiranex Skill Development & Future Tech

Run:
    python main.py

Outputs:
    outputs/  ← all PNG visualisations
    outputs/customer_data.csv
    outputs/segment_profiles.csv
    outputs/metrics_report.txt
"""

import os
import sys
import time
import textwrap
import numpy as np
import pandas as pd

# ── ensure local modules are importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_customer_data
from segmentation   import (preprocess, compute_elbow_data, pick_optimal_k,
                             kmeans_clustering, hierarchical_clustering,
                             compute_linkage_matrix, apply_pca,
                             build_segment_profiles)
from visualizations import (plot_eda_overview, plot_categorical_breakdown,
                             plot_correlation_heatmap, plot_elbow_metrics,
                             plot_dendrogram, plot_pca_clusters,
                             plot_segment_radar, plot_segment_bar_profiles,
                             plot_category_preferences, plot_rfm_scatter)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Segment name mapping (assigned after profiling) ──────────────────────────
SEGMENT_NAMES = {
    0: "Budget Shoppers",
    1: "Loyal High-Spenders",
    2: "Occasional Browsers",
    3: "Premium Deal-Seekers",
    4: "Young Trendsetters",
}

# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str):
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}")


def main():
    t0 = time.time()

    # ── Step 1 · Generate data ───────────────────────────────────────────────
    banner("STEP 1 · Generating Customer Dataset")
    df = generate_customer_data(n_customers=1000, random_state=42)
    df.to_csv(f"{OUT_DIR}/customer_data.csv", index=False)
    print(f"  Records  : {len(df):,}")
    print(f"  Features : {df.shape[1]}")
    print(f"  Sample:\n{df.head(3).to_string()}")

    # ── Step 2 · EDA ─────────────────────────────────────────────────────────
    banner("STEP 2 · Exploratory Data Analysis")
    print("\nDescriptive statistics (numeric):")
    desc = df.select_dtypes(include=np.number).describe().T[
        ["mean","std","min","50%","max"]
    ].round(2)
    print(desc.to_string())

    print("\nMissing values:", df.isnull().sum().sum())
    print("\nPlotting EDA charts …")
    plot_eda_overview(df, OUT_DIR)
    plot_categorical_breakdown(df, OUT_DIR)
    plot_correlation_heatmap(df, OUT_DIR)

    # ── Step 3 · Preprocessing ───────────────────────────────────────────────
    banner("STEP 3 · Preprocessing & Feature Scaling")
    X_scaled, scaler = preprocess(df)
    print(f"  Feature matrix shape: {X_scaled.shape}")
    print("  StandardScaler applied — mean≈0, std≈1 per feature.")

    # ── Step 4 · Determine optimal k ────────────────────────────────────────
    banner("STEP 4 · Finding Optimal Number of Clusters")
    metrics_df = compute_elbow_data(X_scaled, k_range=range(2, 11))
    optimal_k  = pick_optimal_k(metrics_df)
    print(f"\n  Cluster metrics:\n{metrics_df.to_string(index=False)}")
    print(f"\n  ★ Optimal k = {optimal_k}  (highest silhouette score)")
    plot_elbow_metrics(metrics_df, optimal_k, OUT_DIR)

    # ── Step 5 · K-Means clustering ──────────────────────────────────────────
    banner("STEP 5 · K-Means Clustering")
    km_model, km_labels = kmeans_clustering(X_scaled, k=optimal_k)
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil = silhouette_score(X_scaled, km_labels)
    db  = davies_bouldin_score(X_scaled, km_labels)
    print(f"  Silhouette Score    : {sil:.4f}  (closer to 1 = better)")
    print(f"  Davies-Bouldin Score: {db:.4f}   (lower = better)")
    print(f"  Cluster sizes:\n{pd.Series(km_labels).value_counts().sort_index().to_string()}")

    # ── Step 6 · Hierarchical clustering ────────────────────────────────────
    banner("STEP 6 · Hierarchical Clustering")
    hc_model, hc_labels = hierarchical_clustering(X_scaled, k=optimal_k)
    Z = compute_linkage_matrix(X_scaled)
    print("  Agglomerative (Ward) clustering complete.")
    plot_dendrogram(Z, optimal_k, OUT_DIR)

    # ── Step 7 · PCA visualisation ───────────────────────────────────────────
    banner("STEP 7 · PCA Dimensionality Reduction & Visualisation")
    X_pca, pca = apply_pca(X_scaled, n_components=2)
    ev = pca.explained_variance_ratio_
    print(f"  PC1 explained variance: {ev[0]*100:.1f}%")
    print(f"  PC2 explained variance: {ev[1]*100:.1f}%")
    print(f"  Total variance captured: {sum(ev)*100:.1f}%")

    plot_pca_clusters(X_pca, km_labels, pca,
                      title=f"K-Means Segments (k={optimal_k}) — PCA Projection",
                      fname="06a_kmeans_pca.png", out_dir=OUT_DIR)
    plot_pca_clusters(X_pca, hc_labels, pca,
                      title=f"Hierarchical Segments (k={optimal_k}) — PCA Projection",
                      fname="06b_hierarchical_pca.png", out_dir=OUT_DIR)

    # ── Step 8 · Segment profiles ────────────────────────────────────────────
    banner("STEP 8 · Segment Profiling & Analysis")
    profile, df_labeled = build_segment_profiles(df, km_labels)
    profile.to_csv(f"{OUT_DIR}/segment_profiles.csv")
    print("\n  Segment profiles:")
    print(profile.to_string())

    plot_segment_radar(profile, OUT_DIR)
    plot_segment_bar_profiles(df_labeled, OUT_DIR)
    plot_category_preferences(df_labeled, OUT_DIR)
    plot_rfm_scatter(df_labeled, OUT_DIR)

    # ── Step 9 · Text report ─────────────────────────────────────────────────
    banner("STEP 9 · Generating Written Report")
    _write_report(df, df_labeled, profile, metrics_df, optimal_k, sil, db, ev)

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"COMPLETE — {elapsed:.1f}s  |  All outputs in ./outputs/")
    _print_file_summary()


# ─────────────────────────────────────────────────────────────────────────────

def _write_report(df, df_labeled, profile, metrics_df, optimal_k, sil, db, ev):
    lines = []
    w = lambda s: lines.append(s)

    w("=" * 70)
    w("  CUSTOMER SEGMENTATION PROJECT — ANALYTICAL REPORT")
    w("  Thiranex Skill Development & Future Tech")
    w("=" * 70)
    w("")
    w("1. DATASET OVERVIEW")
    w(f"   • Total customers  : {len(df):,}")
    w(f"   • Total features   : {df.shape[1]}")
    w(f"   • Numeric features : 13  (scaled with StandardScaler)")
    w(f"   • Categorical      : 4   (gender, location, income_level, category)")
    w("")
    w("2. METHODOLOGY")
    w("   a. Exploratory Data Analysis — distribution plots, bar charts,")
    w("      correlation heatmap.")
    w("   b. Preprocessing — StandardScaler on all numeric features.")
    w("   c. Optimal k selection — Elbow (inertia), Silhouette, Davies-Bouldin,")
    w("      and Calinski-Harabasz scores evaluated for k in [2, 10].")
    w("   d. K-Means Clustering (primary model).")
    w("   e. Agglomerative Hierarchical Clustering (validation / comparison).")
    w("   f. PCA (2-D) for visual validation of cluster separation.")
    w("   g. Segment profiling — mean numeric stats + modal categorical values.")
    w("   h. RFM analysis (Recency, Frequency, Monetary) scatter.")
    w("")
    w(f"3. OPTIMAL NUMBER OF CLUSTERS:  k = {optimal_k}")
    w(f"   • Silhouette Score     : {sil:.4f}")
    w(f"   • Davies-Bouldin Score : {db:.4f}")
    w(f"   • PCA variance (PC1+PC2): {sum(ev)*100:.1f}%")
    w("")
    w("4. CLUSTER SELECTION METRICS TABLE")
    w(metrics_df.to_string(index=False))
    w("")
    w("5. SEGMENT PROFILES")
    w(profile.to_string())
    w("")
    w("6. BUSINESS INSIGHTS PER SEGMENT")

    for seg in sorted(df_labeled["Segment"].unique()):
        sub  = df_labeled[df_labeled["Segment"] == seg]
        name = SEGMENT_NAMES.get(seg, f"Segment {seg}")
        w("")
        w(f"   ── Segment {seg} | {name} ({len(sub):,} customers, "
          f"{len(sub)/len(df_labeled)*100:.1f}%) ──")
        w(f"   Avg income        : ${sub['annual_income'].mean():,.0f}")
        w(f"   Avg spending score: {sub['spending_score'].mean():.1f}/100")
        w(f"   Avg order value   : ${sub['avg_order_value'].mean():.2f}")
        w(f"   Purchase frequency: {sub['purchase_frequency'].mean():.1f} visits/yr")
        w(f"   Loyalty (yrs)     : {sub['loyalty_years'].mean():.1f}")
        w(f"   Online pct        : {sub['online_purchase_pct'].mean()*100:.0f}%")
        w(f"   Top category      : {sub['preferred_category'].value_counts().index[0]}")
        w(f"   Top location      : {sub['location'].value_counts().index[0]}")

    w("")
    w("7. MARKETING RECOMMENDATIONS")
    recs = [
        "• High-income, high-spend segments → premium loyalty programs, early-access",
        "  offers, concierge service.",
        "• Budget shoppers → flash sales, value bundles, cashback campaigns.",
        "• Low-recency customers (haven't visited recently) → win-back email/SMS",
        "  campaigns with exclusive discounts.",
        "• High online_purchase_pct segments → targeted digital ads, push",
        "  notifications, and mobile-first UX improvements.",
        "• Multi-category buyers → cross-sell recommendations and subscription",
        "  boxes tailored to their purchase history.",
    ]
    for r in recs:
        w(r)

    w("")
    w("8. OUTPUT FILES")
    w("   outputs/customer_data.csv         — Raw generated dataset")
    w("   outputs/segment_profiles.csv      — Per-segment statistics")
    w("   outputs/01_eda_distributions.png  — Feature distributions")
    w("   outputs/02_eda_categorical.png    — Categorical breakdowns")
    w("   outputs/03_correlation_heatmap.png— Correlation matrix")
    w("   outputs/04_cluster_selection.png  — Elbow & metric plots")
    w("   outputs/05_dendrogram.png         — Hierarchical dendrogram")
    w("   outputs/06a_kmeans_pca.png        — K-Means PCA scatter")
    w("   outputs/06b_hierarchical_pca.png  — Hierarchical PCA scatter")
    w("   outputs/07_segment_radar.png      — Segment radar chart")
    w("   outputs/08_segment_boxplots.png   — Metric boxplots by segment")
    w("   outputs/09_category_preferences.png — Category mix by segment")
    w("   outputs/10_rfm_scatter.png        — RFM analysis scatter")
    w("")
    w("=" * 70)

    report_text = "\n".join(lines)
    path = f"{OUT_DIR}/metrics_report.txt"
    with open(path, "w") as f:
        f.write(report_text)
    print(f"  ✔  Saved → {path}")
    print("\n" + report_text)


def _print_file_summary():
    print("\n  Files generated:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"    {fname:<45} {size/1024:>7.1f} KB")


if __name__ == "__main__":
    main()
