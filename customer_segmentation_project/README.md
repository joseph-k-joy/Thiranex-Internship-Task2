# 🛍️ Customer Segmentation Project
### Thiranex Skill Development & Future Tech — Project #2

> **Goal:** Segment customers based on behavioural and demographic data  
> **Due Date:** 24 May 2026 | **Status:** Ready for Submission

---

## 📁 Project Structure

```
customer_segmentation_project/
│
├── main.py               ← Entry point — run this
├── data_generator.py     ← Synthetic customer dataset (1,000 records)
├── segmentation.py       ← Preprocessing, clustering, PCA, profiling
├── visualizations.py     ← All matplotlib/seaborn plots
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
└── outputs/              ← Auto-generated on first run
    ├── customer_data.csv
    ├── segment_profiles.csv
    ├── metrics_report.txt
    ├── 01_eda_distributions.png
    ├── 02_eda_categorical.png
    ├── 03_correlation_heatmap.png
    ├── 04_cluster_selection.png
    ├── 05_dendrogram.png
    ├── 06a_kmeans_pca.png
    ├── 06b_hierarchical_pca.png
    ├── 07_segment_radar.png
    ├── 08_segment_boxplots.png
    ├── 09_category_preferences.png
    └── 10_rfm_scatter.png
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

All outputs are saved to the `outputs/` folder automatically.

---

## 📊 Dataset Features

| Category      | Features |
|---------------|----------|
| Demographics  | age, gender, location, income_level, annual_income |
| Behaviour     | purchase_frequency, avg_order_value, total_annual_spend |
| Engagement    | days_since_last_visit, loyalty_years, online_purchase_pct |
| Preferences   | num_categories_bought, preferred_category, discount_usage_pct |
| Service       | return_rate, support_tickets, spending_score |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution histograms for all numeric features
- Bar charts for categorical variables (gender, location, income level, category)
- Correlation heatmap to identify feature relationships

### 2. Preprocessing
- **StandardScaler** applied to all 13 numeric features
- Ensures clustering is not biased by feature scale

### 3. Optimal Cluster Selection
Four metrics evaluated for k = 2 to 10:
- **Inertia (WCSS)** — Elbow method
- **Silhouette Score** — Cluster cohesion/separation (↑ better)
- **Davies-Bouldin Index** — Cluster compactness (↓ better)
- **Calinski-Harabasz Score** — Between/within cluster variance (↑ better)

### 4. Clustering Algorithms
| Algorithm | Purpose |
|-----------|---------|
| **K-Means** | Primary segmentation model |
| **Agglomerative (Ward)** | Validation & comparison |

### 5. Dimensionality Reduction
- **PCA (2D)** for visual cluster validation

### 6. Segment Analysis
- Mean statistics per segment (income, spending, frequency…)
- Category preference breakdown
- RFM (Recency, Frequency, Monetary) scatter analysis
- Radar chart for multi-metric segment comparison

---

## 📈 Key Outputs

| File | Description |
|------|-------------|
| `customer_data.csv` | Full 1,000-customer dataset |
| `segment_profiles.csv` | Per-cluster summary statistics |
| `metrics_report.txt` | Complete written analytical report |
| `04_cluster_selection.png` | Elbow + metric plots for choosing k |
| `07_segment_radar.png` | Radar chart comparing all segments |
| `10_rfm_scatter.png` | RFM analysis visualisation |

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| scikit-learn | 1.3+ | Clustering, PCA, preprocessing |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical computing |
| matplotlib | 3.7+ | Base plotting |
| seaborn | 0.12+ | Statistical visualisation |
| scipy | 1.11+ | Hierarchical clustering / dendrogram |

---

## 💡 Business Insights

After segmentation, the project delivers actionable insights:

- **High-value segments** → Premium loyalty programs, early access
- **Budget shoppers** → Flash sales, cashback, value bundles
- **Lapsed customers** → Win-back campaigns with exclusive discounts
- **Digital-first customers** → Mobile UX improvements, push notifications
- **Multi-category buyers** → Cross-sell recommendations

---

*Project submitted via Thiranex platform | Customer Segmentation — Project #2*
