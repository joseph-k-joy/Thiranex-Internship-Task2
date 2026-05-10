"""
data_generator.py
-----------------
Generates a realistic synthetic customer dataset with demographic
and behavioural features for the segmentation project.
"""

import numpy as np
import pandas as pd

def generate_customer_data(n_customers: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data with demographics and purchase behaviour.

    Parameters
    ----------
    n_customers : int
        Number of customer records to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per customer.
    """
    rng = np.random.default_rng(random_state)

    # ── Demographics ────────────────────────────────────────────────────────────
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    ages         = rng.integers(18, 71, size=n_customers)
    genders      = rng.choice(["Male", "Female", "Other"], size=n_customers,
                               p=[0.48, 0.48, 0.04])
    locations    = rng.choice(
        ["Urban", "Suburban", "Rural"], size=n_customers, p=[0.50, 0.35, 0.15]
    )
    income_levels = rng.choice(
        ["Low", "Medium", "High", "Very High"], size=n_customers,
        p=[0.25, 0.40, 0.25, 0.10]
    )
    income_map = {"Low": 20_000, "Medium": 50_000, "High": 90_000, "Very High": 150_000}
    annual_income = np.array([
        income_map[lvl] + rng.integers(-8_000, 8_001) for lvl in income_levels
    ])

    # ── Purchase behaviour ───────────────────────────────────────────────────────
    # Spending score 1-100 (loosely correlated with income)
    base_spending = np.clip(
        (annual_income / 2_000).astype(int) + rng.integers(-15, 16, size=n_customers),
        1, 100
    )

    purchase_frequency     = rng.integers(1, 53, size=n_customers)   # visits/year
    avg_order_value        = np.round(
        rng.uniform(10, 500, size=n_customers), 2
    )
    total_spend            = np.round(purchase_frequency * avg_order_value, 2)
    days_since_last_visit  = rng.integers(1, 366, size=n_customers)
    num_categories_bought  = rng.integers(1, 11, size=n_customers)
    loyalty_years          = rng.integers(0, 11, size=n_customers)
    discount_usage_pct     = np.round(rng.uniform(0, 1, size=n_customers), 2)
    online_purchase_pct    = np.round(rng.uniform(0, 1, size=n_customers), 2)
    return_rate            = np.round(rng.uniform(0, 0.30, size=n_customers), 3)
    support_tickets        = rng.integers(0, 11, size=n_customers)

    # ── Preferred category ───────────────────────────────────────────────────────
    categories = ["Electronics", "Fashion", "Grocery", "Home & Garden",
                  "Sports", "Beauty", "Books", "Travel"]
    preferred_category = rng.choice(categories, size=n_customers)

    df = pd.DataFrame({
        "customer_id"           : customer_ids,
        "age"                   : ages,
        "gender"                : genders,
        "location"              : locations,
        "income_level"          : income_levels,
        "annual_income"         : annual_income,
        "spending_score"        : base_spending,
        "purchase_frequency"    : purchase_frequency,
        "avg_order_value"       : avg_order_value,
        "total_annual_spend"    : total_spend,
        "days_since_last_visit" : days_since_last_visit,
        "num_categories_bought" : num_categories_bought,
        "loyalty_years"         : loyalty_years,
        "discount_usage_pct"    : discount_usage_pct,
        "online_purchase_pct"   : online_purchase_pct,
        "return_rate"           : return_rate,
        "support_tickets"       : support_tickets,
        "preferred_category"    : preferred_category,
    })

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    df.to_csv("customer_data.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
