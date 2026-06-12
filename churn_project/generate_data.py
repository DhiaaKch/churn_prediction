"""
generate_data.py
Simulates a telecom customer dataset with realistic churn patterns.
Keeps it small (~2000 rows) so it runs fast anywhere.
"""

import numpy as np
import pandas as pd

def make_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    tenure      = rng.integers(1, 73, n)                        # months with company
    monthly_fee = rng.normal(65, 20, n).clip(20, 120)
    num_products = rng.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.10])
    support_calls = rng.poisson(1.5, n)
    has_contract  = rng.choice([0, 1], n, p=[0.55, 0.45])       # 0 = month-to-month
    paperless     = rng.choice([0, 1], n, p=[0.4, 0.6])
    payment_method = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    senior        = rng.choice([0, 1], n, p=[0.84, 0.16])
    has_partner   = rng.choice([0, 1], n, p=[0.52, 0.48])

    # churn probability driven by real-world logic
    churn_score = (
        -0.04 * tenure
        + 0.012 * monthly_fee
        - 0.3  * has_contract
        + 0.18 * support_calls
        - 0.25 * num_products
        + 0.15 * senior
        - 0.12 * has_partner
        + rng.normal(0, 0.3, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn      = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure":          tenure,
        "monthly_fee":     monthly_fee.round(2),
        "num_products":    num_products,
        "support_calls":   support_calls,
        "has_contract":    has_contract,
        "paperless_billing": paperless,
        "payment_method":  payment_method,
        "senior_citizen":  senior,
        "has_partner":     has_partner,
        "churn":           churn,
    })
    return df


if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("data/customers.csv", index=False)
    print(f"Saved {len(df)} rows. Churn rate: {df['churn'].mean():.1%}")
