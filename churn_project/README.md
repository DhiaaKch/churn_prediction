# Customer Churn Prediction

A practical machine learning project that predicts whether a telecom customer
will cancel their subscription, using tabular data, two classifiers, and a
full evaluation pipeline.

---

## Project structure

```
churn_project/
├── data/
│   └── customers.csv          # synthetic dataset (2 000 rows)
├── outputs/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── cv_comparison.png
├── generate_data.py            # builds the dataset
├── eda.py                      # exploratory analysis + plots
├── train_evaluate.py           # model training, CV, evaluation plots
├── main.py                     # runs everything end-to-end
└── README.md
```

---

## Quickstart

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python3 main.py
```

All outputs land in `./outputs/`. No GPU or heavy dependencies required;
the full run completes in under 30 seconds on a standard laptop.

---

## Dataset

Synthetic telecom data generated with realistic churn drivers:

| Feature | Description |
|---|---|
| `tenure` | Months as a customer (1–72) |
| `monthly_fee` | Monthly bill in USD |
| `num_products` | Number of subscribed services |
| `support_calls` | Support contacts in last 12 months |
| `has_contract` | 1 = annual contract, 0 = month-to-month |
| `paperless_billing` | Paperless billing opted in |
| `payment_method` | One of four payment options |
| `senior_citizen` | Binary demographic flag |
| `has_partner` | Binary demographic flag |
| `churn` | **Target** – 1 if customer left |

Churn rate: ~30% (mild imbalance, representative of real datasets).

---

## Models

Two pipelines are compared:

**Logistic Regression** — a strong linear baseline. Fast, interpretable,
and often surprisingly competitive on tabular data. Includes `StandardScaler`
since LR is sensitive to feature scale.

**Random Forest** — 120 shallow trees with `min_samples_leaf=10` to
avoid overfitting on a small dataset. Naturally handles non-linear
interactions and provides feature importance scores.

Both are evaluated with:
- 5-fold stratified cross-validation (ROC-AUC)
- Test-set confusion matrix
- ROC curve + AUC
- Feature importance chart (Random Forest)

---

## Results

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.730 ± 0.028 | 0.758 |
| Random Forest | 0.714 ± 0.026 | 0.748 |

Logistic Regression edges out the Random Forest here — which makes sense
given that the churn labels were generated from a logistic function.
On real-world data with complex interactions, RF typically gains an advantage.

---

## What to try next

- **Class imbalance** – test `class_weight='balanced'` or SMOTE oversampling
- **Threshold tuning** – the default 0.5 threshold is rarely optimal for
  imbalanced churn; try precision-recall curves to pick a better cutoff
- **Gradient Boosting** – XGBoost or LightGBM usually outperform vanilla RF
- **Feature engineering** – e.g. `fee_per_product = monthly_fee / num_products`
- **SHAP values** – for per-prediction explanations beyond global importance
