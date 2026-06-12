"""
train_evaluate.py
Trains two classifiers (Logistic Regression and Random Forest),
compares their performance, and saves all evaluation plots.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    ConfusionMatrixDisplay,
)

sns.set_theme(style="whitegrid", palette="muted")


# ── helpers ────────────────────────────────────────────────────────────────

def load_and_prepare(path="data/customers.csv"):
    df = pd.read_csv(path)

    # one-hot encode the only categorical column
    df = pd.get_dummies(df, columns=["payment_method"], drop_first=False)

    feature_cols = [c for c in df.columns if c != "churn"]
    X = df[feature_cols].astype(float)
    y = df["churn"]
    return X, y, feature_cols


def print_section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── main ───────────────────────────────────────────────────────────────────

def run(seed=42):
    X, y, feature_cols = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    # ── define models ──────────────────────────────────────────────────────
    # Logistic Regression needs scaling; RandomForest doesn't care but we
    # keep it inside a pipeline for consistency.
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=300, C=0.8, random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=10,
                random_state=seed,
                n_jobs=1,
            )),
        ]),
    }

    results   = {}
    cv_scores = {}

    for name, model in models.items():
        print_section(name)

        # 5-fold cross-validation on training set
        cv = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="roc_auc", n_jobs=1)
        cv_scores[name] = cv
        print(f"  CV AUC: {cv.mean():.3f} ± {cv.std():.3f}")

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred,
                                    target_names=["Retained", "Churned"]))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "y_proba": y_proba,
            "cm":      confusion_matrix(y_test, y_pred),
            "fpr":     fpr,
            "tpr":     tpr,
            "auc":     roc_auc_val,
        }

    # ── Plot 1: Confusion Matrices ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confusion Matrices – Test Set", fontsize=13)

    for ax, (name, r) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(r["cm"],
                                     display_labels=["Retained", "Churned"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    fig.savefig("outputs/confusion_matrices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: ROC Curves ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452"]

    for (name, r), color in zip(results.items(), colors):
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2,
                label=f"{name}  (AUC = {r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Churn Prediction")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("outputs/roc_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Feature Importance (Random Forest) ─────────────────────────
    rf_clf = results["Random Forest"]["model"].named_steps["clf"]
    importances = rf_clf.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(12)                     # top 12 only
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["feature"], feat_df["importance"],
            color="#55A868", edgecolor="white")
    ax.set_title("Feature Importance – Random Forest (top 12)")
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    fig.savefig("outputs/feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: CV Score Comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    data_to_plot = [cv_scores[n] for n in models]
    bp = ax.boxplot(data_to_plot, labels=list(models.keys()),
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("5-Fold Cross-Validation AUC Comparison")
    plt.tight_layout()
    fig.savefig("outputs/cv_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print_section("All plots saved → outputs/")

    return results, cv_scores


if __name__ == "__main__":
    run()
