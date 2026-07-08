import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def run_eda(path="data/customers.csv"):
    df = pd.read_csv(path)

    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Rows: {len(df):,}   |   Columns: {df.shape[1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Churn rate: {df['churn'].mean():.1%}\n")
    print(df.describe(include="all").T.to_string())

  
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Customer Churn – Exploratory Analysis", fontsize=14, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # churn balance
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["churn"].value_counts()
    ax0.bar(["Retained", "Churned"], counts.values,
            color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.8)
    ax0.set_title("Churn Balance")
    ax0.set_ylabel("Customers")
    for i, v in enumerate(counts.values):
        ax0.text(i, v + 15, str(v), ha="center", fontsize=9)

    # tenure by churn
    ax1 = fig.add_subplot(gs[0, 1])
    for label, grp in df.groupby("churn"):
        ax1.hist(grp["tenure"], bins=20, alpha=0.65,
                 label=["Retained", "Churned"][label], density=True)
    ax1.set_title("Tenure Distribution")
    ax1.set_xlabel("Months")
    ax1.legend(fontsize=8)

    # monthly fee by churn
    ax2 = fig.add_subplot(gs[0, 2])
    df.boxplot(column="monthly_fee", by="churn", ax=ax2,
               boxprops=dict(color="#4C72B0"),
               medianprops=dict(color="#DD8452", linewidth=2))
    ax2.set_title("Monthly Fee vs Churn")
    ax2.set_xlabel("Churn (0 = No, 1 = Yes)")
    ax2.set_ylabel("$/month")
    plt.sca(ax2); plt.title("Monthly Fee vs Churn")

    # support calls
    ax3 = fig.add_subplot(gs[1, 0])
    call_churn = df.groupby("support_calls")["churn"].mean()
    ax3.bar(call_churn.index, call_churn.values, color="#55A868", edgecolor="white")
    ax3.set_title("Churn Rate by Support Calls")
    ax3.set_xlabel("Number of calls")
    ax3.set_ylabel("Churn rate")

    # contract type
    ax4 = fig.add_subplot(gs[1, 1])
    ct = df.groupby("has_contract")["churn"].mean()
    ax4.bar(["Month-to-month", "Contract"], ct.values,
            color=["#C44E52", "#4C72B0"], edgecolor="white")
    ax4.set_title("Churn Rate by Contract")
    ax4.set_ylabel("Churn rate")

    # num products
    ax5 = fig.add_subplot(gs[1, 2])
    pt = df.groupby("num_products")["churn"].mean()
    ax5.plot(pt.index, pt.values, marker="o", color="#8172B2", linewidth=2)
    ax5.set_title("Churn Rate by # Products")
    ax5.set_xlabel("Products subscribed")
    ax5.set_ylabel("Churn rate")

    plt.tight_layout()
    fig.savefig("outputs/eda_overview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    num_cols = ["tenure", "monthly_fee", "num_products",
                "support_calls", "senior_citizen", "has_partner",
                "has_contract", "paperless_billing", "churn"]
    corr = df[num_cols].corr()

    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig2.tight_layout()
    fig2.savefig("outputs/correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print("\nEDA plots saved → outputs/")
    return df


if __name__ == "__main__":
    run_eda()
