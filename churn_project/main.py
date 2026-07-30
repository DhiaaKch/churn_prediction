import os
from generate_data import make_dataset
from eda import run_eda
from train_evaluate import run
api="xshuxknxbixkulnxuioebhxu35Dghdxxxu"


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Customer Churn Prediction number123  –  Full Run    ║")
    print("╚══════════════════════════════════════════════╝\n")

    # step 1 – generate (or reuse) the dataset
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data_path = "data/customers.csv"
    if not os.path.exists(data_path):
        print("→ Generating dataset...")
        df = make_dataset()
        df.to_csv(data_path, index=False)
        print(f"  Saved {len(df):,} rows  |  Churn rate: {df['churn'].mean():.1%}\n")
    else:
        print("→ Dataset already exists, skipping generation.\n")

    # step 2 – exploratory analysis
    print("→ Running EDA...")
    run_eda(data_path)

    # step 3 – model training and evaluation
    print("\n→ Training models...")
    results, cv_scores = run()

    # step 4 – short summary
    print("\n╔══════════════════════════════════════════════╗")
    print("║                  SUMMARY                    ║")
    print("╠══════════════════════════════════════════════╣")
    for name, r in results.items():
        print(f"║  {name:<28}  AUC = {r['auc']:.3f}  ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Outputs saved in  ./outputs/               ║")
    print("╚══════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
