from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import shap


def run_shap_analysis():

    # REPORT DIRECTORY

    report_dir = Path("reports/churn")
    report_dir.mkdir(parents=True, exist_ok=True)

    # LOAD MODEL

    model = joblib.load("artifacts/churn/models/xgb_model.pkl")

    # LOAD TEST DATA

    X_test = joblib.load("artifacts/churn/evaluation/x_test.pkl")

    print(f"Loaded X_test shape: {X_test.shape}")

    # CREATE EXPLAINER

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # GLOBAL FEATURE IMPORTANCE (DOT PLOT)

    plt.figure()

    shap.summary_plot(shap_values, X_test, show=False)

    plt.savefig(
        report_dir / "shap_summary.png",
        bbox_inches="tight",
    )

    plt.close()

    # GLOBAL FEATURE IMPORTANCE (BAR)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )

    plt.savefig(report_dir / "shape_bar.png", bbox_inches="tight")

    plt.close()

    # SINGLE CUSTOMER EXPLAINATION

    customer_idx = 0

    explanation = explainer(X_test.iloc[[customer_idx]])

    shap.plots.waterfall(
        explanation[0],
        show=False,
    )

    plt.gcf().savefig(
        report_dir / "customer_0_waterfall.png",
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP analysis complete.")
    print("Generated:")
    print("- shap_summary.png")
    print("- shap_bar.png")
    print("- customer_0_waterfall.png")


if __name__ == "__main__":
    run_shap_analysis()
