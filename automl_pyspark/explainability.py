"""
SHAP explainability utility for PySpark AutoML models.

Uses TreeExplainer for tree-based models (faster, more accurate),
falls back to KernelExplainer otherwise. Saves both SHAP summary plot
and SHAP values CSV.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None


def compute_shap_values(
    spark,
    pipeline_model,
    model,
    sample_df,
    feature_cols,
    output_dir,
    model_type="classification",
    model_name=None,
    max_samples=50,
):
    """Compute SHAP values and save results."""

    if shap is None:
        print("⚠️ SHAP not installed. Run: pip install shap")
        return

    # Collect sample to pandas
    try:
        rows = sample_df.select(*feature_cols).limit(max_samples).collect()
        pd_data = pd.DataFrame([r.asDict() for r in rows])
    except Exception as e:
        print(f"⚠️ Could not collect sample for SHAP: {e}")
        return

    if pd_data.empty:
        print("⚠️ No data available for SHAP.")
        return

    # Unified prediction function
    def predict_fn(X):
        import numpy as np
        X_df = pd.DataFrame(X, columns=feature_cols)
        sdf = spark.createDataFrame(X_df)
        processed = pipeline_model.transform(sdf)
        preds = model.transform(processed).select("prediction").collect()
        return np.array([r.prediction for r in preds])

    # Detect tree-based models
    tree_models = ["xgboost", "lightgbm", "random_forest", "decision_tree", "gbt"]
    use_tree = model_name and model_name.lower() in tree_models

    try:
        if use_tree:
            print(f"🌳 Using TreeExplainer for {model_name}...")
            explainer = shap.TreeExplainer(predict_fn, pd_data)
            shap_values = explainer.shap_values(pd_data)
        else:
            print(f"🔍 Using KernelExplainer for {model_name or 'generic model'}...")
            explainer = shap.KernelExplainer(predict_fn, pd_data)
            shap_values = explainer.shap_values(pd_data, nsamples=max_samples)

        # Multi-class: use first output
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        os.makedirs(output_dir, exist_ok=True)

        # Save SHAP summary plot
        shap.summary_plot(shap_vals, pd_data, show=False)
        plot_path = os.path.join(output_dir, f"shap_summary_{model_type}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Save SHAP values CSV
        shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
        csv_path = os.path.join(output_dir, f"shap_values_{model_type}.csv")
        shap_df.to_csv(csv_path, index=False)

        print(f"✅ Saved SHAP summary to {plot_path}")
        print(f"✅ Saved SHAP values to {csv_path}")

    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")