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
        # Attempt tree-based explanation if requested
        shap_values = None
        explainer = None
        if use_tree:
            try:
                print(f"🌳 Using TreeExplainer for {model_name}...")
                explainer = shap.TreeExplainer(predict_fn, pd_data)
                shap_values = explainer.shap_values(pd_data)
            except Exception as tree_exc:
                # TreeExplainer may not support function predictors; fall back
                print(f"⚠️ TreeExplainer unsupported for this model: {tree_exc}")
                print("🔁 Falling back to KernelExplainer...")
                use_tree = False

        # If not using tree explainer (either originally or due to fallback)
        if not use_tree:
            print(f"🔍 Using KernelExplainer for {model_name or 'generic model'}...")
            # KernelExplainer can be slow; limit nsamples
            explainer = shap.KernelExplainer(predict_fn, pd_data)
            shap_values = explainer.shap_values(pd_data, nsamples=max_samples)

        # If shap_values still None, something went wrong
        if shap_values is None:
            print("⚠️ SHAP values could not be computed.")
            return

        # Multi-class: use first output
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        os.makedirs(output_dir, exist_ok=True)

        # Save SHAP summary plot
        try:
            shap.summary_plot(shap_vals, pd_data, show=False)
            plot_path = os.path.join(output_dir, f"shap_summary_{model_type}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Saved SHAP summary to {plot_path}")
        except Exception as plot_exc:
            print(f"⚠️ Failed to create SHAP summary plot: {plot_exc}")

        # Save SHAP values CSV
        try:
            shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
            csv_path = os.path.join(output_dir, f"shap_values_{model_type}.csv")
            shap_df.to_csv(csv_path, index=False)
            print(f"✅ Saved SHAP values to {csv_path}")
        except Exception as csv_exc:
            print(f"⚠️ Failed to save SHAP values CSV: {csv_exc}")

    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")