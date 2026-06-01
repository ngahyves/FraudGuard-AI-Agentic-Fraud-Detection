# src/utils/explainability/shap_global.py

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from src.config.params_loader import load_params
from src.config.logging_config import get_logger

# Initialize logger and configuration
logger = get_logger("Global_Explainability")
cfg = load_params()

class GlobalExplainer:
    """
    Computes global feature importance using SHAP values.
    Handles One-Hot Encoded (OHE) feature consolidation for better interpretability.
    """
    def __init__(self):
        self.config = cfg
        
        # Paths from configuration
        self.model_path = Path("models/best_xgboost_tuned.joblib")
        self.processed_data_path = (
            Path(cfg["data"]["processed_path"]) / cfg["data"]["processed_file"]
        )

        # MLflow setup
        tracking_uri = cfg["experimentation"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg["experimentation"]["experiment_name"])
        logger.info(f"MLflow Tracking URI initialized: {tracking_uri}")

    def run(self):
        logger.info("Loading champion model and test dataset...")
        model = joblib.load(self.model_path)
        # Load raw processed data
        _, _, X_test, _, _, _ = joblib.load(self.processed_data_path)

        # 1. TRANSFORM the raw data into numerical data using the pipeline
        logger.info("Transforming raw features into numerical format...")
        preprocessor = model.named_steps["preprocess"]
        # This turns 'TRANSFER' into [0, 1, 0...] so XGBoost can read it
        X_test_transformed = preprocessor.transform(X_test)

        # 2. Extract feature names (encoded names)
        num_features = self.config["features"]["numerical"]
        cat_encoder = preprocessor.transformers_[1][1]["encoder"]
        cat_features_encoded = list(cat_encoder.get_feature_names_out())
        all_feature_names = num_features + cat_features_encoded

        # 3. Compute SHAP values using the TRANSFORMED data and the internal MODEL
        logger.info("Calculating SHAP values...")
        classifier = model.named_steps["model"]
        explainer = shap.TreeExplainer(classifier)
        
        # We pass X_test_transformed, not the raw X_test
        shap_values = explainer.shap_values(X_test_transformed)

        # 4. Consolidate OHE features (Grouping Logic)
        logger.info("Grouping SHAP values for categorical features...")
        shap_df = pd.DataFrame(shap_values, columns=all_feature_names)
        grouped_shap = shap_df.groupby(lambda x: x.split('_')[0], axis=1).sum()

        # Rename x0 and x1 par by their business names
        rename_dict = {
            "x0": "type",
            "x1": "category"
        }
        grouped_shap = grouped_shap.rename(columns=rename_dict)
        
        # Calculate Global Importance: mean of absolute SHAP values per grouped feature
        importance = grouped_shap.abs().mean().sort_values(ascending=True)

        # 4. Save visualization and log to MLflow
        output_dir = Path("reports/shap")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "global_importance_grouped.png"

        with mlflow.start_run(run_name="SHAP_Global_Analysis"):
            plt.figure(figsize=(10, 6))
            importance.plot(kind='barh', color='#3498db', edgecolor='black')
            plt.title("Global Feature Importance (Grouped Business Variables)")
            plt.xlabel("mean(|SHAP value|) - Average Impact on Model Prediction")
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Log plot as artifact
            mlflow.log_artifact(str(plot_path), artifact_path="explainability")
            logger.info(f"Consolidated SHAP chart saved and logged to MLflow: {plot_path}")

        return importance.to_dict()

if __name__ == "__main__":
    try:
        explainer = GlobalExplainer()
        explainer.run()
    except Exception as e:
        logger.error(f"Explainability module execution failed: {e}")
        raise