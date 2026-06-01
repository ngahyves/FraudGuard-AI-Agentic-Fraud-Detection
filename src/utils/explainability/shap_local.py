# src/utils/explainability/shap_local.py

import joblib
import pandas as pd
import shap
from pathlib import Path
from src.config.params_loader import load_params
from src.config.logging_config import get_logger

logger = get_logger("Local_Explainability")
cfg = load_params()

class LocalExplainer:
    def __init__(self):
        self.model_path = Path("models/best_xgboost_tuned.joblib")
        self.processed_data_path = (
            Path(cfg["data"]["processed_path"]) / cfg["data"]["processed_file"]
        )

        # Load model and preprocess
        logger.info("Loading model...")
        self.model = joblib.load(self.model_path)
        self.preprocess = self.model.named_steps["preprocess"]
        self.classifier = self.model.named_steps["model"]

        # Load preprocessed data
        logger.info("Loading processed data...")
        X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(self.processed_data_path)

        # Build features name
        num_features = cfg["features"]["numerical"]
        cat_encoder = self.preprocess.transformers_[1][1]["encoder"]
        cat_features = list(cat_encoder.get_feature_names_out())

        self.feature_names = num_features + cat_features
        self.X_test_df = pd.DataFrame(X_test, columns=self.feature_names)

        # Explainer SHAP
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_transaction(self, index: int):
        """
        explain x_test transaction via SHAP.
        """
        logger.info(f"Explaining transaction index: {index}")

        row = self.X_test_df.iloc[[index]]  # DataFrame (1 row)
        shap_values = self.explainer.shap_values(row)[0]

        # Sort by absolute importance
        contributions = sorted(
            zip(self.feature_names, row.values[0], shap_values),
            key=lambda x: abs(x[2]),
            reverse=True
        )

        # Top 5 factors
        top_factors = [
            {"feature": f, "value": float(v), "impact": float(round(s, 4))}
            for f, v, s in contributions[:5]
        ]

        # Fraud probability
        proba = float(self.classifier.predict_proba(row)[0][1])

        return top_factors, proba

