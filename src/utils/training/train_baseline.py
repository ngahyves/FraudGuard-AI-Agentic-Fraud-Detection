# src/utils/training/train_baselines.py

import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config.logging_config import get_logger
from src.config.params_loader import load_params
from src.config.env_vars import PROCESSED_DIR

logger = get_logger(__name__)
cfg = load_params()


class FraudModelTrainer:
    """
    Baseline training with a FULL sklearn Pipeline:
    - preprocessing (impute + scale + encode)
    - model
    - MLflow logging
    - best model saved as a single artifact
    """

    def __init__(self):
        self.num_features = cfg["features"]["numerical"]
        self.cat_features = cfg["features"]["categorical"]
        self.random_state = cfg["data"]["random_state"]

        # MLflow
        self.tracking_uri = cfg["experimentation"]["tracking_uri"]
        self.experiment_name = cfg["experimentation"]["experiment_name"]
        mlflow.set_tracking_uri(self.tracking_uri)

    # ------------------------------------------------------------
    # 1. Baseline models
    # ------------------------------------------------------------
    def get_models(self, pos_weight):
        params = cfg["model_params"]

        return {
            "logistic_regression": LogisticRegression(
                max_iter=params["logistic_regression"]["max_iter"],
                solver=params["logistic_regression"]["solver"],
                class_weight=params["logistic_regression"]["class_weight"],
                n_jobs=-1,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=params["random_forest"]["n_estimators"],
                max_depth=params["random_forest"]["max_depth"],
                class_weight=params["random_forest"]["class_weight"],
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "xgboost": XGBClassifier(
                n_estimators=params["xgboost"]["n_estimators"],
                max_depth=params["xgboost"]["max_depth"],
                learning_rate=params["xgboost"]["learning_rate"],
                scale_pos_weight=params["xgboost"].get("scale_pos_weight", pos_weight),
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=params["lightgbm"]["n_estimators"],
                num_leaves=params["lightgbm"]["num_leaves"],
                learning_rate=params["lightgbm"]["learning_rate"],
                is_unbalance=params["lightgbm"]["is_unbalance"],
                sub_sample=params["lightgbm"]["sub_sample"],
                colsample_bytree=params["lightgbm"]["colsample_bytree"],
                random_state=self.random_state,
            ),
        }

    # ------------------------------------------------------------
    # 2. Build full sklearn pipeline (preprocess + model)
    # ------------------------------------------------------------
    def build_pipeline(self, model):
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transformer, self.num_features),
            ("cat", cat_transformer, self.cat_features),
        ])

        return Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

    # ------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------
    @staticmethod
    def compute_metrics(y_true, y_proba, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "f2": fbeta_score(y_true, y_pred, beta=2),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
        }

    # ------------------------------------------------------------
    # 4. Train & log baselines
    # ------------------------------------------------------------
    def train_and_log_baselines(self, X_train, y_train, X_val, y_val):
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        mlflow.set_experiment(self.experiment_name)

        fraud_rate = y_train.mean()
        pos_weight = (1 - fraud_rate) / fraud_rate
        logger.info(f"Estimated scale_pos_weight: {pos_weight:.2f}")

        models = self.get_models(pos_weight)
        results = {}

        best_name = None
        best_metric = -1
        best_pipeline = None

        for name, model in models.items():
            with mlflow.start_run(run_name=f"baseline_{name}") as run:
                logger.info(f"Training baseline: {name}")

                pipeline = self.build_pipeline(model)
                pipeline.fit(X_train, y_train)

                y_val_proba = pipeline.predict_proba(X_val)[:, 1]
                metrics = self.compute_metrics(y_val, y_val_proba)

                # Log MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

                # ---- Print metrics in terminal ----
                print("\n==============================")
                print(f"Model: {name}")
                for k, v in metrics.items():
                    print(f"{k:10s}: {v:.5f}")
                print("==============================\n")

                results[name] = {
                    "metrics": metrics,
                    "run_id": run.info.run_id,
                }

                if metrics["pr_auc"] > best_metric:
                    best_metric = metrics["pr_auc"]
                    best_name = name
                    best_pipeline = pipeline

        # Save best pipeline
        os.makedirs("models", exist_ok=True)
        best_path = "models/best_baseline_pipeline.joblib"
        joblib.dump(best_pipeline, best_path)

        logger.info(f"Best baseline: {best_name} (PR-AUC={best_metric:.4f})")
        logger.info(f"Saved best pipeline to {best_path}")

        return results, {
            "best_name": best_name,
            "best_metric": best_metric,
            "path": best_path,
        }


# ============================================================
# WRAPPER FUNCTION FOR PREFECT
# ============================================================
def train_baseline_models(processed_path: str):
    """
    Wrapper function for Prefect task.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(processed_path)
    trainer = FraudModelTrainer()
    results, best_info = trainer.train_and_log_baselines(X_train, y_train, X_val, y_val)
    return best_info


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        PROCESSED_DIR / "data_processed.joblib"
    )

    trainer = FraudModelTrainer()
    results, best_info = trainer.train_and_log_baselines(
        X_train, y_train, X_val, y_val
    )

    print("Best baseline:", best_info)