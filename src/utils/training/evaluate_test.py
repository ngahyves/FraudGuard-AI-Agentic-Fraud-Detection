# src/utils/evaluation/evaluate_test.py
"""
Final evaluation module for fraud detection model.
Computes metrics on test set, finds optimal threshold, and logs everything to MLflow.
"""

import joblib
import numpy as np
import mlflow
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)

from src.config.params_loader import load_params
from src.config.env_vars import PROCESSED_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)
cfg = load_params()


def compute_metrics(y_true, y_proba, threshold):
    """
    Compute classification metrics at a given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def find_best_threshold(y_true, y_proba, n_thresholds=300):
    """
    Find optimal threshold that maximizes F2 score.
    """
    thresholds = np.linspace(0, 1, n_thresholds)

    best_f2 = 0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    return best_t, best_f2


def evaluate():
    """
    Run full evaluation pipeline (for standalone execution).
    """
    logger.info("Starting final evaluation on test set...")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        PROCESSED_DIR / "data_processed.joblib"
    )
    logger.info("Loaded processed dataset.")

    # Load tuned pipeline
    model = joblib.load("models/best_xgboost_tuned.joblib")
    logger.info("Loaded best tuned XGBoost pipeline.")

    # MLflow setup
    mlflow.set_tracking_uri(cfg["experimentation"]["tracking_uri"])
    mlflow.set_experiment("final_evaluation")

    with mlflow.start_run(run_name="test_evaluation"):
        logger.info("MLflow run started: test_evaluation")

        # Predict
        y_test_proba = model.predict_proba(X_test)[:, 1]
        print("y_test shape:", y_test.shape)
        print("y_test unique:", np.unique(y_test))
        print("proba min/max:", np.min(y_test_proba), np.max(y_test_proba))
        print("proba contains NaN:", np.isnan(y_test_proba).any())
        logger.info("Predictions on test set completed.")

        # Find best threshold
        best_t, best_f2 = find_best_threshold(y_test, y_test_proba)
        logger.info(f"Best threshold found: {best_t:.4f} with F2={best_f2:.4f}")

        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("best_f2", best_f2)

        # Compute metrics at best threshold
        metrics = compute_metrics(y_test, y_test_proba, best_t)
        mlflow.log_metrics(metrics)

        logger.info("Final Test Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.5f}")

        # Save threshold
        Path("models").mkdir(exist_ok=True)
        joblib.dump(best_t, "models/best_threshold.joblib")
        logger.info(f"Optimal threshold saved to models/best_threshold.joblib")


# ============================================================
# WRAPPER FUNCTION FOR PREFECT
# ============================================================
def evaluate_test_set(processed_path: str):
    """
    Wrapper function for Prefect task.
    Loads data from processed_path, runs evaluation, returns metrics.
    
    Args:
        processed_path: Path to the processed data file (data_processed.joblib)
    
    Returns:
        dict: Metrics computed on test set at optimal threshold
    """
    logger.info("Starting final evaluation on test set (Prefect task)...")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(processed_path)
    logger.info(f"Data loaded: test set shape {X_test.shape}")

    # Load tuned pipeline
    model = joblib.load("models/best_xgboost_tuned.joblib")
    logger.info("Loaded best tuned XGBoost pipeline.")

    # MLflow setup 
    mlflow.set_tracking_uri(cfg["experimentation"]["tracking_uri"])
    mlflow.set_experiment("final_evaluation")

    with mlflow.start_run(run_name="test_evaluation_prefect"):
        logger.info("MLflow run started: test_evaluation_prefect")

        # Predict
        y_test_proba = model.predict_proba(X_test)[:, 1]
        logger.info("Predictions on test set completed.")

        # Find best threshold
        best_t, best_f2 = find_best_threshold(y_test, y_test_proba)
        logger.info(f"Best threshold found: {best_t:.4f} with F2={best_f2:.4f}")

        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("best_f2", best_f2)

        # Compute metrics at best threshold
        metrics = compute_metrics(y_test, y_test_proba, best_t)
        mlflow.log_metrics(metrics)

        logger.info("Final Test Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.5f}")

        # Save threshold
        Path("models").mkdir(exist_ok=True)
        joblib.dump(best_t, "models/best_threshold.joblib")
        logger.info(f"Optimal threshold saved to models/best_threshold.joblib")

        return metrics


if __name__ == "__main__":
    evaluate()