# src/utils/training/tune_best_model.py
import os
import joblib
import optuna
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier

from src.config.params_loader import load_params
from src.config.env_vars import PROCESSED_DIR
from src.config.logging_config import get_logger

# Initialize logger and load parameters
logger = get_logger(__name__)
cfg = load_params()

class OptunaOptimizer:
    """
    Handles hyperparameter optimization using Optuna and logs experiments to MLflow.
    """

    def __init__(self):
        # Load feature and tuning configurations from params.yml
        self.num_features = cfg["features"]["numerical"]
        self.cat_features = cfg["features"]["categorical"]
        self.random_state = cfg["data"]["random_state"]
        self.n_trials = cfg["tuning"]["n_trials"]
        self.timeout = cfg["tuning"]["timeout"]
        self.scale_pos_weight = cfg["model_params"]["xgboost"]["scale_pos_weight"]

        # MLflow Setup: Priority given to environment variable for Docker networking
        # Inside Docker, this should be http://mlflow:5000
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow Tracking URI set to: {tracking_uri}")

    def _build_pipeline(self, model):
        """
        Constructs a Scikit-Learn Pipeline combining preprocessing and the model.
        """
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

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optimization objective function for Optuna.
        Defines the hyperparameter search space.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": self.scale_pos_weight,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": self.random_state,
        }

        # Build and train the pipeline
        model = XGBClassifier(**params)
        pipeline = self._build_pipeline(model)
        pipeline.fit(X_train, y_train)

        # Evaluate using Average Precision (PR-AUC) - best for imbalanced data
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)

        # Log individual trial results to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("pr_auc", pr_auc)

        return pr_auc

    def optimize(self, processed_path: str):
        """
        Main optimization logic: samples data, runs trials, and saves the best champion model.
        """
        logger.info("Loading processed data for Optuna optimization...")
        X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(processed_path)

        # --- DATA SAMPLING FOR PERFORMANCE ---
        # Tuning on 100k rows is enough to find trends and prevents RAM crashes in WSL2/Docker
        SAMPLE_SIZE = 100000
        if len(X_train) > SAMPLE_SIZE:
            logger.info(f"Sampling training set to {SAMPLE_SIZE} rows for faster iteration.")
            X_train = X_train.sample(n=SAMPLE_SIZE, random_state=self.random_state)
            y_train = y_train.loc[X_train.index]

        mlflow.set_experiment("xgboost_optuna_tuning")
        study = optuna.create_study(direction="maximize")

        with mlflow.start_run(run_name="optuna_study"):
            logger.info(f"Starting {self.n_trials} trials...")
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                timeout=self.timeout,
            )

            # Retrieve best trial results
            best_params = study.best_params
            best_pr_auc = study.best_value

            logger.info(f"Best PR-AUC achieved: {best_pr_auc:.4f}")
            mlflow.log_params(best_params)
            mlflow.log_metric("best_pr_auc", best_pr_auc)

            # Re-train the final champion model with best parameters
            best_model = XGBClassifier(
                **best_params,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.random_state,
            )

            logger.info("Training final champion model...")
            best_pipeline = self._build_pipeline(best_model)
            best_pipeline.fit(X_train, y_train)

            # Persistence: Save to local models/ directory and MLflow Artifacts
            Path("models").mkdir(exist_ok=True)
            model_save_path = "models/best_xgboost_tuned.joblib"
            joblib.dump(best_pipeline, model_save_path)
            mlflow.sklearn.log_model(best_pipeline, "best_model")

            # Register the model in the MLflow Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
            mlflow.register_model(model_uri=model_uri, name="fraud_detection_xgboost")

            logger.info(f"Champion model saved at: {model_save_path}")
            return {
                "best_params": best_params,
                "best_pr_auc": best_pr_auc,
                "model_path": model_save_path
            }

# Helper function for orchestration (e.g., via Prefect)
def tune_best_model(processed_path: str):
    """Wrapper function to instantiate and run the optimizer."""
    optimizer = OptunaOptimizer()
    return optimizer.optimize(processed_path)

if __name__ == "__main__":
    # Execute if the script is run directly
    data_path = PROCESSED_DIR / "data_processed.joblib"
    result = tune_best_model(data_path)
    print(result)