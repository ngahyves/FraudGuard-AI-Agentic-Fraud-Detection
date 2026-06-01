"""
Prefect flows for FraudGuard pipeline:
- Data ingestion
- Preprocessing
- Baseline training
- Hyperparameter tuning
- Test evaluation
- Drift monitoring
- Auto-retraining
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.params_loader import load_params
from src.config.env_vars import RAW_DATA_PATH, PROCESSED_DIR
from src.utils.ingestion.ingest import DataIngestor
from src.utils.preprocessing.preprocess import Preprocessor

# Training modules (with wrapper functions)
from src.utils.training.train_baseline import train_baseline_models
from src.utils.training.tune_best_model import tune_best_model
from src.utils.training.evaluate_test import evaluate_test_set

# Drift monitor
from monitoring.drift_monitor import DriftMonitor


# ============================================================
# TASKS
# ============================================================

@task
def load_cfg_task():
    """Load configuration from params.yml"""
    logger = get_run_logger()
    logger.info("Loading configuration from params.yml")
    cfg = load_params()
    logger.info("Configuration loaded successfully")
    return cfg


@task
def ingest_data_task():
    """
    Ingest raw data from CSV or use existing Parquet file.
    Skips re-ingestion if Parquet already exists to save time and memory.
    """
    logger = get_run_logger()
    logger.info("Starting data ingestion")
    logger.info(f"Raw data path: {RAW_DATA_PATH}")

    ingestor = DataIngestor(RAW_DATA_PATH)
    result = ingestor.run()

    if result["status"] != "success":
        logger.error(f"Ingestion failed: {result.get('error')}")
        raise RuntimeError(f"Ingestion failed: {result.get('error')}")

    parquet_path = PROCESSED_DIR / "raw_data.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ingestion complete: {result['total_rows']:,} rows")
    logger.info(f"Parquet file saved to: {parquet_path}")

    return {"path": str(parquet_path), "rows": result["total_rows"]}


@task
def preprocess_data_task(parquet_path: str):
    """
    Preprocess data using a pre-created sample to avoid memory issues.
    The sample should be created once using: 
    pd.read_parquet('data/processed/raw_data.parquet').head(300000).to_parquet('data/processed/sample.parquet')
    """
    logger = get_run_logger()
    logger.info("Starting preprocessing")
    
    # Use pre-created sample file to avoid loading entire dataset
    sample_path = PROCESSED_DIR / "sample.parquet"
    
    if not sample_path.exists():
        logger.error(f"Sample file not found: {sample_path}")
        logger.error("Please create sample file with: pd.read_parquet('data/processed/raw_data.parquet').head(300000).to_parquet('data/processed/sample.parquet')")
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    df = pd.read_parquet(sample_path)
    logger.info(f"Loaded {len(df):,} rows from sample file")
    
    preprocessor = Preprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run(df)
    
    logger.info(f"Preprocessing results:")
    logger.info(f"  - X_train shape: {X_train.shape}")
    logger.info(f"  - X_val shape: {X_val.shape}")
    logger.info(f"  - X_test shape: {X_test.shape}")

    processed_path = PROCESSED_DIR / "data_processed.joblib"
    joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test), processed_path)
    logger.info(f"Preprocessed data saved to: {processed_path}")

    return {"processed_path": str(processed_path)}


@task
def train_baseline_task(processed_path: str):
    """Train baseline models for comparison"""
    logger = get_run_logger()
    logger.info("Starting baseline model training")
    
    result = train_baseline_models(processed_path)
    
    logger.info("Baseline training completed")
    return result


@task
def tune_best_model_task(processed_path: str):
    """Hyperparameter tuning on the best performing model"""
    logger = get_run_logger()
    logger.info("Starting hyperparameter tuning")
    
    result = tune_best_model(processed_path)
    
    logger.info("Hyperparameter tuning completed")
    return result


@task
def evaluate_test_task(processed_path: str):
    """Evaluate final model on test set"""
    logger = get_run_logger()
    logger.info("Starting test set evaluation")
    
    result = evaluate_test_set(processed_path)
    
    logger.info("Test evaluation completed")
    return result


@task
def drift_monitor_task(cfg):
    """Run data drift monitoring"""
    logger = get_run_logger()
    logger.info("Starting drift monitoring")
    
    monitor = DriftMonitor(cfg)
    result = monitor.run_drift_analysis()
    
    logger.info("Drift monitoring completed")
    return result


# ============================================================
# FLOWS
# ============================================================

@flow(name="Fraud Full Pipeline")
def full_pipeline():
    """
    Main orchestration flow for the complete FraudGuard pipeline:
    1. Ingest data (CSV -> Parquet, skip if already exists)
    2. Preprocess using sample file
    3. Train baseline models
    4. Hyperparameter tuning
    5. Test set evaluation
    6. Drift monitoring
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("STARTING FRAUD FULL PIPELINE")
    logger.info("=" * 60)
    
    cfg = load_cfg_task()
    logger.info(f"Project: {cfg.get('project', {}).get('name', 'Fraud AI Agent')}")
    logger.info(f"Version: {cfg.get('project', {}).get('version', '1.0.0')}")

    logger.info("Step 1/6: Data Ingestion")
    ingest = ingest_data_task()
    
    logger.info("Step 2/6: Preprocessing")
    preprocess = preprocess_data_task(ingest["path"])

    logger.info("Step 3/6: Baseline Training")
    baseline = train_baseline_task(preprocess["processed_path"])
    
    logger.info("Step 4/6: Hyperparameter Tuning")
    tuned = tune_best_model_task(preprocess["processed_path"])
    
    logger.info("Step 5/6: Test Evaluation")
    test_eval = evaluate_test_task(preprocess["processed_path"])

    logger.info("Step 6/6: Drift Monitoring")
    drift = drift_monitor_task(cfg)

    logger.info("=" * 60)
    logger.info("FRAUD FULL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return {
        "baseline": baseline,
        "tuning": tuned,
        "test_eval": test_eval,
        "drift": drift,
    }


@flow(name="Auto Retrain on Drift")
def auto_retrain_flow():
    """
    Automatic retraining flow triggered when data drift is detected.
    Only runs retraining pipeline if drift is found.
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("STARTING AUTO RETRAIN PIPELINE")
    logger.info("=" * 60)
    
    cfg = load_cfg_task()
    drift = drift_monitor_task(cfg)

    if drift:
        logger.warning("Drift detected - starting retraining pipeline")
        processed_path = str(PROCESSED_DIR / "data_processed.joblib")
        baseline = train_baseline_task(processed_path)
        tuned = tune_best_model_task(processed_path)
        test_eval = evaluate_test_task(processed_path)
        
        logger.info("Auto retrain completed")
        return {"retrained": True, "test_eval": test_eval}

    logger.info("No drift detected - model is still valid")
    return {"retrained": False}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    full_pipeline()