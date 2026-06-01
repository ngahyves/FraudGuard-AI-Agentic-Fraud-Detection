# src/utils/preprocessing/preprocess.py
"""
Best-practice preprocessing for AMLnet:
- Clean + optimize memory
- Feature engineering
- Stratified Train/Val/Test split (70/15/15)
- Save RAW splits (no sklearn transforms)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config.logging_config import get_logger
from src.config.params_loader import PARAMS
from src.config.env_vars import PROCESSED_DIR

logger = get_logger(__name__)


class Preprocessor:
    """
    Handles preprocessing WITHOUT sklearn transforms.
    (Imputation, scaling, encoding will be done inside the training Pipeline)
    """

    def __init__(self):
        self.target_col = PARAMS["data"]["target_column"]
        self.num_features = PARAMS["features"]["numerical"]
        self.cat_features = PARAMS["features"]["categorical"]
        self.exclude_cols = PARAMS["features"].get("exclude", [])

        # Split parameters
        self.first_split_test_size = 0.3   # 30% temp
        self.second_split_test_size = 0.5  # 50% of temp = 15% test
        self.random_state = PARAMS["data"]["random_state"]

        self.processed_dir = PROCESSED_DIR

    # ----------------------------------------------------------------------
    # Cleaning & memory optimization
    # ----------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        return df

    def check_missing(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logger.info("No missing values found")

    def _log_memory(self, df: pd.DataFrame, stage: str):
        mem = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory ({stage}): {mem:.2f} MB")

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_memory(df, "before optimization")

        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype("float32")
            elif df[col].dtype == "int64":
                df[col] = df[col].astype("int32")

        self._log_memory(df, "after optimization")
        return df

    # ----------------------------------------------------------------------
    # Main pipeline (NO sklearn transforms)
    # ----------------------------------------------------------------------
    def run(self, df: pd.DataFrame):
        logger.info("Starting AMLnet preprocessing (best practice)")

        # 1. Missing values
        self.check_missing(df)

        # 2. Clean + optimize
        df = self.clean_data(df)
        df = self.optimize_dtypes(df)

        # 3. Split X / y
        X = df.drop(columns=[self.target_col] + self.exclude_cols)
        y = df[self.target_col]

        logger.info(f"Features shape: {X.shape}, target shape: {y.shape}")

        # 4. Stratified split 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.first_split_test_size,
            stratify=y,
            random_state=self.random_state,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.second_split_test_size,
            stratify=y_temp,
            random_state=self.random_state,
        )

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # 5. Save RAW splits
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.processed_dir / "data_processed.joblib"

        joblib.dump(
            (X_train, X_val, X_test, y_train, y_val, y_test),
            out_path
        )

        logger.info(f"RAW splits saved to {out_path}")

        return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils.ingestion.ingest import DataIngestor
    from src.config.env_vars import RAW_DATA_PATH

    ingestor = DataIngestor(RAW_DATA_PATH)
    result = ingestor.run()

    if result["status"] == "success":
        df = pd.read_parquet(PROCESSED_DIR / "raw_data.parquet")
        logger.info(f"Loaded Parquet: {df.shape}")

        prep = Preprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = prep.run(df)

        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)
    else:
        print("Ingestion failed:", result["error"])
