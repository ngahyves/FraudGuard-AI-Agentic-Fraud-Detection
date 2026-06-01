import joblib
import pandas as pd
import os

# Load processed file
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
    "data/processed/data_processed.joblib"
)

# Create data folder if it does not exist
os.makedirs("data", exist_ok=True)

# Save data in parquet format
X_train.to_parquet("data/reference.parquet")
X_test.to_parquet("data/current.parquet")

print("reference.parquet and current.parquet created.")
