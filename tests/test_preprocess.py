import pandas as pd
import numpy as np
from src.utils.preprocessing.preprocess import Preprocessor

def test_clean_data_removes_duplicates():
    prep = Preprocessor()
    data = {"step": [1, 1, 2], "amount": [100.0, 100.0, 200.0], "isFraud": [0, 0, 1]}
    df = pd.DataFrame(data)
    cleaned_df = prep.clean_data(df)
    assert len(cleaned_df) == 2

def test_optimize_dtypes_reduces_precision():
    prep = Preprocessor()
    data = {"amount": [100.0, 200.0], "step": [1, 2]}
    df = pd.DataFrame(data)
    optimized_df = prep.optimize_dtypes(df)
    assert optimized_df["amount"].dtype == "float32"

def test_run_produces_correct_splits():
    prep = Preprocessor()
    data = {
        "step": list(range(100)), "amount": [100.0] * 100, "isFraud": [0, 1] * 50,
        "type": ["TRANSFER"] * 100, "category": ["A"] * 100, "nameOrig": ["O"] * 100,
        "nameDest": ["D"] * 100, "oldbalanceOrg": [100.0] * 100, "newbalanceOrig": [100.0] * 100,
        "oldbalanceDest": [100.0] * 100, "newbalanceDest": [100.0] * 100,
        "isMoneyLaundering": [0] * 100, "laundering_typology": ["None"] * 100,
        "metadata": ["None"] * 100, 
        "fraud_probability": [0.5] * 100, 
        "hour": [12] * 100, "day_of_week": [1] * 100, "day_of_month": [1] * 100, "month": [1] * 100
    }
    df = pd.DataFrame(data)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.run(df)
    assert len(X_train) == 70
    assert "isFraud" not in X_train.columns