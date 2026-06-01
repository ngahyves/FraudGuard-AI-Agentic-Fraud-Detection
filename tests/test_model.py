import pytest
import joblib
import pandas as pd
import numpy as np

def test_model_prediction_range():
    """Verify the model give probabilities between 0-1"""
    # Charger le modèle
    model = joblib.load("models/best_xgboost_tuned.joblib")
    
    # Create fake data
    sample_data = pd.DataFrame([{
        "step": 1, "type": "TRANSFER", "amount": 1000.0, 
        "category": "Finance", "hour": 12, "day_of_week": 1,
        "day_of_month": 1, "month": 1
    }])
    
    # Compute predictions
    X = model.named_steps["preprocess"].transform(sample_data)
    prob = model.named_steps["model"].predict_proba(X)[0][1]
    
    assert 0 <= float(prob) <= 1
    assert isinstance(prob, (float, np.floating))