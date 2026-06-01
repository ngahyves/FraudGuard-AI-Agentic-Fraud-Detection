import pytest
import pandas as pd
from src.utils.validation.validate import DataValidator
from src.utils.errors import DataValidationError

def test_validation_passes():
    """Test simple de validation."""
    validator = DataValidator()
    # Create data that respect the schema
    df = pd.DataFrame({
        "step": [1], "type": ["PAYMENT"], "amount": [100.0],
        "category": ["A"], "nameOrig": ["O1"], "nameDest": ["D1"],
        "oldbalanceOrg": [1000.0], "newbalanceOrig": [900.0],
        "isFraud": [0], "isMoneyLaundering": [0],
        "laundering_typology": ["None"], "metadata": ["None"],
        "fraud_probability": [0.1], "hour": [10],
        "day_of_week": [1], "day_of_month": [15], "month": [5]
    })
    result = validator.validate(df)
    assert len(result) == 1