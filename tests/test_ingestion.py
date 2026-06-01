import pytest
import pandas as pd
from pathlib import Path
from src.utils.ingestion.ingest import DataIngestor
import pytest

@pytest.fixture
def sample_csv(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 200.0],
        "category": ["A", "B"],
        "nameOrig": ["O1", "O2"],
        "nameDest": ["D1", "D2"],
        "oldbalanceOrg": [1000.0, 2000.0],
        "newbalanceOrig": [900.0, 1800.0],
        "isFraud": [0, 1],
        "isMoneyLaundering": [0, 0],
        "laundering_typology": ["None", "None"],
        "metadata": ["None", "None"],
        "fraud_probability": [0.0, 0.0],
        "hour": [10, 15],
        "day_of_week": [1, 3],
        "day_of_month": [15, 20],
        "month": [5, 8]
    })
    df.to_csv(file_path, index=False)
    return file_path

def test_ingestion_success(sample_csv, tmp_path):
    output_parquet = tmp_path / "output.parquet"
    ingestor = DataIngestor(sample_csv, chunksize=10, validate=True)
    ingestor._verify_integrity = lambda: True # Mock hash
    result = ingestor.run(output_path=output_parquet)
    
    assert result["status"] == "success", f"Real error : {result.get('error')}"
    
    # Verify the content
    df_read = pd.read_parquet(output_parquet)
    assert len(df_read) == 2
    assert "amount" in df_read.columns

def test_ingestion_file_not_found(tmp_path):
    """Verify the data source"""
    fake_path = tmp_path / "non_existent.csv"
    ingestor = DataIngestor(fake_path)
    
    result = ingestor.run()
    
    assert result["status"] == "failed"
    assert "File not found" in result["error"]

def test_ingestion_invalid_data(tmp_path):
    """Verify if the ingestion will fail if the data contaract is not respected"""
    file_path = tmp_path / "invalid_data.csv"
    # Negative amount (should be rejected)
    df = pd.DataFrame({"step": [1], "amount": [-100.0]}) 
    df.to_csv(file_path, index=False)
    
    ingestor = DataIngestor(file_path, validate=True)
    ingestor._verify_integrity = lambda: True
    
    result = ingestor.run()
    
    assert result["status"] == "failed"
    assert "validation failed" in result["error"]