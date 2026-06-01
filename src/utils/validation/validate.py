#src/utils/validation/validate.py

import pandera.pandas as pa
import pandas as pd
from src.config.logging_config import get_logger
from src.utils.errors import DataValidationError
from src.config.env_vars import settings

logger = get_logger(__name__)

# Defining the schema of our 17 columns
class DataValidator:
    def __init__(self):
        self.schema = pa.DataFrameSchema(
            columns={
                "step": pa.Column(int, pa.Check.ge(0)),
                "type": pa.Column(str,nullable=False),
                "amount": pa.Column(float, pa.Check.ge(0)),
                "category": pa.Column(str, nullable=True),
                "nameOrig": pa.Column(str),
                "nameDest": pa.Column(str),
                "oldbalanceOrg": pa.Column(float, pa.Check.ge(0)),
                "newbalanceOrig": pa.Column(float, pa.Check.ge(0)),
                
                # fraud informations
                "isFraud": pa.Column(int, pa.Check.isin([0, 1])),
                "isMoneyLaundering": pa.Column(int, pa.Check.isin([0, 1]), nullable=True),
                "laundering_typology": pa.Column(str, nullable=True),
                
                # Metadata and pre-calculated probabilities
                "metadata": pa.Column(str, nullable=True),
                "fraud_probability": pa.Column(float, nullable=True),
                
                # Temporal features already extracted the dataset
                "hour": pa.Column(int, pa.Check.in_range(0, 23)),
                "day_of_week": pa.Column(int, pa.Check.in_range(0, 6)), # 0=monday, 6=Sunday
                "day_of_month": pa.Column(int, pa.Check.in_range(1, 31)),
                "month": pa.Column(int, pa.Check.in_range(1, 12)), # 1=January, 12=December
            },
            strict=True, #to ensure we have all the columns
            coerce=True  #convert data types if necessary
        )
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validation of the contract...")
        try:
            return self.schema.validate(df, lazy=True)# validate method from pandera to see all the errors
        except pa.errors.SchemaErrors as err:
            logger.error(f"Validation failed :\n{err.failure_cases}")
            # Custom exception
            raise DataValidationError(f"Data contract violation: {err.failure_cases}")

if __name__ == "__main__":
    local_data_path = settings.RAW_DATA_PATH
    try:
        logger.info(f"Reading local file for validation: {local_data_path}")
        df = pd.read_csv(local_data_path)
        validator = DataValidator()
        validator.validate(df)
        logger.info("Validation passed.")
        print("SUCCESS: Data contract validated")
    except Exception as e:
        logger.error(f"Error during validation : {e}")
        print("Failure. Check “logs/pipeline.log” to see which column is causing the problem.")