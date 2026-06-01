# src/utils/ingestion/ingest.py
import hashlib
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.logging_config import get_logger
from src.config.params_loader import PARAMS
from src.config.env_vars import PROCESSED_DIR
from src.utils.errors import DataIngestionError, DataValidationError
from src.utils.validation.validate import DataValidator

logger = get_logger(__name__)


class DataIngestor:
    """
    Ingests large CSV file by chunks and saves to Parquet.
    Never returns the full DataFrame to save memory.
    """

    def __init__(self, file_path: Path, chunksize: int = 20000, validate: bool = True):
        self.file_path = file_path
        self.chunksize = chunksize
        self.validate = validate
        self.validator = DataValidator() if validate else None

    def _file_exists(self) -> bool:
        return self.file_path.exists()

    def _file_not_empty(self) -> bool:
        return self.file_path.stat().st_size > 0

    def _compute_sha256(self) -> str:
        sha256 = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _verify_integrity(self) -> bool:
        expected_hash = PARAMS.get("expected_hash")
        if not expected_hash:
            logger.warning("No expected hash in params.yml, skipping integrity check")
            return True
        current_hash = self._compute_sha256()
        if current_hash != expected_hash:
            logger.error(f"Hash mismatch: expected {expected_hash}, got {current_hash}")
            return False
        logger.info("Integrity check passed")
        return True

    def _validate_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.validator.validate(chunk)
        except DataValidationError as e:
            raise DataIngestionError(f"Chunk validation failed: {e}")

    def run(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        if output_path is None:
            output_path = PROCESSED_DIR / "raw_data.parquet"

        start_time = time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not self._file_exists():
                raise DataIngestionError(f"File not found: {self.file_path}")
            if not self._file_not_empty():
                raise DataIngestionError(f"File is empty: {self.file_path}")
            if not self._verify_integrity():
                raise DataIngestionError("File integrity check failed")

            logger.info(f"Reading {self.file_path} in chunks of {self.chunksize} rows")

            chunk_iter = pd.read_csv(self.file_path, chunksize=self.chunksize)
            all_chunks = []

            for i, chunk in enumerate(chunk_iter):
                logger.debug(f"Processing chunk {i+1} ({len(chunk)} rows)")
                if self.validate and self.validator:
                    chunk = self._validate_chunk(chunk)
                all_chunks.append(chunk)

            if not all_chunks:
                raise DataIngestionError("No data read from file")

            # Concatenate all chunks
            df_final = pd.concat(all_chunks, ignore_index=True)
            total_rows = len(df_final)

            # Save in parquet
            df_final.to_parquet(output_path, engine='pyarrow', index=False)

            duration = time.time() - start_time
            logger.info(f"Ingestion finished: {total_rows} rows in {duration:.2f}s")
            logger.info(f"Data saved to {output_path}")

            return {
                "status": "success",
                "output_file": str(output_path),
                "total_rows": total_rows,
                "duration_sec": round(duration, 2)
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"status": "failed", "error": str(e)}
        
# Test
if __name__ == "__main__":
    from src.config.env_vars import RAW_DATA_PATH

    ingestor = DataIngestor(RAW_DATA_PATH)
    result = ingestor.run()

    if result["status"] == "success":
        print(f"Ingestion OK: {result['output_file']} ({result['total_rows']} rows)")
    else:
        print(f"Ingestion failed: {result['error']}")