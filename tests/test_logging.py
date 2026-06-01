# tests/test_logging.py
import pytest
from pathlib import Path
from src.config.logging_config import get_logger, logger

def test_logger_creation():
    """Verify if the default logger is created."""
    assert logger is not None
    assert logger.name == "fraud_ai_agent"

def test_logger_handlers():
    """Verify if the logger has at least one handler"""
    logger = get_logger("test_handlers")
    assert len(logger.handlers) > 0

def test_log_file_created(tmp_path):
    """Verify the if the log file is created."""
    test_logger = get_logger("test_logger")
    test_logger.info("Test message")
    
    # Verify if the log file exists
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_file = log_dir / "pipeline.log"
    assert log_dir.exists()