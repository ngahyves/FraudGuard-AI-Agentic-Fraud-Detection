# src/config/logging_config.py

#Importing libraries
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from src.config.env_vars import settings

#Create a function to save the logs
def get_logger(name:str="fraud_ai_agent"):

    """Defining the log's paths"""
    BASE_DIR=Path(__file__).resolve().parent.parent.parent
    LOG_DIR=BASE_DIR/ "logs"
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"

    """Setting up the logger"""
    logger=logging.getLogger(name)
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.DEBUG) #getattr("object", "name_attribut", default_value)
    logger.setLevel(level)

    """Avoiding duplicates logs files"""
    if logger.hasHandlers():
        logger.handlers.clear()
    
    """Format of the logs"""
    formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
    # Console handler (UTF-8 safe)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

logger = get_logger()



