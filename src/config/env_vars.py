#src/config/env_vars.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Root path
ROOT = Path(__file__).resolve().parent.parent.parent

# loading env file
load_dotenv(ROOT / ".env")

class Settings:
    # --- PATHS ---
    BASE_DIR = ROOT
    RAW_DATA_PATH = ROOT / "data" / "raw" / "amlnet.csv"
    PROCESSED_DIR = ROOT / "data" / "processed"
    MODELS_DIR = ROOT / "models"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

    # --- APP SETTINGS ---
    PROJECT_NAME = os.getenv("PROJECT_NAME", "Fraud AI Agent")
    ENV = os.getenv("ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- API KEYS ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "fraud-ai-agent")

    # --- INFRA ---
    DATABASE_URL = os.getenv("DATABASE_URL")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME=os.getenv("MLFLOW_EXPERIMENT_NAME","fraud_detection_experiment")

# Instanciate the class 
settings = Settings()
RAW_DATA_PATH = settings.RAW_DATA_PATH
PROCESSED_DIR = settings.PROCESSED_DIR
MODELS_DIR = settings.MODELS_DIR
PREPROCESSOR_PATH = settings.PREPROCESSOR_PATH
#Keys verifications
missing_keys = []
if settings.GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is missing in .env file")

if settings.LANGSMITH_API_KEY is None:
    missing_keys.append("LANGSMITH_API_KEY")

if missing_keys:
    raise ValueError(
        f"These keys are missing : {', '.join(missing_keys)}"
    )