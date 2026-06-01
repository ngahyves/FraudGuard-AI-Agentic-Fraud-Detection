# tests/test_env_vars.py
import pytest
from pathlib import Path
from src.config.env_vars import settings

def test_settings_loaded():
    """Essential variables."""
    assert settings.PROJECT_NAME == "Fraud AI Agent"
    assert settings.ENV in ["development", "production", "testing"]

def test_api_keys_present():
    """Verify the presence of API keys."""
    assert settings.GROQ_API_KEY is not None
    assert settings.LANGSMITH_API_KEY is not None

def test_paths_are_absolute():
    """Verify if the paths can be resolved to absolute paths."""
    assert settings.RAW_DATA_PATH.resolve().is_absolute()
    assert settings.PROCESSED_DIR.resolve().is_absolute()
    