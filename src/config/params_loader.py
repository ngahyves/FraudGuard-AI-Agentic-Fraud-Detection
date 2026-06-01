# src/config/params_loader.py
import yaml
from pathlib import Path
from src.config.env_vars import settings

def load_params():
    params_path = settings.BASE_DIR / "params.yml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yml not found at {params_path}")
    
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

PARAMS = load_params()