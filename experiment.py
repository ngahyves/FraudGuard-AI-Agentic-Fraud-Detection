from pathlib import Path
from src.config.params_loader import load_params

cfg = load_params()
ref_path = Path(cfg["paths"]["reference_data"])
curr_path = Path(cfg["paths"]["current_data"])

print(f"Reference path: {ref_path.absolute()}")
print(f"Exists: {ref_path.exists()}")
print(f"Current path: {curr_path.absolute()}")
print(f"Exists: {curr_path.exists()}")