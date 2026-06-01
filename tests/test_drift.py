# tests/test_drift.py
import pytest
from pathlib import Path
from monitoring.drift_monitor import DriftMonitor
from src.config.params_loader import load_params
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def test_drift_monitor():
    cfg = load_params()
    monitor = DriftMonitor(cfg)

    # Verify the paths
    ref_path = Path(cfg["paths"]["reference_data"])
    curr_path = Path(cfg["paths"]["current_data"])

    if not ref_path.exists():
        logger.warning(f"Reference data not found: {ref_path}")
        pytest.skip(f"Reference data missing: {ref_path}")
    if not curr_path.exists():
        logger.warning(f"Current data not found: {curr_path}")
        pytest.skip(f"Current data missing: {curr_path}")

    # Run analysis
    drift = monitor.run_drift_analysis()

    print(f"Drift detected: {drift}")
    print(f"Report generated at: {cfg['paths']['drift_report']}")

    # Type verification
    assert isinstance(drift, bool)


if __name__ == "__main__":
    test_drift_monitor()