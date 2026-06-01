# monitoring/drift_monitor.py

import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_drift_analysis(self):
        logger.info("Starting drift analysis")

        # Loading data
        ref_path = Path(self.cfg["paths"]["reference_data"])
        curr_path = Path(self.cfg["paths"]["current_data"])

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found: {ref_path}")
        if not curr_path.exists():
            raise FileNotFoundError(f"Current data not found: {curr_path}")

        reference = pd.read_parquet(ref_path)
        current = pd.read_parquet(curr_path)

        logger.info(f"Reference shape: {reference.shape}, Current shape: {current.shape}")

        # Create and execute the report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        # Save the html report
        report_path = Path(self.cfg["paths"]["drift_report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))
        logger.info(f"Drift report saved to {report_path}")

        # Extract the result
        dataset_drift = False
        try:

            result = report.as_dict()
            if 'metrics' in result and len(result['metrics']) > 0:
                dataset_drift = result['metrics'][0]['result'].get('dataset_drift', False)
        except Exception as e:
            logger.warning(f"Could not extract drift from report: {e}")
            # Save the default value

        logger.info(f"Dataset drift detected: {dataset_drift}")
        return dataset_drift


if __name__ == "__main__":
    from src.config.params_loader import load_params

    cfg = load_params()
    monitor = DriftMonitor(cfg)
    drift = monitor.run_drift_analysis()
    print(f"Drift detected: {drift}")