# run_retrain.py
from flows.fraud_flows import auto_retrain_flow
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Starting auto-retrain check...")
    result = auto_retrain_flow()
    
    if result.get("retrained"):
        print(f"Model retrained successfully. Evaluation: {result.get('test_eval')}")
    else:
        print("No drift detected, model unchanged")