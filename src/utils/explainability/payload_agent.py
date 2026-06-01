#src/utils/explainability/payload_agent.py

import joblib
import pandas as pd
import shap
from pathlib import Path
from src.config.params_loader import load_params
from src.config.logging_config import get_logger

logger = get_logger("payload_agent")
cfg = load_params()

def build_agent_payload(transaction_id, proba, top_factors):
    explanation_text = (
        f"The transaction {transaction_id} poses a risk of fraud {proba:.3f}. "
        f"The main contributing factors are : "
        + ", ".join([f"{f['feature']} (impact {f['impact']})" for f in top_factors])
        + "."
    )

    return {
        "transaction_id": int(transaction_id),
        "fraud_probability": float(proba),
        "top_factors": top_factors,
        "explanation": explanation_text
    }
