# src/api/predict.py

import pandas as pd
import joblib
from fastapi import APIRouter
from api.schemas import AgentRequest
from src.utils.explainability.shap_local import LocalExplainer

router = APIRouter()
MODEL = joblib.load("models/best_xgboost_tuned.joblib")
EXPLAINER = LocalExplainer()

@router.post("/predict")
async def predict(payload: AgentRequest):

    df = pd.DataFrame([payload.transaction.dict()])

    X = MODEL.named_steps["preprocess"].transform(df)
    prob = MODEL.named_steps["model"].predict_proba(X)[0][1]

    shap_row = pd.DataFrame(X, columns=EXPLAINER.feature_names)
    shap_values = EXPLAINER.explainer.shap_values(shap_row)[0]

    contributions = sorted(
        zip(EXPLAINER.feature_names, shap_row.values[0], shap_values),
        key=lambda x: abs(x[2]),
        reverse=True
    )

    top_factors = [
        {"feature": f, "value": float(v), "impact": float(round(s, 4))}
        for f, v, s in contributions[:5]
    ]

    return {
        "fraud_probability": float(prob),
        "top_factors": top_factors
    }
