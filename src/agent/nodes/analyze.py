# src/agent/nodes/analyze.py

import joblib
import pandas as pd
from src.agent.state import AgentState
from src.utils.explainability.shap_local import LocalExplainer
from langsmith import traceable

#  Lazy Loading of the model
MODEL = None
EXPLAINER = None

@traceable(name="Analyze_Node")
def analyze_node(state: AgentState) -> AgentState:
    """
    Node analyse : use existing data or compute if don't exist
    """
    
    # if we have data we go to optimization
    if state.fraud_probability is not None and state.top_factors is not None:
        print("--- AnalyzeNode: Using API scores ---")
        return state

    print("--- AnalyzeNode: compute scores---")
    global MODEL, EXPLAINER
    if MODEL is None:
        MODEL = joblib.load("models/best_xgboost_tuned.joblib")
        EXPLAINER = LocalExplainer()

    transaction = state.transaction
    df = pd.DataFrame([transaction])

    # Preprocessing + prediction
    X = MODEL.named_steps["preprocess"].transform(df)
    prob = MODEL.named_steps["model"].predict_proba(X)[0][1]

    # SHAP local
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

    state.fraud_probability = float(prob)
    state.top_factors = top_factors

    return state