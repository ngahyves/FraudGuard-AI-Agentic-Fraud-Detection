from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List

import io
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv
from langsmith import Client
from groq import Groq

from src.agent.graph import run_agent
from src.utils.explainability.shap_local import LocalExplainer
from api.db import save_audit_record, get_audit_record

# Importing schemas
from api.schemas import (
    PredictRequest,
    AgentDecisionResponse,
    AnalystQueryRequest,
    AnalystQueryResponse,
)

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
ls_client = Client()

app = FastAPI(title="FraudGuard API", version="1.0.0")
Instrumentator().instrument(app).expose(app)


# ---------- SHAP Consolidation Helper ----------
def get_consolidated_factors(shap_values: list, feature_names: list) -> List[dict]:
    mapping = {"x0": "Transaction Type", "x1": "Category"}
    consolidated = {}

    for val, name in zip(shap_values, feature_names):
        root = name.split('_')[0]
        display_name = mapping.get(root, root)
        consolidated[display_name] = consolidated.get(display_name, 0) + val

    sorted_factors = sorted(consolidated.items(), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": name, "impact": float(round(impact, 4))} for name, impact in sorted_factors[:5]]


# ---------- Load Model + SHAP ----------
MODEL = joblib.load("models/best_xgboost_tuned.joblib")
EXPLAINER = LocalExplainer()


def compute_prediction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([transaction])
    X = MODEL.named_steps["preprocess"].transform(df)
    prob = MODEL.named_steps["model"].predict_proba(X)[0][1]

    shap_row = pd.DataFrame(X, columns=EXPLAINER.feature_names)
    raw_shap_values = EXPLAINER.explainer.shap_values(shap_row)[0]

    top_factors = get_consolidated_factors(raw_shap_values, EXPLAINER.feature_names)

    return {
        "fraud_probability": float(prob),
        "top_factors": top_factors,
        "raw_shap": raw_shap_values.tolist()
    }


# ---------- Endpoints ----------
@app.post("/predict")
async def predict(payload: PredictRequest):
    return compute_prediction(payload.transaction.model_dump())


@app.post("/agent/decide", response_model=AgentDecisionResponse)
async def agent_decide(payload: PredictRequest):
    tx_dict = payload.transaction.model_dump()
    pred = compute_prediction(tx_dict)

    agent_state = run_agent(
        transaction_id=payload.transaction_id,
        transaction=tx_dict,
        fraud_probability=pred["fraud_probability"],
        top_factors=pred["top_factors"]
    )

    audit_id = save_audit_record({
        "transaction_id": payload.transaction_id,
        "transaction": tx_dict,
        "fraud_probability": pred["fraud_probability"],
        "top_factors": pred["top_factors"],
        "explanation": agent_state.get("explanation", "N/A"),
        "decision": agent_state.get("decision", "REVIEW"),
        "audit_log": agent_state.get("audit_log", "Logged"),
    })

    return AgentDecisionResponse(
        fraud_probability=pred["fraud_probability"],
        decision=agent_state["decision"],
        explanation=agent_state["explanation"],
        audit_id=audit_id,
    )


@app.get("/explain/{audit_id}")
async def explain(audit_id: int):
    record = get_audit_record(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found")

    raw_shap = record.get("shap_values", [])
    if not raw_shap:
        transaction = record["transaction_json"]
        df = pd.DataFrame([transaction])
        X = MODEL.named_steps["preprocess"].transform(df)
        raw_shap = EXPLAINER.explainer.shap_values(X)[0]

    clean_data = get_consolidated_factors(raw_shap, EXPLAINER.feature_names)
    names = [d["feature"] for d in clean_data]
    vals = [d["impact"] for d in clean_data]

    plt.figure(figsize=(10, 6))
    plt.barh(names, vals, color='skyblue')
    plt.title(f"Key Risk Factors - Transaction {audit_id}")
    plt.xlabel("Impact on Fraud Score")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/analyst/query", response_model=AnalystQueryResponse)
async def analyst_query(payload: AnalystQueryRequest):
    history = "Recent audit logs summary and transaction trends."
    prompt = f"Context: {history}\nAnalyst Question: {payload.question}"

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return AnalystQueryResponse(answer=completion.choices[0].message.content)


@app.get("/health")
async def health():
    return {"status": "ok"}
