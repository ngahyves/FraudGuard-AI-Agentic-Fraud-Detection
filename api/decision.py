# src/api/decision.py

from fastapi import APIRouter
from api.schemas import AgentRequest, AgentResponse
from src.agent.graph import run_agent

router = APIRouter()

@router.post("/decision", response_model=AgentResponse)
async def decision(payload: AgentRequest):

    result = run_agent(
        transaction_id=payload.transaction_id,
        transaction=payload.transaction.dict()
    )

    return AgentResponse(
        fraud_probability=result["fraud_probability"],
        decision=result["decision"],
        explanation=result["explanation"],
        audit_log=result["audit_log"]
    )
