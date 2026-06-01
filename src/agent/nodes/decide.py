# src/agent/nodes/decide.py

from src.agent.state import AgentState
from src.agent.agent_config import THRESHOLD_APPROVE, THRESHOLD_REJECT
from langsmith import traceable

@traceable(name="Decide_Node")
def decide_node(state: AgentState) -> AgentState:
    """
    Decision node: applies business thresholds to determine
    whether the transaction should be approved, rejected, or escalated.
    """

    prob = state.fraud_probability

    if prob < THRESHOLD_APPROVE:
        decision = "APPROVE"
    elif prob > THRESHOLD_REJECT:
        decision = "REJECT"
    else:
        decision = "ESCALATE"

    state.decision = decision
    return state
