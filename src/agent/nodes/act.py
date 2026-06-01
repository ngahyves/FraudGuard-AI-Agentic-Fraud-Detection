# src/agent/nodes/act.py

from datetime import datetime
from src.agent.state import AgentState
from langsmith import traceable

@traceable(name="Act_Node")
def act_node(state: AgentState) -> AgentState:
    """
    Action node: Generates an audit log based on the decision and explanation.
    This log can be stored, sent, or displayed via the API.
    """

    decision = state.decision
    explanation = state.explanation
    transaction_id = state.transaction_id

    timestamp = datetime.utcnow().isoformat()

    audit_message = (
        f"[{timestamp}] Transaction {transaction_id} → DECISION: {decision}\n"
        f"Reason: {explanation}\n"
    )

    state.audit_log = audit_message
    return state
