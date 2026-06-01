# src/agent/state.py

from typing import Optional, List, Dict
from pydantic import BaseModel

class AgentState(BaseModel):
    """
    Represents the complete state passed between nodes in the LangGraph graph.
    Each node reads and modifies this state.
    state → analyze → state → explain → state → decide → state → act → state final

    """

    # Input
    transaction_id: Optional[int] = None
    transaction: Optional[Dict] = None

    # Analyse
    fraud_probability: Optional[float] = None
    top_factors: Optional[List[Dict]] = None  # SHAP local

    # Explanation
    explanation: Optional[str] = None

    # Decision
    decision: Optional[str] = None  # APPROVE / REJECT / ESCALATE

    # Action / Audit
    audit_log: Optional[str] = None
