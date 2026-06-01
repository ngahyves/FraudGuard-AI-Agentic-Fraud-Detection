# src/agent/graph.py

from langgraph.graph import StateGraph
from langsmith import traceable
from src.agent.state import AgentState
from src.agent.nodes.analyze import analyze_node
from src.agent.nodes.explain import explain_node
from src.agent.nodes.decide import decide_node
from src.agent.nodes.act import act_node


# -----------------------------
# Build the FraudGuard workflow
# -----------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("act", act_node)

    # Define workflow transitions
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "explain")
    workflow.add_edge("explain", "decide")
    workflow.add_edge("decide", "act")

    # Compile graph with LangSmith tracing enabled
    graph = workflow.compile(
        debug=True,              # Required for LangSmith graph visualization
        name="FraudGuardGraph"   # Appears in LangSmith
    )

    return graph

fraudguard_graph = build_graph()
# -----------------------------
# Run agent with LangSmith trace
# -----------------------------

@traceable(name="FraudGuard_Run", run_type="chain")
def run_agent(transaction_id: int, transaction: dict, fraud_probability: float, top_factors: list):

    # Results calculated by predict endpoint
    initial_state = {
        "transaction_id": transaction_id,
        "transaction": transaction,
        "fraud_probability": fraud_probability, # Received from API
        "top_factors": top_factors,             # Received from API
        "explanation": None,
        "decision": None,
        "audit_log": None,
    }

    # Execute the graph
    final_state_obj = fraudguard_graph.invoke(initial_state)

    # Convert Pydantic object into dictionary for API and DB
    if hasattr(final_state_obj, "model_dump"):
        return final_state_obj.model_dump()
    return dict(final_state_obj)
