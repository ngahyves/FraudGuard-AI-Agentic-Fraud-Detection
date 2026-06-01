# src/agent/nodes/explain.py

from langchain_groq import ChatGroq
from src.agent.state import AgentState
from src.agent.agent_config import GROQ_API_KEY, LLM_MODEL
from langsmith import traceable

# Initialize LLM once
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.2,
    max_tokens=512,
)

@traceable(name="Explain_Node")
def explain_node(state: AgentState) -> AgentState:
    """
    Explanation node: Consolidates OHE SHAP factors and generates 
    a professional human-readable explanation using Llama 3.1.
    """

    prob = state.fraud_probability
    raw_factors = state.top_factors or []

    # --- 1. GROUPING & CLEANING LOGIC ---
    # We group OHE features (x0, x1) and sum their impacts
    grouped_impacts = {}
    
    # Mapping dict for technical names to business names
    name_mapping = {
        "x0": "Transaction Type",
        "x1": "Category",
        "amount": "Transaction Amount",
        "hour": "Hour of Day",
        "day": "Day of Week",
        "month": "Month"
    }

    for f in raw_factors:
        feature_name = f['feature']
        impact_value = f['impact']

        # Extract root name (e.g., 'x0_TRANSFER' -> 'x0')
        root_name = feature_name.split('_')[0]
        
        # Get business name or keep original if not in mapping
        business_name = name_mapping.get(root_name, root_name)

        # Sum impacts for the same business feature
        grouped_impacts[business_name] = grouped_impacts.get(business_name, 0) + impact_value

    # Sort grouped factors by absolute impact
    sorted_factors = sorted(
        grouped_impacts.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )

    # --- 2. PREPARE TEXT FOR PROMPT ---
    factors_text = "\n".join(
        [f"- {name}: {impact:.3f} impact score" for name, impact in sorted_factors]
    )

    prompt = f"""
You are a Senior Anti-Money Laundering (AML) Analyst. 
Your task is to explain why a transaction was flagged with a fraud probability of {prob:.2%}.

Below are the key risk factors identified by the XGBoost model (SHAP values):
{factors_text}

Instructions:
- Provide a concise, professional explanation for a bank investigator.
- Do NOT use technical machine learning terms like 'SHAP', 'XGBoost', or 'features'.
- Use financial and behavioral terminology (e.g., 'unusual volume', 'high-risk transaction type').
- Focus on the factors with the highest impact.
"""

    # Generate explanation via LLM
    response = llm.invoke(prompt)
    state.explanation = response.content

    return state