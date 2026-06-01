import streamlit as st
import requests
import os
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FraudGuard AI Agent",
    layout="wide"
)

# Define API URL from environment variable (Docker) or default to localhost
# In Docker, this will be http://api:8000
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("FraudGuard AI: Agentic Fraud Detection")
st.markdown("---")

# ==========================================
# SIDEBAR - TRANSACTION INPUT
# ==========================================
st.sidebar.header("Transaction Input")

def get_user_inputs():
    """
    Captures user input from the sidebar and formats it into a JSON payload.
    """
    st.sidebar.subheader("Basic Information")
    transaction_id = st.sidebar.number_input("Transaction ID", value=1001, step=1)
    step = st.sidebar.slider("Step (Time unit)", 1, 744, 1)
    type_tx = st.sidebar.selectbox(
        "Transaction Type", 
        ("TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN")
    )
    amount = st.sidebar.number_input("Amount (USD)", min_value=0.0, value=500.0)
    category = st.sidebar.text_input("Category", "Financial Services")
    
    st.sidebar.markdown("### Temporal Data")
    hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
    
    # Structure matching the PredictRequest Pydantic model
    payload = {
        "transaction_id": transaction_id,
        "transaction": {
            "step": step,
            "type": type_tx,
            "amount": amount,
            "category": category,
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": 1,
            "month": 1
        }
    }
    return transaction_id, payload

tx_id, input_data = get_user_inputs()

# ==========================================
# MAIN INTERFACE - ANALYSIS
# ==========================================
if st.sidebar.button("Run Fraud Analysis"):
    st.subheader(f"Analysis Results for Transaction #{tx_id}")
    
    # Display a loading spinner while the Agent processes the request
    with st.spinner('Agent is analyzing data and generating explanation...'):
        try:
            # API Call to the Agent Decision endpoint
            response = requests.post(f"{API_BASE_URL}/agent/decide", json=input_data)
            
            if response.status_code == 200:
                res = response.json()
                
                # Layout: Metrics (Probability, Decision, Audit ID)
                col1, col2, col3 = st.columns(3)
                
                # 1. Probability Metric
                col1.metric("Fraud Probability", f"{res['fraud_probability']:.2%}")
                
                # 2. Decision Badge
                decision = res['decision']
                if decision == "APPROVE":
                    col2.success(f"Decision: {decision}")
                elif decision == "REJECT":
                    col2.error(f"Decision: {decision}")
                else:
                    col2.warning(f"Decision: {decision}")
                
                # 3. Reference ID
                col3.write(f"**Audit ID:** {res['audit_id']}")

                # 4. Agent Reasoning Section
                st.markdown("### Agent Reasoning")
                st.info(res['explanation'])

                # 5. Visual Evidence (SHAP)
                # FIX: We fetch the image bytes directly from the API to avoid networking issues
                st.markdown("### Visual Evidence (SHAP)")
                explain_endpoint = f"{API_BASE_URL}/explain/{res['audit_id']}"
                
                try:
                    img_response = requests.get(explain_endpoint)
                    if img_response.status_code == 200:
                        # Displaying raw bytes as an image
                        st.image(
                            img_response.content, 
                            caption=f"SHAP Waterfall Analysis for Transaction {tx_id}",
                            use_container_width=True
                        )
                    else:
                        st.warning("SHAP visualization is currently unavailable for this record.")
                except Exception as e:
                    st.error(f"Failed to retrieve SHAP plot: {e}")

            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Error: Could not connect to the API at {API_BASE_URL}. Error: {e}")

else:
    # Initial state when the app is opened
    st.write("Please enter transaction details in the sidebar and click Run Fraud Analysis to start the investigation.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("FraudGuard AI Agent System - Integrated MLOps Lifecycle (XGBoost, SHAP, LangChain)")