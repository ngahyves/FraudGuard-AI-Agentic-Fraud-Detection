#src/utils/explainability/shap_test.py

from src.utils.explainability.shap_local import LocalExplainer
from src.utils.explainability.payload_agent import build_agent_payload

def main():
    print("\n=== SHAP LOCAL TEST ===\n")

    explainer = LocalExplainer()

    # Choose an index to explain
    index = 10

    print(f"Explaining transaction index: {index}\n")

    # SHAP local
    top_factors, proba = explainer.explain_transaction(index)

    print("Top SHAP factors:")
    for f in top_factors:
        print(f"  - {f}")

    print(f"\nFraud probability: {proba:.4f}\n")

    # Payload agent
    payload = build_agent_payload(
        transaction_id=index,
        proba=proba,
        top_factors=top_factors
    )

    print("Agent payload:")
    print(payload)

if __name__ == "__main__":
    main()
