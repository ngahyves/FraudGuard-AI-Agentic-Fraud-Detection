# run_agent.py
from src.agent.graph import run_agent

if __name__ == "__main__":
    # 1. Transaction example
    transaction = {
        "step": 123,
        "type": "DEBIT",
        "amount": 2376.58,
        "category": "Housing",
        "hour": 14,
        "day_of_week": 3,
        "day_of_month": 12,
        "month": 5
    }

    # 2. Callong the run_agent function
    print("--- Lancement de l'agent de test ---")
    result = run_agent(transaction_id=1, transaction=transaction)

    # 3. Print results
    print(f"Probabilité: {result['fraud_probability']}")
    print(f"Décision: {result['decision']}")
    print(f"Explication: {result['explanation']}")
    print(f"Audit Log: {result['audit_log']}")