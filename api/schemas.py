from pydantic import BaseModel

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    category: str
    hour: int
    day_of_week: int
    day_of_month: int
    month: int


class PredictRequest(BaseModel):
    transaction_id: int
    transaction: Transaction

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": 1002,
                "transaction": {
                    "step": 12,
                    "type": "PAYMENT",
                    "amount": 45.20,
                    "category": "Retail",
                    "hour": 14,
                    "day_of_week": 2,
                    "day_of_month": 15,
                    "month": 5
                }
            }
        }
    }


class AgentDecisionResponse(BaseModel):
    fraud_probability: float
    decision: str
    explanation: str
    audit_id: int


class AnalystQueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Why was transaction 1002 flagged as fraud?"
            }
        }
    }


class AnalystQueryResponse(BaseModel):
    answer: str
