# src/utils/errors.py

class FraudAgentException(Exception):
    """Base exception for the Fraud AI Agent project."""
    pass

class DataIngestionError(FraudAgentException):
    """Raised when data loading fails."""
    def __init__(self, message="Failed to load or read data file"):
        self.message = message
        super().__init__(self.message)

class DataValidationError(FraudAgentException):
    """Raised when data does not match the Pandera schema."""
    def __init__(self, message="Data validation failed against expected schema"):
        self.message = message
        super().__init__(self.message)

class ModelTrainingError(FraudAgentException):
    """Raised during model training or hyperparameter tuning."""
    def __init__(self, message="Model training or tuning failed"):
        self.message = message
        super().__init__(self.message)

class AgentDecisionError(FraudAgentException):
    """Raised when the LangGraph agent fails to reach a decision."""
    def __init__(self, message="Agent could not produce a valid decision"):
        self.message = message
        super().__init__(self.message)