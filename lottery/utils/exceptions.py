"""Custom exceptions for the lottery prediction system."""

class LotteryError(Exception):
    """Base exception for all lottery-related errors."""
    pass

class DataError(LotteryError):
    """Raised when there are issues with data loading, validation, or format."""
    pass

class ModelNotFittedError(LotteryError):
    """Raised when predict is called before the model has been trained."""
    pass

class ConfigurationError(LotteryError):
    """Raised when configuration is invalid or missing required fields."""
    pass

class FeatureEngineeringError(LotteryError):
    """Raised when feature computation fails."""
    pass
