"""Machine learning module for advanced failure analysis and prediction."""

from .failure_pattern_recognition import (
    FailurePatternRecognizer,
    FailurePattern,
    PatternType,
    PredictionResult,
    pattern_recognizer
)

__all__ = [
    "FailurePatternRecognizer",
    "FailurePattern", 
    "PatternType",
    "PredictionResult",
    "pattern_recognizer"
]