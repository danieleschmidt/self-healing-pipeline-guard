"""Self-Healing Pipeline Guard with Quantum-Inspired Task Planning.

A defensive security tool that automatically detects, diagnoses, and fixes 
CI/CD pipeline failures using quantum-inspired optimization algorithms.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

from .core.quantum_planner import QuantumTaskPlanner
from .core.failure_detector import FailureDetector
from .core.healing_engine import HealingEngine
from .core.sentiment_analyzer import sentiment_analyzer, PipelineSentimentAnalyzer
from .monitoring.health import health_checker
from .monitoring.metrics import metrics_collector

__all__ = [
    "QuantumTaskPlanner",
    "FailureDetector", 
    "HealingEngine",
    "sentiment_analyzer",
    "PipelineSentimentAnalyzer",
    "health_checker",
    "metrics_collector"
]