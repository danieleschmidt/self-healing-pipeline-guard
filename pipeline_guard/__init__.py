"""
Self-Healing Pipeline Guard
----------------------------
Detects CI/CD pipeline failures, classifies root causes,
selects remediation strategies, and produces healing reports.
"""

from .models import PipelineStep, StepStatus
from .analyzer import FailureAnalyzer, FailureCategory
from .remediation import RemediationStrategy, RemediationAction
from .guard import PipelineGuard
from .report import HealingReport

__all__ = [
    "PipelineStep",
    "StepStatus",
    "FailureAnalyzer",
    "FailureCategory",
    "RemediationStrategy",
    "RemediationAction",
    "PipelineGuard",
    "HealingReport",
]

__version__ = "2.0.0"
