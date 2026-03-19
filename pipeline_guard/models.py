"""
Core data models for the pipeline guard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class StepStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


@dataclass
class PipelineStep:
    """Represents a single CI/CD pipeline step and its outcome."""

    name: str
    status: StepStatus
    duration_s: float = 0.0        # seconds
    error_output: str = ""
    exit_code: int = 0
    timestamp: float = field(default_factory=time.time)

    def failed(self) -> bool:
        return self.status in (StepStatus.FAIL, StepStatus.TIMEOUT)

    def __repr__(self) -> str:
        return (
            f"PipelineStep(name={self.name!r}, status={self.status.value}, "
            f"duration_s={self.duration_s:.1f}s)"
        )
