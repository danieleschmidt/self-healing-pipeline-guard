"""Healing-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class HealingStatus(Enum):
    """Healing operation status."""
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class StrategyType(Enum):
    """Types of healing strategies."""
    RETRY = "retry"
    RESOURCE_INCREASE = "resource_increase"
    CACHE_CLEAR = "cache_clear"
    DEPENDENCY_UPDATE = "dependency_update"
    CONFIGURATION_FIX = "configuration_fix"
    ROLLBACK = "rollback"
    CUSTOM = "custom"


@dataclass
class HealingStrategy:
    """Represents a healing strategy."""
    name: str
    type: StrategyType
    priority: int
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[int] = None
    success_rate: Optional[float] = None


@dataclass
class HealingResult:
    """Represents the result of a healing operation."""
    id: str
    failure_id: str
    strategy: HealingStrategy
    status: HealingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    time_saved_minutes: Optional[int] = None
    cost_saved_usd: Optional[float] = None


class HealingRequest(BaseModel):
    """Pydantic model for healing API requests."""
    failure_id: str
    strategy_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    force: bool = False


class HealingResponse(BaseModel):
    """Pydantic model for healing API responses."""
    id: str
    status: str
    strategy: Dict[str, Any]
    start_time: datetime
    estimated_duration: Optional[int] = None