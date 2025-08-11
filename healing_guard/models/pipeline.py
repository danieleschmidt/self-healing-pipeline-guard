"""Pipeline-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class PipelineStatus(Enum):
    """Pipeline status enum."""
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    PENDING = "pending"


class EventType(Enum):
    """Pipeline event types."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class PipelineFailure:
    """Represents a pipeline failure event."""
    id: str
    pipeline_id: str
    job_id: str
    repository: str
    branch: str
    commit_sha: str
    failure_time: datetime
    logs: str
    exit_code: int
    stage: str
    step_name: str
    duration_seconds: Optional[int] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEvent:
    """Represents a general pipeline event."""
    id: str
    pipeline_id: str
    repository: str
    branch: str
    commit_sha: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


class PipelineFailureRequest(BaseModel):
    """Pydantic model for pipeline failure API requests."""
    pipeline_id: str
    job_id: str
    repository: str
    branch: str
    commit_sha: str
    logs: str
    exit_code: int
    stage: str
    step_name: str
    duration_seconds: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineEventRequest(BaseModel):
    """Pydantic model for pipeline event API requests."""
    pipeline_id: str
    repository: str
    branch: str 
    commit_sha: str
    event_type: str
    data: Optional[Dict[str, Any]] = None