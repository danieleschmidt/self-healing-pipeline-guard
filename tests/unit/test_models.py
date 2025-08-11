"""Tests for data models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from healing_guard.models.pipeline import (
    PipelineFailure, 
    PipelineEvent, 
    PipelineFailureRequest,
    PipelineEventRequest,
    PipelineStatus,
    EventType
)
from healing_guard.models.healing import (
    HealingStrategy,
    HealingResult,
    HealingRequest,
    HealingResponse,
    HealingStatus,
    StrategyType
)


class TestPipelineModels:
    """Test suite for pipeline models."""
    
    def test_pipeline_failure_creation(self):
        """Test PipelineFailure dataclass creation."""
        failure = PipelineFailure(
            id="failure_123",
            pipeline_id="pipeline_456",
            job_id="job_789",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            failure_time=datetime.now(),
            logs="Error: Test failed",
            exit_code=1,
            stage="test",
            step_name="unit_tests"
        )
        
        assert failure.id == "failure_123"
        assert failure.pipeline_id == "pipeline_456"
        assert failure.job_id == "job_789"
        assert failure.repository == "test/repo"
        assert failure.branch == "main"
        assert failure.commit_sha == "abc123"
        assert failure.logs == "Error: Test failed"
        assert failure.exit_code == 1
        assert failure.stage == "test"
        assert failure.step_name == "unit_tests"
        assert failure.duration_seconds is None
        assert failure.environment_vars == {}
        assert failure.metadata == {}
    
    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.SUCCESS.value == "success"
        assert PipelineStatus.FAILURE.value == "failure"
        assert PipelineStatus.IN_PROGRESS.value == "in_progress"
        assert PipelineStatus.CANCELLED.value == "cancelled"
        assert PipelineStatus.PENDING.value == "pending"


class TestHealingModels:
    """Test suite for healing models."""
    
    def test_healing_strategy_creation(self):
        """Test HealingStrategy dataclass creation."""
        strategy = HealingStrategy(
            name="retry_with_backoff",
            type=StrategyType.RETRY,
            priority=1,
            confidence=0.85
        )
        
        assert strategy.name == "retry_with_backoff"
        assert strategy.type == StrategyType.RETRY
        assert strategy.priority == 1
        assert strategy.confidence == 0.85
    
    def test_healing_status_enum(self):
        """Test HealingStatus enum values."""
        assert HealingStatus.SUCCESS.value == "success"
        assert HealingStatus.FAILURE.value == "failure"
        assert HealingStatus.IN_PROGRESS.value == "in_progress"
        assert HealingStatus.TIMEOUT.value == "timeout"
        assert HealingStatus.SKIPPED.value == "skipped"