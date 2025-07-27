"""
Test configuration and fixtures for the Self-Healing Pipeline Guard test suite.

This module provides shared pytest fixtures, configuration, and utilities
used across all test modules.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Test environment configuration
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use test DB
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
        echo=True,
    )
    
    # Import here to avoid circular imports
    from healing_guard.database.models import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for FastAPI application."""
    from healing_guard.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for FastAPI application."""
    from healing_guard.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = False
    mock.expire.return_value = True
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_github_client():
    """Mock GitHub API client."""
    mock = MagicMock()
    mock.get_repo.return_value = MagicMock()
    mock.get_user.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_gitlab_client():
    """Mock GitLab API client."""
    mock = MagicMock()
    mock.projects.get.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_jenkins_client():
    """Mock Jenkins API client."""
    mock = MagicMock()
    mock.get_job.return_value = MagicMock()
    mock.get_build.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing."""
    mock = MagicMock()
    mock.predict.return_value = [0.85]  # High confidence
    mock.predict_proba.return_value = [[0.15, 0.85]]
    mock.classes_ = ["not_flaky", "flaky"]
    return mock


@pytest.fixture
def sample_pipeline_failure():
    """Sample pipeline failure data for testing."""
    return {
        "id": "test-failure-123",
        "platform": "github",
        "repository": "test-org/test-repo",
        "pipeline_id": "pipeline-456",
        "job_id": "job-789",
        "commit_sha": "abc123def456",
        "branch": "main",
        "failure_type": "test_failure",
        "error_message": "AssertionError: Expected 5 but got 3",
        "logs": [
            "Starting test suite...",
            "Running test_calculation...",
            "AssertionError: Expected 5 but got 3",
            "Test failed"
        ],
        "metadata": {
            "duration": 120,
            "exit_code": 1,
            "author": "developer@example.com",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }


@pytest.fixture
def sample_github_webhook():
    """Sample GitHub webhook payload."""
    return {
        "action": "completed",
        "workflow_run": {
            "id": 123456789,
            "name": "CI",
            "status": "completed",
            "conclusion": "failure",
            "html_url": "https://github.com/test-org/test-repo/actions/runs/123456789",
            "repository": {
                "full_name": "test-org/test-repo",
                "html_url": "https://github.com/test-org/test-repo"
            },
            "head_commit": {
                "id": "abc123def456",
                "message": "Fix calculation bug",
                "author": {
                    "email": "developer@example.com",
                    "name": "Developer"
                }
            },
            "head_branch": "main",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:32:00Z"
        }
    }


@pytest.fixture
def temporary_file():
    """Create temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temporary_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_slack_client():
    """Mock Slack client for testing notifications."""
    mock = AsyncMock()
    mock.chat_postMessage.return_value = {"ok": True, "ts": "1234567890.123"}
    return mock


@pytest.fixture
def mock_jira_client():
    """Mock Jira client for testing issue creation."""
    mock = MagicMock()
    mock.create_issue.return_value = MagicMock(key="TEST-123")
    return mock


@pytest.fixture
def healing_strategy_config():
    """Sample healing strategy configuration."""
    return {
        "strategies": {
            "flaky_test_retry": {
                "enabled": True,
                "confidence_threshold": 0.75,
                "max_retries": 3,
                "backoff_factor": 2.0
            },
            "resource_scaling": {
                "enabled": True,
                "memory_increment": "1GB",
                "cpu_increment": "0.5",
                "max_scaling_attempts": 2
            },
            "cache_invalidation": {
                "enabled": True,
                "cache_types": ["npm", "pip", "docker"],
                "selective_clearing": True
            }
        },
        "escalation": {
            "max_auto_attempts": 5,
            "escalation_delay": 300,
            "create_incidents": True
        }
    }


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing observability."""
    mock = MagicMock()
    mock.record_failure.return_value = None
    mock.record_healing_attempt.return_value = None
    mock.record_healing_success.return_value = None
    mock.get_metrics.return_value = {
        "total_failures": 10,
        "healing_success_rate": 0.85,
        "avg_resolution_time": 120
    }
    return mock


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """Isolate tests by mocking external dependencies."""
    # Mock environment variables
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("GITLAB_TOKEN", "fake-token")
    monkeypatch.setenv("JENKINS_TOKEN", "fake-token")
    monkeypatch.setenv("SLACK_TOKEN", "fake-token")
    
    # Mock external API calls
    monkeypatch.setattr("requests.get", MagicMock())
    monkeypatch.setattr("requests.post", MagicMock())
    monkeypatch.setattr("httpx.AsyncClient.get", AsyncMock())
    monkeypatch.setattr("httpx.AsyncClient.post", AsyncMock())


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.ml = pytest.mark.ml
pytest.mark.slow = pytest.mark.slow


# Custom assertions
def assert_healing_result(result, expected_success=True, expected_strategy=None):
    """Assert healing result matches expectations."""
    assert result.success == expected_success
    if expected_strategy:
        assert result.strategy == expected_strategy
    assert result.timestamp is not None
    assert result.duration >= 0


def assert_api_response(response, expected_status=200, expected_keys=None):
    """Assert API response matches expectations."""
    assert response.status_code == expected_status
    if expected_keys:
        response_data = response.json()
        for key in expected_keys:
            assert key in response_data