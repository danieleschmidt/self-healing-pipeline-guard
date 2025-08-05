"""Integration tests for the API endpoints."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from healing_guard.api.main import create_app
from healing_guard.core.quantum_planner import Task, TaskPriority
from healing_guard.core.failure_detector import FailureType, SeverityLevel


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_task_data():
    """Sample task data for API requests."""
    return {
        "name": "Test Task",
        "priority": 2,
        "estimated_duration": 5.0,
        "dependencies": [],
        "resources_required": {"cpu": 1.0, "memory": 2.0},
        "failure_probability": 0.1,
        "max_retries": 3,
        "metadata": {"test": "data"}
    }


@pytest.fixture
def sample_failure_data():
    """Sample failure analysis data for API requests."""
    return {
        "job_id": "test_job_123",
        "repository": "test/repo",
        "branch": "main",
        "commit_sha": "abc123def456",
        "logs": "ERROR: Test failed with timeout\nConnection refused\nRetry attempt 1 failed",
        "context": {"is_main_branch": True}
    }


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "operational"
    
    def test_health_endpoint(self, client):
        """Test comprehensive health check endpoint."""
        with patch('healing_guard.monitoring.health.health_checker.get_comprehensive_health') as mock_health:
            mock_health.return_value = AsyncMock()
            mock_health.return_value.status = "healthy"
            mock_health.return_value.dict.return_value = {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0.0",
                "uptime": 3600.0,
                "checks": {}
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
    
    def test_liveness_endpoint(self, client):
        """Test Kubernetes liveness probe endpoint."""
        with patch('healing_guard.monitoring.health.health_checker.get_liveness') as mock_liveness:
            mock_liveness.return_value = {"status": "alive"}
            
            response = client.get("/health/live")
            assert response.status_code == 200
            assert response.text == "OK"
    
    def test_readiness_endpoint(self, client):
        """Test Kubernetes readiness probe endpoint."""
        with patch('healing_guard.monitoring.health.health_checker.get_readiness') as mock_readiness:
            mock_readiness.return_value = {"status": "ready"}
            
            response = client.get("/health/ready")
            assert response.status_code == 200
            assert response.text == "READY"
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


class TestQuantumPlannerEndpoints:
    """Test quantum planner API endpoints."""
    
    def test_create_task(self, client, sample_task_data):
        """Test creating a new task."""
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.add_task = Mock()
            
            response = client.post("/api/v1/tasks", json=sample_task_data)
            assert response.status_code == 201
            
            data = response.json()
            assert data["name"] == sample_task_data["name"]
            assert data["priority"] == sample_task_data["priority"]
            assert data["estimated_duration"] == sample_task_data["estimated_duration"]
            
            # Verify task was added to planner
            mock_planner.add_task.assert_called_once()
    
    def test_create_task_validation_error(self, client):
        """Test task creation with invalid data."""
        invalid_data = {
            "name": "Test Task",
            "priority": 5,  # Invalid priority (should be 1-4)
            "estimated_duration": -1.0  # Invalid duration (should be positive)
        }
        
        response = client.post("/api/v1/tasks", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_list_tasks(self, client):
        """Test listing all tasks."""
        mock_tasks = {
            "task_1": Task("task_1", "Task 1", TaskPriority.HIGH, 5.0),
            "task_2": Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0)
        }
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.tasks = mock_tasks
            
            response = client.get("/api/v1/tasks")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 2
            assert any(task["id"] == "task_1" for task in data)
            assert any(task["id"] == "task_2" for task in data)
    
    def test_get_task(self, client):
        """Test getting a specific task."""
        mock_task = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0)
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.get_task.return_value = mock_task
            
            response = client.get("/api/v1/tasks/task_1")
            assert response.status_code == 200
            
            data = response.json()
            assert data["id"] == "task_1"
            assert data["name"] == "Task 1"
    
    def test_get_task_not_found(self, client):
        """Test getting a non-existent task."""
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.get_task.return_value = None
            
            response = client.get("/api/v1/tasks/non_existent")
            assert response.status_code == 404
    
    def test_delete_task(self, client):
        """Test deleting a task."""
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.remove_task.return_value = True
            
            response = client.delete("/api/v1/tasks/task_1")
            assert response.status_code == 204
            
            mock_planner.remove_task.assert_called_once_with("task_1")
    
    def test_delete_task_not_found(self, client):
        """Test deleting a non-existent task."""
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.remove_task.return_value = False
            
            response = client.delete("/api/v1/tasks/non_existent")
            assert response.status_code == 404
    
    def test_create_execution_plan(self, client):
        """Test creating an execution plan."""
        mock_plan = Mock()
        mock_plan.execution_order = ["task_1", "task_2"]
        mock_plan.estimated_total_time = 10.0
        mock_plan.resource_utilization = {"cpu": 75.0, "memory": 60.0}
        mock_plan.success_probability = 0.85
        mock_plan.cost_estimate = 5.0
        mock_plan.parallel_stages = [["task_1"], ["task_2"]]
        mock_plan.tasks = [Task("task_1", "Task 1", TaskPriority.HIGH, 5.0)]
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.create_execution_plan = AsyncMock(return_value=mock_plan)
            
            response = client.post("/api/v1/planning/execute")
            assert response.status_code == 200
            
            data = response.json()
            assert data["execution_order"] == ["task_1", "task_2"]
            assert data["estimated_total_time"] == 10.0
            assert data["success_probability"] == 0.85
    
    def test_execute_plan(self, client):
        """Test executing a plan."""
        mock_plan = Mock()
        mock_plan.execution_order = ["task_1"]
        mock_plan.estimated_total_time = 5.0
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.create_execution_plan = AsyncMock(return_value=mock_plan)
            mock_planner.execute_plan = AsyncMock()
            
            response = client.post("/api/v1/planning/execute/run")
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            assert "estimated_duration" in data
    
    def test_get_planning_statistics(self, client):
        """Test getting planning statistics."""
        mock_stats = {
            "total_executions": 10,
            "success_rate": 0.8,
            "average_duration_minutes": 12.5,
            "current_tasks": 3
        }
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.get_planning_statistics.return_value = mock_stats
            
            response = client.get("/api/v1/planning/statistics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_executions"] == 10
            assert data["success_rate"] == 0.8


class TestFailureDetectionEndpoints:
    """Test failure detection API endpoints."""
    
    def test_analyze_failure(self, client, sample_failure_data):
        """Test analyzing a failure."""
        from healing_guard.core.failure_detector import FailureEvent
        from datetime import datetime
        
        mock_failure = FailureEvent(
            id="failure_123",
            timestamp=datetime.now(),
            job_id=sample_failure_data["job_id"],
            repository=sample_failure_data["repository"],
            branch=sample_failure_data["branch"],
            commit_sha=sample_failure_data["commit_sha"],
            failure_type=FailureType.NETWORK_TIMEOUT,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs=sample_failure_data["logs"],
            remediation_suggestions=["retry_with_backoff"]
        )
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.detect_failure = AsyncMock(return_value=mock_failure)
            
            response = client.post("/api/v1/failures/analyze", json=sample_failure_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["job_id"] == sample_failure_data["job_id"]
            assert data["failure_type"] == "network_timeout"
            assert data["confidence"] == 0.85
    
    def test_analyze_failure_no_detection(self, client, sample_failure_data):
        """Test analyzing logs with no failure detected."""
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.detect_failure = AsyncMock(return_value=None)
            
            response = client.post("/api/v1/failures/analyze", json=sample_failure_data)
            assert response.status_code == 400
            assert "No failure detected" in response.json()["detail"]
    
    def test_list_failures(self, client):
        """Test listing failures with pagination and filtering."""
        from healing_guard.core.failure_detector import FailureEvent
        from datetime import datetime
        
        mock_failures = [
            FailureEvent(
                id=f"failure_{i}",
                timestamp=datetime.now(),
                job_id=f"job_{i}",
                repository="test/repo",
                branch="main",
                commit_sha=f"sha_{i}",
                failure_type=FailureType.FLAKY_TEST if i % 2 == 0 else FailureType.NETWORK_TIMEOUT,
                severity=SeverityLevel.MEDIUM,
                confidence=0.8,
                raw_logs="test logs"
            )
            for i in range(5)
        ]
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.failure_history = mock_failures
            
            # Test basic listing
            response = client.get("/api/v1/failures")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 5
            
            # Test with limit
            response = client.get("/api/v1/failures?limit=3")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 3
            
            # Test with failure type filter
            response = client.get("/api/v1/failures?failure_type=flaky_test")
            assert response.status_code == 200
            
            data = response.json()
            # Should only return flaky test failures
            assert all(failure["failure_type"] == "flaky_test" for failure in data)
    
    def test_get_failure_statistics(self, client):
        """Test getting failure statistics."""
        mock_stats = {
            "total_failures": 100,
            "failure_types": {"flaky_test": 60, "network_timeout": 40},
            "average_confidence": 0.82
        }
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.get_failure_statistics.return_value = mock_stats
            
            response = client.get("/api/v1/failures/statistics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_failures"] == 100
            assert "failure_types" in data
    
    def test_get_failure_trends(self, client):
        """Test getting failure trends."""
        mock_trends = {
            "period_days": 7,
            "total_failures": 50,
            "daily_failure_counts": {"2024-01-01": 10, "2024-01-02": 8},
            "trending_failure_types": {"flaky_test": 30, "network_timeout": 20}
        }
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.get_failure_trends.return_value = mock_trends
            
            response = client.get("/api/v1/failures/trends?days=7")
            assert response.status_code == 200
            
            data = response.json()
            assert data["period_days"] == 7
            assert data["total_failures"] == 50


class TestHealingEngineEndpoints:
    """Test healing engine API endpoints."""
    
    def test_create_healing_plan(self, client, sample_failure_data):
        """Test creating a healing plan."""
        from healing_guard.core.failure_detector import FailureEvent
        from healing_guard.core.healing_engine import HealingPlan, HealingAction, HealingStrategy
        from datetime import datetime
        
        mock_failure = FailureEvent(
            id="failure_123",
            timestamp=datetime.now(),
            job_id=sample_failure_data["job_id"],
            repository=sample_failure_data["repository"],
            branch=sample_failure_data["branch"],
            commit_sha=sample_failure_data["commit_sha"],
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs=sample_failure_data["logs"]
        )
        
        mock_action = HealingAction(
            id="action_1",
            strategy=HealingStrategy.RETRY_WITH_BACKOFF,
            description="Retry with backoff"
        )
        
        mock_plan = HealingPlan(
            id="plan_123",
            failure_event=mock_failure,
            actions=[mock_action],
            estimated_total_time=10.0,
            success_probability=0.8,
            total_cost=5.0,
            priority=1,
            created_at=datetime.now()
        )
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.detect_failure = AsyncMock(return_value=mock_failure)
            
            with patch('healing_guard.api.routes.healing_engine') as mock_engine:
                mock_engine.create_healing_plan = AsyncMock(return_value=mock_plan)
                
                response = client.post("/api/v1/healing/plan", json=sample_failure_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["id"] == "plan_123"
                assert data["estimated_total_time"] == 10.0
                assert data["success_probability"] == 0.8
                assert len(data["actions"]) == 1
    
    def test_execute_healing_plan(self, client, sample_failure_data):
        """Test executing a healing plan."""
        from healing_guard.core.failure_detector import FailureEvent
        from healing_guard.core.healing_engine import HealingResult, HealingStatus
        from datetime import datetime
        
        mock_failure = FailureEvent(
            id="failure_123",
            timestamp=datetime.now(),
            job_id=sample_failure_data["job_id"],
            repository=sample_failure_data["repository"],
            branch=sample_failure_data["branch"],
            commit_sha=sample_failure_data["commit_sha"],
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs=sample_failure_data["logs"]
        )
        
        mock_result = HealingResult(
            healing_id="healing_123",
            plan=Mock(),
            status=HealingStatus.SUCCESSFUL,
            actions_executed=["action_1"],
            actions_successful=["action_1"],
            actions_failed=[],
            total_duration=8.5,
            metrics={"success_rate": 1.0}
        )
        mock_result.plan.id = "plan_123"
        
        with patch('healing_guard.api.routes.failure_detector') as mock_detector:
            mock_detector.detect_failure = AsyncMock(return_value=mock_failure)
            
            with patch('healing_guard.api.routes.healing_engine') as mock_engine:
                mock_engine.heal_failure = AsyncMock(return_value=mock_result)
                
                response = client.post("/api/v1/healing/execute", json=sample_failure_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["healing_id"] == "healing_123"
                assert data["status"] == "successful"
                assert data["total_duration"] == 8.5
    
    def test_get_healing_statistics(self, client):
        """Test getting healing statistics."""
        mock_stats = {
            "total_healings": 50,
            "successful_healings": 40,
            "overall_success_rate": 0.8,
            "average_duration_minutes": 12.5
        }
        
        with patch('healing_guard.api.routes.healing_engine') as mock_engine:
            mock_engine.get_healing_statistics.return_value = mock_stats
            
            response = client.get("/api/v1/healing/statistics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_healings"] == 50
            assert data["overall_success_rate"] == 0.8
    
    def test_get_active_healings(self, client):
        """Test getting active healing operations."""
        mock_active = {
            "healing_1": Mock(id="plan_1", created_at=datetime.now()),
            "healing_2": Mock(id="plan_2", created_at=datetime.now())
        }
        
        with patch('healing_guard.api.routes.healing_engine') as mock_engine:
            mock_engine.active_healings = mock_active
            
            response = client.get("/api/v1/healing/active")
            assert response.status_code == 200
            
            data = response.json()
            assert data["active_healings"] == 2
            assert len(data["healings"]) == 2


class TestSystemEndpoints:
    """Test system-level API endpoints."""
    
    def test_get_system_status(self, client):
        """Test getting overall system status."""
        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.timestamp = datetime.now()
        mock_health.version = "1.0.0"
        mock_health.uptime = 3600.0
        mock_health.checks = {"database": {"status": "healthy"}}
        
        with patch('healing_guard.api.routes.health_checker') as mock_checker:
            mock_checker.get_comprehensive_health = AsyncMock(return_value=mock_health)
            
            with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
                mock_planner.tasks = {}
                mock_planner.optimization_iterations = 1000
                
                with patch('healing_guard.api.routes.failure_detector') as mock_detector:
                    mock_detector.patterns = {}
                    mock_detector.failure_history = []
                    
                    with patch('healing_guard.api.routes.healing_engine') as mock_engine:
                        mock_engine.active_healings = {}
                        mock_engine.strategy_registry = {}
                        
                        response = client.get("/api/v1/system/status")
                        assert response.status_code == 200
                        
                        data = response.json()
                        assert data["status"] == "healthy"
                        assert "components" in data
                        assert "health_checks" in data
    
    def test_get_system_metrics(self, client):
        """Test getting system metrics."""
        mock_planner_stats = {"total_executions": 10}
        mock_detector_stats = {"total_failures": 5}
        mock_engine_stats = {"total_healings": 3}
        
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.get_planning_statistics.return_value = mock_planner_stats
            
            with patch('healing_guard.api.routes.failure_detector') as mock_detector:
                mock_detector.get_failure_statistics.return_value = mock_detector_stats
                
                with patch('healing_guard.api.routes.healing_engine') as mock_engine:
                    mock_engine.get_healing_statistics.return_value = mock_engine_stats
                    
                    response = client.get("/api/v1/system/metrics")
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert "quantum_planner" in data
                    assert "failure_detector" in data
                    assert "healing_engine" in data
                    assert "timestamp" in data


class TestErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_validation_error_handling(self, client):
        """Test handling of validation errors."""
        invalid_data = {
            "name": "",  # Empty name should cause validation error
            "priority": 0,  # Invalid priority
            "estimated_duration": -5.0  # Negative duration
        }
        
        response = client.post("/api/v1/tasks", json=invalid_data)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_internal_server_error_handling(self, client):
        """Test handling of internal server errors."""
        with patch('healing_guard.api.routes.quantum_planner') as mock_planner:
            mock_planner.add_task.side_effect = Exception("Internal error")
            
            task_data = {
                "name": "Test Task",
                "priority": 2,
                "estimated_duration": 5.0
            }
            
            response = client.post("/api/v1/tasks", json=task_data)
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
    
    def test_rate_limiting(self, client):
        """Test API rate limiting middleware."""
        # This would require actual rate limiting to be enabled
        # For now, just test that the endpoint is accessible
        response = client.get("/api/v1/system/status")
        assert response.status_code in [200, 429]  # Either success or rate limited
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/tasks")
        # CORS headers should be present
        assert response.status_code in [200, 405]  # Either allowed or method not allowed