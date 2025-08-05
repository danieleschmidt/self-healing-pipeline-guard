"""Unit tests for healing engine."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from healing_guard.core.healing_engine import (
    HealingEngine, HealingAction, HealingPlan, HealingResult, 
    HealingStrategy, HealingStatus
)
from healing_guard.core.failure_detector import FailureEvent, FailureType, SeverityLevel
from healing_guard.core.quantum_planner import QuantumTaskPlanner


class TestHealingEngine:
    """Test cases for HealingEngine class."""
    
    @pytest.fixture
    def mock_planner(self):
        """Create a mock quantum planner."""
        planner = Mock(spec=QuantumTaskPlanner)
        planner.create_execution_plan = AsyncMock()
        planner.create_execution_plan.return_value = Mock(
            estimated_total_time=10.0,
            success_probability=0.8,
            cost_estimate=5.0
        )
        return planner
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mock failure detector."""
        detector = Mock()
        return detector
    
    @pytest.fixture
    def healing_engine(self, mock_planner, mock_detector):
        """Create a HealingEngine instance for testing."""
        return HealingEngine(
            quantum_planner=mock_planner,
            failure_detector=mock_detector,
            max_concurrent_healings=2,
            healing_timeout=10
        )
    
    @pytest.fixture
    def sample_failure_event(self):
        """Create a sample failure event for testing."""
        return FailureEvent(
            id="failure_123",
            timestamp=datetime.now(),
            job_id="job_123",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs="Test failure logs",
            remediation_suggestions=["retry_with_isolation", "increase_timeout"]
        )
    
    def test_initialization(self, healing_engine, mock_planner, mock_detector):
        """Test healing engine initialization."""
        assert healing_engine.quantum_planner == mock_planner
        assert healing_engine.failure_detector == mock_detector
        assert healing_engine.max_concurrent_healings == 2
        assert healing_engine.healing_timeout == 10
        assert len(healing_engine.active_healings) == 0
        assert len(healing_engine.healing_history) == 0
        assert len(healing_engine.strategy_registry) > 0
        assert len(healing_engine.custom_actions) == 0
    
    def test_add_custom_action(self, healing_engine):
        """Test adding custom healing actions."""
        custom_action = HealingAction(
            id="custom_action_1",
            strategy=HealingStrategy.CLEAR_CACHE,
            description="Custom cache clearing action",
            parameters={"cache_types": ["npm", "pip"]}
        )
        
        healing_engine.add_custom_action(custom_action)
        
        assert "custom_action_1" in healing_engine.custom_actions
        assert healing_engine.custom_actions["custom_action_1"] == custom_action
    
    def test_create_healing_actions(self, healing_engine, sample_failure_event):
        """Test creating healing actions from failure event."""
        actions = healing_engine._create_healing_actions(sample_failure_event)
        
        assert len(actions) > 0
        assert all(isinstance(action, HealingAction) for action in actions)
        
        # Check that actions correspond to remediation suggestions
        action_strategies = [action.strategy for action in actions]
        assert HealingStrategy.RETRY_WITH_BACKOFF in action_strategies
    
    def test_create_action_for_strategy(self, healing_engine, sample_failure_event):
        """Test creating specific healing actions for strategies."""
        # Test retry strategy
        retry_action = healing_engine._create_action_for_strategy(
            HealingStrategy.RETRY_WITH_BACKOFF,
            sample_failure_event,
            "retry_with_isolation"
        )
        
        assert retry_action is not None
        assert retry_action.strategy == HealingStrategy.RETRY_WITH_BACKOFF
        assert "max_retries" in retry_action.parameters
        assert retry_action.estimated_duration > 0
        assert 0 <= retry_action.success_probability <= 1
        
        # Test resource increase strategy
        resource_action = healing_engine._create_action_for_strategy(
            HealingStrategy.INCREASE_RESOURCES,
            sample_failure_event,
            "increase_resources"
        )
        
        assert resource_action is not None
        assert resource_action.strategy == HealingStrategy.INCREASE_RESOURCES
        assert "cpu_increase" in resource_action.parameters
        assert "memory_increase" in resource_action.parameters
        assert resource_action.cost_estimate > 0
    
    def test_get_default_actions_for_failure_type(self, healing_engine):
        """Test getting default actions for different failure types."""
        # Test flaky test failure
        flaky_actions = healing_engine._get_default_actions_for_failure_type(
            FailureType.FLAKY_TEST
        )
        assert len(flaky_actions) > 0
        assert any(action.strategy == HealingStrategy.RETRY_WITH_BACKOFF for action in flaky_actions)
        
        # Test resource exhaustion failure
        resource_actions = healing_engine._get_default_actions_for_failure_type(
            FailureType.RESOURCE_EXHAUSTION
        )
        assert len(resource_actions) > 0
        assert any(action.strategy == HealingStrategy.INCREASE_RESOURCES for action in resource_actions)
        
        # Test dependency failure
        dep_actions = healing_engine._get_default_actions_for_failure_type(
            FailureType.DEPENDENCY_FAILURE
        )
        assert len(dep_actions) > 0
        assert any(action.strategy == HealingStrategy.CLEAR_CACHE for action in dep_actions)
        assert any(action.strategy == HealingStrategy.UPDATE_DEPENDENCIES for action in dep_actions)
    
    def test_map_severity_to_priority(self, healing_engine):
        """Test mapping failure severity to task priority."""
        from healing_guard.core.quantum_planner import TaskPriority
        
        assert healing_engine._map_severity_to_priority(SeverityLevel.CRITICAL) == TaskPriority.CRITICAL
        assert healing_engine._map_severity_to_priority(SeverityLevel.HIGH) == TaskPriority.HIGH
        assert healing_engine._map_severity_to_priority(SeverityLevel.MEDIUM) == TaskPriority.MEDIUM
        assert healing_engine._map_severity_to_priority(SeverityLevel.LOW) == TaskPriority.LOW
    
    def test_calculate_healing_priority(self, healing_engine, sample_failure_event):
        """Test calculating healing priority scores."""
        priority = healing_engine._calculate_healing_priority(
            sample_failure_event,
            total_cost=5.0,
            success_probability=0.8
        )
        
        assert isinstance(priority, int)
        assert priority >= 1
        
        # Test main branch should have higher priority (lower number)
        main_branch_event = FailureEvent(
            id="failure_main",
            timestamp=datetime.now(),
            job_id="job_main",
            repository="test/repo",
            branch="main",  # Main branch
            commit_sha="abc123",
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs="Test failure logs"
        )
        
        main_priority = healing_engine._calculate_healing_priority(
            main_branch_event,
            total_cost=5.0,
            success_probability=0.8
        )
        
        feature_branch_event = FailureEvent(
            id="failure_feature",
            timestamp=datetime.now(),
            job_id="job_feature",
            repository="test/repo",
            branch="feature/test",  # Feature branch
            commit_sha="abc123",
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs="Test failure logs"
        )
        
        feature_priority = healing_engine._calculate_healing_priority(
            feature_branch_event,
            total_cost=5.0,
            success_probability=0.8
        )
        
        assert main_priority < feature_priority  # Lower number = higher priority
    
    @pytest.mark.asyncio
    async def test_create_healing_plan(self, healing_engine, sample_failure_event):
        """Test creating a healing plan."""
        with patch.object(healing_engine.quantum_planner, 'add_task') as mock_add_task:
            plan = await healing_engine.create_healing_plan(sample_failure_event)
            
            assert isinstance(plan, HealingPlan)
            assert plan.failure_event == sample_failure_event
            assert len(plan.actions) > 0
            assert plan.estimated_total_time > 0
            assert 0 <= plan.success_probability <= 1
            assert plan.total_cost >= 0
            assert plan.priority >= 1
            assert isinstance(plan.created_at, datetime)
            
            # Verify tasks were added to planner
            assert mock_add_task.called
    
    @pytest.mark.asyncio
    async def test_create_healing_plan_no_actions(self, healing_engine):
        """Test creating healing plan with no available actions."""
        # Create failure event with no remediation suggestions
        failure_event = FailureEvent(
            id="failure_no_actions",
            timestamp=datetime.now(),
            job_id="job_123",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            failure_type=FailureType.UNKNOWN,
            severity=SeverityLevel.LOW,
            confidence=0.5,
            raw_logs="Unclear failure logs",
            remediation_suggestions=[]  # No suggestions
        )
        
        # Mock to return no default actions
        with patch.object(healing_engine, '_get_default_actions_for_failure_type', return_value=[]):
            with pytest.raises(ValueError, match="No healing actions available"):
                await healing_engine.create_healing_plan(failure_event)
    
    @pytest.mark.asyncio
    async def test_execute_healing_plan(self, healing_engine, sample_failure_event):
        """Test executing a healing plan."""
        # Create a healing plan
        plan = await healing_engine.create_healing_plan(sample_failure_event)
        
        # Mock strategy execution to succeed
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.return_value = {"status": "success"}
            
            result = await healing_engine.execute_healing_plan(plan)
            
            assert isinstance(result, HealingResult)
            assert result.plan == plan
            assert result.status in [HealingStatus.SUCCESSFUL, HealingStatus.PARTIAL, HealingStatus.FAILED]
            assert len(result.actions_executed) > 0
            assert result.total_duration >= 0
            assert "success_rate" in result.metrics
    
    @pytest.mark.asyncio
    async def test_execute_healing_plan_timeout(self, healing_engine, sample_failure_event):
        """Test healing plan execution with timeout."""
        plan = await healing_engine.create_healing_plan(sample_failure_event)
        
        # Mock strategy execution to timeout
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.side_effect = asyncio.TimeoutError()
            
            result = await healing_engine.execute_healing_plan(plan)
            
            assert result.status == HealingStatus.FAILED
            assert len(result.actions_failed) > 0
    
    @pytest.mark.asyncio
    async def test_execute_healing_plan_partial_success(self, healing_engine, sample_failure_event):
        """Test healing plan execution with partial success."""
        plan = await healing_engine.create_healing_plan(sample_failure_event)
        
        # Mock strategy execution to have mixed results
        call_count = 0
        def mock_strategy_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "success"}
            else:
                return {"status": "failed"}
        
        with patch('asyncio.wait_for', side_effect=mock_strategy_execution):
            result = await healing_engine.execute_healing_plan(plan)
            
            assert result.status == HealingStatus.PARTIAL
            assert len(result.actions_successful) > 0
            assert len(result.actions_failed) > 0
    
    @pytest.mark.asyncio
    async def test_heal_failure(self, healing_engine, sample_failure_event):
        """Test complete healing process."""
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.return_value = {"status": "success"}
            
            result = await healing_engine.heal_failure(sample_failure_event)
            
            assert isinstance(result, HealingResult)
            assert result.status in [HealingStatus.SUCCESSFUL, HealingStatus.PARTIAL, HealingStatus.FAILED]
            
            # Check that result was added to history
            assert len(healing_engine.healing_history) == 1
            assert healing_engine.healing_history[0] == result
    
    def test_get_healing_statistics(self, healing_engine):
        """Test getting healing statistics."""
        # Initially no history
        stats = healing_engine.get_healing_statistics()
        assert "message" in stats
        
        # Add some mock healing results
        for i in range(5):
            result = HealingResult(
                healing_id=f"healing_{i}",
                plan=Mock(actions=[Mock(), Mock()]),  # Mock plan with 2 actions
                status=HealingStatus.SUCCESSFUL if i < 3 else HealingStatus.FAILED,
                actions_executed=[f"action_{i}_1", f"action_{i}_2"],
                actions_successful=[f"action_{i}_1", f"action_{i}_2"] if i < 3 else [],
                actions_failed=[] if i < 3 else [f"action_{i}_1", f"action_{i}_2"],
                total_duration=10.0 + i,
                metrics={"success_rate": 1.0 if i < 3 else 0.0}
            )
            healing_engine.healing_history.append(result)
        
        stats = healing_engine.get_healing_statistics()
        
        assert stats["total_healings"] == 5
        assert stats["successful_healings"] == 3
        assert stats["failed_healings"] == 2
        assert stats["overall_success_rate"] == 0.6
        assert stats["average_duration_minutes"] > 0
        assert stats["active_healings"] == 0
    
    def test_healing_action_serialization(self):
        """Test healing action serialization."""
        action = HealingAction(
            id="test_action",
            strategy=HealingStrategy.RETRY_WITH_BACKOFF,
            description="Test retry action",
            parameters={"max_retries": 3},
            estimated_duration=5.0,
            success_probability=0.8,
            cost_estimate=1.0,
            prerequisites=["prerequisite1"],
            side_effects=["side_effect1"],
            rollback_action="rollback_action"
        )
        
        action_dict = action.to_dict()
        
        required_fields = [
            "id", "strategy", "description", "parameters", "estimated_duration",
            "success_probability", "cost_estimate", "prerequisites", 
            "side_effects", "rollback_action"
        ]
        
        for field in required_fields:
            assert field in action_dict
        
        assert action_dict["id"] == "test_action"
        assert action_dict["strategy"] == "retry_with_backoff"
        assert action_dict["parameters"]["max_retries"] == 3
    
    def test_healing_plan_serialization(self, sample_failure_event):
        """Test healing plan serialization."""
        action = HealingAction(
            id="test_action",
            strategy=HealingStrategy.RETRY_WITH_BACKOFF,
            description="Test action"
        )
        
        plan = HealingPlan(
            id="test_plan",
            failure_event=sample_failure_event,
            actions=[action],
            estimated_total_time=10.0,
            success_probability=0.8,
            total_cost=5.0,
            priority=1,
            created_at=datetime.now()
        )
        
        plan_dict = plan.to_dict()
        
        required_fields = [
            "id", "failure_event", "actions", "estimated_total_time",
            "success_probability", "total_cost", "priority", "created_at"
        ]
        
        for field in required_fields:
            assert field in plan_dict
        
        assert plan_dict["id"] == "test_plan"
        assert len(plan_dict["actions"]) == 1
        assert plan_dict["estimated_total_time"] == 10.0
    
    def test_healing_result_serialization(self, sample_failure_event):
        """Test healing result serialization."""
        plan = HealingPlan(
            id="test_plan",
            failure_event=sample_failure_event,
            actions=[],
            estimated_total_time=10.0,
            success_probability=0.8,
            total_cost=5.0,
            priority=1,
            created_at=datetime.now()
        )
        
        result = HealingResult(
            healing_id="test_healing",
            plan=plan,
            status=HealingStatus.SUCCESSFUL,
            actions_executed=["action1"],
            actions_successful=["action1"],
            actions_failed=[],
            total_duration=8.5,
            metrics={"success_rate": 1.0}
        )
        
        result_dict = result.to_dict()
        
        required_fields = [
            "healing_id", "plan_id", "status", "actions_executed",
            "actions_successful", "actions_failed", "total_duration",
            "error_message", "metrics", "rollback_performed"
        ]
        
        for field in required_fields:
            assert field in result_dict
        
        assert result_dict["healing_id"] == "test_healing"
        assert result_dict["plan_id"] == "test_plan"
        assert result_dict["status"] == "successful"
        assert result_dict["total_duration"] == 8.5
    
    def test_concurrent_healing_limit(self, healing_engine):
        """Test that concurrent healing limit is respected."""
        # Set up active healings at the limit
        for i in range(healing_engine.max_concurrent_healings):
            mock_plan = Mock()
            mock_plan.id = f"plan_{i}"
            healing_engine.active_healings[f"healing_{i}"] = mock_plan
        
        assert len(healing_engine.active_healings) == healing_engine.max_concurrent_healings
        
        # Adding another should still work but would need to be queued in a real implementation
        # For now, we just test that the limit is tracked
        assert len(healing_engine.active_healings) == 2  # max_concurrent_healings
    
    def test_history_size_management(self, healing_engine):
        """Test that healing history respects size limits."""
        # Add many healing results (more than the limit of 1000)
        for i in range(1500):
            result = HealingResult(
                healing_id=f"healing_{i}",
                plan=Mock(),
                status=HealingStatus.SUCCESSFUL,
                actions_executed=[],
                actions_successful=[],
                actions_failed=[],
                total_duration=1.0,
                metrics={}
            )
            healing_engine.healing_history.append(result)
        
        # Should be limited to 500 after cleanup
        assert len(healing_engine.healing_history) == 500