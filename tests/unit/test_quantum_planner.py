"""Unit tests for quantum task planner."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from healing_guard.core.quantum_planner import (
    QuantumTaskPlanner, Task, TaskPriority, TaskStatus, ExecutionPlan
)


class TestQuantumTaskPlanner:
    """Test cases for QuantumTaskPlanner class."""
    
    @pytest.fixture
    def planner(self):
        """Create a QuantumTaskPlanner instance for testing."""
        return QuantumTaskPlanner(
            max_parallel_tasks=2,
            resource_limits={"cpu": 4.0, "memory": 8.0},
            optimization_iterations=100
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="task_1",
            name="Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=5.0,
            dependencies=[],
            resources_required={"cpu": 1.0, "memory": 2.0},
            failure_probability=0.1
        )
    
    def test_add_task(self, planner, sample_task):
        """Test adding a task to the planner."""
        planner.add_task(sample_task)
        
        assert sample_task.id in planner.tasks
        assert planner.tasks[sample_task.id] == sample_task
        assert len(planner.tasks) == 1
    
    def test_remove_task(self, planner, sample_task):
        """Test removing a task from the planner."""
        planner.add_task(sample_task)
        assert len(planner.tasks) == 1
        
        success = planner.remove_task(sample_task.id)
        assert success is True
        assert sample_task.id not in planner.tasks
        assert len(planner.tasks) == 0
        
        # Test removing non-existent task
        success = planner.remove_task("non_existent")
        assert success is False
    
    def test_get_task(self, planner, sample_task):
        """Test getting a task by ID."""
        planner.add_task(sample_task)
        
        retrieved_task = planner.get_task(sample_task.id)
        assert retrieved_task == sample_task
        
        # Test getting non-existent task
        retrieved_task = planner.get_task("non_existent")
        assert retrieved_task is None
    
    def test_get_ready_tasks(self, planner):
        """Test getting tasks that are ready to execute."""
        # Create tasks with dependencies
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0)
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0, dependencies=["task_1"])
        task3 = Task("task_3", "Task 3", TaskPriority.LOW, 2.0)
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.add_task(task3)
        
        ready_tasks = planner.get_ready_tasks()
        ready_task_ids = [task.id for task in ready_tasks]
        
        # Only task1 and task3 should be ready (no dependencies)
        assert "task_1" in ready_task_ids
        assert "task_3" in ready_task_ids
        assert "task_2" not in ready_task_ids
        
        # Complete task1 and check again
        task1.status = TaskStatus.COMPLETED
        ready_tasks = planner.get_ready_tasks()
        ready_task_ids = [task.id for task in ready_tasks]
        
        # Now task2 should also be ready
        assert "task_2" in ready_task_ids
    
    def test_topological_sort(self, planner):
        """Test topological sorting of tasks."""
        # Create a dependency chain: task3 -> task1 -> task2
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0, dependencies=["task_3"])
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0, dependencies=["task_1"])
        task3 = Task("task_3", "Task 3", TaskPriority.LOW, 2.0)
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.add_task(task3)
        
        execution_order = planner._topological_sort()
        
        # task3 should come before task1, task1 before task2
        assert execution_order.index("task_3") < execution_order.index("task_1")
        assert execution_order.index("task_1") < execution_order.index("task_2")
    
    def test_topological_sort_circular_dependency(self, planner):
        """Test detection of circular dependencies."""
        # Create circular dependency: task1 -> task2 -> task1
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0, dependencies=["task_2"])
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0, dependencies=["task_1"])
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            planner._topological_sort()
    
    def test_calculate_energy(self, planner):
        """Test energy calculation for execution order."""
        task1 = Task("task_1", "Task 1", TaskPriority.CRITICAL, 5.0, 
                     resources_required={"cpu": 1.0, "memory": 2.0})
        task2 = Task("task_2", "Task 2", TaskPriority.LOW, 3.0,
                     resources_required={"cpu": 0.5, "memory": 1.0})
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        execution_order = ["task_1", "task_2"]
        energy = planner._calculate_energy(execution_order)
        
        # Energy should be positive and finite
        assert energy > 0
        assert energy < float('inf')
        
        # Different order should give different energy
        execution_order_2 = ["task_2", "task_1"]
        energy_2 = planner._calculate_energy(execution_order_2)
        
        # Energies will be different due to priority differences
        assert energy != energy_2
    
    def test_is_valid_swap(self, planner):
        """Test validation of task swaps for dependencies."""
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0)
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0, dependencies=["task_1"])
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        order = ["task_1", "task_2"]
        
        # Valid swap (no dependencies violated)
        assert planner._is_valid_swap(order, 0, 1) is False  # task_2 depends on task_1
        
        # Test with independent tasks
        task3 = Task("task_3", "Task 3", TaskPriority.LOW, 2.0)
        planner.add_task(task3)
        
        order = ["task_1", "task_3"]
        assert planner._is_valid_swap(order, 0, 1) is True  # No dependencies
    
    @pytest.mark.asyncio
    async def test_create_execution_plan(self, planner):
        """Test creating an execution plan."""
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0,
                     resources_required={"cpu": 1.0, "memory": 2.0})
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0,
                     resources_required={"cpu": 0.5, "memory": 1.0})
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        plan = await planner.create_execution_plan()
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.tasks) == 2
        assert len(plan.execution_order) == 2
        assert plan.estimated_total_time > 0
        assert 0 <= plan.success_probability <= 1
        assert plan.cost_estimate >= 0
        assert len(plan.parallel_stages) > 0
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_empty(self, planner):
        """Test creating execution plan with no tasks."""
        with pytest.raises(ValueError, match="No tasks available for planning"):
            await planner.create_execution_plan()
    
    def test_calculate_parallel_stages(self, planner):
        """Test calculation of parallel execution stages."""
        # Create tasks with varying resource requirements
        task1 = Task("task_1", "Task 1", TaskPriority.HIGH, 5.0,
                     resources_required={"cpu": 2.0, "memory": 3.0})
        task2 = Task("task_2", "Task 2", TaskPriority.MEDIUM, 3.0,
                     resources_required={"cpu": 1.0, "memory": 2.0})
        task3 = Task("task_3", "Task 3", TaskPriority.LOW, 2.0,
                     resources_required={"cpu": 3.0, "memory": 4.0})  # Won't fit with others
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.add_task(task3)
        
        execution_order = ["task_1", "task_2", "task_3"]
        stages = planner._calculate_parallel_stages(execution_order)
        
        assert len(stages) > 0
        assert all(isinstance(stage, list) for stage in stages)
        
        # Check that all tasks are included
        all_tasks_in_stages = [task_id for stage in stages for task_id in stage]
        assert set(all_tasks_in_stages) == {"task_1", "task_2", "task_3"}
    
    @pytest.mark.asyncio
    async def test_execute_task(self, planner, sample_task):
        """Test individual task execution."""
        planner.add_task(sample_task)
        
        # Mock the task execution to be faster for testing
        with patch('asyncio.sleep', return_value=None):
            result = await planner._execute_task(sample_task.id)
            
            assert "task_id" in result
            assert result["task_id"] == sample_task.id
            assert "status" in result
            assert "duration" in result
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, planner):
        """Test task execution with high failure probability."""
        failing_task = Task(
            id="failing_task",
            name="Failing Task",
            priority=TaskPriority.HIGH,
            estimated_duration=1.0,
            failure_probability=1.0  # Always fails
        )
        
        planner.add_task(failing_task)
        
        with patch('asyncio.sleep', return_value=None):
            with pytest.raises(Exception, match="failed due to simulated failure"):
                await planner._execute_task(failing_task.id)
    
    def test_add_historical_data(self, planner):
        """Test adding historical execution data."""
        execution_result = {
            "total_duration": 10.5,
            "tasks_completed": 3,
            "tasks_failed": 1
        }
        
        planner.add_historical_data(execution_result)
        
        assert len(planner.historical_data) == 1
        assert "timestamp" in planner.historical_data[0]
        assert planner.historical_data[0]["result"] == execution_result
    
    def test_get_planning_statistics(self, planner):
        """Test getting planning statistics."""
        # Initially no historical data
        stats = planner.get_planning_statistics()
        assert "message" in stats
        
        # Add some historical data
        for i in range(3):
            execution_result = {
                "total_duration": 10.0 + i,
                "tasks_completed": 2,
                "tasks_failed": 0 if i < 2 else 1
            }
            planner.add_historical_data(execution_result)
        
        stats = planner.get_planning_statistics()
        
        assert stats["total_executions"] == 3
        assert stats["success_rate"] == 2/3  # 2 successful out of 3
        assert stats["average_duration_minutes"] > 0
        assert stats["current_tasks"] == 0
        assert stats["optimization_iterations"] == 100
        assert stats["max_parallel_tasks"] == 2
    
    def test_task_to_dict(self, sample_task):
        """Test task serialization to dictionary."""
        task_dict = sample_task.to_dict()
        
        required_fields = [
            "id", "name", "priority", "estimated_duration", "dependencies",
            "resources_required", "failure_probability", "retry_count",
            "max_retries", "status", "start_time", "end_time", "actual_duration", "metadata"
        ]
        
        for field in required_fields:
            assert field in task_dict
        
        assert task_dict["id"] == sample_task.id
        assert task_dict["name"] == sample_task.name
        assert task_dict["priority"] == sample_task.priority.value
    
    def test_task_properties(self, sample_task):
        """Test task properties and computed fields."""
        # Test is_ready property
        assert sample_task.is_ready is True
        
        sample_task.status = TaskStatus.RUNNING
        assert sample_task.is_ready is False
        
        # Test actual_duration property
        assert sample_task.actual_duration is None
        
        sample_task.start_time = datetime.now()
        sample_task.end_time = datetime.now()
        assert sample_task.actual_duration is not None
        assert sample_task.actual_duration >= 0