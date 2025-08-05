"""API routes for the Healing Guard system."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..core.quantum_planner import QuantumTaskPlanner, Task, TaskPriority, ExecutionPlan
from ..core.failure_detector import FailureDetector, FailureEvent, FailureType, SeverityLevel
from ..core.healing_engine import HealingEngine, HealingPlan, HealingResult
from ..monitoring.health import health_checker
from ..monitoring.metrics import metrics_collector
from .exceptions import (
    ValidationException,
    ResourceNotFoundException,
    QuantumPlannerException,
    HealingEngineException
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (in production, these would be dependency-injected)
quantum_planner = QuantumTaskPlanner()
failure_detector = FailureDetector()
healing_engine = HealingEngine(quantum_planner, failure_detector)


# Pydantic models for request/response

class TaskCreateRequest(BaseModel):
    """Request model for creating a task."""
    name: str = Field(..., description="Task name")
    priority: int = Field(..., ge=1, le=4, description="Priority level (1=CRITICAL, 4=LOW)")
    estimated_duration: float = Field(..., gt=0, description="Estimated duration in minutes")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    resources_required: Dict[str, float] = Field(default_factory=dict, description="Required resources")
    failure_probability: float = Field(0.1, ge=0, le=1, description="Failure probability")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskResponse(BaseModel):
    """Response model for task information."""
    id: str
    name: str
    priority: int
    estimated_duration: float
    dependencies: List[str]
    resources_required: Dict[str, float]
    failure_probability: float
    retry_count: int
    max_retries: int
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    actual_duration: Optional[float]
    metadata: Dict[str, Any]


class ExecutionPlanResponse(BaseModel):
    """Response model for execution plan."""
    execution_order: List[str]
    estimated_total_time: float
    resource_utilization: Dict[str, float]
    success_probability: float
    cost_estimate: float
    parallel_stages: List[List[str]]
    tasks: List[TaskResponse]


class FailureAnalysisRequest(BaseModel):
    """Request model for failure analysis."""
    job_id: str = Field(..., description="CI/CD job ID")
    repository: str = Field(..., description="Repository name")
    branch: str = Field(..., description="Branch name")
    commit_sha: str = Field(..., description="Commit SHA")
    logs: str = Field(..., description="Failure logs")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class FailureEventResponse(BaseModel):
    """Response model for failure event."""
    id: str
    timestamp: str
    job_id: str
    repository: str
    branch: str
    commit_sha: str
    failure_type: str
    severity: int
    confidence: float
    extracted_features: Dict[str, Any]
    matched_patterns: List[str]
    context: Dict[str, Any]
    remediation_suggestions: List[str]


class HealingPlanResponse(BaseModel):
    """Response model for healing plan."""
    id: str
    failure_event: FailureEventResponse
    estimated_total_time: float
    success_probability: float
    total_cost: float
    priority: int
    created_at: str
    actions: List[Dict[str, Any]]


class HealingResultResponse(BaseModel):
    """Response model for healing result."""
    healing_id: str
    plan_id: str
    status: str
    actions_executed: List[str]
    actions_successful: List[str]
    actions_failed: List[str]
    total_duration: float
    error_message: Optional[str]
    metrics: Dict[str, Any]
    rollback_performed: bool


# Quantum Planner endpoints

@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(task_request: TaskCreateRequest):
    """Create a new task for quantum planning."""
    try:
        task = Task(
            id=f"task_{datetime.now().timestamp()}",
            name=task_request.name,
            priority=TaskPriority(task_request.priority),
            estimated_duration=task_request.estimated_duration,
            dependencies=task_request.dependencies,
            resources_required=task_request.resources_required,
            failure_probability=task_request.failure_probability,
            max_retries=task_request.max_retries,
            metadata=task_request.metadata
        )
        
        quantum_planner.add_task(task)
        
        return TaskResponse(**task.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise QuantumPlannerException(f"Failed to create task: {e}")


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks():
    """List all tasks in the quantum planner."""
    try:
        tasks = [TaskResponse(**task.to_dict()) for task in quantum_planner.tasks.values()]
        return tasks
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise QuantumPlannerException(f"Failed to list tasks: {e}")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    task = quantum_planner.get_task(task_id)
    if not task:
        raise ResourceNotFoundException(f"Task {task_id} not found")
    
    return TaskResponse(**task.to_dict())


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str):
    """Delete a task from the quantum planner."""
    if not quantum_planner.remove_task(task_id):
        raise ResourceNotFoundException(f"Task {task_id} not found")


@router.post("/planning/execute", response_model=ExecutionPlanResponse)
async def create_execution_plan():
    """Create an optimized execution plan using quantum algorithms."""
    try:
        plan = await quantum_planner.create_execution_plan()
        
        return ExecutionPlanResponse(
            execution_order=plan.execution_order,
            estimated_total_time=plan.estimated_total_time,
            resource_utilization=plan.resource_utilization,
            success_probability=plan.success_probability,
            cost_estimate=plan.cost_estimate,
            parallel_stages=plan.parallel_stages,
            tasks=[TaskResponse(**task.to_dict()) for task in plan.tasks]
        )
        
    except Exception as e:
        logger.error(f"Failed to create execution plan: {e}")
        raise QuantumPlannerException(f"Failed to create execution plan: {e}")


@router.post("/planning/execute/run")
async def execute_plan(background_tasks: BackgroundTasks):
    """Execute the current quantum-optimized plan."""
    try:
        plan = await quantum_planner.create_execution_plan()
        
        # Execute in background
        background_tasks.add_task(quantum_planner.execute_plan, plan)
        
        return {
            "message": "Execution started",
            "plan_id": plan.execution_order[0] if plan.execution_order else "empty",
            "estimated_duration": plan.estimated_total_time
        }
        
    except Exception as e:
        logger.error(f"Failed to execute plan: {e}")
        raise QuantumPlannerException(f"Failed to execute plan: {e}")


@router.get("/planning/statistics")
async def get_planning_statistics():
    """Get quantum planner statistics and metrics."""
    try:
        return quantum_planner.get_planning_statistics()
    except Exception as e:
        logger.error(f"Failed to get planning statistics: {e}")
        raise QuantumPlannerException(f"Failed to get planning statistics: {e}")


# Failure Detection endpoints

@router.post("/failures/analyze", response_model=FailureEventResponse)
async def analyze_failure(analysis_request: FailureAnalysisRequest):
    """Analyze a pipeline failure and classify its type."""
    try:
        failure_event = await failure_detector.detect_failure(
            job_id=analysis_request.job_id,
            repository=analysis_request.repository,
            branch=analysis_request.branch,
            commit_sha=analysis_request.commit_sha,
            logs=analysis_request.logs,
            context=analysis_request.context
        )
        
        if not failure_event:
            raise HTTPException(status_code=400, detail="No failure detected in provided logs")
        
        return FailureEventResponse(**failure_event.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to analyze failure: {e}")
        raise ValidationException(f"Failed to analyze failure: {e}")


@router.get("/failures", response_model=List[FailureEventResponse])
async def list_failures(
    limit: int = Query(50, ge=1, le=1000, description="Number of failures to return"),
    offset: int = Query(0, ge=0, description="Number of failures to skip"),
    failure_type: Optional[str] = Query(None, description="Filter by failure type"),
    severity: Optional[int] = Query(None, ge=1, le=4, description="Filter by severity level")
):
    """List recent failure events with optional filtering."""
    try:
        failures = failure_detector.failure_history
        
        # Apply filters
        if failure_type:
            failures = [f for f in failures if f.failure_type.value == failure_type]
        if severity:
            failures = [f for f in failures if f.severity.value == severity]
        
        # Apply pagination
        failures = failures[offset:offset+limit]
        
        return [FailureEventResponse(**failure.to_dict()) for failure in failures]
        
    except Exception as e:
        logger.error(f"Failed to list failures: {e}")
        raise ValidationException(f"Failed to list failures: {e}")


@router.get("/failures/statistics")
async def get_failure_statistics():
    """Get failure detection statistics and metrics."""
    try:
        return failure_detector.get_failure_statistics()
    except Exception as e:
        logger.error(f"Failed to get failure statistics: {e}")
        raise ValidationException(f"Failed to get failure statistics: {e}")


@router.get("/failures/trends")
async def get_failure_trends(days: int = Query(7, ge=1, le=30, description="Number of days to analyze")):
    """Get failure trends over specified time period."""
    try:
        return failure_detector.get_failure_trends(days)
    except Exception as e:
        logger.error(f"Failed to get failure trends: {e}")
        raise ValidationException(f"Failed to get failure trends: {e}")


# Healing Engine endpoints

@router.post("/healing/plan", response_model=HealingPlanResponse)
async def create_healing_plan(analysis_request: FailureAnalysisRequest):
    """Create a healing plan for a failure."""
    try:
        # First analyze the failure
        failure_event = await failure_detector.detect_failure(
            job_id=analysis_request.job_id,
            repository=analysis_request.repository,
            branch=analysis_request.branch,
            commit_sha=analysis_request.commit_sha,
            logs=analysis_request.logs,
            context=analysis_request.context
        )
        
        if not failure_event:
            raise HTTPException(status_code=400, detail="No failure detected in provided logs")
        
        # Create healing plan
        healing_plan = await healing_engine.create_healing_plan(failure_event)
        
        return HealingPlanResponse(**healing_plan.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to create healing plan: {e}")
        raise HealingEngineException(f"Failed to create healing plan: {e}")


@router.post("/healing/execute", response_model=HealingResultResponse)
async def execute_healing_plan(analysis_request: FailureAnalysisRequest):
    """Analyze failure and execute healing plan in one step."""
    try:
        # First analyze the failure
        failure_event = await failure_detector.detect_failure(
            job_id=analysis_request.job_id,
            repository=analysis_request.repository,
            branch=analysis_request.branch,
            commit_sha=analysis_request.commit_sha,
            logs=analysis_request.logs,
            context=analysis_request.context
        )
        
        if not failure_event:
            raise HTTPException(status_code=400, detail="No failure detected in provided logs")
        
        # Execute complete healing process
        healing_result = await healing_engine.heal_failure(failure_event)
        
        return HealingResultResponse(**healing_result.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to execute healing: {e}")
        raise HealingEngineException(f"Failed to execute healing: {e}")


@router.get("/healing/statistics")
async def get_healing_statistics():
    """Get healing engine statistics and metrics."""
    try:
        return healing_engine.get_healing_statistics()
    except Exception as e:
        logger.error(f"Failed to get healing statistics: {e}")
        raise HealingEngineException(f"Failed to get healing statistics: {e}")


@router.get("/healing/active")
async def get_active_healings():
    """Get currently active healing operations."""
    try:
        return {
            "active_healings": len(healing_engine.active_healings),
            "healings": [
                {"id": healing_id, "plan_id": plan.id, "created_at": plan.created_at.isoformat()}
                for healing_id, plan in healing_engine.active_healings.items()
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get active healings: {e}")
        raise HealingEngineException(f"Failed to get active healings: {e}")


# System endpoints

@router.get("/system/status")
async def get_system_status():
    """Get overall system status and health."""
    try:
        health_status = await health_checker.get_comprehensive_health()
        
        return {
            "status": health_status.status,
            "timestamp": health_status.timestamp.isoformat(),
            "version": health_status.version,
            "uptime": health_status.uptime,
            "components": {
                "quantum_planner": {
                    "tasks": len(quantum_planner.tasks),
                    "optimization_iterations": quantum_planner.optimization_iterations
                },
                "failure_detector": {
                    "patterns": len(failure_detector.patterns),
                    "history_size": len(failure_detector.failure_history)
                },
                "healing_engine": {
                    "active_healings": len(healing_engine.active_healings),
                    "strategies": len(healing_engine.strategy_registry)
                }
            },
            "health_checks": health_status.checks
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


@router.get("/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics."""
    try:
        return {
            "quantum_planner": quantum_planner.get_planning_statistics(),
            "failure_detector": failure_detector.get_failure_statistics(),
            "healing_engine": healing_engine.get_healing_statistics(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {e}")