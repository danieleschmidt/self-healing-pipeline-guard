"""Self-healing engine that orchestrates failure remediation.

Combines quantum-inspired planning with intelligent remediation strategies
to automatically fix CI/CD pipeline failures.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from .quantum_planner import QuantumTaskPlanner, Task, TaskPriority, TaskStatus
from .failure_detector import FailureDetector, FailureEvent, FailureType, SeverityLevel

logger = logging.getLogger(__name__)


class HealingStrategy(Enum):
    """Healing strategy types."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    INCREASE_RESOURCES = "increase_resources"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    RESTART_SERVICES = "restart_services"
    UPDATE_DEPENDENCIES = "update_dependencies"
    SCALE_HORIZONTALLY = "scale_horizontally"
    ISOLATE_ENVIRONMENT = "isolate_environment"
    MOCK_EXTERNAL_SERVICES = "mock_external_services"
    OPTIMIZE_CONFIGURATION = "optimize_configuration"


class HealingStatus(Enum):
    """Healing operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class HealingAction:
    """Represents a specific healing action."""
    id: str
    strategy: HealingStrategy
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 5.0  # minutes
    success_probability: float = 0.8
    cost_estimate: float = 1.0
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    rollback_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "description": self.description,
            "parameters": self.parameters,
            "estimated_duration": self.estimated_duration,
            "success_probability": self.success_probability,
            "cost_estimate": self.cost_estimate,
            "prerequisites": self.prerequisites,
            "side_effects": self.side_effects,
            "rollback_action": self.rollback_action
        }


@dataclass
class HealingPlan:
    """Represents a complete healing plan."""
    id: str
    failure_event: FailureEvent
    actions: List[HealingAction]
    estimated_total_time: float
    success_probability: float
    total_cost: float
    priority: int
    created_at: datetime
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "id": self.id,
            "failure_event": self.failure_event.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "estimated_total_time": self.estimated_total_time,
            "success_probability": self.success_probability,
            "total_cost": self.total_cost,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "dependencies": self.dependencies
        }


@dataclass
class HealingResult:
    """Results of a healing operation."""
    healing_id: str
    plan: HealingPlan
    status: HealingStatus
    actions_executed: List[str]
    actions_successful: List[str]
    actions_failed: List[str]
    total_duration: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_performed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "healing_id": self.healing_id,
            "plan_id": self.plan.id,
            "status": self.status.value,
            "actions_executed": self.actions_executed,
            "actions_successful": self.actions_successful,
            "actions_failed": self.actions_failed,
            "total_duration": self.total_duration,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "rollback_performed": self.rollback_performed
        }


class HealingEngine:
    """Main healing engine that orchestrates failure remediation."""
    
    def __init__(
        self,
        quantum_planner: Optional[QuantumTaskPlanner] = None,
        failure_detector: Optional[FailureDetector] = None,
        max_concurrent_healings: int = 3,
        healing_timeout: int = 30  # minutes
    ):
        self.quantum_planner = quantum_planner or QuantumTaskPlanner()
        self.failure_detector = failure_detector or FailureDetector()
        self.max_concurrent_healings = max_concurrent_healings
        self.healing_timeout = healing_timeout
        
        self.active_healings: Dict[str, HealingPlan] = {}
        self.healing_history: List[HealingResult] = []
        self.strategy_registry: Dict[HealingStrategy, Callable] = {}
        self.custom_actions: Dict[str, HealingAction] = {}
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self) -> None:
        """Initialize default healing strategies."""
        
        async def retry_with_backoff(action: HealingAction, context: Dict[str, Any]) -> Dict[str, Any]:
            """Retry failed operation with exponential backoff."""
            max_retries = action.parameters.get("max_retries", 3)
            base_delay = action.parameters.get("base_delay", 1.0)
            
            for attempt in range(max_retries):
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                await asyncio.sleep(delay)
                
                # Simulate retry logic - in real implementation, this would re-run the failed task
                success_rate = 0.7 ** attempt  # Decreasing success rate
                if await self._simulate_action_execution(success_rate):
                    return {"status": "success", "attempts": attempt + 1}
                    
            return {"status": "failed", "attempts": max_retries}
        
        async def increase_resources(action: HealingAction, context: Dict[str, Any]) -> Dict[str, Any]:
            """Increase resource allocation for failing jobs."""
            cpu_increase = action.parameters.get("cpu_increase", 2.0)
            memory_increase = action.parameters.get("memory_increase", 4.0)
            
            logger.info(f"Increasing resources: CPU +{cpu_increase}, Memory +{memory_increase}GB")
            
            # Simulate resource increase
            await asyncio.sleep(2.0)
            
            return {
                "status": "success",
                "new_cpu": cpu_increase,
                "new_memory": memory_increase,
                "message": "Resources increased successfully"
            }
        
        async def clear_cache(action: HealingAction, context: Dict[str, Any]) -> Dict[str, Any]:
            """Clear various types of caches."""
            cache_types = action.parameters.get("cache_types", ["npm", "pip", "docker"])
            
            logger.info(f"Clearing caches: {cache_types}")
            
            results = {}
            for cache_type in cache_types:
                await asyncio.sleep(0.5)  # Simulate cache clearing
                results[cache_type] = "cleared"
                
            return {"status": "success", "caches_cleared": results}
        
        async def restart_services(action: HealingAction, context: Dict[str, Any]) -> Dict[str, Any]:
            """Restart related services."""
            services = action.parameters.get("services", ["database", "redis", "api"])
            
            logger.info(f"Restarting services: {services}")
            
            results = {}
            for service in services:
                await asyncio.sleep(1.0)  # Simulate service restart
                results[service] = "restarted"
                
            return {"status": "success", "services_restarted": results}
        
        async def update_dependencies(action: HealingAction, context: Dict[str, Any]) -> Dict[str, Any]:
            """Update project dependencies."""
            package_manager = action.parameters.get("package_manager", "npm")
            update_type = action.parameters.get("update_type", "patch")
            
            logger.info(f"Updating dependencies using {package_manager} ({update_type} updates)")
            
            await asyncio.sleep(10.0)  # Simulate dependency update
            
            return {
                "status": "success",
                "package_manager": package_manager,
                "update_type": update_type,
                "packages_updated": action.parameters.get("packages_updated", 5)
            }
        
        # Register strategies
        self.strategy_registry = {
            HealingStrategy.RETRY_WITH_BACKOFF: retry_with_backoff,
            HealingStrategy.INCREASE_RESOURCES: increase_resources,
            HealingStrategy.CLEAR_CACHE: clear_cache,
            HealingStrategy.RESTART_SERVICES: restart_services,
            HealingStrategy.UPDATE_DEPENDENCIES: update_dependencies
        }
        
    async def _simulate_action_execution(self, success_probability: float) -> bool:
        """Simulate action execution for testing."""
        import random
        return random.random() < success_probability
        
    def add_custom_action(self, action: HealingAction) -> None:
        """Add a custom healing action."""
        self.custom_actions[action.id] = action
        logger.info(f"Added custom healing action: {action.id}")
        
    def _create_healing_actions(self, failure_event: FailureEvent) -> List[HealingAction]:
        """Create healing actions based on failure type and suggestions."""
        actions = []
        
        # Map remediation suggestions to healing actions
        strategy_mapping = {
            "retry_with_isolation": HealingStrategy.RETRY_WITH_BACKOFF,
            "retry_with_backoff": HealingStrategy.RETRY_WITH_BACKOFF,
            "increase_memory": HealingStrategy.INCREASE_RESOURCES,
            "increase_resources": HealingStrategy.INCREASE_RESOURCES,
            "scale_resources": HealingStrategy.SCALE_HORIZONTALLY,
            "clear_cache": HealingStrategy.CLEAR_CACHE,
            "update_dependencies": HealingStrategy.UPDATE_DEPENDENCIES,
            "restart_services": HealingStrategy.RESTART_SERVICES,
            "optimize_config": HealingStrategy.OPTIMIZE_CONFIGURATION
        }
        
        # Create actions from failure event suggestions
        for suggestion in failure_event.remediation_suggestions:
            if suggestion in strategy_mapping:
                strategy = strategy_mapping[suggestion]
                action = self._create_action_for_strategy(strategy, failure_event, suggestion)
                if action:
                    actions.append(action)
                    
        # Add default actions based on failure type
        default_actions = self._get_default_actions_for_failure_type(failure_event.failure_type)
        actions.extend(default_actions)
        
        # Remove duplicates based on strategy
        unique_actions = {}
        for action in actions:
            if action.strategy not in unique_actions:
                unique_actions[action.strategy] = action
                
        return list(unique_actions.values())
    
    def _create_action_for_strategy(
        self,
        strategy: HealingStrategy,
        failure_event: FailureEvent,
        suggestion: str
    ) -> Optional[HealingAction]:
        """Create a healing action for a specific strategy."""
        action_id = f"{strategy.value}_{uuid.uuid4().hex[:8]}"
        
        # Configure action based on strategy and failure context
        if strategy == HealingStrategy.RETRY_WITH_BACKOFF:
            return HealingAction(
                id=action_id,
                strategy=strategy,
                description=f"Retry failed operation with exponential backoff",
                parameters={
                    "max_retries": 3 if failure_event.severity == SeverityLevel.LOW else 5,
                    "base_delay": 1.0,
                    "max_delay": 60.0
                },
                estimated_duration=3.0,
                success_probability=0.75
            )
            
        elif strategy == HealingStrategy.INCREASE_RESOURCES:
            multiplier = 2.0 if failure_event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 1.5
            return HealingAction(
                id=action_id,
                strategy=strategy,
                description=f"Increase resource allocation by {multiplier}x",
                parameters={
                    "cpu_increase": multiplier,
                    "memory_increase": multiplier * 2.0
                },
                estimated_duration=2.0,
                success_probability=0.85,
                cost_estimate=multiplier
            )
            
        elif strategy == HealingStrategy.CLEAR_CACHE:
            cache_types = []
            if "npm" in failure_event.raw_logs.lower():
                cache_types.append("npm")
            if "pip" in failure_event.raw_logs.lower():
                cache_types.append("pip")
            if "docker" in failure_event.raw_logs.lower():
                cache_types.append("docker")
            if not cache_types:
                cache_types = ["build", "dependency"]
                
            return HealingAction(
                id=action_id,
                strategy=strategy,
                description=f"Clear caches: {', '.join(cache_types)}",
                parameters={"cache_types": cache_types},
                estimated_duration=1.5,
                success_probability=0.70
            )
            
        elif strategy == HealingStrategy.UPDATE_DEPENDENCIES:
            return HealingAction(
                id=action_id,
                strategy=strategy,
                description="Update project dependencies to resolve conflicts",
                parameters={
                    "package_manager": "auto-detect",
                    "update_type": "patch"
                },
                estimated_duration=8.0,
                success_probability=0.65,
                cost_estimate=2.0
            )
            
        elif strategy == HealingStrategy.RESTART_SERVICES:
            return HealingAction(
                id=action_id,
                strategy=strategy,
                description="Restart related infrastructure services",
                parameters={"services": ["database", "cache", "message_queue"]},
                estimated_duration=3.0,
                success_probability=0.80,
                side_effects=["temporary_service_downtime"]
            )
            
        return None
    
    def _get_default_actions_for_failure_type(self, failure_type: FailureType) -> List[HealingAction]:
        """Get default actions for specific failure types."""
        actions = []
        
        if failure_type == FailureType.FLAKY_TEST:
            actions.append(HealingAction(
                id=f"flaky_retry_{uuid.uuid4().hex[:8]}",
                strategy=HealingStrategy.RETRY_WITH_BACKOFF,
                description="Retry flaky test with isolation",
                parameters={"max_retries": 2, "isolation": True},
                estimated_duration=2.0,
                success_probability=0.85
            ))
            
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            actions.append(HealingAction(
                id=f"resource_scale_{uuid.uuid4().hex[:8]}",
                strategy=HealingStrategy.INCREASE_RESOURCES,
                description="Scale up resources for resource-exhausted job",
                parameters={"cpu_increase": 2.0, "memory_increase": 4.0},
                estimated_duration=2.0,
                success_probability=0.90,
                cost_estimate=2.0
            ))
            
        elif failure_type == FailureType.DEPENDENCY_FAILURE:
            actions.extend([
                HealingAction(
                    id=f"dep_cache_clear_{uuid.uuid4().hex[:8]}",
                    strategy=HealingStrategy.CLEAR_CACHE,
                    description="Clear dependency caches",
                    parameters={"cache_types": ["npm", "pip", "maven"]},
                    estimated_duration=1.0,
                    success_probability=0.70
                ),
                HealingAction(
                    id=f"dep_update_{uuid.uuid4().hex[:8]}",
                    strategy=HealingStrategy.UPDATE_DEPENDENCIES,
                    description="Update dependencies to resolve conflicts",
                    parameters={"update_type": "patch"},
                    estimated_duration=5.0,
                    success_probability=0.60,
                    cost_estimate=1.5
                )
            ])
            
        return actions
    
    async def create_healing_plan(self, failure_event: FailureEvent) -> HealingPlan:
        """Create an optimized healing plan for a failure event."""
        logger.info(f"Creating healing plan for failure {failure_event.id}")
        
        # Create healing actions
        actions = self._create_healing_actions(failure_event)
        
        if not actions:
            raise ValueError(f"No healing actions available for failure type {failure_event.failure_type}")
            
        # Convert actions to quantum planner tasks
        planner_tasks = []
        for action in actions:
            task = Task(
                id=action.id,
                name=action.description,
                priority=self._map_severity_to_priority(failure_event.severity),
                estimated_duration=action.estimated_duration,
                resources_required={"cpu": 1.0, "memory": 1.0},
                failure_probability=1.0 - action.success_probability
            )
            planner_tasks.append(task)
            self.quantum_planner.add_task(task)
            
        # Create execution plan using quantum optimizer
        execution_plan = await self.quantum_planner.create_execution_plan()
        
        # Calculate healing plan metrics
        total_time = execution_plan.estimated_total_time
        success_probability = execution_plan.success_probability
        total_cost = sum(action.cost_estimate for action in actions)
        
        # Determine priority based on failure severity and impact
        priority = self._calculate_healing_priority(failure_event, total_cost, success_probability)
        
        healing_plan = HealingPlan(
            id=f"healing_{uuid.uuid4().hex[:8]}",
            failure_event=failure_event,
            actions=actions,
            estimated_total_time=total_time,
            success_probability=success_probability,
            total_cost=total_cost,
            priority=priority,
            created_at=datetime.now()
        )
        
        logger.info(
            f"Created healing plan {healing_plan.id}: "
            f"{len(actions)} actions, {total_time:.1f}min, "
            f"{success_probability:.1%} success rate"
        )
        
        return healing_plan
    
    def _map_severity_to_priority(self, severity: SeverityLevel) -> TaskPriority:
        """Map failure severity to task priority."""
        mapping = {
            SeverityLevel.CRITICAL: TaskPriority.CRITICAL,
            SeverityLevel.HIGH: TaskPriority.HIGH,
            SeverityLevel.MEDIUM: TaskPriority.MEDIUM,
            SeverityLevel.LOW: TaskPriority.LOW
        }
        return mapping.get(severity, TaskPriority.MEDIUM)
    
    def _calculate_healing_priority(
        self,
        failure_event: FailureEvent,
        total_cost: float,
        success_probability: float
    ) -> int:
        """Calculate healing priority score (lower = higher priority)."""
        base_priority = failure_event.severity.value * 10
        
        # Adjust for success probability
        probability_factor = (1.0 - success_probability) * 5
        
        # Adjust for cost efficiency
        cost_factor = min(5.0, total_cost / 2.0)
        
        # Adjust for branch importance
        branch_factor = 0
        if failure_event.branch in ["main", "master", "production"]:
            branch_factor = -10  # Higher priority for important branches
            
        total_priority = base_priority + probability_factor + cost_factor + branch_factor
        return max(1, int(total_priority))
    
    async def execute_healing_plan(self, plan: HealingPlan) -> HealingResult:
        """Execute a healing plan."""
        healing_id = f"exec_{uuid.uuid4().hex[:8]}"
        logger.info(f"Executing healing plan {plan.id} as {healing_id}")
        
        start_time = datetime.now()
        self.active_healings[healing_id] = plan
        
        result = HealingResult(
            healing_id=healing_id,
            plan=plan,
            status=HealingStatus.IN_PROGRESS,
            actions_executed=[],
            actions_successful=[],
            actions_failed=[],
            total_duration=0.0
        )
        
        try:
            # Execute actions according to quantum-optimized plan
            for action in plan.actions:
                logger.info(f"Executing action {action.id}: {action.description}")
                result.actions_executed.append(action.id)
                
                # Check if strategy is registered
                if action.strategy in self.strategy_registry:
                    strategy_func = self.strategy_registry[action.strategy]
                    
                    try:
                        # Execute the healing action
                        action_result = await asyncio.wait_for(
                            strategy_func(action, {"failure_event": plan.failure_event}),
                            timeout=self.healing_timeout * 60
                        )
                        
                        if action_result.get("status") == "success":
                            result.actions_successful.append(action.id)
                            logger.info(f"Action {action.id} completed successfully")
                        else:
                            result.actions_failed.append(action.id)
                            logger.warning(f"Action {action.id} failed: {action_result}")
                            
                    except asyncio.TimeoutError:
                        result.actions_failed.append(action.id)
                        logger.error(f"Action {action.id} timed out")
                        
                    except Exception as e:
                        result.actions_failed.append(action.id)
                        logger.error(f"Action {action.id} failed with exception: {e}")
                        
                else:
                    result.actions_failed.append(action.id)
                    logger.warning(f"No strategy implementation for {action.strategy}")
                    
            # Determine overall healing status
            if len(result.actions_successful) == len(plan.actions):
                result.status = HealingStatus.SUCCESSFUL
            elif len(result.actions_successful) > 0:
                result.status = HealingStatus.PARTIAL  
            else:
                result.status = HealingStatus.FAILED
                
        except Exception as e:
            result.status = HealingStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Healing plan execution failed: {e}")
            
        finally:
            # Clean up
            if healing_id in self.active_healings:
                del self.active_healings[healing_id]
                
            # Calculate metrics
            end_time = datetime.now()
            result.total_duration = (end_time - start_time).total_seconds() / 60
            
            result.metrics = {
                "success_rate": len(result.actions_successful) / len(plan.actions) if plan.actions else 0,
                "failure_rate": len(result.actions_failed) / len(plan.actions) if plan.actions else 0,
                "execution_efficiency": min(1.0, plan.estimated_total_time / result.total_duration) if result.total_duration > 0 else 0
            }
            
            # Add to history
            self.healing_history.append(result)
            
            # Keep only recent history
            if len(self.healing_history) > 1000:
                self.healing_history = self.healing_history[-500:]
                
        logger.info(
            f"Healing {healing_id} completed: {result.status.value} "
            f"({len(result.actions_successful)}/{len(plan.actions)} actions succeeded)"
        )
        
        return result
    
    async def heal_failure(self, failure_event: FailureEvent) -> HealingResult:
        """Complete healing process: plan creation and execution."""
        logger.info(f"Starting healing process for failure {failure_event.id}")
        
        # Create healing plan
        plan = await self.create_healing_plan(failure_event)
        
        # Execute healing plan
        result = await self.execute_healing_plan(plan)
        
        return result
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get healing engine statistics."""
        if not self.healing_history:
            return {"message": "No healing history available"}
            
        total_healings = len(self.healing_history)
        successful_healings = sum(
            1 for result in self.healing_history
            if result.status == HealingStatus.SUCCESSFUL
        )
        
        partial_healings = sum(
            1 for result in self.healing_history
            if result.status == HealingStatus.PARTIAL
        )
        
        # Calculate average metrics
        avg_duration = sum(result.total_duration for result in self.healing_history) / total_healings
        avg_success_rate = sum(
            result.metrics.get("success_rate", 0) for result in self.healing_history
        ) / total_healings
        
        # Strategy effectiveness
        strategy_stats = {}
        for result in self.healing_history:
            for action in result.plan.actions:
                strategy = action.strategy.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"total": 0, "successful": 0}
                    
                strategy_stats[strategy]["total"] += 1
                if action.id in result.actions_successful:
                    strategy_stats[strategy]["successful"] += 1
                    
        # Calculate strategy success rates
        for strategy, stats in strategy_stats.items():
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            
        return {
            "total_healings": total_healings,
            "successful_healings": successful_healings,
            "partial_healings": partial_healings,
            "failed_healings": total_healings - successful_healings - partial_healings,
            "overall_success_rate": successful_healings / total_healings,
            "partial_success_rate": partial_healings / total_healings,
            "average_duration_minutes": avg_duration,
            "average_action_success_rate": avg_success_rate,
            "active_healings": len(self.active_healings),
            "strategy_effectiveness": strategy_stats,
            "registered_strategies": len(self.strategy_registry),
            "custom_actions": len(self.custom_actions)
        }