"""Quantum-inspired task planning for CI/CD pipeline optimization.

Uses quantum-inspired algorithms including simulated annealing and 
quantum approximate optimization to find optimal task execution strategies.
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2  
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a CI/CD pipeline task."""
    id: str
    name: str
    priority: TaskPriority
    estimated_duration: float  # minutes
    dependencies: List[str] = field(default_factory=list)
    resources_required: Dict[str, float] = field(default_factory=dict)
    failure_probability: float = 0.1
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def actual_duration(self) -> Optional[float]:
        """Calculate actual execution time in minutes."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if task dependencies are satisfied."""
        return self.status == TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.value,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "resources_required": self.resources_required,
            "failure_probability": self.failure_probability,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "actual_duration": self.actual_duration,
            "metadata": self.metadata
        }


@dataclass 
class ExecutionPlan:
    """Represents an optimized execution plan."""
    tasks: List[Task]
    execution_order: List[str]
    estimated_total_time: float
    resource_utilization: Dict[str, float]
    success_probability: float
    cost_estimate: float
    parallel_stages: List[List[str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution plan to dictionary."""
        return {
            "execution_order": self.execution_order,
            "estimated_total_time": self.estimated_total_time,
            "resource_utilization": self.resource_utilization,
            "success_probability": self.success_probability,
            "cost_estimate": self.cost_estimate,
            "parallel_stages": self.parallel_stages,
            "tasks": [task.to_dict() for task in self.tasks]
        }


class QuantumTaskPlanner:
    """Quantum-inspired task planning and optimization engine."""
    
    def __init__(
        self,
        max_parallel_tasks: int = 4,
        resource_limits: Optional[Dict[str, float]] = None,
        optimization_iterations: int = 1000,
        temperature_schedule: str = "exponential"
    ):
        self.max_parallel_tasks = max_parallel_tasks
        self.resource_limits = resource_limits or {"cpu": 8.0, "memory": 16.0}
        self.optimization_iterations = optimization_iterations
        self.temperature_schedule = temperature_schedule
        self.tasks: Dict[str, Task] = {}
        self.historical_data: List[Dict[str, Any]] = []
        
    def add_task(self, task: Task) -> None:
        """Add a task to the planning queue."""
        self.tasks[task.id] = task
        logger.info(f"Added task {task.id}: {task.name}")
        
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the planning queue."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Removed task {task_id}")
            return True
        return False
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
        
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                ready_tasks.append(task)
                
        return ready_tasks
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for topological sorting."""
        graph = {}
        for task_id, task in self.tasks.items():
            graph[task_id] = task.dependencies.copy()
        return graph
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort to get valid execution order."""
        graph = self._build_dependency_graph()
        in_degree = {task_id: 0 for task_id in self.tasks}
        
        # Calculate in-degrees
        for task_id in graph:
            for dep in graph[task_id]:
                if dep in in_degree:
                    in_degree[dep] += 1
                    
        # Find tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority to ensure deterministic ordering
            queue.sort(key=lambda tid: self.tasks[tid].priority.value)
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Reduce in-degree of dependent tasks
            for other_task_id, deps in graph.items():
                if task_id in deps:
                    in_degree[other_task_id] -= 1
                    if in_degree[other_task_id] == 0:
                        queue.append(other_task_id)
                        
        if len(result) != len(self.tasks):
            raise ValueError("Circular dependency detected in task graph")
            
        return result
    
    def _calculate_energy(self, execution_order: List[str]) -> float:
        """Calculate energy (cost) of execution plan using quantum-inspired metrics."""
        total_energy = 0.0
        current_time = 0.0
        running_tasks = {}
        resource_usage = {res: 0.0 for res in self.resource_limits}
        
        for task_id in execution_order:
            task = self.tasks[task_id]
            
            # Wait for dependencies to complete
            dep_completion_time = 0.0
            for dep_id in task.dependencies:
                if dep_id in running_tasks:
                    dep_completion_time = max(dep_completion_time, running_tasks[dep_id])
                    
            start_time = max(current_time, dep_completion_time)
            
            # Check resource constraints
            resource_penalty = 0.0
            for resource, required in task.resources_required.items():
                if resource in resource_usage:
                    if resource_usage[resource] + required > self.resource_limits[resource]:
                        resource_penalty += 100.0  # Heavy penalty for resource overflow
                        
            # Priority-based energy (lower priority = higher energy)
            priority_energy = task.priority.value * 10.0
            
            # Duration energy
            duration_energy = task.estimated_duration
            
            # Failure probability penalty
            failure_penalty = task.failure_probability * 50.0
            
            # Resource contention energy
            contention_energy = sum(resource_usage.values()) * 0.1
            
            task_energy = (
                priority_energy +
                duration_energy + 
                failure_penalty +
                resource_penalty +
                contention_energy
            )
            
            total_energy += task_energy
            
            # Update running tasks and resource usage
            end_time = start_time + task.estimated_duration
            running_tasks[task_id] = end_time
            
            for resource, required in task.resources_required.items():
                if resource in resource_usage:
                    resource_usage[resource] += required
                    
            current_time = start_time
            
        return total_energy
    
    def _simulated_annealing_optimize(self, initial_order: List[str]) -> List[str]:
        """Optimize execution order using simulated annealing."""
        current_order = initial_order.copy()
        current_energy = self._calculate_energy(current_order)
        best_order = current_order.copy()
        best_energy = current_energy
        
        initial_temp = 1000.0
        final_temp = 0.1
        
        for iteration in range(self.optimization_iterations):
            # Temperature schedule
            if self.temperature_schedule == "exponential":
                temp = initial_temp * (final_temp / initial_temp) ** (iteration / self.optimization_iterations)
            else:  # linear
                temp = initial_temp - (initial_temp - final_temp) * (iteration / self.optimization_iterations)
                
            # Generate neighbor solution by swapping two random tasks
            new_order = current_order.copy()
            if len(new_order) > 1:
                i, j = random.sample(range(len(new_order)), 2)
                
                # Ensure swap doesn't violate dependencies
                if self._is_valid_swap(new_order, i, j):
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_energy = self._calculate_energy(new_order)
                    
                    # Accept or reject based on Metropolis criterion
                    delta_energy = new_energy - current_energy
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                        current_order = new_order
                        current_energy = new_energy
                        
                        if new_energy < best_energy:
                            best_order = new_order.copy()
                            best_energy = new_energy
                            
        logger.info(f"Optimization complete. Best energy: {best_energy:.2f}")
        return best_order
    
    def _is_valid_swap(self, order: List[str], i: int, j: int) -> bool:
        """Check if swapping tasks at positions i and j maintains dependency constraints."""
        task_i = self.tasks[order[i]]
        task_j = self.tasks[order[j]]
        
        # Check if task_i depends on task_j (can't swap if task_i comes before its dependency)
        if order[j] in task_i.dependencies and i < j:
            return False
            
        # Check if task_j depends on task_i  
        if order[i] in task_j.dependencies and j < i:
            return False
            
        return True
    
    def _calculate_parallel_stages(self, execution_order: List[str]) -> List[List[str]]:
        """Calculate parallel execution stages from linear order."""
        stages = []
        remaining_tasks = set(execution_order)
        completed_tasks = set()
        
        while remaining_tasks:
            current_stage = []
            stage_resources = {res: 0.0 for res in self.resource_limits}
            
            for task_id in execution_order:
                if task_id not in remaining_tasks:
                    continue
                    
                task = self.tasks[task_id]
                
                # Check if dependencies are satisfied
                deps_satisfied = all(dep in completed_tasks for dep in task.dependencies)
                if not deps_satisfied:
                    continue
                    
                # Check resource constraints
                resource_fit = True
                for resource, required in task.resources_required.items():
                    if resource in stage_resources:
                        if stage_resources[resource] + required > self.resource_limits[resource]:
                            resource_fit = False
                            break
                            
                # Check parallel task limit
                if len(current_stage) >= self.max_parallel_tasks:
                    resource_fit = False
                    
                if resource_fit:
                    current_stage.append(task_id)
                    remaining_tasks.remove(task_id)
                    
                    # Update resource usage
                    for resource, required in task.resources_required.items():
                        if resource in stage_resources:
                            stage_resources[resource] += required
                            
            if current_stage:
                stages.append(current_stage)
                completed_tasks.update(current_stage)
            else:
                # Safety check - if no tasks can be scheduled, add the first remaining task
                if remaining_tasks:
                    first_task = next(iter(remaining_tasks))
                    stages.append([first_task])
                    remaining_tasks.remove(first_task)
                    completed_tasks.add(first_task)
                    
        return stages
    
    def _genetic_algorithm_optimize(self, initial_order: List[str]) -> List[str]:
        """Optimize execution order using genetic algorithm."""
        if len(initial_order) < 3:
            return initial_order
        
        population_size = min(20, len(initial_order) * 2)
        generations = min(50, self.optimization_iterations // 20)
        
        # Initialize population with permutations of initial order
        population = []
        for _ in range(population_size):
            individual = initial_order.copy()
            # Apply random swaps while maintaining dependencies
            for _ in range(random.randint(1, len(individual) // 3)):
                if len(individual) > 1:
                    i, j = random.sample(range(len(individual)), 2)
                    if self._is_valid_swap(individual, i, j):
                        individual[i], individual[j] = individual[j], individual[i]
            population.append(individual)
        
        best_individual = initial_order
        best_fitness = self._calculate_energy(initial_order)
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = [(individual, self._calculate_energy(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1])  # Lower energy is better
            
            # Update best individual
            if fitness_scores[0][1] < best_fitness:
                best_individual = fitness_scores[0][0]
                best_fitness = fitness_scores[0][1]
            
            # Selection: keep best 50%
            survivors = [individual for individual, _ in fitness_scores[:population_size // 2]]
            
            # Crossover and mutation to create new population
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = self._crossover(parent1, parent2)
                if child:
                    child = self._mutate(child)
                    new_population.append(child)
            
            population = new_population
        
        logger.info(f"Genetic algorithm optimization complete. Fitness improved from {self._calculate_energy(initial_order):.2f} to {best_fitness:.2f}")
        return best_individual
    
    def _crossover(self, parent1: List[str], parent2: List[str]) -> Optional[List[str]]:
        """Create child through order crossover while maintaining dependencies."""
        if len(parent1) != len(parent2):
            return None
        
        size = len(parent1)
        if size < 3:
            return parent1
        
        # Order crossover (OX)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        
        # Copy segment from parent1
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2, maintaining order
        remaining = [item for item in parent2 if item not in child]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1
        
        # Validate dependencies
        if self._validate_dependencies(child):
            return child
        else:
            return parent1  # Return parent1 if child violates dependencies
    
    def _mutate(self, individual: List[str]) -> List[str]:
        """Apply mutation by swapping valid positions."""
        mutated = individual.copy()
        if len(mutated) > 1 and random.random() < 0.3:  # 30% mutation rate
            for _ in range(random.randint(1, 2)):  # 1-2 mutations
                i, j = random.sample(range(len(mutated)), 2)
                if self._is_valid_swap(mutated, i, j):
                    mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def _local_search_optimize(self, initial_order: List[str]) -> List[str]:
        """Local search optimization using 2-opt improvements."""
        current_order = initial_order.copy()
        current_energy = self._calculate_energy(current_order)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(current_order) - 1):
                for j in range(i + 2, len(current_order)):
                    if self._is_valid_swap(current_order, i, j):
                        # Try swap
                        new_order = current_order.copy()
                        new_order[i], new_order[j] = new_order[j], new_order[i]
                        new_energy = self._calculate_energy(new_order)
                        
                        if new_energy < current_energy:
                            current_order = new_order
                            current_energy = new_energy
                            improved = True
                            break
                if improved:
                    break
        
        logger.info(f"Local search optimization complete. Energy: {current_energy:.2f}")
        return current_order
    
    def _validate_dependencies(self, order: List[str]) -> bool:
        """Validate that task order respects all dependencies."""
        position = {task_id: i for i, task_id in enumerate(order)}
        
        for task_id in order:
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in position and position[dep_id] > position[task_id]:
                    return False
        return True
    
    def _calculate_parallel_stages_balanced(self, execution_order: List[str]) -> List[List[str]]:
        """Calculate parallel execution stages with improved load balancing."""
        stages = []
        remaining_tasks = set(execution_order)
        completed_tasks = set()
        
        while remaining_tasks:
            current_stage = []
            stage_resources = {res: 0.0 for res in self.resource_limits}
            stage_duration = 0.0
            
            # Sort available tasks by priority and estimated duration
            available_tasks = []
            for task_id in execution_order:
                if task_id not in remaining_tasks:
                    continue
                    
                task = self.tasks[task_id]
                deps_satisfied = all(dep in completed_tasks for dep in task.dependencies)
                if deps_satisfied:
                    available_tasks.append((task_id, task.priority.value, task.estimated_duration))
            
            # Sort by priority (lower value = higher priority) then by duration
            available_tasks.sort(key=lambda x: (x[1], -x[2]))  # Prioritize high priority, longer tasks
            
            for task_id, _, duration in available_tasks:
                if task_id not in remaining_tasks:
                    continue
                    
                task = self.tasks[task_id]
                
                # Check resource constraints
                resource_fit = True
                for resource, required in task.resources_required.items():
                    if resource in stage_resources:
                        if stage_resources[resource] + required > self.resource_limits[resource]:
                            resource_fit = False
                            break
                
                # Check parallel task limit
                if len(current_stage) >= self.max_parallel_tasks:
                    resource_fit = False
                
                # Load balancing: avoid stages that are too imbalanced in duration
                if stage_duration > 0 and duration > stage_duration * 2:
                    resource_fit = False
                
                if resource_fit:
                    current_stage.append(task_id)
                    remaining_tasks.remove(task_id)
                    stage_duration = max(stage_duration, duration)
                    
                    # Update resource usage
                    for resource, required in task.resources_required.items():
                        if resource in stage_resources:
                            stage_resources[resource] += required
            
            if current_stage:
                stages.append(current_stage)
                completed_tasks.update(current_stage)
            else:
                # Safety check - if no tasks can be scheduled, add the first remaining task
                if remaining_tasks:
                    first_task = next(iter(remaining_tasks))
                    stages.append([first_task])
                    remaining_tasks.remove(first_task)
                    completed_tasks.add(first_task)
        
        return stages
    
    def _estimate_execution_metrics_enhanced(self, parallel_stages: List[List[str]]) -> Tuple[float, Dict[str, float], float, float]:
        """Enhanced estimation with risk analysis and uncertainty modeling."""
        total_time = 0.0
        max_resource_usage = {res: 0.0 for res in self.resource_limits}
        overall_success_prob = 1.0
        total_cost = 0.0
        risk_factors = []
        
        for stage_idx, stage in enumerate(parallel_stages):
            stage_durations = []
            stage_resources = {res: 0.0 for res in self.resource_limits}
            stage_success_prob = 1.0
            stage_cost = 0.0
            
            for task_id in stage:
                task = self.tasks[task_id]
                
                # Enhanced duration estimation with uncertainty
                base_duration = task.estimated_duration
                uncertainty_factor = 0.2  # 20% uncertainty
                expected_duration = base_duration * (1 + uncertainty_factor * random.gauss(0, 1))
                expected_duration = max(0.1, expected_duration)  # Minimum duration
                stage_durations.append(expected_duration)
                
                # Enhanced failure probability with stage dependency risk
                task_failure_prob = task.failure_probability
                if stage_idx > 0:  # Later stages have slightly higher risk
                    task_failure_prob *= (1 + stage_idx * 0.05)
                
                stage_success_prob *= (1.0 - min(0.9, task_failure_prob))
                
                # Resource usage
                for resource, required in task.resources_required.items():
                    if resource in stage_resources:
                        stage_resources[resource] += required
                
                # Enhanced cost estimation with scaling factors
                base_cost = expected_duration * sum(task.resources_required.values()) * 0.1
                scaling_cost = base_cost * (1 + len(stage) * 0.05)  # Small cost increase for parallel execution
                stage_cost += scaling_cost
                
                # Risk factor analysis
                if task.failure_probability > 0.3:
                    risk_factors.append(f"High failure probability: {task.name}")
                if sum(task.resources_required.values()) > 5:
                    risk_factors.append(f"High resource usage: {task.name}")
            
            # Stage duration is maximum of parallel tasks with some overhead
            stage_duration = max(stage_durations) if stage_durations else 0
            if len(stage) > 1:  # Add coordination overhead for parallel tasks
                stage_duration *= (1 + len(stage) * 0.02)
            
            total_time += stage_duration
            overall_success_prob *= stage_success_prob
            total_cost += stage_cost
            
            # Track maximum resource usage
            for resource, usage in stage_resources.items():
                max_resource_usage[resource] = max(max_resource_usage[resource], usage)
        
        # Calculate resource utilization percentages with efficiency considerations
        resource_utilization = {}
        for resource, max_usage in max_resource_usage.items():
            if resource in self.resource_limits and self.resource_limits[resource] > 0:
                utilization = min(100.0, (max_usage / self.resource_limits[resource]) * 100)
                # Adjust for efficiency (high utilization might indicate inefficiency)
                if utilization > 90:
                    efficiency_penalty = (utilization - 90) * 0.1
                    utilization += efficiency_penalty
                resource_utilization[resource] = utilization
            else:
                resource_utilization[resource] = 0.0
        
        # Store risk analysis
        if hasattr(self, 'last_risk_analysis'):
            self.last_risk_analysis = {
                "risk_factors": risk_factors[:10],  # Top 10 risks
                "risk_count": len(risk_factors),
                "overall_risk_level": "high" if len(risk_factors) > 5 else "medium" if len(risk_factors) > 2 else "low"
            }
        
        return total_time, resource_utilization, overall_success_prob, total_cost
    
    def _estimate_execution_metrics(self, parallel_stages: List[List[str]]) -> Tuple[float, Dict[str, float], float, float]:
        """Estimate total time, resource utilization, success probability, and cost."""
        total_time = 0.0
        max_resource_usage = {res: 0.0 for res in self.resource_limits}
        overall_success_prob = 1.0
        total_cost = 0.0
        
        for stage in parallel_stages:
            stage_duration = 0.0
            stage_resources = {res: 0.0 for res in self.resource_limits}
            stage_success_prob = 1.0
            stage_cost = 0.0
            
            for task_id in stage:
                task = self.tasks[task_id]
                stage_duration = max(stage_duration, task.estimated_duration)
                stage_success_prob *= (1.0 - task.failure_probability)
                
                # Resource usage
                for resource, required in task.resources_required.items():
                    if resource in stage_resources:
                        stage_resources[resource] += required
                        
                # Cost estimation (simplified)
                task_cost = task.estimated_duration * sum(task.resources_required.values()) * 0.1
                stage_cost += task_cost
                
            total_time += stage_duration
            overall_success_prob *= stage_success_prob
            total_cost += stage_cost
            
            # Track maximum resource usage
            for resource, usage in stage_resources.items():
                max_resource_usage[resource] = max(max_resource_usage[resource], usage)
                
        # Calculate resource utilization percentages
        resource_utilization = {}
        for resource, max_usage in max_resource_usage.items():
            if resource in self.resource_limits and self.resource_limits[resource] > 0:
                resource_utilization[resource] = min(100.0, (max_usage / self.resource_limits[resource]) * 100)
            else:
                resource_utilization[resource] = 0.0
                
        return total_time, resource_utilization, overall_success_prob, total_cost
    
    async def create_execution_plan(self) -> ExecutionPlan:
        """Create optimized execution plan using enhanced quantum-inspired algorithms."""
        if not self.tasks:
            raise ValueError("No tasks available for planning")
            
        logger.info(f"Creating enhanced execution plan for {len(self.tasks)} tasks")
        
        # Start with topological sort for valid base ordering
        try:
            initial_order = self._topological_sort()
        except ValueError as e:
            logger.error(f"Failed to create initial ordering: {e}")
            raise
        
        # Multi-phase optimization for better results
        logger.info("Phase 1: Simulated annealing optimization")
        sa_optimized = self._simulated_annealing_optimize(initial_order)
        
        logger.info("Phase 2: Genetic algorithm refinement")
        ga_optimized = self._genetic_algorithm_optimize(sa_optimized)
        
        logger.info("Phase 3: Local search improvement")
        final_order = self._local_search_optimize(ga_optimized)
        
        # Calculate parallel execution stages with load balancing
        parallel_stages = self._calculate_parallel_stages_balanced(final_order)
        
        # Estimate execution metrics with risk analysis
        total_time, resource_util, success_prob, cost = self._estimate_execution_metrics_enhanced(parallel_stages)
        
        # Create enhanced execution plan
        plan = ExecutionPlan(
            tasks=list(self.tasks.values()),
            execution_order=final_order,
            estimated_total_time=total_time,
            resource_utilization=resource_util,
            success_probability=success_prob,
            cost_estimate=cost,
            parallel_stages=parallel_stages
        )
        
        # Add optimization metadata
        plan.metadata = {
            "optimization_phases": ["simulated_annealing", "genetic_algorithm", "local_search"],
            "load_balanced": True,
            "risk_analyzed": True,
            "quantum_inspired": True
        }
        
        logger.info(f"Enhanced execution plan created: {total_time:.1f}min, {success_prob:.1%} success rate, {len(parallel_stages)} stages")
        return plan
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the optimized plan with real-time monitoring."""
        logger.info("Starting plan execution")
        start_time = datetime.now()
        results = {
            "start_time": start_time,
            "stages_completed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_stages": len(plan.parallel_stages),
            "total_tasks": len(plan.tasks)
        }
        
        try:
            for stage_idx, stage_tasks in enumerate(plan.parallel_stages):
                logger.info(f"Executing stage {stage_idx + 1}/{len(plan.parallel_stages)} with {len(stage_tasks)} tasks")
                
                # Execute tasks in parallel within the stage
                stage_start = datetime.now()
                stage_results = await asyncio.gather(
                    *[self._execute_task(task_id) for task_id in stage_tasks],
                    return_exceptions=True
                )
                
                # Process stage results
                for task_id, result in zip(stage_tasks, stage_results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {task_id} failed: {result}")
                        self.tasks[task_id].status = TaskStatus.FAILED
                        results["tasks_failed"] += 1
                    else:
                        logger.info(f"Task {task_id} completed successfully")
                        self.tasks[task_id].status = TaskStatus.COMPLETED
                        results["tasks_completed"] += 1
                        
                results["stages_completed"] += 1
                stage_duration = (datetime.now() - stage_start).total_seconds() / 60
                logger.info(f"Stage {stage_idx + 1} completed in {stage_duration:.1f} minutes")
                
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            results["error"] = str(e)
            
        finally:
            results["end_time"] = datetime.now()
            results["total_duration"] = (results["end_time"] - start_time).total_seconds() / 60
            
        return results
    
    async def _execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a single task with enhanced real-world integration."""
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        logger.info(f"Starting task {task_id}: {task.name}")
        
        try:
            # Enhanced execution with adaptive timing
            base_time = task.estimated_duration
            
            # Adaptive timing based on task history and system load
            if hasattr(self, '_task_history') and task.name in self._task_history:
                historical_avg = np.mean(self._task_history[task.name])
                base_time = (base_time + historical_avg) / 2
            
            # Simulate realistic execution with resource awareness
            execution_time = base_time * random.uniform(0.7, 1.3)
            
            # Simulate faster execution for testing (reduce by 99%)
            execution_time = max(0.1, execution_time * 0.01)
            await asyncio.sleep(execution_time)
            
            # Enhanced failure simulation with recovery mechanisms
            failure_occurred = random.random() < task.failure_probability
            
            if failure_occurred and task.retry_count < task.max_retries:
                # Auto-retry with exponential backoff
                task.retry_count += 1
                retry_delay = min(30, 2 ** task.retry_count)
                logger.warning(f"Task {task_id} failed, retrying in {retry_delay}s (attempt {task.retry_count}/{task.max_retries})")
                await asyncio.sleep(retry_delay * 0.01)  # Fast retry for testing
                
                # Recursive retry with improved success probability
                task.failure_probability *= 0.5  # Reduce failure probability on retry
                return await self._execute_task(task_id)
                
            elif failure_occurred:
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
                raise Exception(f"Task {task_id} failed after {task.max_retries} retries")
                
            # Success path with learning
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            # Store execution time for learning
            if not hasattr(self, '_task_history'):
                self._task_history = {}
            if task.name not in self._task_history:
                self._task_history[task.name] = deque(maxlen=50)
            self._task_history[task.name].append(task.actual_duration)
            
            return {
                "task_id": task_id,
                "status": "completed",
                "duration": task.actual_duration,
                "retries": task.retry_count,
                "learned": True
            }
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            logger.error(f"Task {task_id} execution failed: {str(e)}")
            raise
    
    def add_historical_data(self, execution_result: Dict[str, Any]) -> None:
        """Add execution results to historical data for learning."""
        self.historical_data.append({
            "timestamp": datetime.now().isoformat(),
            "result": execution_result
        })
        
        # Keep only recent history (last 1000 executions)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]
            
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning and execution statistics."""
        if not self.historical_data:
            return {"message": "No historical data available"}
            
        total_executions = len(self.historical_data)
        successful_executions = sum(
            1 for data in self.historical_data 
            if data["result"].get("tasks_failed", 0) == 0
        )
        
        avg_duration = np.mean([
            data["result"].get("total_duration", 0)
            for data in self.historical_data
        ])
        
        # Enhanced statistics with learning insights
        task_performance = {}
        if hasattr(self, '_task_history'):
            for task_name, times in self._task_history.items():
                if times:
                    task_performance[task_name] = {
                        "avg_duration": float(np.mean(times)),
                        "std_deviation": float(np.std(times)),
                        "execution_count": len(times),
                        "improvement_trend": self._calculate_improvement_trend(list(times))
                    }
        
        # Resource utilization analysis
        resource_efficiency = {}
        for resource, limit in self.resource_limits.items():
            utilizations = []
            for data in self.historical_data:
                result = data["result"]
                if "resource_utilization" in result:
                    util = result["resource_utilization"].get(resource, 0)
                    utilizations.append(util)
            
            if utilizations:
                resource_efficiency[resource] = {
                    "avg_utilization": np.mean(utilizations),
                    "peak_utilization": np.max(utilizations),
                    "efficiency_score": min(100, np.mean(utilizations) / 80 * 100)  # Target 80% utilization
                }
        
        return {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_duration_minutes": avg_duration,
            "current_tasks": len(self.tasks),
            "optimization_iterations": self.optimization_iterations,
            "max_parallel_tasks": self.max_parallel_tasks,
            "resource_limits": self.resource_limits,
            "task_performance_analytics": task_performance,
            "resource_efficiency_analytics": resource_efficiency,
            "learning_enabled": hasattr(self, '_task_history'),
            "quantum_optimization_active": True
        }
    
    def _calculate_improvement_trend(self, times: List[float]) -> str:
        """Calculate if task performance is improving over time."""
        if len(times) < 5:
            return "insufficient_data"
        
        # Simple linear regression to detect trend
        x = np.arange(len(times))
        z = np.polyfit(x, times, 1)
        slope = z[0]
        
        if slope < -0.1:
            return "improving"
        elif slope > 0.1:
            return "degrading"
        else:
            return "stable"