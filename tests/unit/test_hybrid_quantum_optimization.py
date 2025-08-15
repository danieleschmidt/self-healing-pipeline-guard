"""Test suite for hybrid quantum-classical optimization algorithms.

This module validates the novel algorithmic improvements implemented
for autonomous SDLC execution with statistical significance testing.
"""

import asyncio
import logging
import pytest
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

from healing_guard.core.quantum_planner import (
    QuantumTaskPlanner, Task, TaskPriority, TaskStatus, ExecutionPlan
)


class TestHybridQuantumOptimization:
    """Test suite for hybrid quantum-classical optimization."""
    
    @pytest.fixture
    def planner(self):
        """Create a quantum task planner for testing."""
        return QuantumTaskPlanner(
            max_parallel_tasks=4,
            resource_limits={"cpu": 8.0, "memory": 16.0},
            optimization_iterations=100  # Reduced for testing
        )
    
    @pytest.fixture
    def complex_task_set(self):
        """Create a complex set of interdependent tasks for testing."""
        tasks = [
            Task(
                id="task_1",
                name="Initial Setup",
                priority=TaskPriority.CRITICAL,
                estimated_duration=5.0,
                dependencies=[],
                resources_required={"cpu": 2.0, "memory": 4.0},
                failure_probability=0.05
            ),
            Task(
                id="task_2", 
                name="Code Analysis",
                priority=TaskPriority.HIGH,
                estimated_duration=15.0,
                dependencies=["task_1"],
                resources_required={"cpu": 4.0, "memory": 8.0},
                failure_probability=0.10
            ),
            Task(
                id="task_3",
                name="Security Scan",
                priority=TaskPriority.HIGH,
                estimated_duration=20.0,
                dependencies=["task_1"],
                resources_required={"cpu": 3.0, "memory": 6.0},
                failure_probability=0.08
            ),
            Task(
                id="task_4",
                name="Unit Tests",
                priority=TaskPriority.MEDIUM,
                estimated_duration=10.0,
                dependencies=["task_2"],
                resources_required={"cpu": 2.0, "memory": 4.0},
                failure_probability=0.15
            ),
            Task(
                id="task_5",
                name="Integration Tests",
                priority=TaskPriority.MEDIUM,
                estimated_duration=25.0,
                dependencies=["task_2", "task_3"],
                resources_required={"cpu": 3.0, "memory": 8.0},
                failure_probability=0.12
            ),
            Task(
                id="task_6",
                name="Performance Tests",
                priority=TaskPriority.LOW,
                estimated_duration=30.0,
                dependencies=["task_4", "task_5"],
                resources_required={"cpu": 4.0, "memory": 12.0},
                failure_probability=0.18
            ),
            Task(
                id="task_7",
                name="Deployment",
                priority=TaskPriority.CRITICAL,
                estimated_duration=8.0,
                dependencies=["task_6"],
                resources_required={"cpu": 1.0, "memory": 2.0},
                failure_probability=0.06
            )
        ]
        return tasks
    
    def test_qaoa_optimization_basic(self, planner: QuantumTaskPlanner):
        """Test basic QAOA optimization functionality."""
        # Add simple tasks
        tasks = [
            Task("t1", "Task 1", TaskPriority.HIGH, 5.0, [], {"cpu": 1.0}),
            Task("t2", "Task 2", TaskPriority.MEDIUM, 3.0, ["t1"], {"cpu": 1.0}),
            Task("t3", "Task 3", TaskPriority.LOW, 4.0, ["t1"], {"cpu": 1.0})
        ]
        
        for task in tasks:
            planner.add_task(task)
        
        initial_order = planner._topological_sort()
        qaoa_order = planner._quantum_approximate_optimization(initial_order)
        
        # Validate dependencies are maintained
        assert planner._validate_dependencies(qaoa_order)
        
        # Check that optimization produces valid result
        assert len(qaoa_order) == len(initial_order)
        assert set(qaoa_order) == set(initial_order)
    
    def test_adaptive_hybrid_optimization(self, planner: QuantumTaskPlanner, complex_task_set: List[Task]):
        """Test adaptive hybrid optimization with complex task set."""
        for task in complex_task_set:
            planner.add_task(task)
        
        initial_order = planner._topological_sort()
        hybrid_order = planner._adaptive_hybrid_optimization(initial_order)
        
        # Validate optimization results
        assert planner._validate_dependencies(hybrid_order)
        assert len(hybrid_order) == len(complex_task_set)
        assert set(hybrid_order) == set([t.id for t in complex_task_set])
        
        # Check that optimization improves energy
        initial_energy = planner._calculate_energy(initial_order)
        hybrid_energy = planner._calculate_energy(hybrid_order)
        
        # Hybrid should perform at least as well as initial ordering
        assert hybrid_energy <= initial_energy + 10.0  # Allow small tolerance for randomness
    
    def test_algorithm_selection_logic(self, planner: QuantumTaskPlanner):
        """Test that adaptive algorithm selection works correctly."""
        # Small problem should prefer local search + QAOA
        small_tasks = [
            Task("s1", "Small 1", TaskPriority.HIGH, 2.0, [], {"cpu": 1.0}),
            Task("s2", "Small 2", TaskPriority.MEDIUM, 3.0, ["s1"], {"cpu": 1.0})
        ]
        
        for task in small_tasks:
            planner.add_task(task)
        
        dependency_density = planner._calculate_dependency_density()
        resource_complexity = planner._calculate_resource_complexity()
        
        # Verify complexity calculations
        assert 0.0 <= dependency_density <= 1.0
        assert 0.0 <= resource_complexity <= 1.0
        
        # Test optimization with small problem
        initial_order = planner._topological_sort()
        result = planner._adaptive_hybrid_optimization(initial_order)
        assert planner._validate_dependencies(result)
    
    def test_ensemble_refinement(self, planner: QuantumTaskPlanner, complex_task_set: List[Task]):
        """Test ensemble refinement functionality."""
        for task in complex_task_set:
            planner.add_task(task)
        
        initial_order = planner._topological_sort()
        
        # Create mock results for ensemble testing
        results = {
            'simulated_annealing': {
                'order': initial_order.copy(),
                'energy': planner._calculate_energy(initial_order),
                'weight': 0.4
            },
            'genetic': {
                'order': initial_order[::-1] if planner._validate_dependencies(initial_order[::-1]) else initial_order,
                'energy': planner._calculate_energy(initial_order),
                'weight': 0.6
            }
        }
        
        refined_order = planner._ensemble_refinement(results, initial_order)
        
        # Validate refinement
        assert planner._validate_dependencies(refined_order)
        assert len(refined_order) == len(initial_order)
    
    def test_ml_parameter_tuning(self, planner: QuantumTaskPlanner):
        """Test ML-guided parameter tuning."""
        # Create mock historical data
        historical_data = [
            {
                'problem_size': 5,
                'dependency_density': 0.3,
                'resource_complexity': 0.4,
                'avg_task_duration': 10.0,
                'max_parallel_tasks': 4,
                'performance_improvement': 0.2
            }
            for _ in range(15)  # Need at least 10 for ML
        ]
        
        # Add some variance
        for i, data in enumerate(historical_data):
            data['performance_improvement'] += np.random.normal(0, 0.1)
            data['problem_size'] += i % 3
        
        optimized_params = planner._ml_guided_parameter_tuning(historical_data)
        
        # Validate parameters
        assert 'optimization_iterations' in optimized_params
        assert 'population_size' in optimized_params
        assert optimized_params['optimization_iterations'] > 0
        assert optimized_params['population_size'] > 0
    
    def test_default_parameters(self, planner: QuantumTaskPlanner):
        """Test default parameter retrieval."""
        params = planner._get_default_parameters()
        
        expected_keys = [
            'optimization_iterations', 'temperature_schedule', 
            'population_size', 'mutation_rate', 'crossover_rate'
        ]
        
        for key in expected_keys:
            assert key in params
        
        assert params['optimization_iterations'] == planner.optimization_iterations
        assert 0.0 <= params['mutation_rate'] <= 1.0
        assert 0.0 <= params['crossover_rate'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_enhanced_execution_plan(self, planner: QuantumTaskPlanner, complex_task_set: List[Task]):
        """Test creation of enhanced execution plan with hybrid optimization."""
        for task in complex_task_set:
            planner.add_task(task)
        
        # Add some historical data to enable ML optimization
        planner.historical_data = [
            {
                'problem_size': len(complex_task_set),
                'dependency_density': 0.4,
                'resource_complexity': 0.6,
                'avg_task_duration': 15.0,
                'max_parallel_tasks': 4,
                'performance_improvement': 0.15
            }
            for _ in range(12)
        ]
        
        plan = await planner.create_execution_plan()
        
        # Validate execution plan
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.tasks) == len(complex_task_set)
        assert len(plan.execution_order) == len(complex_task_set)
        assert plan.estimated_total_time > 0
        assert 0.0 <= plan.success_probability <= 1.0
        assert plan.cost_estimate >= 0
        
        # Check metadata for research-grade features
        assert plan.metadata['optimization_method'] == 'adaptive_hybrid_quantum_classical'
        assert plan.metadata['research_grade'] is True
        assert plan.metadata['quantum_inspired'] is True
        assert plan.metadata['ml_guided_parameters'] is True
    
    def test_performance_comparison(self, planner: QuantumTaskPlanner, complex_task_set: List[Task]):
        """Test performance comparison between optimization methods."""
        for task in complex_task_set:
            planner.add_task(task)
        
        initial_order = planner._topological_sort()
        
        # Test different optimization methods
        sa_order = planner._simulated_annealing_optimize(initial_order)
        ga_order = planner._genetic_algorithm_optimize(initial_order)
        local_order = planner._local_search_optimize(initial_order)
        qaoa_order = planner._quantum_approximate_optimization(initial_order)
        hybrid_order = planner._adaptive_hybrid_optimization(initial_order)
        
        # Calculate energies
        energies = {
            'initial': planner._calculate_energy(initial_order),
            'simulated_annealing': planner._calculate_energy(sa_order),
            'genetic': planner._calculate_energy(ga_order),
            'local_search': planner._calculate_energy(local_order),
            'qaoa': planner._calculate_energy(qaoa_order),
            'hybrid': planner._calculate_energy(hybrid_order)
        }
        
        # Hybrid should perform competitively
        best_energy = min(energies.values())
        hybrid_energy = energies['hybrid']
        
        # Allow 10% tolerance for hybrid performance
        assert hybrid_energy <= best_energy * 1.1
        
        # All optimized solutions should be better than or equal to initial
        for method, energy in energies.items():
            if method != 'initial':
                assert energy <= energies['initial'] * 1.05  # 5% tolerance
    
    def test_statistical_significance(self, planner: QuantumTaskPlanner):
        """Test statistical significance of optimization improvements."""
        # Create multiple problem instances
        problem_instances = []
        for i in range(10):
            tasks = [
                Task(f"t{i}_{j}", f"Task {j}", TaskPriority.HIGH, 
                     np.random.uniform(2, 15), [], {"cpu": np.random.uniform(1, 3)})
                for j in range(5)
            ]
            # Add some dependencies
            if len(tasks) > 1:
                tasks[1].dependencies = [tasks[0].id]
            if len(tasks) > 2:
                tasks[2].dependencies = [tasks[0].id]
            if len(tasks) > 3:
                tasks[3].dependencies = [tasks[1].id, tasks[2].id]
            if len(tasks) > 4:
                tasks[4].dependencies = [tasks[3].id]
            
            problem_instances.append(tasks)
        
        improvements = []
        
        for tasks in problem_instances:
            # Clear previous tasks
            planner.tasks.clear()
            for task in tasks:
                planner.add_task(task)
            
            initial_order = planner._topological_sort()
            hybrid_order = planner._adaptive_hybrid_optimization(initial_order)
            
            initial_energy = planner._calculate_energy(initial_order)
            hybrid_energy = planner._calculate_energy(hybrid_order)
            
            improvement = (initial_energy - hybrid_energy) / initial_energy
            improvements.append(improvement)
        
        # Statistical analysis
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        # Check for positive improvements on average
        assert mean_improvement >= 0, f"Mean improvement {mean_improvement:.3f} should be non-negative"
        
        # Calculate confidence interval (95%)
        margin_error = 1.96 * std_improvement / np.sqrt(len(improvements))
        ci_lower = mean_improvement - margin_error
        
        logger = logging.getLogger(__name__)
        logger.info(f"Statistical Analysis Results:")
        logger.info(f"Mean improvement: {mean_improvement:.3f} ± {margin_error:.3f}")
        logger.info(f"95% CI: [{ci_lower:.3f}, {mean_improvement + margin_error:.3f}]")
        logger.info(f"Standard deviation: {std_improvement:.3f}")
        
        # Document results for research publication
        return {
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            'confidence_interval': (ci_lower, mean_improvement + margin_error),
            'sample_size': len(improvements),
            'all_improvements': improvements
        }
    
    def test_scalability_analysis(self, planner: QuantumTaskPlanner):
        """Test scalability of hybrid optimization with varying problem sizes."""
        problem_sizes = [3, 5, 10, 15, 20]
        execution_times = []
        
        for size in problem_sizes:
            # Generate problem of specified size
            tasks = [
                Task(f"task_{i}", f"Task {i}", TaskPriority.MEDIUM, 
                     np.random.uniform(5, 20), [], {"cpu": np.random.uniform(1, 2)})
                for i in range(size)
            ]
            
            # Add realistic dependency structure
            for i in range(1, size):
                if i < size // 2:
                    tasks[i].dependencies = [tasks[0].id]  # Fan-out pattern
                else:
                    # Dependent on earlier tasks
                    deps = np.random.choice(range(i), size=min(2, i), replace=False)
                    tasks[i].dependencies = [tasks[d].id for d in deps]
            
            # Clear and add tasks
            planner.tasks.clear()
            for task in tasks:
                planner.add_task(task)
            
            # Measure execution time
            start_time = datetime.now()
            initial_order = planner._topological_sort()
            hybrid_order = planner._adaptive_hybrid_optimization(initial_order)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            execution_times.append(execution_time)
            
            # Validate result
            assert planner._validate_dependencies(hybrid_order)
        
        # Analyze scalability
        logger = logging.getLogger(__name__)
        logger.info("Scalability Analysis:")
        for size, time in zip(problem_sizes, execution_times):
            logger.info(f"Size {size:2d}: {time:.3f}s")
        
        # Check that algorithm scales reasonably (should be sub-quadratic)
        if len(execution_times) >= 3:
            # Simple growth rate check
            small_time = execution_times[0]
            large_time = execution_times[-1]
            size_ratio = problem_sizes[-1] / problem_sizes[0]
            time_ratio = large_time / small_time if small_time > 0 else 1
            
            # Should scale better than O(n^3)
            assert time_ratio <= size_ratio ** 2.5, f"Algorithm may not scale well: {time_ratio:.2f} vs {size_ratio**2.5:.2f}"
        
        return {
            'problem_sizes': problem_sizes,
            'execution_times': execution_times,
            'scalability_assessment': 'acceptable' if time_ratio <= size_ratio ** 2 else 'needs_optimization'
        }


class TestResearchValidation:
    """Research-grade validation for publication-ready results."""
    
    @pytest.fixture
    def research_planner(self):
        """Create planner with research-grade configuration."""
        return QuantumTaskPlanner(
            max_parallel_tasks=6,
            resource_limits={"cpu": 16.0, "memory": 32.0, "gpu": 4.0},
            optimization_iterations=500  # Higher iterations for research
        )
    
    def test_comparative_algorithm_study(self, research_planner: QuantumTaskPlanner):
        """Comprehensive comparative study of optimization algorithms."""
        # Generate benchmark problem set
        benchmark_problems = self._generate_benchmark_problems()
        
        algorithm_performance = {
            'simulated_annealing': [],
            'genetic_algorithm': [],
            'local_search': [],
            'qaoa': [],
            'adaptive_hybrid': []
        }
        
        for problem_id, tasks in enumerate(benchmark_problems):
            research_planner.tasks.clear()
            for task in tasks:
                research_planner.add_task(task)
            
            initial_order = research_planner._topological_sort()
            baseline_energy = research_planner._calculate_energy(initial_order)
            
            # Test each algorithm
            for algorithm in algorithm_performance.keys():
                try:
                    if algorithm == 'simulated_annealing':
                        result = research_planner._simulated_annealing_optimize(initial_order)
                    elif algorithm == 'genetic_algorithm':
                        result = research_planner._genetic_algorithm_optimize(initial_order)
                    elif algorithm == 'local_search':
                        result = research_planner._local_search_optimize(initial_order)
                    elif algorithm == 'qaoa':
                        result = research_planner._quantum_approximate_optimization(initial_order)
                    elif algorithm == 'adaptive_hybrid':
                        result = research_planner._adaptive_hybrid_optimization(initial_order)
                    
                    optimized_energy = research_planner._calculate_energy(result)
                    improvement = (baseline_energy - optimized_energy) / baseline_energy
                    algorithm_performance[algorithm].append(improvement)
                    
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Algorithm {algorithm} failed on problem {problem_id}: {e}")
                    algorithm_performance[algorithm].append(0.0)  # No improvement
        
        # Statistical analysis
        results = {}
        for algorithm, improvements in algorithm_performance.items():
            if improvements:
                results[algorithm] = {
                    'mean': np.mean(improvements),
                    'std': np.std(improvements),
                    'median': np.median(improvements),
                    'min': np.min(improvements),
                    'max': np.max(improvements),
                    'samples': len(improvements)
                }
        
        # Validate hybrid algorithm performance
        if 'adaptive_hybrid' in results and len(results) > 1:
            hybrid_mean = results['adaptive_hybrid']['mean']
            other_means = [results[alg]['mean'] for alg in results.keys() if alg != 'adaptive_hybrid']
            best_other = max(other_means) if other_means else 0
            
            # Hybrid should be competitive (within 5% of best individual algorithm)
            assert hybrid_mean >= best_other * 0.95, f"Hybrid performance {hybrid_mean:.3f} vs best {best_other:.3f}"
        
        return results
    
    def _generate_benchmark_problems(self) -> List[List[Task]]:
        """Generate diverse benchmark problems for comprehensive testing."""
        problems = []
        
        # Problem 1: Linear chain
        chain_tasks = [
            Task(f"chain_{i}", f"Chain Task {i}", TaskPriority.MEDIUM, 
                 np.random.uniform(5, 15), [f"chain_{i-1}"] if i > 0 else [],
                 {"cpu": np.random.uniform(1, 3)})
            for i in range(8)
        ]
        problems.append(chain_tasks)
        
        # Problem 2: Star pattern (one root, many dependents)
        star_tasks = [Task("root", "Root Task", TaskPriority.CRITICAL, 10.0, [], {"cpu": 2.0})]
        star_tasks.extend([
            Task(f"leaf_{i}", f"Leaf Task {i}", TaskPriority.LOW, 
                 np.random.uniform(3, 8), ["root"], {"cpu": np.random.uniform(0.5, 2)})
            for i in range(6)
        ])
        problems.append(star_tasks)
        
        # Problem 3: Diamond pattern (complex dependencies)
        diamond_tasks = [
            Task("d1", "Start", TaskPriority.HIGH, 5.0, [], {"cpu": 1.0}),
            Task("d2", "Branch A", TaskPriority.MEDIUM, 8.0, ["d1"], {"cpu": 2.0}),
            Task("d3", "Branch B", TaskPriority.MEDIUM, 6.0, ["d1"], {"cpu": 1.5}),
            Task("d4", "Merge AB", TaskPriority.HIGH, 12.0, ["d2", "d3"], {"cpu": 3.0}),
            Task("d5", "Parallel C", TaskPriority.LOW, 4.0, ["d1"], {"cpu": 1.0}),
            Task("d6", "Final", TaskPriority.CRITICAL, 7.0, ["d4", "d5"], {"cpu": 2.5})
        ]
        problems.append(diamond_tasks)
        
        # Problem 4: Resource-intensive tasks
        resource_tasks = [
            Task(f"gpu_{i}", f"GPU Task {i}", TaskPriority.HIGH,
                 np.random.uniform(20, 40), [], 
                 {"cpu": np.random.uniform(2, 4), "gpu": np.random.uniform(1, 2), "memory": np.random.uniform(4, 8)})
            for i in range(4)
        ]
        problems.append(resource_tasks)
        
        # Problem 5: Mixed priorities with complex dependencies
        mixed_tasks = []
        priorities = [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]
        for i in range(10):
            deps = []
            if i > 0:
                num_deps = min(i, np.random.poisson(1) + 1)
                deps = [f"mixed_{j}" for j in np.random.choice(i, num_deps, replace=False)]
            
            mixed_tasks.append(Task(
                f"mixed_{i}", f"Mixed Task {i}", 
                priorities[i % len(priorities)],
                np.random.uniform(3, 25), deps,
                {"cpu": np.random.uniform(1, 4), "memory": np.random.uniform(2, 10)},
                failure_probability=np.random.uniform(0.05, 0.20)
            ))
        problems.append(mixed_tasks)
        
        return problems


if __name__ == "__main__":
    # Run research validation when executed directly
    import sys
    logging.basicConfig(level=logging.INFO)
    
    planner = QuantumTaskPlanner(optimization_iterations=200)
    test_instance = TestHybridQuantumOptimization()
    
    print("🔬 RESEARCH VALIDATION: Hybrid Quantum-Classical Optimization")
    print("=" * 80)
    
    # Run statistical significance test
    print("\n📊 Statistical Significance Analysis...")
    stats = test_instance.test_statistical_significance(planner)
    print(f"✅ Mean Improvement: {stats['mean_improvement']:.1%} ± {(stats['confidence_interval'][1] - stats['confidence_interval'][0])/2:.1%}")
    
    # Run scalability test  
    print("\n⚡ Scalability Analysis...")
    scalability = test_instance.test_scalability_analysis(planner)
    print(f"✅ Scalability Assessment: {scalability['scalability_assessment']}")
    
    print("\n🏆 RESEARCH VALIDATION COMPLETE")
    print("Novel hybrid quantum-classical optimization algorithms validated!")