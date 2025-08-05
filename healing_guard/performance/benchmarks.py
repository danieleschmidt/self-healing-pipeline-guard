"""Performance benchmarking system for quality gates."""

import asyncio
import gc
import logging
import psutil
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple
import tracemalloc
import threading
from contextlib import contextmanager

from ..core.quantum_planner import QuantumTaskPlanner, Task, TaskPriority
from ..core.failure_detector import FailureDetector
from ..core.healing_engine import HealingEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmarking."""
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    peak_memory: float  # MB
    memory_allocations: int
    gc_collections: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "peak_memory": self.peak_memory,
            "memory_allocations": self.memory_allocations,
            "gc_collections": self.gc_collections,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    iterations: int
    metrics: List[BenchmarkMetrics]
    started_at: datetime
    completed_at: datetime
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def avg_execution_time(self) -> float:
        """Average execution time across iterations."""
        return statistics.mean([m.execution_time for m in self.metrics])
    
    @property
    def min_execution_time(self) -> float:
        """Minimum execution time."""
        return min([m.execution_time for m in self.metrics])
    
    @property
    def max_execution_time(self) -> float:
        """Maximum execution time."""
        return max([m.execution_time for m in self.metrics])
    
    @property
    def p95_execution_time(self) -> float:
        """95th percentile execution time."""
        times = sorted([m.execution_time for m in self.metrics])
        index = int(len(times) * 0.95)
        return times[min(index, len(times) - 1)]
    
    @property
    def avg_memory_usage(self) -> float:
        """Average memory usage."""
        return statistics.mean([m.memory_usage for m in self.metrics])
    
    @property
    def peak_memory_usage(self) -> float:
        """Peak memory usage across all iterations."""
        return max([m.peak_memory for m in self.metrics])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "iterations": self.iterations,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
            "summary": {
                "avg_execution_time": self.avg_execution_time,
                "min_execution_time": self.min_execution_time,
                "max_execution_time": self.max_execution_time,
                "p95_execution_time": self.p95_execution_time,
                "avg_memory_usage": self.avg_memory_usage,
                "peak_memory_usage": self.peak_memory_usage
            },
            "detailed_metrics": [m.to_dict() for m in self.metrics]
        }


class PerformanceBenchmark(ABC):
    """Abstract base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.setup_complete = False
    
    async def setup(self):
        """Setup benchmark environment."""
        self.setup_complete = True
    
    async def teardown(self):
        """Clean up benchmark environment."""
        pass
    
    @abstractmethod
    async def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        pass
    
    async def validate_result(self, result: Any) -> bool:
        """Validate benchmark result."""
        return True


class QuantumPlannerBenchmark(PerformanceBenchmark):
    """Benchmark for quantum task planner performance."""
    
    def __init__(self):
        super().__init__("quantum_planner_benchmark")
        self.planner: Optional[QuantumTaskPlanner] = None
    
    async def setup(self):
        """Setup quantum planner benchmark."""
        self.planner = QuantumTaskPlanner(
            max_parallel_tasks=4,
            resource_limits={"cpu": 8.0, "memory": 16.0},
            optimization_iterations=100
        )
        await super().setup()
    
    async def teardown(self):
        """Clean up quantum planner benchmark."""
        if self.planner:
            self.planner.tasks.clear()
    
    async def run_iteration(self) -> Dict[str, Any]:
        """Run quantum planner benchmark iteration."""
        if not self.planner:
            raise RuntimeError("Benchmark not properly setup")
        
        # Clear previous tasks
        self.planner.tasks.clear()
        
        # Create test tasks with dependencies
        tasks = []
        for i in range(20):  # 20 tasks for meaningful optimization
            dependencies = []
            if i > 0:
                # Add 1-2 random dependencies from previous tasks
                import random
                num_deps = random.randint(0, min(2, i))
                if num_deps > 0:
                    dependencies = random.sample([f"task_{j}" for j in range(i)], num_deps)
            
            task = Task(
                id=f"task_{i}",
                name=f"Benchmark Task {i}",
                priority=TaskPriority(random.randint(1, 4)),
                estimated_duration=random.uniform(1.0, 10.0),
                dependencies=dependencies,
                resources_required={
                    "cpu": random.uniform(0.5, 2.0),
                    "memory": random.uniform(1.0, 4.0)
                }
            )
            tasks.append(task)
            self.planner.add_task(task)
        
        # Measure execution plan creation
        start_time = time.time()
        execution_plan = await self.planner.create_execution_plan()
        end_time = time.time()
        
        return {
            "execution_time": end_time - start_time,
            "tasks_planned": len(tasks),
            "parallel_stages": len(execution_plan.parallel_stages),
            "estimated_total_time": execution_plan.estimated_total_time,
            "success_probability": execution_plan.success_probability
        }


class FailureDetectorBenchmark(PerformanceBenchmark):
    """Benchmark for failure detector performance."""
    
    def __init__(self):
        super().__init__("failure_detector_benchmark")
        self.detector: Optional[FailureDetector] = None
        self.test_logs = self._generate_test_logs()
    
    def _generate_test_logs(self) -> List[str]:
        """Generate test log samples for benchmarking."""
        log_templates = [
            "2024-01-01 10:00:00 ERROR: Connection timeout after 30 seconds",
            "2024-01-01 10:00:01 INFO: Retrying connection attempt 1/3",
            "2024-01-01 10:00:02 ERROR: OutOfMemoryError: Java heap space",
            "2024-01-01 10:00:03 WARN: High CPU usage detected: 95%",
            "2024-01-01 10:00:04 ERROR: Test failed: assertion failed",
            "2024-01-01 10:00:05 DEBUG: Network packet lost",
            "2024-01-01 10:00:06 FATAL: System shutdown initiated",
            "2024-01-01 10:00:07 ERROR: Database connection refused",
            "2024-01-01 10:00:08 INFO: Attempting graceful recovery",
            "2024-01-01 10:00:09 ERROR: npm install failed with exit code 1"
        ]
        
        # Generate larger log samples
        logs = []
        for i in range(10):
            log_content = "\n".join(log_templates * (10 + i))  # Varying sizes
            logs.append(log_content)
        
        return logs
    
    async def setup(self):
        """Setup failure detector benchmark."""
        self.detector = FailureDetector()
        await super().setup()
    
    async def run_iteration(self) -> Dict[str, Any]:
        """Run failure detector benchmark iteration."""
        if not self.detector:
            raise RuntimeError("Benchmark not properly setup")
        
        import random
        
        # Select random log sample
        log_sample = random.choice(self.test_logs)
        
        # Measure failure detection
        start_time = time.time()
        failure_event = await self.detector.detect_failure(
            job_id=f"benchmark_job_{random.randint(1000, 9999)}",
            repository="benchmark/repo",
            branch="main",
            commit_sha=f"abc{random.randint(100000, 999999)}",
            logs=log_sample
        )
        end_time = time.time()
        
        return {
            "execution_time": end_time - start_time,
            "log_size": len(log_sample),
            "failure_detected": failure_event is not None,
            "confidence": failure_event.confidence if failure_event else 0,
            "patterns_matched": len(failure_event.matched_patterns) if failure_event else 0
        }


class HealingEngineBenchmark(PerformanceBenchmark):
    """Benchmark for healing engine performance."""
    
    def __init__(self):
        super().__init__("healing_engine_benchmark")
        self.healing_engine: Optional[HealingEngine] = None
        self.test_failures = self._generate_test_failures()
    
    def _generate_test_failures(self) -> List[Dict[str, Any]]:
        """Generate test failure events for benchmarking."""
        from ..core.failure_detector import FailureEvent, FailureType, SeverityLevel
        
        failures = []
        failure_types = list(FailureType)
        severity_levels = list(SeverityLevel)
        
        for i in range(10):
            import random
            
            failure = FailureEvent(
                id=f"benchmark_failure_{i}",
                timestamp=datetime.now(),
                job_id=f"job_{i}",
                repository="benchmark/repo",
                branch="main",
                commit_sha=f"sha_{i}",
                failure_type=random.choice(failure_types),
                severity=random.choice(severity_levels),
                confidence=random.uniform(0.7, 1.0),
                raw_logs=f"Benchmark failure logs {i}" * 100,
                remediation_suggestions=["retry_with_backoff", "increase_resources"]
            )
            failures.append(failure)
        
        return failures
    
    async def setup(self):
        """Setup healing engine benchmark."""
        from ..core.quantum_planner import QuantumTaskPlanner
        from ..core.failure_detector import FailureDetector
        
        planner = QuantumTaskPlanner()
        detector = FailureDetector()
        self.healing_engine = HealingEngine(planner, detector)
        await super().setup()
    
    async def run_iteration(self) -> Dict[str, Any]:
        """Run healing engine benchmark iteration."""
        if not self.healing_engine:
            raise RuntimeError("Benchmark not properly setup")
        
        import random
        
        # Select random failure event
        failure_event = random.choice(self.test_failures)
        
        # Measure healing plan creation
        start_time = time.time()
        healing_plan = await self.healing_engine.create_healing_plan(failure_event)
        end_time = time.time()
        
        return {
            "execution_time": end_time - start_time,
            "failure_type": failure_event.failure_type.value,
            "actions_generated": len(healing_plan.actions),
            "estimated_healing_time": healing_plan.estimated_total_time,
            "success_probability": healing_plan.success_probability,
            "total_cost": healing_plan.total_cost
        }


class BenchmarkRunner:
    """Runner for executing performance benchmarks."""
    
    def __init__(self):
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.results: List[BenchmarkResult] = []
        
        # Resource monitoring
        self.process = psutil.Process()
        
    def register_benchmark(self, benchmark: PerformanceBenchmark):
        """Register a benchmark for execution."""
        self.benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name}")
    
    def register_default_benchmarks(self):
        """Register default benchmarks."""
        self.register_benchmark(QuantumPlannerBenchmark())
        self.register_benchmark(FailureDetectorBenchmark())
        self.register_benchmark(HealingEngineBenchmark())
    
    @contextmanager
    def _monitor_resources(self):
        """Context manager for monitoring resource usage."""
        # Start memory tracing
        tracemalloc.start()
        
        # Initial measurements
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        gc_initial = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
        
        # CPU monitoring setup
        cpu_measurements = []
        stop_monitoring = threading.Event()
        
        def cpu_monitor():
            while not stop_monitoring.is_set():
                cpu_measurements.append(self.process.cpu_percent())
                time.sleep(0.1)
        
        cpu_thread = threading.Thread(target=cpu_monitor, daemon=True)
        cpu_thread.start()
        
        try:
            yield cpu_measurements
        finally:
            # Stop monitoring
            stop_monitoring.set()
            cpu_thread.join(timeout=1.0)
            
            # Get final memory measurements
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            gc_final = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
            
            # Store measurements in the monitoring context
            self._last_measurements = {
                "memory_usage": final_memory - initial_memory,
                "peak_memory": peak / 1024 / 1024,  # MB
                "memory_allocations": current,
                "gc_collections": gc_final - gc_initial,
                "avg_cpu_usage": statistics.mean(cpu_measurements) if cpu_measurements else 0
            }
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        iterations: int = 10,
        warmup_iterations: int = 2
    ) -> BenchmarkResult:
        """Run a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not registered")
        
        benchmark = self.benchmarks[benchmark_name]
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            iterations=iterations,
            metrics=[],
            started_at=datetime.now(),
            completed_at=datetime.now()  # Will be updated
        )
        
        try:
            # Setup benchmark
            await benchmark.setup()
            
            # Warmup iterations
            logger.info(f"Running {warmup_iterations} warmup iterations for {benchmark_name}")
            for _ in range(warmup_iterations):
                await benchmark.run_iteration()
            
            # Actual benchmark iterations
            logger.info(f"Running {iterations} benchmark iterations for {benchmark_name}")
            
            for i in range(iterations):
                # Force garbage collection before iteration
                gc.collect()
                
                with self._monitor_resources():
                    start_time = time.time()
                    iteration_result = await benchmark.run_iteration()
                    end_time = time.time()
                
                # Create metrics for this iteration
                metrics = BenchmarkMetrics(
                    execution_time=end_time - start_time,
                    memory_usage=self._last_measurements["memory_usage"],
                    cpu_usage=self._last_measurements["avg_cpu_usage"],
                    peak_memory=self._last_measurements["peak_memory"],
                    memory_allocations=self._last_measurements["memory_allocations"],
                    gc_collections=self._last_measurements["gc_collections"],
                    custom_metrics=iteration_result
                )
                
                result.metrics.append(metrics)
                
                # Validate result if needed
                if not await benchmark.validate_result(iteration_result):
                    logger.warning(f"Benchmark iteration {i} failed validation")
            
            result.completed_at = datetime.now()
            result.success = True
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.success = False
            result.error_message = str(e)
            logger.error(f"Benchmark {benchmark_name} failed: {e}")
        
        finally:
            await benchmark.teardown()
        
        self.results.append(result)
        return result
    
    async def run_all_benchmarks(
        self,
        iterations: int = 10,
        warmup_iterations: int = 2
    ) -> Dict[str, BenchmarkResult]:
        """Run all registered benchmarks."""
        results = {}
        
        for benchmark_name in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark_name}")
            result = await self.run_benchmark(benchmark_name, iterations, warmup_iterations)
            results[benchmark_name] = result
        
        return results
    
    def get_performance_report(self, results: Optional[Dict[str, BenchmarkResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if results is None:
            results = {result.benchmark_name: result for result in self.results}
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {
                "total_benchmarks": len(results),
                "successful_benchmarks": sum(1 for r in results.values() if r.success),
                "failed_benchmarks": sum(1 for r in results.values() if not r.success)
            }
        }
        
        # Performance thresholds for quality gates
        performance_thresholds = {
            "quantum_planner_benchmark": {
                "max_avg_execution_time": 1.0,  # 1 second
                "max_memory_usage": 100.0,  # 100 MB
                "max_peak_memory": 200.0  # 200 MB
            },
            "failure_detector_benchmark": {
                "max_avg_execution_time": 0.5,  # 500ms
                "max_memory_usage": 50.0,  # 50 MB
                "max_peak_memory": 100.0  # 100 MB
            },
            "healing_engine_benchmark": {
                "max_avg_execution_time": 2.0,  # 2 seconds
                "max_memory_usage": 150.0,  # 150 MB
                "max_peak_memory": 300.0  # 300 MB
            }
        }
        
        # Analyze each benchmark
        for benchmark_name, result in results.items():
            benchmark_report = {
                "success": result.success,
                "iterations": result.iterations,
                "performance_metrics": {
                    "avg_execution_time": result.avg_execution_time,
                    "min_execution_time": result.min_execution_time,
                    "max_execution_time": result.max_execution_time,
                    "p95_execution_time": result.p95_execution_time,
                    "avg_memory_usage": result.avg_memory_usage,
                    "peak_memory_usage": result.peak_memory_usage
                },
                "quality_gate_status": "PASS"
            }
            
            # Check against performance thresholds
            if benchmark_name in performance_thresholds:
                thresholds = performance_thresholds[benchmark_name]
                violations = []
                
                if result.avg_execution_time > thresholds["max_avg_execution_time"]:
                    violations.append(f"Average execution time ({result.avg_execution_time:.3f}s) exceeds threshold ({thresholds['max_avg_execution_time']}s)")
                
                if result.avg_memory_usage > thresholds["max_memory_usage"]:
                    violations.append(f"Average memory usage ({result.avg_memory_usage:.1f}MB) exceeds threshold ({thresholds['max_memory_usage']}MB)")
                
                if result.peak_memory_usage > thresholds["max_peak_memory"]:
                    violations.append(f"Peak memory usage ({result.peak_memory_usage:.1f}MB) exceeds threshold ({thresholds['max_peak_memory']}MB)")
                
                if violations:
                    benchmark_report["quality_gate_status"] = "FAIL"
                    benchmark_report["violations"] = violations
            
            report["benchmarks"][benchmark_name] = benchmark_report
        
        # Overall quality gate status
        all_passed = all(
            benchmark["quality_gate_status"] == "PASS"
            for benchmark in report["benchmarks"].values()
        )
        
        report["overall_quality_gate"] = "PASS" if all_passed else "FAIL"
        
        return report
    
    def get_benchmark_history(self, benchmark_name: str, limit: int = 50) -> List[BenchmarkResult]:
        """Get historical results for a specific benchmark."""
        return [
            result for result in self.results[-limit:]
            if result.benchmark_name == benchmark_name
        ]


# Global benchmark runner instance
benchmark_runner = BenchmarkRunner()