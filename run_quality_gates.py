#!/usr/bin/env python3
"""
Quality Gates & Testing Suite for Healing Guard

This comprehensive test suite validates all three generations:
- Generation 1: Core API functionality
- Generation 2: Robust reliability features
- Generation 3: Scaling & optimization features

Includes security testing, performance validation, and integration tests.
"""

import asyncio
import os
import sys
import time
import subprocess
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class QualityGate:
    """Individual quality gate test."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.error = None
        self.execution_time = 0.0
        self.details = {}
    
    async def run(self) -> bool:
        """Run the quality gate test."""
        start_time = time.time()
        try:
            await self.execute()
            self.passed = True
        except Exception as e:
            self.error = str(e)
            self.passed = False
        finally:
            self.execution_time = time.time() - start_time
        
        return self.passed
    
    async def execute(self):
        """Override in subclasses."""
        raise NotImplementedError


class APIFunctionalityGate(QualityGate):
    """Test core API functionality."""
    
    def __init__(self):
        super().__init__(
            "API Functionality",
            "Validate core API endpoints and routing"
        )
    
    async def execute(self):
        from healing_guard.api.main import create_app
        
        # Test app creation
        app = create_app()
        assert app is not None, "Failed to create FastAPI app"
        
        # Test route configuration
        routes = [route for route in app.routes if hasattr(route, 'path')]
        api_routes = [route for route in routes if route.path.startswith('/api/v1/')]
        
        assert len(api_routes) >= 10, f"Expected at least 10 API routes, got {len(api_routes)}"
        
        # Test essential routes exist
        essential_routes = [
            '/api/v1/tasks',
            '/api/v1/planning/execute', 
            '/api/v1/failures/analyze',
            '/api/v1/healing/plan',
            '/health'
        ]
        
        route_paths = [route.path for route in routes]
        for essential_route in essential_routes:
            assert essential_route in route_paths, f"Missing essential route: {essential_route}"
        
        self.details["total_routes"] = len(routes)
        self.details["api_routes"] = len(api_routes)


class ValidationSecurityGate(QualityGate):
    """Test input validation and security features."""
    
    def __init__(self):
        super().__init__(
            "Validation & Security",
            "Validate input sanitization and security measures"
        )
    
    async def execute(self):
        from healing_guard.core.validation import (
            StringValidator, SecurityValidator, ValidationError
        )
        
        # Test string validation
        validator = StringValidator(min_length=5, max_length=20)
        result = validator.validate("valid_test", "test_field")
        assert result.is_valid, "Valid string should pass validation"
        
        # Test security validation - should reject dangerous paths
        try:
            SecurityValidator.validate_file_path("../../../etc/passwd")
            assert False, "Should have rejected path traversal"
        except ValidationError:
            pass  # Expected
        
        # Test SQL injection prevention
        dangerous_input = "'; DROP TABLE users; --"
        cleaned = SecurityValidator.sanitize_sql_input(dangerous_input)
        assert "DROP TABLE" not in cleaned, "SQL injection not properly sanitized"
        
        # Test shell injection prevention
        shell_input = "; rm -rf / &"
        cleaned_shell = SecurityValidator.sanitize_shell_input(shell_input)
        assert "rm -rf" not in cleaned_shell, "Shell injection not properly sanitized"
        
        self.details["validation_tests"] = 4
        self.details["security_tests"] = 3


class ObservabilityGate(QualityGate):
    """Test observability and monitoring systems."""
    
    def __init__(self):
        super().__init__(
            "Observability",
            "Validate tracing, metrics, and monitoring"
        )
    
    async def execute(self):
        from healing_guard.monitoring.observability import observability, traced, record_metric
        
        # Start observability
        observability.start()
        
        # Test tracing
        @traced("test_trace_function")
        async def test_function():
            await asyncio.sleep(0.01)
            return "traced_result"
        
        result = await test_function()
        assert result == "traced_result", "Traced function should return correct result"
        
        # Test metrics recording
        record_metric("test_counter", 1, "counter")
        record_metric("test_gauge", 42.5, "gauge")
        record_metric("test_histogram", 123.45, "histogram")
        
        # Verify traces were recorded
        traces = observability.get_traces(limit=10)
        assert len(traces) > 0, "Traces should be recorded"
        
        # Find our test trace
        test_traces = [t for t in traces if t.operation_name == "test_trace_function"]
        assert len(test_traces) > 0, "Test trace should be recorded"
        
        # Verify trace has expected properties
        test_trace = test_traces[0]
        assert test_trace.status == "success", "Test trace should be successful"
        assert test_trace.duration_ms > 0, "Test trace should have duration"
        
        self.details["traces_recorded"] = len(traces)
        self.details["metrics_recorded"] = 3


class OptimizationGate(QualityGate):
    """Test optimization and scaling features."""
    
    def __init__(self):
        super().__init__(
            "Optimization & Scaling",
            "Validate quantum optimization and auto-scaling"
        )
    
    async def execute(self):
        from healing_guard.core.optimization import (
            quantum_optimizer, load_balancer, resource_scaler, profiler
        )
        
        # Test quantum optimization
        tasks = [
            {
                'cpu_required': 1.0,
                'memory_required': 2.0, 
                'estimated_duration': 3.0
            }
            for _ in range(5)
        ]
        resources = {'cpu': 8.0, 'memory': 16.0}
        
        optimization_result = quantum_optimizer.optimize_task_schedule(tasks, resources)
        assert 'optimized_schedule' in optimization_result, "Optimization should return schedule"
        assert len(optimization_result['optimized_schedule']) == len(tasks), "Should optimize all tasks"
        
        # Test load balancer
        servers = ['server1', 'server2', 'server3']
        for i, server in enumerate(servers):
            load_balancer.update_server_metrics(server, 0.5, 0.1 * (i + 1))
        
        selected_server = load_balancer.select_server(servers)
        assert selected_server in servers, "Should select valid server"
        
        # Test resource scaler
        high_load_metrics = {
            'cpu_usage': 0.9,
            'memory_usage': 0.85,
            'request_rate': 150,
            'response_time': 2.5,
            'instances': 2
        }
        
        action, instances = resource_scaler.should_scale(high_load_metrics)
        # Should either scale up or do nothing (due to cooldown)
        assert action in ["up", "none"], f"Unexpected scaling action: {action}"
        
        # Test performance profiler
        @profiler.profile("test_profiled_function")
        def test_profiled():
            time.sleep(0.001)  # 1ms
            return "profiled"
        
        result = test_profiled()
        assert result == "profiled", "Profiled function should work"
        
        report = profiler.get_report(top_n=5)
        assert report["total_functions"] > 0, "Should have profiled functions"
        
        self.details["optimization_tasks"] = len(tasks)
        self.details["load_balancer_servers"] = len(servers)
        self.details["profiled_functions"] = report["total_functions"]


class CachePerformanceGate(QualityGate):
    """Test caching system performance."""
    
    def __init__(self):
        super().__init__(
            "Cache Performance",
            "Validate caching system efficiency and correctness"
        )
    
    async def execute(self):
        from healing_guard.core.cache import cache_manager
        
        # Test basic cache operations
        cache = cache_manager.get_cache("test_performance")
        
        # Cache hit/miss test
        key = "performance_test_key"
        value = {"data": "test_value", "number": 42}
        
        # First access should be a miss
        result = await cache.get(key)
        assert result is None, "Cache should initially be empty"
        
        # Set value
        success = await cache.set(key, value, ttl=60)
        assert success, "Cache set should succeed"
        
        # Second access should be a hit
        cached_result = await cache.get(key)
        assert cached_result == value, "Cached value should match original"
        
        # Test cache decorator performance
        call_count = 0
        
        @cache_manager.cached(ttl=30, key_prefix="perf_test")
        def expensive_computation(n: int) -> int:
            nonlocal call_count
            call_count += 1
            return n * n
        
        # First call
        result1 = expensive_computation(10)
        assert call_count == 1, "Function should be called first time"
        assert result1 == 100, "Function should return correct result"
        
        # Second call should use cache
        result2 = expensive_computation(10)
        assert call_count == 1, "Function should not be called second time (cached)"
        assert result2 == 100, "Cached result should match"
        
        self.details["cache_operations"] = 4
        self.details["decorator_efficiency"] = "100%"


class IntegrationGate(QualityGate):
    """Test system integration and end-to-end workflows."""
    
    def __init__(self):
        super().__init__(
            "Integration Tests", 
            "Validate complete system integration"
        )
    
    async def execute(self):
        from healing_guard.core.quantum_planner import QuantumTaskPlanner
        from healing_guard.core.failure_detector import FailureDetector
        from healing_guard.core.healing_engine import HealingEngine
        from healing_guard.monitoring.health import health_checker
        
        # Test core component initialization
        planner = QuantumTaskPlanner()
        assert planner is not None, "Quantum planner should initialize"
        
        detector = FailureDetector()  
        assert detector is not None, "Failure detector should initialize"
        
        engine = HealingEngine()
        assert engine is not None, "Healing engine should initialize"
        
        # Test health checker
        health_status = await health_checker.get_liveness()
        assert health_status["status"] == "alive", "System should be alive"
        
        readiness_status = await health_checker.get_readiness()
        # Should be ready or not ready (both acceptable for test environment)
        assert "status" in readiness_status, "Readiness check should return status"
        
        self.details["components_initialized"] = 3
        self.details["health_checks"] = 2


async def run_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and return results."""
    
    gates = [
        APIFunctionalityGate(),
        ValidationSecurityGate(), 
        ObservabilityGate(),
        OptimizationGate(),
        CachePerformanceGate(),
        IntegrationGate()
    ]
    
    print("🚀 Running Quality Gates & Testing Suite")
    print("=" * 60)
    
    results = {
        "total_gates": len(gates),
        "passed_gates": 0,
        "failed_gates": 0,
        "total_time": 0.0,
        "gate_results": []
    }
    
    start_time = time.time()
    
    for gate in gates:
        print(f"🧪 Running {gate.name}...")
        
        passed = await gate.run()
        
        gate_result = {
            "name": gate.name,
            "description": gate.description,
            "passed": passed,
            "execution_time": gate.execution_time,
            "details": gate.details
        }
        
        if gate.error:
            gate_result["error"] = gate.error
        
        results["gate_results"].append(gate_result)
        
        if passed:
            results["passed_gates"] += 1
            print(f"  ✅ PASSED ({gate.execution_time:.3f}s)")
            if gate.details:
                for key, value in gate.details.items():
                    print(f"     {key}: {value}")
        else:
            results["failed_gates"] += 1
            print(f"  ❌ FAILED ({gate.execution_time:.3f}s)")
            if gate.error:
                print(f"     Error: {gate.error}")
        
        print()
    
    results["total_time"] = time.time() - start_time
    
    return results


def generate_quality_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive quality report."""
    
    report = f"""
# Healing Guard Quality Gates Report

## Summary
- **Total Gates**: {results['total_gates']}
- **Passed**: {results['passed_gates']} ✅
- **Failed**: {results['failed_gates']} ❌
- **Success Rate**: {(results['passed_gates']/results['total_gates']*100):.1f}%
- **Total Execution Time**: {results['total_time']:.2f}s

## Quality Gates Results

"""
    
    for gate_result in results["gate_results"]:
        status = "✅ PASSED" if gate_result["passed"] else "❌ FAILED"
        report += f"### {gate_result['name']} - {status}\n"
        report += f"**Description**: {gate_result['description']}\n"
        report += f"**Execution Time**: {gate_result['execution_time']:.3f}s\n"
        
        if gate_result.get("error"):
            report += f"**Error**: {gate_result['error']}\n"
        
        if gate_result.get("details"):
            report += f"**Details**:\n"
            for key, value in gate_result["details"].items():
                report += f"- {key}: {value}\n"
        
        report += "\n"
    
    # Overall assessment
    if results["failed_gates"] == 0:
        report += "## 🎯 Overall Assessment: EXCELLENT\n"
        report += "All quality gates passed successfully. The system is ready for production deployment.\n"
    elif results["failed_gates"] <= 2:
        report += "## ⚠️ Overall Assessment: GOOD\n" 
        report += "Most quality gates passed with minor issues to address.\n"
    else:
        report += "## 🚨 Overall Assessment: NEEDS IMPROVEMENT\n"
        report += "Multiple quality gates failed. Significant issues need to be resolved.\n"
    
    return report


async def main():
    """Main execution function."""
    
    # Run quality gates
    results = await run_quality_gates()
    
    # Print summary
    print("=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Total Gates: {results['total_gates']}")
    print(f"Passed: {results['passed_gates']} ✅")
    print(f"Failed: {results['failed_gates']} ❌")
    print(f"Success Rate: {(results['passed_gates']/results['total_gates']*100):.1f}%")
    print(f"Total Time: {results['total_time']:.2f}s")
    
    # Generate and save report
    report = generate_quality_report(results)
    
    with open("QUALITY_GATES_REPORT.md", "w") as f:
        f.write(report)
    
    print(f"\n📝 Detailed report saved to: QUALITY_GATES_REPORT.md")
    
    # Final assessment
    if results["failed_gates"] == 0:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        print("✨ System is ready for production deployment!")
    else:
        print(f"\n⚠️  {results['failed_gates']} quality gates failed.")
        print("🔧 Please review and fix issues before deployment.")
    
    return results["failed_gates"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)