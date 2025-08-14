"""Enhanced integration tests for the self-healing pipeline system.

Tests the complete end-to-end functionality including all enhancements
from Generation 1, 2, and 3 implementations.
"""

import asyncio
import json
import logging
import os
import sys
# import pytest  # Not available in this environment
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from healing_guard.core.quantum_planner import QuantumTaskPlanner, Task, TaskPriority, TaskStatus
from healing_guard.core.failure_detector import FailureDetector, FailureType, SeverityLevel
from healing_guard.core.healing_engine import HealingEngine, HealingStatus
from healing_guard.core.advanced_scaling import auto_scaler, WorkerNode
from healing_guard.security.advanced_security import security_manager
from healing_guard.monitoring.enhanced_monitoring import monitoring_dashboard

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestQuantumPlannerEnhanced:
    """Test enhanced quantum planner functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.planner = QuantumTaskPlanner(
            max_parallel_tasks=4,
            resource_limits={"cpu": 8.0, "memory": 16.0},
            optimization_iterations=50  # Reduced for faster tests
        )
    
    def test_enhanced_task_execution(self):
        """Test enhanced task execution with learning."""
        # Create test tasks
        tasks = [
            Task(
                id="task_1",
                name="Build Application",
                priority=TaskPriority.HIGH,
                estimated_duration=5.0,
                resources_required={"cpu": 2.0, "memory": 4.0}
            ),
            Task(
                id="task_2", 
                name="Run Tests",
                priority=TaskPriority.MEDIUM,
                estimated_duration=3.0,
                dependencies=["task_1"],
                resources_required={"cpu": 1.0, "memory": 2.0}
            ),
            Task(
                id="task_3",
                name="Deploy",
                priority=TaskPriority.CRITICAL,
                estimated_duration=2.0,
                dependencies=["task_1", "task_2"],
                resources_required={"cpu": 1.0, "memory": 1.0}
            )
        ]
        
        # Add tasks to planner
        for task in tasks:
            self.planner.add_task(task)
        
        # Create execution plan
        plan = asyncio.run(self.planner.create_execution_plan())
        
        # Validate plan
        assert plan is not None
        assert len(plan.tasks) == 3
        assert plan.estimated_total_time > 0
        assert plan.success_probability > 0
        assert len(plan.parallel_stages) > 0
        
        # Check that dependencies are respected
        task_positions = {task_id: stage_idx for stage_idx, stage in enumerate(plan.parallel_stages) for task_id in stage}
        
        # task_2 should come after task_1
        assert task_positions["task_2"] > task_positions["task_1"]
        # task_3 should come after both task_1 and task_2
        assert task_positions["task_3"] > task_positions["task_1"]
        assert task_positions["task_3"] > task_positions["task_2"]
        
        logger.info(f"Enhanced planning test passed: {plan.estimated_total_time:.1f}min, {plan.success_probability:.1%} success rate")
    
    def test_multi_phase_optimization(self):
        """Test multi-phase optimization (SA + GA + Local Search)."""
        # Create complex task graph
        tasks = []
        for i in range(8):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                priority=TaskPriority.MEDIUM,
                estimated_duration=float(i + 1),
                resources_required={"cpu": 1.0, "memory": 1.0}
            )
            
            # Add some dependencies
            if i > 0:
                task.dependencies = [f"task_{i-1}"]
            if i > 2:
                task.dependencies.append(f"task_{i-2}")
            
            tasks.append(task)
            self.planner.add_task(task)
        
        plan = asyncio.run(self.planner.create_execution_plan())
        
        # Validate optimization worked
        assert plan is not None
        assert hasattr(plan, 'metadata')
        assert 'optimization_phases' in plan.metadata
        assert len(plan.metadata['optimization_phases']) >= 3  # SA, GA, Local Search
        
        logger.info(f"Multi-phase optimization test passed: {len(plan.metadata['optimization_phases'])} phases")
    
    def test_performance_learning(self):
        """Test performance learning and adaptation."""
        # Create and execute a task multiple times to build learning data
        for i in range(5):
            task = Task(
                id=f"learn_task_{i}",
                name="Learning Task",
                priority=TaskPriority.MEDIUM,
                estimated_duration=2.0
            )
            
            self.planner.add_task(task)
            
            # Simulate task execution
            result = asyncio.run(self.planner._execute_task(task.id))
            assert result is not None
            assert result['status'] == 'completed'
        
        # Check that learning data is being collected
        assert hasattr(self.planner, '_task_history')
        assert 'Learning Task' in self.planner._task_history
        
        # Get planning statistics with learning insights
        stats = self.planner.get_planning_statistics()
        assert 'task_performance_analytics' in stats
        assert stats['learning_enabled'] is True
        
        logger.info("Performance learning test passed")


class TestFailureDetectorEnhanced:
    """Test enhanced failure detection capabilities."""
    
    def setup_method(self):
        """Setup test environment."""
        self.detector = FailureDetector()
    
    def test_enhanced_log_analysis(self):
        """Test enhanced log feature extraction."""
        complex_logs = """
        ERROR 2024-01-15 10:30:15 [BuildService] Build failed with exit code 1
        java.lang.OutOfMemoryError: Java heap space
        \tat com.example.service.DataProcessor.processLargeDataset(DataProcessor.java:45)
        \tat com.example.service.BuildService.executeBuild(BuildService.java:123)
        \tat com.example.controller.BuildController.startBuild(BuildController.java:67)
        WARN 2024-01-15 10:30:16 [BuildService] Memory usage at 95%
        ERROR 2024-01-15 10:30:17 [NetworkClient] Connection timeout after 30 seconds
        CRITICAL 2024-01-15 10:30:18 [SecurityScanner] High severity vulnerability found: CVE-2023-12345
        """
        
        # Extract enhanced features
        features = self.detector._extract_log_features(complex_logs)
        
        # Validate enhanced features
        assert 'weighted_error_score' in features
        assert 'memory_issue_indicators' in features
        assert 'network_issue_indicators' in features
        assert 'stack_trace_depth' in features
        assert 'programming_languages' in features
        assert 'log_complexity_score' in features
        
        # Check specific detections
        assert features['memory_issue_indicators'] > 0  # OutOfMemoryError detected
        assert features['network_issue_indicators'] > 0  # Connection timeout detected
        assert features['stack_trace_depth'] > 0  # Stack trace found
        assert 'java' in features['programming_languages']  # Java detected
        
        logger.info(f"Enhanced log analysis test passed: {len(features)} features extracted")
    
    def test_ml_enhanced_classification(self):
        """Test ML-enhanced failure classification."""
        # Memory exhaustion scenario
        memory_logs = """
        java.lang.OutOfMemoryError: Java heap space
        ERROR: Cannot allocate memory for buffer
        Process killed due to memory limit exceeded
        """
        
        failure_event = asyncio.run(self.detector.detect_failure(
            job_id="test_001",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            logs=memory_logs,
            context={"is_main_branch": True}
        ))
        
        assert failure_event is not None
        assert failure_event.failure_type in [FailureType.RESOURCE_EXHAUSTION, FailureType.COMPILATION_ERROR]
        assert failure_event.confidence > 0.5
        assert len(failure_event.matched_patterns) > 0
        assert len(failure_event.remediation_suggestions) > 0
        
        # Check enhanced metadata
        assert hasattr(failure_event, 'metadata')
        assert failure_event.metadata['ml_enhanced'] is True
        
        logger.info(f"ML classification test passed: {failure_event.failure_type.value} with {failure_event.confidence:.2f} confidence")
    
    def test_adaptive_learning(self):
        """Test adaptive learning from feedback."""
        # Create initial failure
        logs = "Build failed with dependency resolution error"
        
        failure_event = asyncio.run(self.detector.detect_failure(
            job_id="test_002",
            repository="test/repo", 
            branch="develop",
            commit_sha="def456",
            logs=logs
        ))
        
        initial_confidence = failure_event.confidence
        
        # Provide feedback
        feedback = {
            "correct_failure_type": FailureType.DEPENDENCY_FAILURE.value,
            "user_rating": 4
        }
        
        learning_result = asyncio.run(self.detector.learn_from_feedback(failure_event.id, feedback))
        assert learning_result is True
        
        # Check that learning models were updated
        assert hasattr(self.detector, '_pattern_effectiveness')
        
        logger.info(f"Adaptive learning test passed: initial confidence {initial_confidence:.2f}")


class TestHealingEngineEnhanced:
    """Test enhanced healing engine capabilities."""
    
    def setup_method(self):
        """Setup test environment."""
        self.planner = QuantumTaskPlanner(optimization_iterations=50)
        self.detector = FailureDetector()
        self.engine = HealingEngine(self.planner, self.detector)
    
    def test_comprehensive_healing_workflow(self):
        """Test complete healing workflow with all enhancements."""
        # Create complex failure scenario
        complex_logs = """
        ERROR: npm install failed with exit code 1
        npm ERR! code ERESOLVE
        npm ERR! ERESOLVE could not resolve dependency
        npm ERR! peer dep missing: react@^18.0.0
        ERROR: Memory usage exceeded 90% during dependency resolution
        WARN: Connection timeout to registry.npmjs.org
        """
        
        # Detect failure
        failure_event = asyncio.run(self.detector.detect_failure(
            job_id="complex_001",
            repository="frontend/app",
            branch="feature/upgrade",
            commit_sha="xyz789",
            logs=complex_logs,
            context={"is_feature_branch": True, "build_env": "ci"}
        ))
        
        assert failure_event is not None
        
        # Create healing plan
        healing_plan = asyncio.run(self.engine.create_healing_plan(failure_event))
        
        assert healing_plan is not None
        assert len(healing_plan.actions) > 0
        assert healing_plan.estimated_total_time > 0
        assert healing_plan.success_probability > 0
        
        # Execute healing plan
        healing_result = asyncio.run(self.engine.execute_healing_plan(healing_plan))
        
        assert healing_result is not None
        assert healing_result.status in [HealingStatus.SUCCESSFUL, HealingStatus.PARTIAL, HealingStatus.FAILED]
        assert len(healing_result.actions_executed) > 0
        
        logger.info(f"Comprehensive healing test passed: {healing_result.status.value}")
    
    def test_resilience_integration(self):
        """Test integration with resilience mechanisms."""
        # Create failure that should trigger resilience features
        logs = "Intermittent network failure during API call"
        
        failure_event = asyncio.run(self.detector.detect_failure(
            job_id="resilience_001", 
            repository="api/service",
            branch="main",
            commit_sha="resilience123",
            logs=logs
        ))
        
        # Execute healing with resilience
        healing_result = asyncio.run(self.engine.heal_failure(failure_event))
        
        assert healing_result is not None
        # Check that resilience metadata is present
        if hasattr(healing_result, 'metadata') and 'resilient' in str(healing_result.metadata):
            logger.info("Resilience integration detected in healing result")
        
        # Get healing statistics
        stats = self.engine.get_healing_statistics()
        assert 'total_healings' in stats
        assert stats['total_healings'] > 0
        
        logger.info(f"Resilience integration test passed: {stats['total_healings']} healings recorded")


class TestAdvancedScaling:
    """Test advanced scaling and load balancing."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset auto_scaler state
        auto_scaler.load_balancer.nodes.clear()
        
    def test_adaptive_load_balancing(self):
        """Test adaptive load balancing strategies."""
        lb = auto_scaler.load_balancer
        
        # Add test nodes
        nodes = [
            WorkerNode(id="node_1", capacity=10),
            WorkerNode(id="node_2", capacity=15),
            WorkerNode(id="node_3", capacity=8)
        ]
        
        for node in nodes:
            lb.add_node(node)
        
        # Test node selection with different strategies
        selected_node = lb.select_node({"session_id": "test_session"})
        assert selected_node is not None
        assert selected_node.id in [n.id for n in nodes]
        
        # Simulate request processing
        lb.record_request_result(selected_node.id, response_time=0.5, had_error=False)
        
        # Get load balancer stats
        stats = lb.get_load_balancer_stats()
        assert stats['total_nodes'] == 3
        assert stats['healthy_nodes'] == 3
        assert stats['request_counter'] >= 1
        
        logger.info(f"Load balancing test passed: {stats['healthy_nodes']}/{stats['total_nodes']} healthy nodes")
    
    def test_predictive_scaling(self):
        """Test predictive auto-scaling."""
        scaler = auto_scaler.predictive_scaler
        
        # Simulate high load scenario
        for i in range(10):
            scaler.update_metrics(
                cpu=75 + i * 2,  # Increasing CPU
                memory=60 + i * 3,  # Increasing memory  
                req_rate=50 + i * 5,  # Increasing requests
                resp_time=1 + i * 0.2,  # Increasing response time
                queue_depth=2 + i,  # Increasing queue
                error_rate=0.5 + i * 0.1  # Increasing errors
            )
        
        # Predict scaling need
        direction, confidence, analysis = scaler.predict_scaling_need()
        
        # Should recommend scaling up due to high metrics
        assert direction is not None
        assert confidence > 0
        assert 'analysis' in analysis
        
        logger.info(f"Predictive scaling test passed: {direction.value} with confidence {confidence:.2f}")
    
    def test_resource_optimization(self):
        """Test resource pool optimization."""
        optimizer = auto_scaler.resource_optimizer
        
        # Create test resource pools
        optimizer.create_resource_pool("test_workers", "thread", 4, 16)
        optimizer.create_resource_pool("test_processors", "process", 2, 8)
        
        # Simulate resource usage
        for i in range(5):
            resource_id = optimizer.allocate_resource("test_workers", duration_estimate=2.0)
            assert resource_id is not None
            
            # Simulate some work
            time.sleep(0.01)
            
            optimizer.release_resource("test_workers", resource_id, actual_duration=1.5 + i * 0.1)
        
        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        assert stats['total_pools'] == 2
        assert 'test_workers' in stats['pool_stats']
        
        logger.info(f"Resource optimization test passed: {stats['total_pools']} pools managed")


class TestSecurityIntegration:
    """Test security system integration."""
    
    def test_threat_detection(self):
        """Test threat detection capabilities."""
        # Simulate malicious request
        malicious_request = {
            "source_ip": "192.168.1.100",
            "content": "'; DROP TABLE users; --",
            "headers": {"user-agent": "curl/7.68.0"},
            "params": {"search": "<script>alert('xss')</script>"}
        }
        
        threat_analysis = asyncio.run(security_manager.threat_detector.analyze_request(malicious_request))
        
        assert threat_analysis is not None
        assert len(threat_analysis['threats_detected']) > 0
        assert threat_analysis['recommended_action'] in ['allow', 'monitor', 'quarantine', 'block']
        
        logger.info(f"Threat detection test passed: {len(threat_analysis['threats_detected'])} threats detected")
    
    def test_access_control(self):
        """Test role-based access control."""
        ac = security_manager.access_controller
        
        # Create test user
        user_created = ac.create_user("test_user", "secure_password_123", "operator", "test@example.com")
        assert user_created is True
        
        # Authenticate user
        session_token = ac.authenticate_user("test_user", "secure_password_123")
        assert session_token is not None
        
        # Test permissions
        has_read_permission = ac.check_permission(session_token, "read_healing_status")
        assert has_read_permission is True
        
        has_admin_permission = ac.check_permission(session_token, "manage_users")
        assert has_admin_permission is False  # Operator role shouldn't have admin permissions
        
        logger.info("Access control test passed")
    
    def test_comprehensive_security_processing(self):
        """Test comprehensive security request processing."""
        request_data = {
            "source_ip": "10.0.0.1",
            "user_id": "test_user",
            "resource": "healing_status",
            "action": "read",
            "content": "GET /api/v1/status",
            "headers": {"authorization": "Bearer valid_token"}
        }
        
        security_result = asyncio.run(security_manager.process_security_request(request_data))
        
        assert security_result is not None
        assert 'allowed' in security_result
        assert 'threat_analysis' in security_result
        assert 'rate_limit_status' in security_result
        
        logger.info(f"Comprehensive security test passed: request {'allowed' if security_result['allowed'] else 'blocked'}")


class TestMonitoringIntegration:
    """Test monitoring and observability integration."""
    
    def test_metrics_collection(self):
        """Test metrics collection and analysis."""
        # Record some test metrics
        for i in range(10):
            monitoring_dashboard.metrics_collector.record_metric(
                "test.execution_time",
                1.5 + i * 0.1,
                {"component": "test", "operation": "sample"}
            )
            
            monitoring_dashboard.metrics_collector.record_metric(
                "test.success_rate", 
                1.0 if i < 8 else 0.0,  # Simulate some failures
                {"component": "test"}
            )
        
        # Get metric statistics
        time_stats = monitoring_dashboard.metrics_collector.calculate_metric_stats("test.execution_time")
        assert time_stats['count'] == 10
        assert time_stats['min'] > 0
        assert time_stats['max'] > time_stats['min']
        
        # Get all metrics
        all_metrics = monitoring_dashboard.metrics_collector.get_all_metrics()
        assert 'test.execution_time' in all_metrics
        assert 'test.success_rate' in all_metrics
        
        logger.info(f"Metrics collection test passed: {len(all_metrics)} metrics tracked")
    
    def test_dashboard_data(self):
        """Test dashboard data aggregation."""
        dashboard_data = monitoring_dashboard.get_dashboard_data()
        
        assert 'timestamp' in dashboard_data
        assert 'system_status' in dashboard_data
        assert 'metrics' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'performance' in dashboard_data
        
        logger.info(f"Dashboard test passed: status = {dashboard_data['system_status']}")


class TestEndToEndIntegration:
    """Complete end-to-end integration tests."""
    
    def test_complete_system_workflow(self):
        """Test complete system workflow from failure to healing."""
        logger.info("Starting complete end-to-end integration test...")
        
        # 1. Initialize all components
        planner = QuantumTaskPlanner(optimization_iterations=25)
        detector = FailureDetector()
        engine = HealingEngine(planner, detector)
        
        # 2. Start monitoring (briefly)
        monitoring_dashboard.start()
        auto_scaler.start_monitoring()
        
        try:
            # 3. Simulate realistic failure scenario
            realistic_logs = """
            2024-01-15T14:30:45.123Z [ERROR] [BuildService] Maven build failed
            [ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.8.1:compile
            [ERROR] Compilation failure: Package org.springframework.boot does not exist
            [ERROR] Memory warning: Heap usage at 89% (1.2GB/1.4GB)
            [WARN] Network latency detected: 2.3s response time to artifact repository
            [INFO] Retry attempt 1/3 failed
            [ERROR] Build process terminated with exit code 1
            """
            
            # 4. Security check
            security_request = {
                "source_ip": "172.16.0.10",
                "content": realistic_logs[:200],  # First part of logs
                "headers": {"content-type": "application/json"},
                "resource": "healing_api"
            }
            
            security_result = asyncio.run(security_manager.process_security_request(security_request))
            assert security_result['allowed'] is True
            
            # 5. Failure detection and analysis
            failure_event = asyncio.run(detector.detect_failure(
                job_id="e2e_test_001",
                repository="microservice/order-service", 
                branch="release/v2.1.0",
                commit_sha="e2e123test",
                logs=realistic_logs,
                context={
                    "is_release_candidate": True,
                    "build_env": "production",
                    "retry_count": 1
                }
            ))
            
            assert failure_event is not None
            assert failure_event.failure_type in [FailureType.DEPENDENCY_FAILURE, FailureType.COMPILATION_ERROR]
            
            # 6. Healing plan creation and execution
            healing_result = asyncio.run(engine.heal_failure(failure_event))
            
            assert healing_result is not None
            assert healing_result.status in [HealingStatus.SUCCESSFUL, HealingStatus.PARTIAL]
            
            # 7. Record metrics
            monitoring_dashboard.metrics_collector.record_metric(
                "e2e_test.healing_duration",
                healing_result.total_duration,
                {"test_type": "integration", "status": healing_result.status.value}
            )
            
            # 8. Validate system state
            system_stats = {
                "failure_detection": detector.get_failure_statistics(),
                "healing_engine": engine.get_healing_statistics(),
                "quantum_planning": planner.get_planning_statistics(),
                "monitoring": monitoring_dashboard.get_dashboard_data()
            }
            
            # Validate key metrics
            assert system_stats['failure_detection']['total_failures'] > 0
            assert system_stats['healing_engine']['total_healings'] > 0
            assert system_stats['quantum_planning']['total_executions'] >= 0
            
            logger.info("✅ Complete end-to-end integration test PASSED")
            logger.info(f"   - Failure detected: {failure_event.failure_type.value}")
            logger.info(f"   - Healing status: {healing_result.status.value}")
            logger.info(f"   - Actions executed: {len(healing_result.actions_executed)}")
            logger.info(f"   - Total duration: {healing_result.total_duration:.2f} minutes")
            
        finally:
            # Cleanup
            monitoring_dashboard.stop()
            auto_scaler.stop_monitoring()
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        logger.info("Starting performance load test...")
        
        start_time = time.time()
        
        # Create multiple concurrent healing requests
        async def healing_task(task_id: int):
            planner = QuantumTaskPlanner(optimization_iterations=10)
            detector = FailureDetector()
            engine = HealingEngine(planner, detector)
            
            logs = f"Build error {task_id}: Dependency resolution failed"
            
            failure_event = await detector.detect_failure(
                job_id=f"load_test_{task_id}",
                repository=f"service_{task_id % 5}",  # 5 different services
                branch="main",
                commit_sha=f"load{task_id}",
                logs=logs
            )
            
            if failure_event:
                healing_result = await engine.heal_failure(failure_event)
                return healing_result.status.value
            return "failed"
        
        # Run concurrent healing tasks
        async def run_load_test():
            tasks = [healing_task(i) for i in range(5)]  # 5 concurrent healing operations
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = asyncio.run(run_load_test())
        
        execution_time = time.time() - start_time
        successful_healings = sum(1 for r in results if isinstance(r, str) and r != "failed")
        
        # Performance assertions
        assert execution_time < 60.0  # Should complete within 60 seconds
        assert successful_healings >= 3  # At least 60% success rate
        
        logger.info(f"✅ Performance load test PASSED")
        logger.info(f"   - Execution time: {execution_time:.2f} seconds")  
        logger.info(f"   - Successful healings: {successful_healings}/{len(results)}")
        logger.info(f"   - Throughput: {len(results)/execution_time:.2f} healings/second")


# Test execution
if __name__ == "__main__":
    # Run specific test classes
    test_classes = [
        TestQuantumPlannerEnhanced,
        TestFailureDetectorEnhanced,
        TestHealingEngineEnhanced,
        TestAdvancedScaling,
        TestSecurityIntegration,
        TestMonitoringIntegration,
        TestEndToEndIntegration
    ]
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    for test_class in test_classes:
        logger.info(f"\\n=== Running {test_class.__name__} ===")
        
        try:
            instance = test_class()
            
            # Run setup if it exists
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            
            # Run all test methods
            for method_name in dir(instance):
                if method_name.startswith('test_'):
                    try:
                        logger.info(f"Running {method_name}...")
                        method = getattr(instance, method_name)
                        method()
                        results["passed"] += 1
                        logger.info(f"✅ {method_name} PASSED")
                    except Exception as e:
                        results["failed"] += 1
                        error_msg = f"❌ {test_class.__name__}.{method_name} FAILED: {e}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        
        except Exception as e:
            results["failed"] += 1
            error_msg = f"❌ {test_class.__name__} setup/teardown FAILED: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
    
    # Final results
    logger.info(f"\\n🎯 TEST RESULTS SUMMARY:")
    logger.info(f"   ✅ Passed: {results['passed']}")
    logger.info(f"   ❌ Failed: {results['failed']}")
    logger.info(f"   📊 Success Rate: {results['passed']/(results['passed']+results['failed'])*100:.1f}%")
    
    if results["errors"]:
        logger.info(f"\\n💥 ERRORS:")
        for error in results["errors"]:
            logger.info(f"   {error}")
    
    # Exit with appropriate code
    exit(0 if results["failed"] == 0 else 1)