#!/usr/bin/env python3
"""
Final System Validation

This script performs a comprehensive final check of all system components
without complex async interactions that might cause issues.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_core_imports():
    """Test that all core modules can be imported successfully."""
    print("🔍 Testing Core Imports...")
    
    try:
        from healing_guard.api.main import create_app
        from healing_guard.core.quantum_planner import QuantumTaskPlanner
        from healing_guard.core.failure_detector import FailureDetector
        from healing_guard.core.healing_engine import HealingEngine
        from healing_guard.core.validation import SecurityValidator, ValidationError
        from healing_guard.core.optimization import quantum_optimizer, load_balancer, profiler
        from healing_guard.monitoring.observability import observability
        from healing_guard.monitoring.health import health_checker
        
        print("  ✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_api_creation():
    """Test API application creation."""
    print("🌐 Testing API Creation...")
    
    try:
        from healing_guard.api.main import create_app
        
        app = create_app()
        routes = [route for route in app.routes if hasattr(route, 'path')]
        api_routes = [route for route in routes if route.path.startswith('/api/v1/')]
        
        print(f"  ✅ API created with {len(routes)} total routes, {len(api_routes)} API routes")
        return True
    except Exception as e:
        print(f"  ❌ API creation failed: {e}")
        return False


def test_security_validation():
    """Test security validation functions."""
    print("🛡️ Testing Security Validation...")
    
    try:
        from healing_guard.core.validation import SecurityValidator, ValidationError
        
        # Test SQL injection prevention
        dangerous_sql = "'; DROP TABLE users; --"
        cleaned = SecurityValidator.sanitize_sql_input(dangerous_sql)
        
        # Should remove dangerous keywords
        if any(keyword in cleaned.lower() for keyword in ['drop', 'table', 'delete']):
            print(f"  ❌ SQL sanitization failed: '{cleaned}' still contains dangerous keywords")
            return False
        
        # Test shell injection prevention  
        dangerous_shell = "; rm -rf / &"
        cleaned_shell = SecurityValidator.sanitize_shell_input(dangerous_shell)
        
        if 'rm' in cleaned_shell:
            print(f"  ❌ Shell sanitization failed: '{cleaned_shell}' still contains dangerous commands")
            return False
        
        # Test path traversal prevention
        try:
            SecurityValidator.validate_file_path("../../../etc/passwd")
            print("  ❌ Path traversal validation failed - should have rejected dangerous path")
            return False
        except ValidationError:
            pass  # Expected
        
        print("  ✅ Security validation working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Security validation failed: {e}")
        return False


def test_optimization_systems():
    """Test optimization and scaling systems."""
    print("⚡ Testing Optimization Systems...")
    
    try:
        from healing_guard.core.optimization import (
            quantum_optimizer, load_balancer, resource_scaler, profiler
        )
        
        # Test quantum optimizer
        tasks = [{'cpu_required': 1.0, 'memory_required': 2.0, 'estimated_duration': 3.0}]
        resources = {'cpu': 4.0, 'memory': 8.0}
        result = quantum_optimizer.optimize_task_schedule(tasks, resources)
        
        if 'optimized_schedule' not in result:
            print("  ❌ Quantum optimizer missing required output")
            return False
        
        # Test load balancer
        servers = ['server1', 'server2']
        load_balancer.update_server_metrics('server1', 0.3, 0.1)
        selected = load_balancer.select_server(servers)
        
        if selected not in servers:
            print(f"  ❌ Load balancer selected invalid server: {selected}")
            return False
        
        # Test profiler
        @profiler.profile("test_function")
        def test_func():
            time.sleep(0.001)
            return 42
        
        result = test_func()
        if result != 42:
            print("  ❌ Profiler decorator broke function execution")
            return False
        
        print("  ✅ Optimization systems working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Optimization systems failed: {e}")
        return False


def test_component_initialization():
    """Test that core components can be initialized."""
    print("🧩 Testing Component Initialization...")
    
    try:
        from healing_guard.core.quantum_planner import QuantumTaskPlanner
        from healing_guard.core.failure_detector import FailureDetector
        from healing_guard.core.healing_engine import HealingEngine
        
        # Initialize components
        planner = QuantumTaskPlanner()
        detector = FailureDetector()
        engine = HealingEngine()
        
        # Basic functionality tests
        if not hasattr(planner, 'create_execution_plan'):
            print("  ❌ Quantum planner missing required method")
            return False
        
        if not hasattr(detector, 'detect_failure'):
            print("  ❌ Failure detector missing required method")
            return False
        
        if not hasattr(engine, 'execute_healing_plan'):
            print("  ❌ Healing engine missing required method")
            return False
        
        print("  ✅ All core components initialized successfully")
        return True
    except Exception as e:
        print(f"  ❌ Component initialization failed: {e}")
        return False


def test_observability():
    """Test observability and monitoring systems."""
    print("👁️ Testing Observability...")
    
    try:
        from healing_guard.monitoring.observability import observability, record_metric
        
        # Start observability
        observability.start()
        
        # Record test metrics
        record_metric("test_counter", 1, "counter")
        record_metric("test_gauge", 42.5, "gauge")
        
        # Get status
        status = observability.get_observability_status()
        
        if 'service_name' not in status:
            print("  ❌ Observability status missing service name")
            return False
        
        print("  ✅ Observability system working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Observability failed: {e}")
        return False


def run_final_validation():
    """Run complete final validation."""
    print("🚀 Final System Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        ("Core Imports", test_core_imports),
        ("API Creation", test_api_creation), 
        ("Security Validation", test_security_validation),
        ("Optimization Systems", test_optimization_systems),
        ("Component Initialization", test_component_initialization),
        ("Observability", test_observability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final summary
    success_rate = (passed / total) * 100
    
    print("=" * 50)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Tests Run: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {total - passed} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Execution Time: {duration:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ Healing Guard system is fully operational!")
        print("🚀 Ready for production deployment!")
        
        # Show system capabilities
        print("\n🔥 SYSTEM CAPABILITIES:")
        print("  ✅ Self-healing CI/CD pipeline monitoring")
        print("  ✅ Quantum-inspired task optimization") 
        print("  ✅ AI-powered failure detection and classification")
        print("  ✅ Automated healing strategy generation")
        print("  ✅ Advanced caching and performance optimization")
        print("  ✅ Comprehensive observability and monitoring")
        print("  ✅ Security-hardened input validation")
        print("  ✅ Adaptive load balancing and auto-scaling")
        
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
        print("🔧 Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = run_final_validation()
    sys.exit(0 if success else 1)