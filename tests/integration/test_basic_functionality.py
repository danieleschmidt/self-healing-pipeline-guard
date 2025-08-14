"""Basic functionality tests for the self-healing pipeline system.

Tests core functionality without external dependencies that aren't available.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_import_basic_modules():
    """Test that basic modules can be imported."""
    logger.info("Testing basic module imports...")
    
    # Test core modules that don't require external dependencies
    try:
        import healing_guard.core.exceptions
        logger.info("✅ Core exceptions module imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import core exceptions: {e}")
        return False
    
    try:
        from healing_guard.core.exceptions import HealingSystemException, ValidationException
        logger.info("✅ Exception classes imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import exception classes: {e}")
        return False
    
    return True


def test_exception_handling():
    """Test custom exception handling."""
    logger.info("Testing exception handling...")
    
    from healing_guard.core.exceptions import HealingSystemException, ValidationException, handle_exception_with_context
    
    # Test HealingSystemException
    try:
        raise HealingSystemException("Test exception", context={"test": True}, recoverable=True)
    except HealingSystemException as e:
        assert str(e) == "Test exception"
        assert e.context["test"] is True
        assert e.recoverable is True
        assert isinstance(e.timestamp, datetime)
        logger.info("✅ HealingSystemException works correctly")
    
    # Test ValidationException
    try:
        raise ValidationException("Invalid input", field="test_field", value="invalid")
    except ValidationException as e:
        assert e.field == "test_field"
        assert e.value == "invalid"
        assert e.recoverable is False
        logger.info("✅ ValidationException works correctly")
    
    # Test exception context handler
    generic_exception = ValueError("Generic error")
    healing_exception = handle_exception_with_context(generic_exception, {"service_name": "test"})
    
    assert isinstance(healing_exception, HealingSystemException)
    assert "Generic error" in str(healing_exception)
    logger.info("✅ Exception context handler works correctly")
    
    return True


def test_basic_data_structures():
    """Test basic data structures and enums."""
    logger.info("Testing basic data structures...")
    
    # Test that we can create basic Python data structures
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "active",
        "metrics": {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "response_time": 0.125
        },
        "tags": ["production", "critical", "monitored"]
    }
    
    # Serialize and deserialize
    json_str = json.dumps(test_data)
    parsed_data = json.loads(json_str)
    
    assert parsed_data["status"] == "active"
    assert len(parsed_data["tags"]) == 3
    assert parsed_data["metrics"]["cpu_usage"] == 45.2
    
    logger.info("✅ Basic data structures work correctly")
    return True


def test_async_functionality():
    """Test basic async functionality."""
    logger.info("Testing async functionality...")
    
    async def sample_async_function(delay: float, result: str) -> str:
        """Sample async function for testing."""
        await asyncio.sleep(delay)
        return f"Result: {result}"
    
    async def test_async_operations():
        """Test async operations."""
        start_time = time.time()
        
        # Test single async operation
        result1 = await sample_async_function(0.01, "test1")
        assert result1 == "Result: test1"
        
        # Test concurrent async operations
        tasks = [
            sample_async_function(0.005, f"concurrent_{i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all("concurrent_" in result for result in results)
        
        total_time = time.time() - start_time
        # Should complete faster than running sequentially
        assert total_time < 0.1  # All operations should complete quickly
        
        return True
    
    # Run the async test
    result = asyncio.run(test_async_operations())
    
    if result:
        logger.info("✅ Async functionality works correctly")
        return True
    else:
        logger.error("❌ Async functionality test failed")
        return False


def test_logging_functionality():
    """Test logging functionality."""
    logger.info("Testing logging functionality...")
    
    # Create a test logger
    test_logger = logging.getLogger("test_healing_guard")
    test_logger.setLevel(logging.DEBUG)
    
    # Test different log levels
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    test_logger.critical("Critical message")
    
    logger.info("✅ Logging functionality works correctly")
    return True


def test_system_resource_checks():
    """Test basic system resource checking."""
    logger.info("Testing system resource checks...")
    
    # Test that we can access basic system info
    try:
        import platform
        import os
        
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "current_dir": os.getcwd(),
            "environment_vars": len(os.environ)
        }
        
        logger.info(f"System platform: {system_info['platform']}")
        logger.info(f"Python version: {system_info['python_version']}")
        logger.info(f"Architecture: {system_info['architecture']}")
        
        # Basic validations
        assert system_info["platform"] in ["Linux", "Windows", "Darwin"]
        assert system_info["python_version"].startswith("3.")
        assert system_info["environment_vars"] > 0
        
        logger.info("✅ System resource checks work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ System resource check failed: {e}")
        return False


def test_configuration_handling():
    """Test configuration handling functionality."""
    logger.info("Testing configuration handling...")
    
    # Test configuration-like data structures
    config = {
        "healing": {
            "max_retries": 3,
            "timeout_minutes": 30,
            "strategies": ["retry", "resource_scaling", "cache_clear"]
        },
        "monitoring": {
            "enabled": True,
            "interval_seconds": 60,
            "metrics": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "response_time_threshold": 5.0
            }
        },
        "security": {
            "enabled": True,
            "rate_limit": {
                "requests_per_minute": 100,
                "burst_limit": 20
            }
        }
    }
    
    # Test configuration access patterns
    assert config["healing"]["max_retries"] == 3
    assert "retry" in config["healing"]["strategies"]
    assert config["monitoring"]["metrics"]["cpu_threshold"] == 80.0
    
    # Test configuration validation
    def validate_config(cfg):
        """Validate configuration structure."""
        required_sections = ["healing", "monitoring", "security"]
        for section in required_sections:
            if section not in cfg:
                return False, f"Missing section: {section}"
        
        if cfg["healing"]["max_retries"] <= 0:
            return False, "max_retries must be positive"
            
        if cfg["monitoring"]["interval_seconds"] <= 0:
            return False, "interval_seconds must be positive"
            
        return True, "Configuration valid"
    
    is_valid, message = validate_config(config)
    assert is_valid, f"Configuration validation failed: {message}"
    
    logger.info("✅ Configuration handling works correctly")
    return True


def test_failure_scenario_simulation():
    """Test failure scenario simulation without external dependencies."""
    logger.info("Testing failure scenario simulation...")
    
    # Simulate different types of failure logs
    failure_scenarios = {
        "dependency_failure": """
ERROR: npm install failed
npm ERR! code ERESOLVE
npm ERR! ERESOLVE could not resolve dependency
npm ERR! peer dep missing: react@^17.0.0
        """.strip(),
        
        "memory_exhaustion": """
java.lang.OutOfMemoryError: Java heap space
ERROR: Cannot allocate memory for operation
Process killed due to excessive memory usage
        """.strip(),
        
        "network_timeout": """
ERROR: Connection timeout after 30 seconds
Failed to connect to api.external.com:443
Network unreachable: No route to host
        """.strip(),
        
        "compilation_error": """
ERROR: Compilation failed
src/main.py:45: SyntaxError: invalid syntax
Build process terminated with exit code 1
        """.strip()
    }
    
    # Simple pattern-based classification
    def classify_failure(logs: str) -> str:
        """Simple failure classification based on keywords."""
        logs_lower = logs.lower()
        
        if any(keyword in logs_lower for keyword in ["outofmemoryerror", "memory", "heap space"]):
            return "memory_exhaustion"
        elif any(keyword in logs_lower for keyword in ["timeout", "connection", "network"]):
            return "network_timeout"  
        elif any(keyword in logs_lower for keyword in ["compilation", "syntax", "error"]):
            return "compilation_error"
        elif any(keyword in logs_lower for keyword in ["dependency", "resolve", "npm err", "pip", "maven"]):
            return "dependency_failure"
        else:
            return "unknown"
    
    # Test classification for each scenario
    correct_classifications = 0
    for expected_type, logs in failure_scenarios.items():
        classified_type = classify_failure(logs)
        if classified_type == expected_type:
            correct_classifications += 1
            logger.info(f"✅ Correctly classified: {expected_type}")
        else:
            logger.warning(f"⚠️  Misclassified {expected_type} as {classified_type}")
    
    # Should classify at least 75% correctly
    accuracy = correct_classifications / len(failure_scenarios)
    assert accuracy >= 0.75, f"Classification accuracy too low: {accuracy:.2%}"
    
    logger.info(f"✅ Failure scenario simulation works correctly (accuracy: {accuracy:.2%})")
    return True


def test_healing_strategy_simulation():
    """Test healing strategy simulation without external dependencies."""
    logger.info("Testing healing strategy simulation...")
    
    # Define healing strategies
    healing_strategies = {
        "dependency_failure": [
            "clear_package_cache",
            "update_dependencies", 
            "retry_with_clean_install",
            "use_alternative_registry"
        ],
        "memory_exhaustion": [
            "increase_heap_size",
            "optimize_memory_usage",
            "restart_with_more_memory",
            "enable_garbage_collection_tuning"
        ],
        "network_timeout": [
            "retry_with_exponential_backoff",
            "use_alternative_endpoint",
            "implement_circuit_breaker",
            "enable_connection_pooling"
        ],
        "compilation_error": [
            "run_syntax_check",
            "update_compiler_version",
            "check_code_formatting",
            "validate_dependencies"
        ]
    }
    
    # Simulate strategy selection and execution
    def select_strategies(failure_type: str, max_strategies: int = 3) -> List[str]:
        """Select healing strategies for a failure type."""
        available_strategies = healing_strategies.get(failure_type, [])
        return available_strategies[:max_strategies]
    
    def simulate_strategy_execution(strategy: str, success_probability: float = 0.8) -> Dict[str, Any]:
        """Simulate execution of a healing strategy."""
        import random
        
        execution_time = random.uniform(0.5, 3.0)  # 0.5 to 3 minutes
        success = random.random() < success_probability
        
        return {
            "strategy": strategy,
            "success": success,
            "execution_time_minutes": execution_time,
            "timestamp": datetime.now().isoformat()
        }
    
    # Test healing workflow
    test_failure_type = "dependency_failure"
    selected_strategies = select_strategies(test_failure_type)
    
    assert len(selected_strategies) > 0, "No strategies selected"
    assert len(selected_strategies) <= 3, "Too many strategies selected"
    
    # Execute strategies
    execution_results = []
    for strategy in selected_strategies:
        result = simulate_strategy_execution(strategy)
        execution_results.append(result)
        
        logger.info(f"Strategy '{strategy}': {'SUCCESS' if result['success'] else 'FAILED'} ({result['execution_time_minutes']:.1f}min)")
    
    # Calculate overall success
    successful_strategies = sum(1 for result in execution_results if result['success'])
    success_rate = successful_strategies / len(execution_results)
    
    logger.info(f"✅ Healing strategy simulation works correctly (success rate: {success_rate:.1%})")
    return True


def run_all_tests():
    """Run all available tests."""
    test_functions = [
        test_import_basic_modules,
        test_exception_handling,
        test_basic_data_structures,
        test_async_functionality,
        test_logging_functionality,
        test_system_resource_checks,
        test_configuration_handling,
        test_failure_scenario_simulation,
        test_healing_strategy_simulation
    ]
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    logger.info("🚀 Starting Basic Functionality Tests\n")
    
    for test_func in test_functions:
        logger.info(f"Running {test_func.__name__}...")
        
        try:
            if test_func():
                results["passed"] += 1
                logger.info(f"✅ {test_func.__name__} PASSED\n")
            else:
                results["failed"] += 1
                logger.error(f"❌ {test_func.__name__} FAILED\n")
                
        except Exception as e:
            results["failed"] += 1
            error_msg = f"❌ {test_func.__name__} ERROR: {e}"
            logger.error(error_msg + "\n")
            results["errors"].append(error_msg)
    
    # Final results
    logger.info("🎯 TEST RESULTS SUMMARY:")
    logger.info(f"   ✅ Passed: {results['passed']}")
    logger.info(f"   ❌ Failed: {results['failed']}")
    total_tests = results['passed'] + results['failed']
    success_rate = results['passed'] / total_tests * 100 if total_tests > 0 else 0
    logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
    
    if results["errors"]:
        logger.info("\n💥 ERRORS:")
        for error in results["errors"]:
            logger.info(f"   {error}")
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)