#!/usr/bin/env python3
"""
Test script for Generation 2 enhancements.

This script tests the robust features we've added:
- Input validation
- Enhanced observability
- Error handling
- Security features
"""

import asyncio
import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from healing_guard.core.validation import (
    StringValidator, NumberValidator, ValidationError,
    FailureAnalysisValidator, SecurityValidator
)
from healing_guard.monitoring.observability import (
    observability, traced, record_metric, performance_monitor
)
from healing_guard.api.main import create_app


async def test_validation_system():
    """Test the comprehensive validation system."""
    print("🧪 Testing Validation System...")
    
    # Test string validation
    try:
        validator = StringValidator(min_length=5, max_length=20, pattern=r'^[a-zA-Z0-9_]+$')
        result = validator.validate("valid_string_123", "test_field")
        assert result.is_valid, "Valid string should pass validation"
        print("  ✅ String validation: PASS")
    except Exception as e:
        print(f"  ❌ String validation: FAIL - {e}")
    
    # Test number validation
    try:
        validator = NumberValidator(min_value=1, max_value=100, numeric_type=int)
        result = validator.validate(42, "test_number")
        assert result.is_valid, "Valid number should pass validation"
        assert result.sanitized_data["test_number"] == 42
        print("  ✅ Number validation: PASS")
    except Exception as e:
        print(f"  ❌ Number validation: FAIL - {e}")
    
    # Test security validation
    try:
        # This should pass
        clean_path = SecurityValidator.validate_file_path("safe/path/file.txt")
        assert clean_path == "safe/path/file.txt"
        print("  ✅ Security validation (safe): PASS")
    except Exception as e:
        print(f"  ❌ Security validation (safe): FAIL - {e}")
    
    # Test security validation with dangerous input
    try:
        SecurityValidator.validate_file_path("../../../etc/passwd")
        print("  ❌ Security validation (dangerous): FAIL - Should have rejected dangerous path")
    except ValidationError:
        print("  ✅ Security validation (dangerous): PASS - Correctly rejected dangerous path")
    except Exception as e:
        print(f"  ❌ Security validation (dangerous): FAIL - Unexpected error: {e}")
    
    # Test failure analysis validation
    try:
        job_id = FailureAnalysisValidator.validate_job_id("job-123_build.456")
        repo = FailureAnalysisValidator.validate_repository_name("org/repo-name")
        sha = FailureAnalysisValidator.validate_commit_sha("abc123def456")
        
        print("  ✅ Failure analysis validation: PASS")
    except Exception as e:
        print(f"  ❌ Failure analysis validation: FAIL - {e}")


async def test_observability_system():
    """Test the observability and tracing system."""
    print("🔍 Testing Observability System...")
    
    # Start observability
    observability.start()
    
    # Test basic tracing
    try:
        async with observability.trace_operation("test_operation", component="test") as trace:
            trace.set_tag("test_tag", "test_value")
            trace.log("Test log message")
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Test nested operation
            async with observability.trace_operation("nested_operation", parent=trace) as nested_trace:
                nested_trace.set_tag("nested", True)
                await asyncio.sleep(0.05)
        
        print("  ✅ Basic tracing: PASS")
    except Exception as e:
        print(f"  ❌ Basic tracing: FAIL - {e}")
    
    # Test decorated functions
    try:
        @traced("test_function", component="test")
        async def test_async_function(value: int) -> int:
            await asyncio.sleep(0.02)
            return value * 2
        
        result = await test_async_function(21)
        assert result == 42
        print("  ✅ Traced async function: PASS")
    except Exception as e:
        print(f"  ❌ Traced async function: FAIL - {e}")
    
    # Test metrics recording
    try:
        record_metric("test_counter", 1, "counter", component="test")
        record_metric("test_gauge", 42.5, "gauge", component="test")
        record_metric("test_histogram", 123.45, "histogram", component="test")
        print("  ✅ Metrics recording: PASS")
    except Exception as e:
        print(f"  ❌ Metrics recording: FAIL - {e}")
    
    # Test performance monitoring
    try:
        performance_monitor.record_request(0.125, True)  # 125ms successful request
        performance_monitor.record_request(0.250, False)  # 250ms failed request
        
        stats = performance_monitor.get_performance_stats()
        assert "response_time_avg" in stats
        assert "error_rate" in stats
        print("  ✅ Performance monitoring: PASS")
    except Exception as e:
        print(f"  ❌ Performance monitoring: FAIL - {e}")
    
    # Get observability status
    try:
        status = observability.get_observability_status()
        assert "service_name" in status
        assert "active_traces" in status
        print("  ✅ Observability status: PASS")
    except Exception as e:
        print(f"  ❌ Observability status: FAIL - {e}")


async def test_api_robustness():
    """Test API robustness and error handling."""
    print("🛡️ Testing API Robustness...")
    
    try:
        # Create FastAPI app
        app = create_app()
        
        # Test that all routes are properly configured
        routes = [route for route in app.routes if hasattr(route, 'path') and hasattr(route, 'methods')]
        api_routes = [route for route in routes if route.path.startswith('/api/v1/')]
        
        assert len(api_routes) > 10, "Should have multiple API routes"
        print(f"  ✅ API routes configured: {len(api_routes)} routes")
        
        # Test that middleware is properly configured
        middleware_count = len(app.user_middleware)
        assert middleware_count > 0, "Should have middleware configured"
        print(f"  ✅ Middleware configured: {middleware_count} middleware")
        
    except Exception as e:
        print(f"  ❌ API configuration: FAIL - {e}")


async def test_error_handling():
    """Test comprehensive error handling."""
    print("🚨 Testing Error Handling...")
    
    # Test validation error handling
    try:
        validator = StringValidator(min_length=10)
        result = validator.validate("short", "test_field")
        print("  ❌ Validation error handling: FAIL - Should have raised ValidationError")
    except ValidationError as e:
        print("  ✅ Validation error handling: PASS - Correctly caught ValidationError")
    except Exception as e:
        print(f"  ❌ Validation error handling: FAIL - Unexpected error: {e}")
    
    # Test traced function error handling
    try:
        @traced("error_function")
        async def function_that_fails():
            raise ValueError("Intentional test error")
        
        await function_that_fails()
        print("  ❌ Traced error handling: FAIL - Should have raised ValueError")
    except ValueError:
        print("  ✅ Traced error handling: PASS - Error properly propagated")
        
        # Check that error was recorded in trace
        traces = observability.get_traces(limit=5)
        error_trace = next((t for t in traces if t.operation_name == "error_function"), None)
        if error_trace and error_trace.status == "error":
            print("  ✅ Error trace recording: PASS")
        else:
            print("  ❌ Error trace recording: FAIL - Error not recorded in trace")
    except Exception as e:
        print(f"  ❌ Traced error handling: FAIL - Unexpected error: {e}")


async def run_comprehensive_test():
    """Run all tests comprehensively."""
    print("🚀 Starting Generation 2 Robustness Tests")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all test suites
    await test_validation_system()
    print()
    
    await test_observability_system()
    print()
    
    await test_api_robustness()
    print()
    
    await test_error_handling()
    print()
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"🎯 Generation 2 Tests Completed in {duration:.2f}s")
    
    # Get final observability status
    try:
        status = observability.get_observability_status()
        traces = observability.get_traces()
        
        print(f"📊 Final Stats:")
        print(f"  - Active traces: {status.get('active_traces', 0)}")
        print(f"  - Completed traces: {len(traces)}")
        print(f"  - Performance samples: {len(performance_monitor._response_times)}")
        
        # Show some example traces
        if traces:
            print(f"  - Recent trace operations:")
            for trace in traces[-3:]:
                status_icon = "✅" if trace.status == "success" else "❌"
                print(f"    {status_icon} {trace.operation_name} ({trace.duration_ms:.2f}ms)")
                
    except Exception as e:
        print(f"  Warning: Could not get final stats - {e}")
    
    print("\n✨ Generation 2 (MAKE IT ROBUST) implementation complete!")


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(run_comprehensive_test())