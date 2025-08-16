
# Healing Guard Quality Gates Report

## Summary
- **Total Gates**: 6
- **Passed**: 5 ✅
- **Failed**: 1 ❌
- **Success Rate**: 83.3%
- **Total Execution Time**: 4.07s

## Quality Gates Results

### API Functionality - ✅ PASSED
**Description**: Validate core API endpoints and routing
**Execution Time**: 3.929s
**Details**:
- total_routes: 23
- api_routes: 17

### Validation & Security - ✅ PASSED
**Description**: Validate input sanitization and security measures
**Execution Time**: 0.002s
**Details**:
- validation_tests: 4
- security_tests: 3

### Observability - ✅ PASSED
**Description**: Validate tracing, metrics, and monitoring
**Execution Time**: 0.112s
**Details**:
- traces_recorded: 1
- metrics_recorded: 3

### Optimization & Scaling - ✅ PASSED
**Description**: Validate quantum optimization and auto-scaling
**Execution Time**: 0.004s
**Details**:
- optimization_tasks: 5
- load_balancer_servers: 3
- profiled_functions: 1

### Cache Performance - ❌ FAILED
**Description**: Validate caching system efficiency and correctness
**Execution Time**: 0.009s
**Error**: This event loop is already running

### Integration Tests - ✅ PASSED
**Description**: Validate complete system integration
**Execution Time**: 0.008s
**Details**:
- components_initialized: 3
- health_checks: 2

## ⚠️ Overall Assessment: GOOD
Most quality gates passed with minor issues to address.
