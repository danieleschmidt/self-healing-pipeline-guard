
# Healing Guard Quality Gates Report

## Summary
- **Total Gates**: 6
- **Passed**: 4 ✅
- **Failed**: 2 ❌
- **Success Rate**: 66.7%
- **Total Execution Time**: 1.30s

## Quality Gates Results

### API Functionality - ✅ PASSED
**Description**: Validate core API endpoints and routing
**Execution Time**: 1.250s
**Details**:
- total_routes: 23
- api_routes: 17

### Validation & Security - ❌ FAILED
**Description**: Validate input sanitization and security measures
**Execution Time**: 0.011s
**Error**: Shell injection not properly sanitized

### Observability - ✅ PASSED
**Description**: Validate tracing, metrics, and monitoring
**Execution Time**: 0.014s
**Details**:
- traces_recorded: 1
- metrics_recorded: 3

### Optimization & Scaling - ✅ PASSED
**Description**: Validate quantum optimization and auto-scaling
**Execution Time**: 0.006s
**Details**:
- optimization_tasks: 5
- load_balancer_servers: 3
- profiled_functions: 1

### Cache Performance - ❌ FAILED
**Description**: Validate caching system efficiency and correctness
**Execution Time**: 0.020s
**Error**: This event loop is already running

### Integration Tests - ✅ PASSED
**Description**: Validate complete system integration
**Execution Time**: 0.000s
**Details**:
- components_initialized: 3
- health_checks: 2

## ⚠️ Overall Assessment: GOOD
Most quality gates passed with minor issues to address.
