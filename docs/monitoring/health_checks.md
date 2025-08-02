# Health Check Configuration

This document describes the health check endpoints and monitoring setup for the Self-Healing Pipeline Guard.

## Health Check Endpoints

### Application Health

**Endpoint:** `GET /health`

**Response Format:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "environment": "production",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 45,
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "total": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 12,
      "memory_usage": "245MB",
      "connected_clients": 8
    },
    "external_apis": {
      "github": {
        "status": "healthy",
        "response_time_ms": 234,
        "rate_limit_remaining": 4500
      },
      "slack": {
        "status": "healthy",
        "response_time_ms": 156
      }
    },
    "ml_models": {
      "status": "healthy",
      "loaded_models": 3,
      "last_updated": "2024-01-15T08:00:00Z"
    }
  }
}
```

### Detailed Health Checks

**Endpoint:** `GET /health/detailed`

Provides comprehensive health information including:
- System resource usage
- Queue depths and processing rates
- Recent error rates
- Performance metrics

### Readiness Check

**Endpoint:** `GET /health/ready`

Returns 200 if the application is ready to serve traffic:
```json
{
  "ready": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Liveness Check

**Endpoint:** `GET /health/live`

Returns 200 if the application is alive and responsive:
```json
{
  "alive": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Kubernetes Health Checks

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healing-guard
spec:
  template:
    spec:
      containers:
      - name: healing-guard
        image: healing-guard:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
```

## Docker Health Checks

### Dockerfile Configuration

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  healing-guard:
    image: healing-guard:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Health Check Implementation

### FastAPI Health Router

```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import asyncio

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check."""
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_apis": await check_external_apis(),
        "ml_models": await check_ml_models()
    }
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "checks": checks
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Check if application is ready to serve traffic."""
    # Check critical dependencies
    database_ok = await quick_database_check()
    redis_ok = await quick_redis_check()
    
    ready = database_ok and redis_ok
    
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"ready": True, "timestamp": time.time()}

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Check if application is alive."""
    return {"alive": True, "timestamp": time.time()}
```

## Monitoring Integration

### Prometheus Metrics

Health check metrics are automatically exposed at `/metrics`:

```
# Application health status
healing_guard_health_status{check="database"} 1
healing_guard_health_status{check="redis"} 1
healing_guard_health_status{check="external_apis"} 1

# Response times
healing_guard_health_response_time_seconds{check="database"} 0.045
healing_guard_health_response_time_seconds{check="redis"} 0.012

# Resource usage
healing_guard_database_connections_active 5
healing_guard_database_connections_idle 15
healing_guard_redis_memory_usage_bytes 256000000
```

### Grafana Dashboard Queries

**Overall Health Status:**
```promql
healing_guard_health_status
```

**Average Response Times:**
```promql
avg(healing_guard_health_response_time_seconds) by (check)
```

**Database Connection Pool:**
```promql
healing_guard_database_connections_active / healing_guard_database_connections_total
```

## Alerting Rules

### Critical Alerts

```yaml
groups:
- name: healing-guard-health
  rules:
  - alert: HealingGuardDown
    expr: up{job="healing-guard"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Healing Guard is down"
      description: "Healing Guard has been down for more than 1 minute"

  - alert: HealthCheckFailing
    expr: healing_guard_health_status == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Health check failing"
      description: "Health check {{ $labels.check }} has been failing for 2 minutes"

  - alert: HighResponseTime
    expr: healing_guard_health_response_time_seconds > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High health check response time"
      description: "Health check {{ $labels.check }} response time is {{ $value }}s"
```

## Load Balancer Health Checks

### AWS Application Load Balancer

```json
{
  "HealthCheckPath": "/health/ready",
  "HealthCheckProtocol": "HTTP",
  "HealthCheckIntervalSeconds": 30,
  "HealthCheckTimeoutSeconds": 5,
  "HealthyThresholdCount": 2,
  "UnhealthyThresholdCount": 3,
  "Matcher": {
    "HttpCode": "200"
  }
}
```

### NGINX Health Check

```nginx
upstream healing_guard {
    server healing-guard-1:8000 max_fails=3 fail_timeout=30s;
    server healing-guard-2:8000 max_fails=3 fail_timeout=30s;
}

server {
    location /health {
        access_log off;
        proxy_pass http://healing_guard/health/ready;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Best Practices

### Health Check Guidelines

1. **Keep checks lightweight** - Health checks should complete quickly (< 1 second)
2. **Check critical dependencies** - Include database, cache, and essential external services
3. **Use different endpoints** - Separate liveness and readiness checks
4. **Return detailed information** - Include diagnostic data in responses
5. **Set appropriate timeouts** - Configure reasonable timeout values
6. **Monitor the monitors** - Alert on health check failures

### Common Pitfalls

- **Cascading failures** - Avoid health checks that depend on failing services
- **Resource exhaustion** - Don't perform expensive operations in health checks
- **False positives** - Ensure checks accurately reflect application state
- **Missing dependencies** - Include all critical external services
- **Poor error handling** - Handle exceptions gracefully in health checks

## Troubleshooting

### Health Check Failures

1. **Check application logs** for errors during health check execution
2. **Verify database connectivity** and connection pool status
3. **Test external API connectivity** and authentication
4. **Review resource usage** for memory/CPU constraints
5. **Check network connectivity** to dependent services

### Performance Issues

1. **Analyze response times** for individual health checks
2. **Review database query performance** in health checks
3. **Check external API rate limits** and response times
4. **Monitor system resource usage** during health checks

### False Alarms

1. **Review health check thresholds** and adjust if necessary
2. **Analyze historical data** to identify patterns
3. **Validate check logic** for accuracy
4. **Consider environmental factors** (network latency, load)