# Performance Optimization Guide

## Overview

This guide provides comprehensive performance optimization strategies for the Self-Healing Pipeline Guard, focusing on throughput, latency, and resource efficiency.

## Application Performance

### FastAPI Optimization

#### Async/Await Best Practices
```python
# Optimal async patterns
async def process_failure_analysis(failure_data: dict) -> AnalysisResult:
    async with httpx.AsyncClient() as client:
        # Concurrent API calls for better performance
        tasks = [
            client.get(f"/api/logs/{failure_data['job_id']}"),
            client.get(f"/api/metrics/{failure_data['repo']}"),
            client.get(f"/api/history/{failure_data['failure_type']}")
        ]
        responses = await asyncio.gather(*tasks)
    return analyze_responses(responses)
```

#### Connection Pooling Configuration
```python
# Optimized database connection pool
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True
}

# Redis connection pool optimization
REDIS_CONFIG = {
    "connection_pool_kwargs": {
        "max_connections": 50,
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {}
    }
}
```

### ML Model Performance

#### Model Loading Optimization
```python
# Lazy loading and caching strategy
class ModelManager:
    _models = {}
    
    @classmethod
    async def get_model(cls, model_type: str):
        if model_type not in cls._models:
            cls._models[model_type] = await cls._load_model(model_type)
        return cls._models[model_type]
    
    @classmethod
    async def _load_model(cls, model_type: str):
        # Optimized model loading with memory mapping
        return joblib.load(f"models/{model_type}.pkl", mmap_mode='r')
```

#### Batch Processing for ML Inference
```python
# Batch inference for better throughput
async def batch_failure_classification(failures: List[FailureData]) -> List[Classification]:
    batch_size = 32
    results = []
    
    for i in range(0, len(failures), batch_size):
        batch = failures[i:i + batch_size]
        batch_results = await classify_batch(batch)
        results.extend(batch_results)
    
    return results
```

## Database Optimization

### PostgreSQL Performance Tuning

#### Connection and Memory Settings
```sql
-- Optimized PostgreSQL configuration
-- postgresql.conf settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100
```

#### Index Optimization
```sql
-- Performance-critical indexes
CREATE INDEX CONCURRENTLY idx_pipeline_failures_repo_created 
ON pipeline_failures(repository, created_at DESC);

CREATE INDEX CONCURRENTLY idx_healing_actions_status_timestamp 
ON healing_actions(status, timestamp) 
WHERE status IN ('pending', 'in_progress');

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_active_pipelines 
ON pipelines(id, status) 
WHERE status = 'active';
```

#### Query Optimization
```sql
-- Optimized queries with proper joins and filtering
EXPLAIN (ANALYZE, BUFFERS) 
SELECT p.id, p.repository, f.failure_type, h.action_taken
FROM pipelines p
JOIN pipeline_failures f ON p.id = f.pipeline_id
LEFT JOIN healing_actions h ON f.id = h.failure_id
WHERE p.created_at >= NOW() - INTERVAL '7 days'
  AND p.status = 'failed'
ORDER BY p.created_at DESC
LIMIT 100;
```

### Redis Optimization

#### Memory Optimization
```python
# Efficient Redis usage patterns
import redis.asyncio as redis

class OptimizedRedisCache:
    def __init__(self):
        self.redis = redis.Redis(
            decode_responses=True,
            max_connections=20,
            socket_keepalive=True
        )
    
    async def cache_failure_analysis(self, key: str, data: dict, ttl: int = 3600):
        # Use hash for structured data to save memory
        pipe = self.redis.pipeline()
        pipe.hset(f"analysis:{key}", mapping=data)
        pipe.expire(f"analysis:{key}", ttl)
        await pipe.execute()
```

## Container and Infrastructure Optimization

### Docker Image Optimization

#### Multi-stage Build for Minimal Images
```dockerfile
# Optimized multi-stage Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry export -f requirements.txt --output requirements.txt

FROM python:3.11-slim as production
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "healing_guard.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Container Resource Optimization
```yaml
# docker-compose.yml resource limits
services:
  healing-guard:
    image: healing-guard:latest
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    environment:
      - WORKERS=4
      - MAX_REQUESTS=1000
      - MAX_REQUESTS_JITTER=50
```

### Kubernetes Optimization

#### Resource Requests and Limits
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healing-guard
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: healing-guard
        image: healing-guard:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Monitoring and Profiling

### Application Performance Monitoring

#### Custom Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Performance metrics
request_count = Counter('healing_guard_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('healing_guard_request_duration_seconds', 'Request duration')
active_healing_sessions = Gauge('healing_guard_active_sessions', 'Active healing sessions')

def monitor_performance(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            request_count.labels(method='POST', endpoint='/heal').inc()
            return result
        finally:
            request_duration.observe(time.time() - start_time)
    return wrapper
```

#### Memory Profiling Integration
```python
import tracemalloc
import asyncio
from typing import Dict, Any

class MemoryProfiler:
    def __init__(self):
        self.snapshots = {}
    
    async def start_profiling(self, session_id: str):
        tracemalloc.start()
        self.snapshots[session_id] = tracemalloc.take_snapshot()
    
    async def analyze_memory_usage(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.snapshots:
            return {"error": "No snapshot found"}
        
        current = tracemalloc.take_snapshot()
        top_stats = current.compare_to(self.snapshots[session_id], 'lineno')
        
        return {
            "top_memory_consumers": [
                {
                    "file": stat.traceback.format()[-1],
                    "size_mb": stat.size_diff / 1024 / 1024
                }
                for stat in top_stats[:10]
            ]
        }
```

## Performance Testing

### Load Testing Configuration
```python
# locust load testing configuration
from locust import HttpUser, task, between

class HealingGuardUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
    
    @task(3)
    def analyze_failure(self):
        self.client.post("/api/v1/analyze", json={
            "job_id": "test-job-123",
            "failure_type": "test_failure",
            "logs": "Sample failure logs"
        })
    
    @task(1)
    def get_healing_status(self):
        self.client.get("/api/v1/healing/status")
```

### Benchmark Scripts
```bash
#!/bin/bash
# Performance benchmark script

echo "Starting performance benchmarks..."

# API throughput test
echo "API Throughput Test:"
ab -n 1000 -c 10 -H "Authorization: Bearer $TOKEN" \
   -p failure_data.json -T application/json \
   http://localhost:8000/api/v1/analyze

# Database performance test
echo "Database Performance Test:"
poetry run python -m pytest tests/performance/ -v --benchmark-only

# Memory usage test
echo "Memory Usage Test:"
poetry run python scripts/memory_benchmark.py

# ML model inference test
echo "ML Model Performance Test:"
poetry run python scripts/ml_benchmark.py

echo "Benchmarks completed. Check results in performance_report.html"
```

## Caching Strategies

### Multi-level Caching Architecture
```python
class CacheManager:
    def __init__(self):
        self.memory_cache = {}  # L1 cache
        self.redis_cache = redis.Redis()  # L2 cache
        self.database = get_database()  # L3 persistent storage
    
    async def get_analysis_result(self, key: str):
        # L1: Memory cache (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # L2: Redis cache (fast)
        cached = await self.redis_cache.get(f"analysis:{key}")
        if cached:
            result = json.loads(cached)
            self.memory_cache[key] = result  # Populate L1
            return result
        
        # L3: Database (slowest, authoritative)
        result = await self.database.get_analysis(key)
        if result:
            # Populate all cache levels
            self.memory_cache[key] = result
            await self.redis_cache.setex(f"analysis:{key}", 3600, json.dumps(result))
        
        return result
```

## Performance Budgets and SLAs

### Service Level Objectives
- **API Response Time**: 95th percentile < 200ms
- **Healing Action Latency**: < 30 seconds
- **System Availability**: 99.9% uptime
- **Error Rate**: < 0.1% of requests
- **ML Model Inference**: < 5 seconds per analysis

### Performance Budget Monitoring
```python
# Automated performance budget checks
class PerformanceBudget:
    BUDGETS = {
        "api_response_time_p95": 0.2,  # 200ms
        "healing_action_latency": 30.0,  # 30 seconds
        "error_rate": 0.001,  # 0.1%
        "ml_inference_time": 5.0  # 5 seconds
    }
    
    async def check_budgets(self) -> Dict[str, bool]:
        metrics = await self.get_current_metrics()
        violations = {}
        
        for metric, budget in self.BUDGETS.items():
            current_value = metrics.get(metric, 0)
            violations[metric] = current_value <= budget
        
        return violations
```

---

**Performance Target**: Support 10,000+ concurrent healing sessions  
**Review Frequency**: Monthly optimization reviews  
**Last Updated**: January 2025