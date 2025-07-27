"""Prometheus metrics and monitoring utilities."""

import time
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)

# Create custom registry for the application
registry = CollectorRegistry()

# Application info
app_info = Info(
    'healing_guard_info',
    'Application information',
    registry=registry
)

# Request metrics
request_count = Counter(
    'healing_guard_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

request_duration = Histogram(
    'healing_guard_request_duration_seconds',
    'Time spent processing HTTP requests',
    ['method', 'endpoint'],
    registry=registry
)

# Pipeline healing metrics
healing_attempts = Counter(
    'healing_guard_healing_attempts_total',
    'Total number of healing attempts',
    ['ci_platform', 'failure_type', 'strategy'],
    registry=registry
)

healing_success = Counter(
    'healing_guard_healing_success_total',
    'Total number of successful healing attempts',
    ['ci_platform', 'failure_type', 'strategy'],
    registry=registry
)

healing_duration = Histogram(
    'healing_guard_healing_duration_seconds',
    'Time spent on healing attempts',
    ['ci_platform', 'strategy'],
    registry=registry
)

time_saved = Counter(
    'healing_guard_time_saved_minutes_total',
    'Total time saved through automated healing',
    ['ci_platform', 'strategy'],
    registry=registry
)

cost_saved = Counter(
    'healing_guard_cost_saved_usd_total',
    'Total cost saved through automated healing',
    ['ci_platform', 'strategy'],
    registry=registry
)

# Failure detection metrics
failures_detected = Counter(
    'healing_guard_failures_detected_total',
    'Total number of failures detected',
    ['ci_platform', 'failure_type'],
    registry=registry
)

failure_classification_accuracy = Gauge(
    'healing_guard_classification_accuracy',
    'Accuracy of failure classification model',
    ['model_version'],
    registry=registry
)

# ML model metrics
model_inference_duration = Histogram(
    'healing_guard_model_inference_seconds',
    'Time spent on ML model inference',
    ['model_name', 'model_version'],
    registry=registry
)

model_predictions = Counter(
    'healing_guard_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'prediction_type'],
    registry=registry
)

# Database metrics
db_connections_active = Gauge(
    'healing_guard_db_connections_active',
    'Number of active database connections',
    registry=registry
)

db_query_duration = Histogram(
    'healing_guard_db_query_duration_seconds',
    'Database query execution time',
    ['query_type'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'healing_guard_cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'healing_guard_cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

# Queue metrics
queue_size = Gauge(
    'healing_guard_queue_size',
    'Current queue size',
    ['queue_name'],
    registry=registry
)

queue_processing_duration = Histogram(
    'healing_guard_queue_processing_seconds',
    'Time spent processing queue items',
    ['queue_name', 'task_type'],
    registry=registry
)

# API integration metrics
api_requests = Counter(
    'healing_guard_api_requests_total',
    'Total API requests to external services',
    ['service', 'endpoint', 'status_code'],
    registry=registry
)

api_request_duration = Histogram(
    'healing_guard_api_request_duration_seconds',
    'External API request duration',
    ['service', 'endpoint'],
    registry=registry
)

# System health metrics
system_health = Enum(
    'healing_guard_system_health',
    'Overall system health status',
    states=['healthy', 'degraded', 'unhealthy'],
    registry=registry
)

uptime_seconds = Gauge(
    'healing_guard_uptime_seconds',
    'Application uptime in seconds',
    registry=registry
)

# Business metrics
repositories_monitored = Gauge(
    'healing_guard_repositories_monitored',
    'Number of repositories currently monitored',
    registry=registry
)

pipelines_healed_today = Gauge(
    'healing_guard_pipelines_healed_today',
    'Number of pipelines healed today',
    registry=registry
)

mttr_improvement = Gauge(
    'healing_guard_mttr_improvement_percent',
    'Mean time to recovery improvement percentage',
    ['time_period'],
    registry=registry
)


class MetricsCollector:
    """Centralized metrics collection and utilities."""
    
    def __init__(self):
        self.start_time = time.time()
        self._initialize_app_info()
    
    def _initialize_app_info(self):
        """Initialize application info metrics."""
        app_info.info({
            'version': '1.0.0',
            'build_time': '2024-01-01T00:00:00Z',
            'commit_hash': 'unknown',
            'environment': 'development'
        })
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_healing_attempt(
        self,
        ci_platform: str,
        failure_type: str,
        strategy: str,
        success: bool,
        duration: float,
        time_saved_minutes: float = 0,
        cost_saved_usd: float = 0
    ):
        """Record pipeline healing metrics."""
        healing_attempts.labels(
            ci_platform=ci_platform,
            failure_type=failure_type,
            strategy=strategy
        ).inc()
        
        if success:
            healing_success.labels(
                ci_platform=ci_platform,
                failure_type=failure_type,
                strategy=strategy
            ).inc()
            
            if time_saved_minutes > 0:
                time_saved.labels(
                    ci_platform=ci_platform,
                    strategy=strategy
                ).inc(time_saved_minutes)
            
            if cost_saved_usd > 0:
                cost_saved.labels(
                    ci_platform=ci_platform,
                    strategy=strategy
                ).inc(cost_saved_usd)
        
        healing_duration.labels(
            ci_platform=ci_platform,
            strategy=strategy
        ).observe(duration)
    
    def record_failure_detection(self, ci_platform: str, failure_type: str):
        """Record failure detection metrics."""
        failures_detected.labels(
            ci_platform=ci_platform,
            failure_type=failure_type
        ).inc()
    
    def record_model_inference(
        self,
        model_name: str,
        model_version: str,
        prediction_type: str,
        duration: float
    ):
        """Record ML model inference metrics."""
        model_inference_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
        
        model_predictions.labels(
            model_name=model_name,
            prediction_type=prediction_type
        ).inc()
    
    def record_db_query(self, query_type: str, duration: float):
        """Record database query metrics."""
        db_query_duration.labels(query_type=query_type).observe(duration)
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        if hit:
            cache_hits.labels(cache_type=cache_type).inc()
        else:
            cache_misses.labels(cache_type=cache_type).inc()
    
    def record_api_request(
        self,
        service: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """Record external API request metrics."""
        api_requests.labels(
            service=service,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        api_request_duration.labels(
            service=service,
            endpoint=endpoint
        ).observe(duration)
    
    def update_system_health(self, health_status: str):
        """Update system health status."""
        system_health.state(health_status)
    
    def update_uptime(self):
        """Update application uptime."""
        uptime = time.time() - self.start_time
        uptime_seconds.set(uptime)
    
    def update_business_metrics(
        self,
        repositories_count: int,
        pipelines_healed_today_count: int,
        mttr_improvement_percent: float
    ):
        """Update business-related metrics."""
        repositories_monitored.set(repositories_count)
        pipelines_healed_today.set(pipelines_healed_today_count)
        mttr_improvement.labels(time_period='daily').set(mttr_improvement_percent)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        self.update_uptime()
        return generate_latest(registry).decode('utf-8')


def timing_metric(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution and record metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        return async_wrapper if hasattr(func, '__await__') else sync_wrapper
    return decorator


def counter_metric(metric: Counter, labels: Optional[Dict[str, str]] = None):
    """Decorator to increment counter metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
                return result
            except Exception:
                # Don't increment on failure
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
                return result
            except Exception:
                # Don't increment on failure
                raise
        
        return async_wrapper if hasattr(func, '__await__') else sync_wrapper
    return decorator


# Global metrics collector instance
metrics_collector = MetricsCollector()