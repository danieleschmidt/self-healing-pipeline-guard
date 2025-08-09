"""Advanced observability system for the Healing Guard."""

import asyncio
import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Optional, Any, Callable, Union
from threading import Lock
from collections import defaultdict, deque

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Represents a trace event in our observability system."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "unknown"  # success, error, cancelled
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def finish(self, error: Optional[Exception] = None):
        """Mark the event as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if error:
            self.status = "error"
            self.error = str(error)
        else:
            self.status = "success"
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the event."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the event."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        })


class TraceContext:
    """Context manager for distributed tracing."""
    
    def __init__(self, operation_name: str, parent_trace: Optional['TraceContext'] = None):
        self.operation_name = operation_name
        self.parent_trace = parent_trace
        self.trace_id = self._generate_trace_id()
        self.span_id = self._generate_span_id()
        self.start_time = time.time()
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.error: Optional[Exception] = None
        
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the current span."""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the current span."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        })
    
    def set_error(self, error: Exception):
        """Mark the span as having an error."""
        self.error = error
        self.tags["error"] = True
        self.tags["error.type"] = type(error).__name__
        self.tags["error.message"] = str(error)
    
    def create_child_span(self, operation_name: str) -> 'TraceContext':
        """Create a child span."""
        return TraceContext(operation_name, parent_trace=self)
    
    def to_trace_event(self) -> TraceEvent:
        """Convert to a TraceEvent."""
        return TraceEvent(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_trace.span_id if self.parent_trace else None,
            operation_name=self.operation_name,
            start_time=self.start_time,
            tags=self.tags.copy(),
            logs=self.logs.copy(),
            error=str(self.error) if self.error else None
        )


@dataclass
class MetricPoint:
    """Represents a metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


class CustomMetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self._counters[key] += value
            
            # Store as metric point
            metric_point = MetricPoint(
                name=name,
                value=self._counters[key],
                timestamp=time.time(),
                tags=tags or {},
                metric_type="counter"
            )
            self._metrics[key].append(metric_point)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self._gauges[key] = value
            
            # Store as metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type="gauge"
            )
            self._metrics[key].append(metric_point)
    
    def observe_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Observe a value in a histogram."""
        with self._lock:
            key = self._make_key(name, tags or {})
            self._histograms[key].append(value)
            
            # Keep only recent values (last 1000)
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
            
            # Store as metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type="histogram"
            )
            self._metrics[key].append(metric_point)
    
    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create a unique key for the metric."""
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}" if tag_str else name
    
    def get_counter(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get current counter value."""
        with self._lock:
            key = self._make_key(name, tags or {})
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            key = self._make_key(name, tags or {})
            return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            key = self._make_key(name, tags or {})
            values = self._histograms.get(key, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "count": count,
                "sum": sum(sorted_values),
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p90": sorted_values[int(count * 0.9)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)]
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    key: self.get_histogram_stats("", {}) 
                    for key in self._histograms.keys()
                },
                "timestamp": time.time()
            }


class ObservabilityManager:
    """Central observability management system."""
    
    def __init__(self, service_name: str = "healing-guard"):
        self.service_name = service_name
        self.traces: deque = deque(maxlen=1000)
        self.metrics_collector = CustomMetricsCollector()
        self._current_traces: Dict[str, TraceContext] = {}
        self._lock = Lock()
        
        # Initialize OpenTelemetry if available
        if HAS_OTEL:
            self._init_otel()
        else:
            logger.warning("OpenTelemetry not available - using custom observability only")
    
    def _init_otel(self):
        """Initialize OpenTelemetry components."""
        try:
            # Set up tracing
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(self.service_name)
            
            # Set up metrics  
            metrics.set_meter_provider(MeterProvider())
            self.meter = metrics.get_meter(self.service_name)
            
            logger.info("OpenTelemetry initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.tracer = None
            self.meter = None
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **tags):
        """Context manager for tracing operations."""
        trace_context = TraceContext(operation_name)
        
        # Add tags
        for key, value in tags.items():
            trace_context.set_tag(key, value)
        
        # Store in current traces
        with self._lock:
            self._current_traces[trace_context.trace_id] = trace_context
        
        try:
            yield trace_context
            
            # Mark as successful
            trace_context.set_tag("status", "success")
            
        except Exception as e:
            # Mark as error
            trace_context.set_error(e)
            raise
            
        finally:
            # Finish the trace
            trace_event = trace_context.to_trace_event()
            trace_event.finish()
            
            # Store completed trace
            with self._lock:
                self.traces.append(trace_event)
                if trace_context.trace_id in self._current_traces:
                    del self._current_traces[trace_context.trace_id]
    
    def traced(self, operation_name: str = None, **default_tags):
        """Decorator for tracing function calls."""
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.trace_operation(op_name, **default_tags) as trace:
                        # Add function metadata
                        trace.set_tag("function.name", func.__name__)
                        trace.set_tag("function.module", func.__module__)
                        
                        try:
                            result = await func(*args, **kwargs)
                            trace.set_tag("result.type", type(result).__name__)
                            return result
                        except Exception as e:
                            trace.log(f"Function failed: {e}", level="error")
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # For sync functions, create trace manually
                    trace_context = TraceContext(op_name)
                    for key, value in default_tags.items():
                        trace_context.set_tag(key, value)
                    
                    trace_context.set_tag("function.name", func.__name__)
                    trace_context.set_tag("function.module", func.__module__)
                    
                    with self._lock:
                        self._current_traces[trace_context.trace_id] = trace_context
                    
                    try:
                        result = func(*args, **kwargs)
                        trace_context.set_tag("result.type", type(result).__name__)
                        trace_context.set_tag("status", "success")
                        return result
                        
                    except Exception as e:
                        trace_context.set_error(e)
                        raise
                        
                    finally:
                        trace_event = trace_context.to_trace_event()
                        trace_event.finish()
                        
                        with self._lock:
                            self.traces.append(trace_event)
                            if trace_context.trace_id in self._current_traces:
                                del self._current_traces[trace_context.trace_id]
                
                return sync_wrapper
        
        return decorator
    
    def record_metric(self, name: str, value: Union[int, float], metric_type: str = "gauge", **tags):
        """Record a metric."""
        if metric_type == "counter":
            self.metrics_collector.increment_counter(name, value, tags)
        elif metric_type == "gauge":
            self.metrics_collector.set_gauge(name, value, tags)
        elif metric_type == "histogram":
            self.metrics_collector.observe_histogram(name, value, tags)
        else:
            logger.warning(f"Unknown metric type: {metric_type}")
    
    def get_traces(self, limit: int = 100) -> List[TraceEvent]:
        """Get recent traces."""
        with self._lock:
            return list(self.traces)[-limit:]
    
    def get_active_traces(self) -> List[TraceContext]:
        """Get currently active traces."""
        with self._lock:
            return list(self._current_traces.values())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics_collector.get_all_metrics()
    
    def get_observability_status(self) -> Dict[str, Any]:
        """Get overall observability status."""
        with self._lock:
            active_traces = len(self._current_traces)
            completed_traces = len(self.traces)
        
        return {
            "service_name": self.service_name,
            "opentelemetry_enabled": HAS_OTEL,
            "active_traces": active_traces,
            "completed_traces": completed_traces,
            "metrics": self.metrics_collector.get_all_metrics(),
            "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0
        }
    
    def start(self):
        """Start the observability manager."""
        self._start_time = time.time()
        logger.info("Observability manager started")
    
    def stop(self):
        """Stop the observability manager."""
        logger.info("Observability manager stopped")


class PerformanceMonitor:
    """Advanced performance monitoring."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._response_times: deque = deque(maxlen=window_size)
        self._error_rates: deque = deque(maxlen=window_size)
        self._throughput: deque = deque(maxlen=window_size)
        self._lock = Lock()
    
    def record_request(self, response_time: float, success: bool):
        """Record a request's performance metrics."""
        with self._lock:
            self._response_times.append(response_time)
            self._error_rates.append(0 if success else 1)
            self._throughput.append(time.time())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            if not self._response_times:
                return {}
            
            response_times = list(self._response_times)
            error_rates = list(self._error_rates)
            
            # Calculate throughput (requests per second)
            now = time.time()
            recent_requests = [t for t in self._throughput if now - t < 60]  # Last minute
            throughput = len(recent_requests) / 60.0
            
            return {
                "response_time_avg": sum(response_times) / len(response_times),
                "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)],
                "response_time_p99": sorted(response_times)[int(len(response_times) * 0.99)],
                "error_rate": sum(error_rates) / len(error_rates),
                "throughput_rps": throughput,
                "window_size": len(response_times)
            }


# Global observability manager
observability = ObservabilityManager()
performance_monitor = PerformanceMonitor()

# Convenience decorators
def traced(operation_name: str = None, **tags):
    """Convenience decorator for tracing."""
    return observability.traced(operation_name, **tags)

def record_metric(name: str, value: Union[int, float], metric_type: str = "gauge", **tags):
    """Convenience function for recording metrics."""
    return observability.record_metric(name, value, metric_type, **tags)