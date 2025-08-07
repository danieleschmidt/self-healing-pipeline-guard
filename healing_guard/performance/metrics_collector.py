"""Advanced metrics collection system for sentiment analysis and healing operations."""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
from functools import wraps

from ..core.config import settings
from ..monitoring.structured_logging import performance_logger

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricPoint:
    """A single metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """Histogram bucket for latency measurements."""
    upper_bound: float
    count: int = 0


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            metrics_collector.record_timer(
                self.metric_name,
                duration * 1000,  # Convert to milliseconds
                self.labels
            )


class SlidingWindow:
    """Sliding window for calculating rates and moving averages."""
    
    def __init__(self, window_size: timedelta):
        self.window_size = window_size
        self.data: deque = deque()
        self._lock = threading.Lock()
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the sliding window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.data.append((timestamp, value))
            self._cleanup_old_data()
    
    def get_rate(self) -> float:
        """Get rate (events per second) in the current window."""
        with self._lock:
            self._cleanup_old_data()
            if not self.data:
                return 0.0
            
            window_seconds = self.window_size.total_seconds()
            return len(self.data) / window_seconds
    
    def get_sum(self) -> float:
        """Get sum of all values in the current window."""
        with self._lock:
            self._cleanup_old_data()
            return sum(value for _, value in self.data)
    
    def get_average(self) -> float:
        """Get average of all values in the current window."""
        with self._lock:
            self._cleanup_old_data()
            if not self.data:
                return 0.0
            return self.get_sum() / len(self.data)
    
    def _cleanup_old_data(self):
        """Remove data points outside the window."""
        cutoff = datetime.now() - self.window_size
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()


class MetricsCollector:
    """Comprehensive metrics collector for the Healing Guard system."""
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._sliding_windows: Dict[str, SlidingWindow] = {}
        
        # Histogram buckets for latency measurements
        self._histogram_buckets = [
            5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
        ]  # milliseconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rate tracking
        self._rate_windows = {
            "sentiment_analysis_rate": SlidingWindow(timedelta(minutes=5)),
            "healing_plan_creation_rate": SlidingWindow(timedelta(minutes=5)),
            "healing_execution_rate": SlidingWindow(timedelta(minutes=5)),
            "error_rate": SlidingWindow(timedelta(minutes=5))
        }
        
        # Performance thresholds
        self._performance_thresholds = {
            "sentiment_analysis_latency_ms": 500,
            "healing_plan_creation_latency_ms": 2000,
            "healing_execution_latency_ms": 30000,
            "cache_hit_rate": 70,  # percentage
            "error_rate": 5  # per minute
        }
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            self._counters[metric_key] += value
        
        logger.debug(f"Counter incremented: {metric_key} += {value}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            self._gauges[metric_key] = value
        
        logger.debug(f"Gauge set: {metric_key} = {value}")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            self._histograms[metric_key].append(value)
            
            # Keep only recent values to manage memory
            if len(self._histograms[metric_key]) > 10000:
                self._histograms[metric_key] = self._histograms[metric_key][-5000:]
        
        logger.debug(f"Histogram recorded: {metric_key} = {value}")
    
    def record_timer(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer measurement."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            self._timers[metric_key].append(duration_ms)
            
            # Keep only recent values
            if len(self._timers[metric_key]) > 1000:
                self._timers[metric_key] = self._timers[metric_key][-500:]
        
        # Check performance thresholds
        threshold = self._performance_thresholds.get(name)
        if threshold and duration_ms > threshold:
            performance_logger.log_slow_operation(
                operation_name=name,
                duration_ms=duration_ms,
                threshold_ms=threshold,
                details=labels
            )
        
        logger.debug(f"Timer recorded: {metric_key} = {duration_ms}ms")
    
    def record_rate(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Record an event for rate calculation."""
        if name in self._rate_windows:
            self._rate_windows[name].add(1.0)
        
        # Also increment counter
        self.increment_counter(f"{name}_total", 1.0, labels)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        metric_key = self._create_metric_key(name, labels)
        with self._lock:
            return self._counters.get(metric_key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        metric_key = self._create_metric_key(name, labels)
        with self._lock:
            return self._gauges.get(metric_key)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            values = self._histograms.get(metric_key, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            values = self._timers.get(metric_key, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum_ms": sum(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": self._percentile(values, 95),
            "p99_ms": self._percentile(values, 99)
        }
    
    def get_rate(self, name: str) -> float:
        """Get rate (events per second)."""
        if name in self._rate_windows:
            return self._rate_windows[name].get_rate()
        return 0.0
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name.split("|")[0], self._parse_labels(name))
                    for name in self._histograms.keys()
                },
                "timers": {
                    name: self.get_timer_stats(name.split("|")[0], self._parse_labels(name))
                    for name in self._timers.keys()
                },
                "rates": {
                    name: window.get_rate()
                    for name, window in self._rate_windows.items()
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return metrics
    
    def _create_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _parse_labels(self, metric_key: str) -> Optional[Dict[str, str]]:
        """Parse labels from a metric key."""
        if "|" not in metric_key:
            return None
        
        _, label_str = metric_key.split("|", 1)
        labels = {}
        
        for pair in label_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                labels[key] = value
        
        return labels if labels else None
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        
        # Interpolate between adjacent values
        lower = int(index)
        upper = lower + 1
        weight = index - lower
        
        if upper >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class SentimentAnalysisMetrics:
    """Specialized metrics for sentiment analysis operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_analysis_start(self, analysis_type: str = "single"):
        """Record start of sentiment analysis."""
        self.collector.increment_counter("sentiment_analysis_started_total", labels={
            "type": analysis_type
        })
        self.collector.record_rate("sentiment_analysis_rate")
    
    def record_analysis_complete(
        self,
        duration_ms: float,
        sentiment_label: str,
        confidence: float,
        text_length: int,
        analysis_type: str = "single",
        is_urgent: bool = False,
        is_frustrated: bool = False
    ):
        """Record completion of sentiment analysis."""
        labels = {
            "type": analysis_type,
            "sentiment": sentiment_label,
            "urgent": str(is_urgent),
            "frustrated": str(is_frustrated)
        }
        
        self.collector.increment_counter("sentiment_analysis_completed_total", labels=labels)
        self.collector.record_timer("sentiment_analysis_duration_ms", duration_ms, labels)
        self.collector.record_histogram("sentiment_confidence", confidence, labels)
        self.collector.record_histogram("sentiment_text_length", text_length, labels)
        
        # Record specific metrics for concerning sentiments
        if is_urgent:
            self.collector.increment_counter("urgent_sentiment_detected_total")
        if is_frustrated:
            self.collector.increment_counter("frustrated_sentiment_detected_total")
    
    def record_analysis_error(
        self,
        error_type: str,
        analysis_type: str = "single"
    ):
        """Record sentiment analysis error."""
        self.collector.increment_counter("sentiment_analysis_errors_total", labels={
            "type": analysis_type,
            "error_type": error_type
        })
        self.collector.record_rate("error_rate")
    
    def record_batch_analysis(
        self,
        batch_size: int,
        successful_count: int,
        failed_count: int,
        total_duration_ms: float,
        sentiment_distribution: Dict[str, int]
    ):
        """Record batch analysis metrics."""
        self.collector.increment_counter("sentiment_batch_analysis_total")
        self.collector.record_histogram("sentiment_batch_size", batch_size)
        self.collector.record_timer("sentiment_batch_duration_ms", total_duration_ms)
        self.collector.set_gauge("sentiment_batch_success_rate", 
                                successful_count / batch_size if batch_size > 0 else 0)
        
        # Record sentiment distribution
        for sentiment, count in sentiment_distribution.items():
            self.collector.record_histogram("sentiment_distribution", count, {"sentiment": sentiment})
    
    def record_cache_hit(self, cache_type: str = "sentiment"):
        """Record cache hit."""
        self.collector.increment_counter("cache_hits_total", labels={"type": cache_type})
    
    def record_cache_miss(self, cache_type: str = "sentiment"):
        """Record cache miss."""
        self.collector.increment_counter("cache_misses_total", labels={"type": cache_type})
    
    def get_cache_hit_rate(self, cache_type: str = "sentiment") -> float:
        """Calculate cache hit rate."""
        hits = self.collector.get_counter("cache_hits_total", {"type": cache_type})
        misses = self.collector.get_counter("cache_misses_total", {"type": cache_type})
        total = hits + misses
        
        return (hits / total * 100) if total > 0 else 0.0


class HealingEngineMetrics:
    """Specialized metrics for healing engine operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_healing_plan_created(
        self,
        failure_type: str,
        action_count: int,
        estimated_duration_minutes: float,
        success_probability: float,
        priority: int,
        sentiment_enhanced: bool = False
    ):
        """Record healing plan creation."""
        labels = {
            "failure_type": failure_type,
            "sentiment_enhanced": str(sentiment_enhanced)
        }
        
        self.collector.increment_counter("healing_plans_created_total", labels=labels)
        self.collector.record_histogram("healing_plan_action_count", action_count, labels)
        self.collector.record_histogram("healing_plan_estimated_duration_minutes", estimated_duration_minutes, labels)
        self.collector.record_histogram("healing_plan_success_probability", success_probability, labels)
        self.collector.record_histogram("healing_plan_priority", priority, labels)
        self.collector.record_rate("healing_plan_creation_rate")
    
    def record_sentiment_priority_adjustment(
        self,
        priority_change: int,
        sentiment_label: str,
        urgency_score: float
    ):
        """Record sentiment-based priority adjustment."""
        self.collector.increment_counter("sentiment_priority_adjustments_total", labels={
            "sentiment": sentiment_label
        })
        self.collector.record_histogram("priority_adjustment_magnitude", abs(priority_change))
        self.collector.record_histogram("sentiment_urgency_score", urgency_score)
    
    def record_healing_execution_start(self, action_count: int):
        """Record start of healing execution."""
        self.collector.increment_counter("healing_executions_started_total")
        self.collector.record_histogram("healing_execution_action_count", action_count)
        self.collector.record_rate("healing_execution_rate")
    
    def record_healing_execution_complete(
        self,
        success: bool,
        duration_minutes: float,
        successful_actions: int,
        failed_actions: int,
        rollback_performed: bool = False
    ):
        """Record completion of healing execution."""
        labels = {
            "success": str(success),
            "rollback_performed": str(rollback_performed)
        }
        
        self.collector.increment_counter("healing_executions_completed_total", labels=labels)
        self.collector.record_timer("healing_execution_duration_minutes", duration_minutes, labels)
        self.collector.record_histogram("healing_successful_actions", successful_actions, labels)
        self.collector.record_histogram("healing_failed_actions", failed_actions, labels)
        
        total_actions = successful_actions + failed_actions
        success_rate = (successful_actions / total_actions) if total_actions > 0 else 0
        self.collector.record_histogram("healing_action_success_rate", success_rate, labels)
    
    def record_healing_action_executed(
        self,
        strategy: str,
        success: bool,
        duration_seconds: float
    ):
        """Record individual healing action execution."""
        labels = {
            "strategy": strategy,
            "success": str(success)
        }
        
        self.collector.increment_counter("healing_actions_executed_total", labels=labels)
        self.collector.record_timer("healing_action_duration_seconds", duration_seconds, labels)


# Global metrics collector and specialized metrics
metrics_collector = MetricsCollector()
sentiment_metrics = SentimentAnalysisMetrics(metrics_collector)
healing_metrics = HealingEngineMetrics(metrics_collector)


# Decorators for automatic metrics collection
def collect_sentiment_metrics(analysis_type: str = "single"):
    """Decorator to automatically collect sentiment analysis metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            sentiment_metrics.record_analysis_start(analysis_type)
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful completion
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                if hasattr(result, 'label'):
                    # Single result
                    sentiment_metrics.record_analysis_complete(
                        duration_ms=duration_ms,
                        sentiment_label=result.label.value,
                        confidence=result.confidence,
                        text_length=len(args[1]) if len(args) > 1 else 0,
                        analysis_type=analysis_type,
                        is_urgent=result.is_urgent,
                        is_frustrated=result.is_frustrated
                    )
                elif isinstance(result, list) and result:
                    # Batch results
                    sentiment_distribution = {}
                    urgent_count = frustrated_count = 0
                    
                    for r in result:
                        if hasattr(r, 'label'):
                            label = r.label.value
                            sentiment_distribution[label] = sentiment_distribution.get(label, 0) + 1
                            if r.is_urgent:
                                urgent_count += 1
                            if r.is_frustrated:
                                frustrated_count += 1
                    
                    sentiment_metrics.record_batch_analysis(
                        batch_size=len(args[1]) if len(args) > 1 else len(result),
                        successful_count=len(result),
                        failed_count=0,
                        total_duration_ms=duration_ms,
                        sentiment_distribution=sentiment_distribution
                    )
                
                return result
                
            except Exception as e:
                # Record error
                sentiment_metrics.record_analysis_error(
                    error_type=type(e).__name__,
                    analysis_type=analysis_type
                )
                raise
        
        return wrapper
    return decorator


def collect_healing_metrics():
    """Decorator to automatically collect healing engine metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                duration = (time.perf_counter() - start_time) * 1000
                
                # Record metrics based on function name and result
                if "create_healing_plan" in func.__name__ and hasattr(result, 'actions'):
                    failure_event = args[1] if len(args) > 1 else kwargs.get('failure_event')
                    if failure_event:
                        healing_metrics.record_healing_plan_created(
                            failure_type=failure_event.failure_type.value if hasattr(failure_event.failure_type, 'value') else str(failure_event.failure_type),
                            action_count=len(result.actions),
                            estimated_duration_minutes=result.estimated_total_time,
                            success_probability=result.success_probability,
                            priority=result.priority,
                            sentiment_enhanced=True  # Assuming sentiment enhancement is used
                        )
                
                elif "execute" in func.__name__ and hasattr(result, 'status'):
                    healing_metrics.record_healing_execution_complete(
                        success=result.status.value == "successful" if hasattr(result.status, 'value') else result.status == "successful",
                        duration_minutes=result.total_duration,
                        successful_actions=len(result.actions_successful),
                        failed_actions=len(result.actions_failed),
                        rollback_performed=result.rollback_performed
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Healing operation failed: {e}")
                raise
        
        return wrapper
    return decorator


def time_operation(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with TimingContext(metric_name, labels):
                return await func(*args, **kwargs)
        return wrapper
    return decorator