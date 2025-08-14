"""Enhanced monitoring and observability system.

Provides comprehensive monitoring, alerting, and observability features
for the self-healing pipeline system with OpenTelemetry integration.
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import statistics

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric data types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Individual metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class Alert:
    """Alert definition and state."""
    id: str
    name: str
    description: str
    level: AlertLevel
    condition: str
    threshold: float
    metric: str
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    fired: bool = False
    last_fired: Optional[datetime] = None
    fire_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "metric": self.metric,
            "labels": self.labels,
            "enabled": self.enabled,
            "fired": self.fired,
            "last_fired": self.last_fired.isoformat() if self.last_fired else None,
            "fire_count": self.fire_count
        }


class MetricsCollector:
    """Advanced metrics collection and storage."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self.collection_interval = 10  # seconds
        self.running = False
        self.collection_thread = None
        
        # System metrics tracking
        self.system_metrics_enabled = True
        
        # OpenTelemetry setup
        self._setup_opentelemetry()
        
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing and metrics."""
        # Tracer setup
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Add OTLP exporter (would connect to observability backend)
        otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Metrics setup
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint="http://localhost:4317", insecure=True),
            export_interval_millis=10000
        )
        
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        self.meter = metrics.get_meter(__name__)
        
        # Create core metrics
        self.healing_success_counter = self.meter.create_counter(
            "healing_success_total",
            description="Total number of successful healing operations"
        )
        
        self.healing_duration_histogram = self.meter.create_histogram(
            "healing_duration_seconds",
            description="Duration of healing operations in seconds"
        )
        
        self.system_resource_gauge = self.meter.create_observable_gauge(
            "system_resource_usage",
            description="System resource usage",
            callbacks=[self._get_system_resources]
        )
    
    def _get_system_resources(self, options):
        """Callback for system resource metrics."""
        yield metrics.Observation(psutil.cpu_percent(), {"resource": "cpu"})
        yield metrics.Observation(psutil.virtual_memory().percent, {"resource": "memory"})
        yield metrics.Observation(psutil.disk_usage('/').percent, {"resource": "disk"})
    
    def start_collection(self):
        """Start background metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self.running:
            try:
                if self.system_metrics_enabled:
                    self._collect_system_metrics()
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        now = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.record_metric("system.cpu_usage", cpu_percent, {"component": "system"})
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric("system.memory_usage", memory.percent, {"component": "system"})
        self.record_metric("system.memory_available", memory.available / 1024**3, {"component": "system", "unit": "GB"})
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.record_metric("system.disk_usage", disk.percent, {"component": "system"})
        self.record_metric("system.disk_free", disk.free / 1024**3, {"component": "system", "unit": "GB"})
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.record_metric("system.network_bytes_sent", net_io.bytes_sent, {"component": "system", "direction": "sent"})
        self.record_metric("system.network_bytes_recv", net_io.bytes_recv, {"component": "system", "direction": "received"})
        
        # Process count
        process_count = len(psutil.pids())
        self.record_metric("system.process_count", process_count, {"component": "system"})
        
        # Load average (Unix-like systems only)
        try:
            load_avg = psutil.getloadavg()
            self.record_metric("system.load_average_1m", load_avg[0], {"component": "system", "period": "1m"})
            self.record_metric("system.load_average_5m", load_avg[1], {"component": "system", "period": "5m"})
            self.record_metric("system.load_average_15m", load_avg[2], {"component": "system", "period": "15m"})
        except AttributeError:
            # Windows doesn't have load average
            pass
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        labels = labels or {}
        point = MetricPoint(datetime.now(), value, labels)
        
        self.metrics[metric_name].append(point)
        
        # Update OpenTelemetry metrics
        if metric_name.startswith("healing."):
            if "success" in metric_name:
                self.healing_success_counter.add(1, labels)
            elif "duration" in metric_name:
                self.healing_duration_histogram.record(value, labels)
    
    def _cleanup_old_metrics(self):
        """Remove old metric points beyond retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        for metric_name in self.metrics:
            self.metrics[metric_name] = [
                point for point in self.metrics[metric_name]
                if point.timestamp > cutoff
            ]
    
    def get_metric_values(self, metric_name: str, start_time: datetime = None, end_time: datetime = None, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get metric values within time range and matching labels."""
        points = self.metrics.get(metric_name, [])
        
        # Filter by time range
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Filter by labels
        if labels:
            points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]
        
        return points
    
    def calculate_metric_stats(self, metric_name: str, start_time: datetime = None, end_time: datetime = None) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        points = self.get_metric_values(metric_name, start_time, end_time)
        
        if not points:
            return {"count": 0}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "sum": sum(values)
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with basic statistics."""
        result = {}
        
        for metric_name in self.metrics:
            stats = self.calculate_metric_stats(metric_name)
            latest_points = self.metrics[metric_name][-10:]  # Last 10 points
            
            result[metric_name] = {
                "stats": stats,
                "latest_value": latest_points[-1].value if latest_points else None,
                "latest_timestamp": latest_points[-1].timestamp.isoformat() if latest_points else None,
                "recent_points": [p.to_dict() for p in latest_points]
            }
        
        return result


class AlertManager:
    """Advanced alerting system with smart notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=5000)
        self.notification_handlers: Dict[str, Callable] = {}
        self.running = False
        self.check_thread = None
        self.check_interval = 30  # seconds
        
        # Initialize default alerts
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """Initialize default system alerts."""
        default_alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is above 80%",
                level=AlertLevel.WARNING,
                condition="greater_than",
                threshold=80.0,
                metric="system.cpu_usage"
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage", 
                description="System memory usage is above 85%",
                level=AlertLevel.WARNING,
                condition="greater_than",
                threshold=85.0,
                metric="system.memory_usage"
            ),
            Alert(
                id="low_disk_space",
                name="Low Disk Space",
                description="Disk usage is above 90%",
                level=AlertLevel.CRITICAL,
                condition="greater_than",
                threshold=90.0,
                metric="system.disk_usage"
            ),
            Alert(
                id="healing_failure_rate",
                name="High Healing Failure Rate",
                description="Healing operation failure rate is above 20%",
                level=AlertLevel.ERROR,
                condition="greater_than",
                threshold=20.0,
                metric="healing.failure_rate"
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.id] = alert
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        logger.info("Started alert monitoring")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        logger.info("Stopped alert monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_all_alerts(self):
        """Check all enabled alerts."""
        for alert in self.alerts.values():
            if alert.enabled:
                self._check_alert(alert)
    
    def _check_alert(self, alert: Alert):
        """Check individual alert condition."""
        # Get recent metric values (last 5 minutes)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        points = self.metrics_collector.get_metric_values(
            alert.metric, start_time, end_time, alert.labels
        )
        
        if not points:
            return
        
        # Use latest value for evaluation
        latest_value = points[-1].value
        
        # Evaluate condition
        condition_met = False
        if alert.condition == "greater_than":
            condition_met = latest_value > alert.threshold
        elif alert.condition == "less_than":
            condition_met = latest_value < alert.threshold
        elif alert.condition == "equals":
            condition_met = abs(latest_value - alert.threshold) < 0.001
        
        # Handle alert state changes
        if condition_met and not alert.fired:
            self._fire_alert(alert, latest_value)
        elif not condition_met and alert.fired:
            self._resolve_alert(alert, latest_value)
    
    def _fire_alert(self, alert: Alert, current_value: float):
        """Fire an alert."""
        alert.fired = True
        alert.last_fired = datetime.now()
        alert.fire_count += 1
        
        alert_event = {
            "id": alert.id,
            "name": alert.name,
            "level": alert.level.value,
            "description": alert.description,
            "current_value": current_value,
            "threshold": alert.threshold,
            "timestamp": alert.last_fired.isoformat(),
            "action": "fired"
        }
        
        self.alert_history.append(alert_event)
        logger.warning(f"Alert fired: {alert.name} (value: {current_value}, threshold: {alert.threshold})")
        
        # Send notifications
        self._send_notifications(alert_event)
    
    def _resolve_alert(self, alert: Alert, current_value: float):
        """Resolve an alert."""
        alert.fired = False
        
        alert_event = {
            "id": alert.id,
            "name": alert.name,
            "level": alert.level.value,
            "description": f"Resolved: {alert.description}",
            "current_value": current_value,
            "threshold": alert.threshold,
            "timestamp": datetime.now().isoformat(),
            "action": "resolved"
        }
        
        self.alert_history.append(alert_event)
        logger.info(f"Alert resolved: {alert.name} (value: {current_value})")
        
        # Send resolution notifications
        self._send_notifications(alert_event)
    
    def _send_notifications(self, alert_event: Dict[str, Any]):
        """Send alert notifications through registered handlers."""
        for handler_name, handler_func in self.notification_handlers.items():
            try:
                handler_func(alert_event)
            except Exception as e:
                logger.error(f"Failed to send notification via {handler_name}: {e}")
    
    def register_notification_handler(self, name: str, handler: Callable):
        """Register a notification handler."""
        self.notification_handlers[name] = handler
        logger.info(f"Registered notification handler: {name}")
    
    def create_alert(self, alert: Alert):
        """Create a new alert."""
        self.alerts[alert.id] = alert
        logger.info(f"Created alert: {alert.name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        return [alert.to_dict() for alert in self.alerts.values() if alert.fired]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.alert_history
            if datetime.fromisoformat(event["timestamp"]) > cutoff
        ]


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_traces: Dict[str, Dict[str, Any]] = {}
    
    def start_trace(self, trace_name: str, context: Dict[str, Any] = None):
        """Start a performance trace."""
        trace_id = f"{trace_name}_{int(time.time() * 1000)}"
        
        self.active_traces[trace_id] = {
            "name": trace_name,
            "start_time": time.time(),
            "context": context or {},
            "spans": []
        }
        
        return trace_id
    
    def add_span(self, trace_id: str, span_name: str, duration: float = None, metadata: Dict[str, Any] = None):
        """Add a span to an active trace."""
        if trace_id not in self.active_traces:
            return
        
        span = {
            "name": span_name,
            "timestamp": time.time(),
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self.active_traces[trace_id]["spans"].append(span)
    
    def end_trace(self, trace_id: str) -> Dict[str, Any]:
        """End a performance trace and return results."""
        if trace_id not in self.active_traces:
            return {}
        
        trace = self.active_traces.pop(trace_id)
        trace["end_time"] = time.time()
        trace["total_duration"] = trace["end_time"] - trace["start_time"]
        
        # Store profile
        self.profiles[trace["name"]].append(trace)
        
        # Keep only recent profiles
        if len(self.profiles[trace["name"]]) > 1000:
            self.profiles[trace["name"]] = self.profiles[trace["name"]][-500:]
        
        return trace
    
    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """Get performance analysis summary."""
        if function_name and function_name in self.profiles:
            profiles = self.profiles[function_name]
        else:
            profiles = []
            for func_profiles in self.profiles.values():
                profiles.extend(func_profiles)
        
        if not profiles:
            return {"message": "No performance data available"}
        
        durations = [p["total_duration"] for p in profiles]
        
        return {
            "function_name": function_name or "all",
            "call_count": len(profiles),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
            "total_time": sum(durations),
            "recent_calls": len([p for p in profiles if time.time() - p["start_time"] < 3600])
        }


class MonitoringDashboard:
    """Main monitoring orchestrator and dashboard."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.profiler = PerformanceProfiler()
        self.started = False
        
        # Register default notification handlers
        self._setup_default_notifications()
    
    def _setup_default_notifications(self):
        """Setup default notification handlers."""
        
        def log_notification(alert_event):
            level = alert_event["level"]
            message = f"Alert {alert_event['action']}: {alert_event['name']} - {alert_event['description']}"
            
            if level == "critical":
                logger.critical(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
            else:
                logger.info(message)
        
        def console_notification(alert_event):
            print(f"🚨 [{alert_event['level'].upper()}] {alert_event['name']}: {alert_event['description']}")
        
        self.alert_manager.register_notification_handler("log", log_notification)
        self.alert_manager.register_notification_handler("console", console_notification)
    
    def start(self):
        """Start all monitoring components."""
        if self.started:
            return
        
        self.metrics_collector.start_collection()
        self.alert_manager.start_monitoring()
        self.started = True
        
        logger.info("Monitoring dashboard started")
    
    def stop(self):
        """Stop all monitoring components."""
        if not self.started:
            return
        
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_monitoring()
        self.started = False
        
        logger.info("Monitoring dashboard stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy" if not self.alert_manager.get_active_alerts() else "degraded",
            "metrics": {
                "total_metrics": len(self.metrics_collector.metrics),
                "collection_active": self.metrics_collector.running,
                "retention_hours": self.metrics_collector.retention_hours
            },
            "alerts": {
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "total_alerts": len(self.alert_manager.alerts),
                "monitoring_active": self.alert_manager.running
            },
            "performance": {
                "active_traces": len(self.profiler.active_traces),
                "completed_profiles": sum(len(profiles) for profiles in self.profiler.profiles.values())
            },
            "recent_metrics": self._get_recent_key_metrics(),
            "active_alerts": self.alert_manager.get_active_alerts()
        }
    
    def _get_recent_key_metrics(self) -> Dict[str, Any]:
        """Get recent values for key metrics."""
        key_metrics = [
            "system.cpu_usage",
            "system.memory_usage", 
            "system.disk_usage",
            "healing.success_rate",
            "healing.avg_duration"
        ]
        
        recent_metrics = {}
        for metric in key_metrics:
            latest_points = self.metrics_collector.metrics.get(metric, [])
            if latest_points:
                recent_metrics[metric] = latest_points[-1].value
        
        return recent_metrics


# Global monitoring instance
monitoring_dashboard = MonitoringDashboard()