"""Auto-scaling system for sentiment analysis workloads and healing operations."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .metrics_collector import metrics_collector, MetricType
from ..core.config import settings
from ..core.exceptions import ResourceException
from ..monitoring.structured_logging import performance_logger

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    SENTIMENT_ANALYZER_WORKERS = "sentiment_analyzer_workers"
    HEALING_ENGINE_WORKERS = "healing_engine_workers"
    CACHE_SIZE = "cache_size"
    THREAD_POOL_SIZE = "thread_pool_size"
    CONNECTION_POOL_SIZE = "connection_pool_size"


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    resource_type: ResourceType
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    scale_up_factor: float = 2.0
    scale_down_factor: float = 0.5
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    evaluation_window: timedelta = field(default_factory=lambda: timedelta(minutes=2))


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    resource_type: ResourceType
    action: ScalingAction
    previous_capacity: int
    new_capacity: int
    trigger_metric: str
    trigger_value: float
    reason: str


class LoadPredictor:
    """Predicts future load based on historical patterns."""
    
    def __init__(self, history_window: timedelta = timedelta(hours=24)):
        self.history_window = history_window
        self.load_history: List[tuple] = []  # (timestamp, load_value)
        self._lock = threading.Lock()
    
    def record_load(self, load_value: float, timestamp: Optional[datetime] = None):
        """Record current load measurement."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.load_history.append((timestamp, load_value))
            
            # Clean up old history
            cutoff = timestamp - self.history_window
            self.load_history = [
                (ts, load) for ts, load in self.load_history
                if ts > cutoff
            ]
    
    def predict_load(self, prediction_horizon: timedelta) -> float:
        """Predict load for a given time horizon."""
        with self._lock:
            if len(self.load_history) < 2:
                return 0.0
            
            # Simple linear regression for trend prediction
            now = datetime.now()
            recent_data = [
                (ts, load) for ts, load in self.load_history
                if (now - ts) <= timedelta(hours=1)  # Use last hour of data
            ]
            
            if len(recent_data) < 2:
                return self.load_history[-1][1]  # Return last known value
            
            # Calculate trend
            x_values = [(ts - recent_data[0][0]).total_seconds() for ts, _ in recent_data]
            y_values = [load for _, load in recent_data]
            
            # Simple linear regression
            n = len(recent_data)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return y_values[-1]  # No trend
            
            # Calculate slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict value at prediction horizon
            prediction_seconds = prediction_horizon.total_seconds()
            predicted_load = intercept + slope * prediction_seconds
            
            # Apply bounds (load can't be negative, and cap at reasonable maximum)
            return max(0.0, min(predicted_load, max(y_values) * 2))
    
    def get_load_trend(self) -> str:
        """Get current load trend (increasing, decreasing, stable)."""
        with self._lock:
            if len(self.load_history) < 10:
                return "stable"
            
            # Compare recent average to older average
            recent_values = [load for _, load in self.load_history[-5:]]
            older_values = [load for _, load in self.load_history[-10:-5]]
            
            if not older_values:
                return "stable"
            
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values)
            
            if recent_avg > older_avg * 1.1:
                return "increasing"
            elif recent_avg < older_avg * 0.9:
                return "decreasing"
            else:
                return "stable"


class AutoScaler:
    """Intelligent auto-scaling system for sentiment analysis and healing operations."""
    
    def __init__(self):
        self.scaling_rules: List[ScalingRule] = []
        self.scaling_history: List[ScalingEvent] = []
        self.load_predictors: Dict[ResourceType, LoadPredictor] = {}
        
        # Current resource capacities
        self.current_capacities: Dict[ResourceType, int] = {
            ResourceType.SENTIMENT_ANALYZER_WORKERS: 4,
            ResourceType.HEALING_ENGINE_WORKERS: 2,
            ResourceType.CACHE_SIZE: 10000,
            ResourceType.THREAD_POOL_SIZE: 8,
            ResourceType.CONNECTION_POOL_SIZE: 20
        }
        
        # Last scaling times for cooldown
        self.last_scaling_times: Dict[ResourceType, datetime] = {}
        
        # Scaling callbacks
        self.scaling_callbacks: Dict[ResourceType, Callable[[int], None]] = {}
        
        self._initialize_default_rules()
        self._initialize_predictors()
        
        # Background task
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        
        # Sentiment analyzer scaling based on request rate
        self.scaling_rules.append(ScalingRule(
            resource_type=ResourceType.SENTIMENT_ANALYZER_WORKERS,
            metric_name="sentiment_analysis_rate",
            scale_up_threshold=10.0,  # requests per second
            scale_down_threshold=2.0,
            min_instances=2,
            max_instances=16,
            scale_up_factor=1.5,
            scale_down_factor=0.7,
            cooldown_period=timedelta(minutes=3)
        ))
        
        # Sentiment analyzer scaling based on queue depth
        self.scaling_rules.append(ScalingRule(
            resource_type=ResourceType.SENTIMENT_ANALYZER_WORKERS,
            metric_name="sentiment_analysis_queue_depth",
            scale_up_threshold=20.0,  # queued requests
            scale_down_threshold=5.0,
            min_instances=2,
            max_instances=16,
            scale_up_factor=1.5,
            scale_down_factor=0.8,
            cooldown_period=timedelta(minutes=2)
        ))
        
        # Healing engine scaling
        self.scaling_rules.append(ScalingRule(
            resource_type=ResourceType.HEALING_ENGINE_WORKERS,
            metric_name="healing_execution_rate",
            scale_up_threshold=1.0,  # executions per second
            scale_down_threshold=0.1,
            min_instances=1,
            max_instances=8,
            scale_up_factor=2.0,
            scale_down_factor=0.5,
            cooldown_period=timedelta(minutes=5)
        ))
        
        # Cache scaling based on hit rate
        self.scaling_rules.append(ScalingRule(
            resource_type=ResourceType.CACHE_SIZE,
            metric_name="cache_hit_rate",
            scale_up_threshold=60.0,  # percentage (scale up if hit rate is low)
            scale_down_threshold=90.0,  # percentage (scale down if hit rate is very high)
            min_instances=5000,
            max_instances=100000,
            scale_up_factor=1.5,
            scale_down_factor=0.8,
            cooldown_period=timedelta(minutes=10)
        ))
        
        # Thread pool scaling based on utilization
        self.scaling_rules.append(ScalingRule(
            resource_type=ResourceType.THREAD_POOL_SIZE,
            metric_name="thread_pool_utilization",
            scale_up_threshold=80.0,  # percentage
            scale_down_threshold=30.0,
            min_instances=4,
            max_instances=32,
            scale_up_factor=1.3,
            scale_down_factor=0.8,
            cooldown_period=timedelta(minutes=5)
        ))
    
    def _initialize_predictors(self):
        """Initialize load predictors for each resource type."""
        for resource_type in ResourceType:
            self.load_predictors[resource_type] = LoadPredictor()
    
    def register_scaling_callback(
        self, 
        resource_type: ResourceType, 
        callback: Callable[[int], None]
    ):
        """Register callback for when scaling occurs."""
        self.scaling_callbacks[resource_type] = callback
        logger.info(f"Registered scaling callback for {resource_type.value}")
    
    async def start(self):
        """Start the auto-scaling background task."""
        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop the auto-scaling background task."""
        self._running = False
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-scaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Record current load metrics
                self._record_current_loads()
                
                # Evaluate scaling rules
                for rule in self.scaling_rules:
                    await self._evaluate_scaling_rule(rule)
                
                # Predictive scaling
                await self._evaluate_predictive_scaling()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _record_current_loads(self):
        """Record current load metrics for prediction."""
        try:
            # Sentiment analysis load
            sentiment_rate = metrics_collector.get_rate("sentiment_analysis_rate")
            self.load_predictors[ResourceType.SENTIMENT_ANALYZER_WORKERS].record_load(sentiment_rate)
            
            # Healing engine load
            healing_rate = metrics_collector.get_rate("healing_execution_rate")
            self.load_predictors[ResourceType.HEALING_ENGINE_WORKERS].record_load(healing_rate)
            
            # Update current metrics for monitoring
            metrics_collector.set_gauge("current_sentiment_workers", 
                                       self.current_capacities[ResourceType.SENTIMENT_ANALYZER_WORKERS])
            metrics_collector.set_gauge("current_healing_workers", 
                                       self.current_capacities[ResourceType.HEALING_ENGINE_WORKERS])
            
        except Exception as e:
            logger.warning(f"Failed to record current loads: {e}")
    
    async def _evaluate_scaling_rule(self, rule: ScalingRule):
        """Evaluate a single scaling rule and take action if needed."""
        try:
            # Check cooldown period
            last_scaling = self.last_scaling_times.get(rule.resource_type)
            if last_scaling and (datetime.now() - last_scaling) < rule.cooldown_period:
                return
            
            # Get current metric value
            current_value = self._get_metric_value(rule.metric_name)
            if current_value is None:
                return
            
            current_capacity = self.current_capacities.get(rule.resource_type, 0)
            scaling_action = self._determine_scaling_action(rule, current_value, current_capacity)
            
            if scaling_action != ScalingAction.MAINTAIN:
                new_capacity = self._calculate_new_capacity(rule, current_capacity, scaling_action)
                await self._execute_scaling(rule, scaling_action, current_capacity, new_capacity, 
                                          current_value, "rule_based")
        
        except Exception as e:
            logger.error(f"Error evaluating scaling rule for {rule.resource_type.value}: {e}")
    
    async def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling based on load forecasts."""
        try:
            prediction_horizon = timedelta(minutes=10)
            
            for resource_type, predictor in self.load_predictors.items():
                # Skip if we recently scaled this resource
                last_scaling = self.last_scaling_times.get(resource_type)
                if last_scaling and (datetime.now() - last_scaling) < timedelta(minutes=2):
                    continue
                
                # Get prediction
                predicted_load = predictor.predict_load(prediction_horizon)
                current_capacity = self.current_capacities.get(resource_type, 0)
                
                # Find relevant rule for this resource
                relevant_rules = [r for r in self.scaling_rules if r.resource_type == resource_type]
                if not relevant_rules:
                    continue
                
                rule = relevant_rules[0]  # Use first matching rule
                
                # Check if predicted load would trigger scaling
                if predicted_load > rule.scale_up_threshold and current_capacity < rule.max_instances:
                    # Predictive scale-up
                    new_capacity = min(
                        rule.max_instances,
                        int(current_capacity * 1.2)  # Conservative predictive scaling
                    )
                    
                    if new_capacity > current_capacity:
                        await self._execute_scaling(
                            rule, ScalingAction.SCALE_UP, current_capacity, 
                            new_capacity, predicted_load, "predictive"
                        )
                        
                        logger.info(f"Predictive scale-up: {resource_type.value} "
                                   f"predicted load {predicted_load:.2f} > threshold {rule.scale_up_threshold}")
        
        except Exception as e:
            logger.error(f"Error in predictive scaling: {e}")
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        # Try different metric types
        if metric_name.endswith("_rate"):
            return metrics_collector.get_rate(metric_name)
        
        gauge_value = metrics_collector.get_gauge(metric_name)
        if gauge_value is not None:
            return gauge_value
        
        counter_value = metrics_collector.get_counter(metric_name)
        if counter_value is not None:
            return counter_value
        
        # Special cases
        if metric_name == "cache_hit_rate":
            from .cache_manager import sentiment_cache
            stats = sentiment_cache.get_stats()
            return stats.get("hit_rate", 0) * 100
        
        elif metric_name == "sentiment_analysis_queue_depth":
            # This would be implemented by the actual worker queue
            return 0.0  # Placeholder
        
        elif metric_name == "thread_pool_utilization":
            # This would be implemented by monitoring thread pool
            return 50.0  # Placeholder
        
        return None
    
    def _determine_scaling_action(
        self, 
        rule: ScalingRule, 
        current_value: float, 
        current_capacity: int
    ) -> ScalingAction:
        """Determine what scaling action to take."""
        
        # Special handling for cache hit rate (inverted logic)
        if rule.metric_name == "cache_hit_rate":
            if current_value < rule.scale_up_threshold and current_capacity < rule.max_instances:
                return ScalingAction.SCALE_UP  # Scale up cache if hit rate is low
            elif current_value > rule.scale_down_threshold and current_capacity > rule.min_instances:
                return ScalingAction.SCALE_DOWN  # Scale down cache if hit rate is very high
        else:
            # Normal scaling logic
            if current_value > rule.scale_up_threshold and current_capacity < rule.max_instances:
                return ScalingAction.SCALE_UP
            elif current_value < rule.scale_down_threshold and current_capacity > rule.min_instances:
                return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def _calculate_new_capacity(
        self, 
        rule: ScalingRule, 
        current_capacity: int, 
        action: ScalingAction
    ) -> int:
        """Calculate new capacity based on scaling action."""
        if action == ScalingAction.SCALE_UP:
            new_capacity = int(current_capacity * rule.scale_up_factor)
            return min(new_capacity, rule.max_instances)
        elif action == ScalingAction.SCALE_DOWN:
            new_capacity = int(current_capacity * rule.scale_down_factor)
            return max(new_capacity, rule.min_instances)
        else:
            return current_capacity
    
    async def _execute_scaling(
        self,
        rule: ScalingRule,
        action: ScalingAction,
        current_capacity: int,
        new_capacity: int,
        trigger_value: float,
        reason: str
    ):
        """Execute the scaling action."""
        if new_capacity == current_capacity:
            return
        
        try:
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=datetime.now(),
                resource_type=rule.resource_type,
                action=action,
                previous_capacity=current_capacity,
                new_capacity=new_capacity,
                trigger_metric=rule.metric_name,
                trigger_value=trigger_value,
                reason=reason
            )
            
            self.scaling_history.append(scaling_event)
            self.last_scaling_times[rule.resource_type] = datetime.now()
            self.current_capacities[rule.resource_type] = new_capacity
            
            # Execute scaling callback if registered
            callback = self.scaling_callbacks.get(rule.resource_type)
            if callback:
                callback(new_capacity)
            
            # Log scaling event
            logger.info(
                f"Auto-scaling executed: {rule.resource_type.value} "
                f"{action.value} from {current_capacity} to {new_capacity} "
                f"(trigger: {rule.metric_name}={trigger_value:.2f}, reason: {reason})"
            )
            
            # Record metrics
            metrics_collector.increment_counter("autoscaling_events_total", labels={
                "resource_type": rule.resource_type.value,
                "action": action.value,
                "reason": reason
            })
            
            metrics_collector.record_histogram(
                "autoscaling_capacity_change",
                new_capacity - current_capacity,
                labels={"resource_type": rule.resource_type.value}
            )
            
            # Log performance impact
            performance_logger.log_resource_usage(
                resource_type=f"{rule.resource_type.value}_capacity",
                current_usage=new_capacity,
                limit=rule.max_instances,
                usage_percentage=(new_capacity / rule.max_instances) * 100
            )
            
        except Exception as e:
            logger.error(f"Failed to execute scaling for {rule.resource_type.value}: {e}")
            raise
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        return {
            "current_capacities": {
                rt.value: capacity for rt, capacity in self.current_capacities.items()
            },
            "recent_scaling_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "resource_type": event.resource_type.value,
                    "action": event.action.value,
                    "capacity_change": event.new_capacity - event.previous_capacity,
                    "trigger_metric": event.trigger_metric,
                    "trigger_value": event.trigger_value,
                    "reason": event.reason
                }
                for event in self.scaling_history[-10:]  # Last 10 events
            ],
            "load_trends": {
                rt.value: predictor.get_load_trend()
                for rt, predictor in self.load_predictors.items()
            },
            "next_predictions": {
                rt.value: predictor.predict_load(timedelta(minutes=10))
                for rt, predictor in self.load_predictors.items()
            }
        }
    
    def force_scaling(
        self, 
        resource_type: ResourceType, 
        target_capacity: int, 
        reason: str = "manual"
    ) -> bool:
        """Manually force scaling of a resource."""
        try:
            current_capacity = self.current_capacities.get(resource_type, 0)
            
            if target_capacity == current_capacity:
                return True
            
            # Find relevant rule for bounds checking
            relevant_rules = [r for r in self.scaling_rules if r.resource_type == resource_type]
            if relevant_rules:
                rule = relevant_rules[0]
                target_capacity = max(rule.min_instances, min(target_capacity, rule.max_instances))
            
            action = ScalingAction.SCALE_UP if target_capacity > current_capacity else ScalingAction.SCALE_DOWN
            
            # Execute scaling
            asyncio.create_task(self._execute_scaling(
                rule if relevant_rules else None,
                action,
                current_capacity,
                target_capacity,
                0.0,  # No trigger value for manual scaling
                reason
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to force scaling of {resource_type.value}: {e}")
            return False


# Global auto-scaler instance
auto_scaler = AutoScaler()