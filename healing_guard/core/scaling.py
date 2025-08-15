"""Auto-scaling and load balancing for distributed healing operations."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict, deque
import json
import uuid

logger = logging.getLogger(__name__)

# Optional imports for scaling features
try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    logger.warning("aioredis not available - Redis features disabled")
    AIOREDIS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp not available - HTTP features disabled")
    AIOHTTP_AVAILABLE = False

try:
    from kubernetes import client as k8s_client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    logger.warning("kubernetes client not available - K8s features disabled")
    KUBERNETES_AVAILABLE = False

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    logger.warning("ML libraries not available - predictive features disabled")
    ML_AVAILABLE = False


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class LoadMetric(Enum):
    """Load metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    queue_length: int
    avg_response_time: float
    error_rate: float
    active_connections: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "request_rate": self.request_rate,
            "queue_length": self.queue_length,
            "avg_response_time": self.avg_response_time,
            "error_rate": self.error_rate,
            "active_connections": self.active_connections,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class ScalingRule:
    """Configuration for scaling rules."""
    name: str
    metric: LoadMetric
    threshold_up: float
    threshold_down: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    min_instances: int = 1
    max_instances: int = 10
    cooldown_period: int = 300  # seconds
    evaluation_window: int = 60  # seconds
    enabled: bool = True
    weight: float = 1.0
    
    def evaluate(self, current_value: float, current_instances: int) -> Tuple[ScalingAction, int]:
        """Evaluate if scaling action is needed."""
        if not self.enabled:
            return ScalingAction.MAINTAIN, current_instances
        
        if current_value > self.threshold_up:
            new_instances = min(
                self.max_instances,
                max(current_instances + 1, int(current_instances * self.scale_up_factor))
            )
            return ScalingAction.SCALE_UP, new_instances
        
        elif current_value < self.threshold_down:
            new_instances = max(
                self.min_instances,
                min(current_instances - 1, int(current_instances * self.scale_down_factor))
            )
            return ScalingAction.SCALE_DOWN, new_instances
        
        return ScalingAction.MAINTAIN, current_instances


class AutoScaler:
    """Intelligent auto-scaling system for healing operations."""
    
    def __init__(
        self,
        service_name: str = "healing-guard",
        namespace: str = "default",
        evaluation_interval: int = 30,
        metrics_history_size: int = 1000
    ):
        self.service_name = service_name
        self.namespace = namespace
        self.evaluation_interval = evaluation_interval
        self.metrics_history_size = metrics_history_size
        
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: deque = deque(maxlen=metrics_history_size)
        self.scaling_history: List[Dict[str, Any]] = []
        self.current_instances = 1
        self.target_instances = 1
        self.last_scaling_action = datetime.now()
        
        # Kubernetes client (if available)
        self.k8s_apps_v1 = None
        self.k8s_custom_objects = None
        if KUBERNETES_AVAILABLE:
            self._init_kubernetes()
        
        # Redis client for distributed coordination
        self.redis_client = None
        
        # Statistics
        self.stats = {
            "scaling_actions": 0,
            "scale_up_actions": 0,
            "scale_down_actions": 0,
            "avg_utilization": 0.0,
            "cost_saved": 0.0,
            "performance_gained": 0.0
        }
        
        # Background task
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _init_kubernetes(self):
        """Initialize Kubernetes client if available."""
        if not KUBERNETES_AVAILABLE:
            logger.warning("kubernetes client not available - K8s features disabled")
            return
        
        try:
            k8s_config.load_incluster_config()
            self.k8s_apps_v1 = k8s_client.AppsV1Api()
            self.k8s_custom_objects = k8s_client.CustomObjectsApi()
            logger.info("Kubernetes client initialized (in-cluster)")
        except Exception:
            try:
                k8s_config.load_kube_config()
                self.k8s_apps_v1 = k8s_client.AppsV1Api()
                self.k8s_custom_objects = k8s_client.CustomObjectsApi()
                logger.info("Kubernetes client initialized (kubeconfig)")
            except Exception as e:
                logger.warning(f"Kubernetes client not available: {e}")
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis client for distributed coordination."""
        if not AIOREDIS_AVAILABLE:
            logger.warning("aioredis not available - Redis features disabled")
            return
        
        try:
            self.redis_client = aioredis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Redis client initialized for auto-scaling coordination")
        except Exception as e:
            logger.warning(f"Redis client initialization failed: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule."""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # This would typically collect from various sources:
        # - Prometheus metrics
        # - Kubernetes metrics server
        # - Application metrics
        # - Custom metrics
        
        # For now, simulate metrics collection
        import psutil
        import random
        
        cpu_util = psutil.cpu_percent(interval=1)
        memory_util = psutil.virtual_memory().percent
        
        # Simulate request rate and other metrics
        request_rate = random.uniform(50, 200)  # requests per second
        queue_length = random.randint(0, 50)
        avg_response_time = random.uniform(0.1, 2.0)
        error_rate = random.uniform(0, 0.1)
        active_connections = random.randint(10, 100)
        
        return ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            request_rate=request_rate,
            queue_length=queue_length,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            active_connections=active_connections
        )
    
    def _calculate_weighted_decision(self, metrics: ScalingMetrics) -> Tuple[ScalingAction, int]:
        """Calculate scaling decision based on weighted rules."""
        decisions = []
        total_weight = 0
        
        metric_values = {
            LoadMetric.CPU_UTILIZATION: metrics.cpu_utilization,
            LoadMetric.MEMORY_UTILIZATION: metrics.memory_utilization,
            LoadMetric.REQUEST_RATE: metrics.request_rate,
            LoadMetric.QUEUE_LENGTH: metrics.queue_length,
            LoadMetric.RESPONSE_TIME: metrics.avg_response_time,
            LoadMetric.ERROR_RATE: metrics.error_rate
        }
        
        for rule in self.scaling_rules.values():
            if rule.metric in metric_values:
                action, instances = rule.evaluate(
                    metric_values[rule.metric],
                    self.current_instances
                )
                decisions.append((action, instances, rule.weight))
                total_weight += rule.weight
        
        if not decisions:
            return ScalingAction.MAINTAIN, self.current_instances
        
        # Weighted voting
        scale_up_weight = sum(w for a, _, w in decisions if a == ScalingAction.SCALE_UP)
        scale_down_weight = sum(w for a, _, w in decisions if a == ScalingAction.SCALE_DOWN)
        maintain_weight = sum(w for a, _, w in decisions if a == ScalingAction.MAINTAIN)
        
        if scale_up_weight > scale_down_weight and scale_up_weight > maintain_weight:
            # Calculate weighted average of scale-up targets
            scale_up_instances = [i for a, i, w in decisions if a == ScalingAction.SCALE_UP]
            scale_up_weights = [w for a, i, w in decisions if a == ScalingAction.SCALE_UP]
            
            if scale_up_instances:
                target = sum(i * w for i, w in zip(scale_up_instances, scale_up_weights)) / sum(scale_up_weights)
                return ScalingAction.SCALE_UP, int(target)
        
        elif scale_down_weight > maintain_weight:
            # Calculate weighted average of scale-down targets
            scale_down_instances = [i for a, i, w in decisions if a == ScalingAction.SCALE_DOWN]
            scale_down_weights = [w for a, i, w in decisions if a == ScalingAction.SCALE_DOWN]
            
            if scale_down_instances:
                target = sum(i * w for i, w in zip(scale_down_instances, scale_down_weights)) / sum(scale_down_weights)
                return ScalingAction.SCALE_DOWN, int(target)
        
        return ScalingAction.MAINTAIN, self.current_instances
    
    def _should_scale(self, action: ScalingAction) -> bool:
        """Check if scaling action should be performed considering cooldown."""
        if action == ScalingAction.MAINTAIN:
            return False
        
        # Check cooldown period
        cooldown_periods = [rule.cooldown_period for rule in self.scaling_rules.values()]
        min_cooldown = min(cooldown_periods) if cooldown_periods else 300
        
        time_since_last_scaling = (datetime.now() - self.last_scaling_action).total_seconds()
        if time_since_last_scaling < min_cooldown:
            logger.debug(f"Scaling action {action} skipped due to cooldown")
            return False
        
        return True
    
    async def _execute_scaling_action(self, action: ScalingAction, target_instances: int):
        """Execute the scaling action."""
        if action == ScalingAction.MAINTAIN:
            return
        
        logger.info(f"Executing scaling action: {action} to {target_instances} instances")
        
        # Record scaling action
        scaling_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action.value,
            "from_instances": self.current_instances,
            "to_instances": target_instances,
            "reason": "auto-scaling decision"
        }
        
        success = False
        
        # Try Kubernetes scaling first
        if self.k8s_apps_v1:
            success = await self._scale_kubernetes_deployment(target_instances)
        
        # Fallback to other scaling methods
        if not success:
            success = await self._scale_external_service(target_instances)
        
        if success:
            self.current_instances = target_instances
            self.target_instances = target_instances
            self.last_scaling_action = datetime.now()
            self.scaling_history.append(scaling_record)
            
            # Update statistics
            self.stats["scaling_actions"] += 1
            if action == ScalingAction.SCALE_UP:
                self.stats["scale_up_actions"] += 1
            else:
                self.stats["scale_down_actions"] += 1
            
            # Distribute scaling event if Redis is available
            if self.redis_client:
                await self._broadcast_scaling_event(scaling_record)
            
            logger.info(f"Scaling successful: {self.current_instances} instances")
        else:
            logger.error(f"Scaling failed: {action} to {target_instances} instances")
    
    async def _scale_kubernetes_deployment(self, target_instances: int) -> bool:
        """Scale Kubernetes deployment."""
        try:
            # Scale deployment
            body = {"spec": {"replicas": target_instances}}
            await asyncio.to_thread(
                self.k8s_apps_v1.patch_namespaced_deployment_scale,
                name=self.service_name,
                namespace=self.namespace,
                body=body
            )
            
            logger.info(f"Kubernetes deployment scaled to {target_instances} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scaling failed: {e}")
            return False
    
    async def _scale_external_service(self, target_instances: int) -> bool:
        """Scale external service (placeholder for custom scaling logic)."""
        # This would implement scaling for non-Kubernetes environments
        # For example: Docker Swarm, cloud-specific auto-scaling groups, etc.
        logger.info(f"External service scaling to {target_instances} instances (simulated)")
        return True
    
    async def _broadcast_scaling_event(self, scaling_record: Dict[str, Any]):
        """Broadcast scaling event to other instances."""
        try:
            if self.redis_client:
                channel = f"healing-guard:scaling:{self.service_name}"
                await self.redis_client.publish(channel, json.dumps(scaling_record))
        except Exception as e:
            logger.error(f"Failed to broadcast scaling event: {e}")
    
    async def start(self):
        """Start the auto-scaling loop."""
        if self._running:
            return
        
        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop the auto-scaling loop."""
        self._running = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaler stopped")
    
    async def _scaling_loop(self):
        """Main auto-scaling evaluation loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Make scaling decision
                action, target_instances = self._calculate_weighted_decision(metrics)
                
                # Execute scaling if needed
                if self._should_scale(action) and target_instances != self.current_instances:
                    await self._execute_scaling_action(action, target_instances)
                
                # Update statistics
                self._update_statistics(metrics)
                
                # Wait for next evaluation
                await asyncio.sleep(self.evaluation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    def _update_statistics(self, metrics: ScalingMetrics):
        """Update auto-scaler statistics."""
        # Calculate moving averages
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
            
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
            
            self.stats["avg_utilization"] = (avg_cpu + avg_memory) / 2
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        return {
            "service_name": self.service_name,
            "namespace": self.namespace,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "running": self._running,
            "evaluation_interval": self.evaluation_interval,
            "scaling_rules": len(self.scaling_rules),
            "last_scaling_action": self.last_scaling_action.isoformat(),
            "metrics_history_size": len(self.metrics_history),
            "recent_metrics": self.metrics_history[-1].to_dict() if self.metrics_history else None
        }
    
    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return self.scaling_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            **self.stats,
            "total_scaling_actions": len(self.scaling_history),
            "uptime_hours": (datetime.now() - self.last_scaling_action).total_seconds() / 3600
        }


class LoadBalancer:
    """Intelligent load balancer for distributed healing operations."""
    
    def __init__(self, algorithm: str = "weighted_round_robin"):
        self.algorithm = algorithm
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.health_check_interval = 30
        self.request_count = 0
        self.last_backend_index = 0
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_backend(
        self,
        backend_id: str,
        url: str,
        weight: float = 1.0,
        max_connections: int = 100,
        health_check_path: str = "/health"
    ):
        """Add a backend server."""
        self.backends[backend_id] = {
            "url": url,
            "weight": weight,
            "max_connections": max_connections,
            "current_connections": 0,
            "health_check_path": health_check_path,
            "healthy": True,
            "last_health_check": datetime.now(),
            "response_times": deque(maxlen=100),
            "request_count": 0,
            "error_count": 0
        }
        
        logger.info(f"Added backend: {backend_id} ({url})")
    
    def remove_backend(self, backend_id: str):
        """Remove a backend server."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            logger.info(f"Removed backend: {backend_id}")
    
    def get_backend(self) -> Optional[Tuple[str, str]]:
        """Get next backend using configured algorithm."""
        healthy_backends = {
            bid: backend for bid, backend in self.backends.items()
            if backend["healthy"] and backend["current_connections"] < backend["max_connections"]
        }
        
        if not healthy_backends:
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin_selection(healthy_backends)
        elif self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(healthy_backends)
        elif self.algorithm == "least_connections":
            return self._least_connections_selection(healthy_backends)
        elif self.algorithm == "response_time":
            return self._response_time_selection(healthy_backends)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_backends)
    
    def _round_robin_selection(self, backends: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Simple round-robin selection."""
        backend_ids = list(backends.keys())
        selected_id = backend_ids[self.last_backend_index % len(backend_ids)]
        self.last_backend_index += 1
        return selected_id, backends[selected_id]["url"]
    
    def _weighted_round_robin_selection(self, backends: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Weighted round-robin selection."""
        # Calculate total weight
        total_weight = sum(backend["weight"] for backend in backends.values())
        
        # Use request count to determine position in weighted sequence
        position = self.request_count % int(total_weight)
        current_weight = 0
        
        for backend_id, backend in backends.items():
            current_weight += backend["weight"]
            if position < current_weight:
                self.request_count += 1
                return backend_id, backend["url"]
        
        # Fallback to first backend
        first_id = next(iter(backends))
        return first_id, backends[first_id]["url"]
    
    def _least_connections_selection(self, backends: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Least connections selection."""
        selected_id = min(backends.keys(), key=lambda bid: backends[bid]["current_connections"])
        return selected_id, backends[selected_id]["url"]
    
    def _response_time_selection(self, backends: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Response time based selection."""
        def avg_response_time(backend):
            times = backend["response_times"]
            return sum(times) / len(times) if times else float('inf')
        
        selected_id = min(backends.keys(), key=lambda bid: avg_response_time(backends[bid]))
        return selected_id, backends[selected_id]["url"]
    
    async def make_request(self, path: str, method: str = "GET", **kwargs):
        """Make a load-balanced request."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available - cannot make HTTP requests")
            return None
        
        backend_info = self.get_backend()
        if not backend_info:
            logger.warning("No healthy backends available")
            return None
        
        backend_id, backend_url = backend_info
        backend = self.backends[backend_id]
        
        # Track connection
        backend["current_connections"] += 1
        backend["request_count"] += 1
        
        start_time = time.time()
        
        try:
            url = f"{backend_url.rstrip('/')}/{path.lstrip('/')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    response_time = time.time() - start_time
                    backend["response_times"].append(response_time)
                    
                    if response.status >= 400:
                        backend["error_count"] += 1
                    
                    return response
        
        except Exception as e:
            response_time = time.time() - start_time
            backend["response_times"].append(response_time)
            backend["error_count"] += 1
            logger.error(f"Request failed to backend {backend_id}: {e}")
            return None
        
        finally:
            backend["current_connections"] -= 1
    
    async def health_check_backend(self, backend_id: str) -> bool:
        """Perform health check on a backend."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - cannot perform health checks")
            return True  # Assume healthy if we can't check
        
        backend = self.backends[backend_id]
        health_url = f"{backend['url'].rstrip('/')}/{backend['health_check_path'].lstrip('/')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    healthy = response.status < 400
                    backend["healthy"] = healthy
                    backend["last_health_check"] = datetime.now()
                    return healthy
        
        except Exception as e:
            logger.warning(f"Health check failed for backend {backend_id}: {e}")
            backend["healthy"] = False
            backend["last_health_check"] = datetime.now()
            return False
    
    async def start_health_checks(self):
        """Start periodic health checks."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer health checks started")
    
    async def stop_health_checks(self):
        """Stop periodic health checks."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Load balancer health checks stopped")
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                # Check all backends
                health_check_tasks = [
                    self.health_check_backend(backend_id)
                    for backend_id in self.backends.keys()
                ]
                
                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        stats = {}
        
        for backend_id, backend in self.backends.items():
            total_requests = backend["request_count"]
            error_rate = backend["error_count"] / total_requests if total_requests > 0 else 0
            
            avg_response_time = 0
            if backend["response_times"]:
                avg_response_time = sum(backend["response_times"]) / len(backend["response_times"])
            
            stats[backend_id] = {
                "url": backend["url"],
                "healthy": backend["healthy"],
                "current_connections": backend["current_connections"],
                "max_connections": backend["max_connections"],
                "total_requests": total_requests,
                "error_count": backend["error_count"],
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "weight": backend["weight"],
                "last_health_check": backend["last_health_check"].isoformat()
            }
        
        return stats


class PredictiveAutoScaler:
    """Advanced predictive auto-scaling using machine learning."""
    
    def __init__(self, prediction_horizon: int = 300):
        self.prediction_horizon = prediction_horizon  # seconds
        self.metrics_history: List[ScalingMetrics] = []
        self.prediction_models: Dict[str, Any] = {}
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.historical_scaling_decisions: List[Dict[str, Any]] = []
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history for training."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history (24 hours worth)
        max_history = 24 * 60  # 24 hours of minute-level metrics
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
    
    def train_prediction_models(self) -> bool:
        """Train ML models for load prediction."""
        if not ML_AVAILABLE or len(self.metrics_history) < 100:
            logger.warning("Insufficient data or ML unavailable for predictive scaling")
            return False
        
        try:
            # Prepare training data
            features, targets = self._prepare_training_data()
            
            if len(features) < 50:
                return False
            
            # Train models for different metrics
            metrics_to_predict = ['cpu_utilization', 'memory_utilization', 'request_rate', 'queue_length']
            
            for i, metric in enumerate(metrics_to_predict):
                # Use ensemble of models
                models = {
                    'linear': LinearRegression(),
                    'forest': RandomForestRegressor(n_estimators=50, random_state=42)
                }
                
                target_values = [t[i] for t in targets]
                
                # Scale features
                features_scaled = self.scaler.fit_transform(features)
                
                trained_models = {}
                for model_name, model in models.items():
                    try:
                        model.fit(features_scaled, target_values)
                        trained_models[model_name] = model
                    except Exception as e:
                        logger.warning(f"Failed to train {model_name} for {metric}: {e}")
                
                if trained_models:
                    self.prediction_models[metric] = {
                        'models': trained_models,
                        'scaler': StandardScaler().fit(features)
                    }
            
            logger.info(f"Trained predictive models for {len(self.prediction_models)} metrics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train prediction models: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Prepare features and targets for ML training."""
        features = []
        targets = []
        
        # Use sliding window approach
        window_size = 10  # Use last 10 metrics points as features
        
        for i in range(window_size, len(self.metrics_history)):
            # Features: time-series window of previous metrics
            feature_window = []
            for j in range(i - window_size, i):
                metric = self.metrics_history[j]
                feature_window.extend([
                    metric.cpu_utilization,
                    metric.memory_utilization,
                    metric.request_rate,
                    metric.queue_length / 100.0,  # Normalize
                    metric.avg_response_time,
                    metric.error_rate,
                    metric.active_connections / 1000.0  # Normalize
                ])
            
            # Add time-based features
            current_metric = self.metrics_history[i]
            hour_of_day = current_metric.timestamp.hour
            day_of_week = current_metric.timestamp.weekday()
            
            feature_window.extend([
                np.sin(2 * np.pi * hour_of_day / 24),  # Cyclic encoding
                np.cos(2 * np.pi * hour_of_day / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7)
            ])
            
            features.append(feature_window)
            
            # Target: next metric values
            target_metric = self.metrics_history[i]
            targets.append([
                target_metric.cpu_utilization,
                target_metric.memory_utilization,
                target_metric.request_rate,
                target_metric.queue_length
            ])
        
        return features, targets
    
    def predict_future_load(self, horizon_minutes: int = 5) -> Dict[str, float]:
        """Predict future load metrics."""
        if not self.prediction_models or len(self.metrics_history) < 10:
            return {}
        
        try:
            predictions = {}
            
            # Prepare current features
            current_features = self._prepare_current_features()
            
            for metric, model_info in self.prediction_models.items():
                models = model_info['models']
                scaler = model_info['scaler']
                
                # Scale features
                features_scaled = scaler.transform([current_features])
                
                # Ensemble prediction
                ensemble_predictions = []
                for model_name, model in models.items():
                    try:
                        pred = model.predict(features_scaled)[0]
                        ensemble_predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {e}")
                
                if ensemble_predictions:
                    # Average ensemble predictions
                    predictions[metric] = sum(ensemble_predictions) / len(ensemble_predictions)
            
            logger.debug(f"Load predictions: {predictions}")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict future load: {e}")
            return {}
    
    def _prepare_current_features(self) -> List[float]:
        """Prepare features from current state."""
        window_size = 10
        recent_metrics = self.metrics_history[-window_size:]
        
        features = []
        for metric in recent_metrics:
            features.extend([
                metric.cpu_utilization,
                metric.memory_utilization,
                metric.request_rate,
                metric.queue_length / 100.0,
                metric.avg_response_time,
                metric.error_rate,
                metric.active_connections / 1000.0
            ])
        
        # Add current time features
        now = datetime.now()
        features.extend([
            np.sin(2 * np.pi * now.hour / 24),
            np.cos(2 * np.pi * now.hour / 24),
            np.sin(2 * np.pi * now.weekday() / 7),
            np.cos(2 * np.pi * now.weekday() / 7)
        ])
        
        return features
    
    def get_predictive_scaling_recommendation(self, current_instances: int) -> Dict[str, Any]:
        """Get scaling recommendation based on predictions."""
        predictions = self.predict_future_load(horizon_minutes=5)
        
        if not predictions:
            return {"action": ScalingAction.MAINTAIN, "confidence": 0.0, "reason": "no_predictions"}
        
        # Analyze predictions
        predicted_cpu = predictions.get('cpu_utilization', 0)
        predicted_memory = predictions.get('memory_utilization', 0)
        predicted_requests = predictions.get('request_rate', 0)
        predicted_queue = predictions.get('queue_length', 0)
        
        # Scaling thresholds
        scale_up_threshold = 0.7
        scale_down_threshold = 0.3
        
        # Determine scaling action
        scale_indicators = []
        
        if predicted_cpu > scale_up_threshold:
            scale_indicators.append(("cpu", "up", predicted_cpu))
        elif predicted_cpu < scale_down_threshold:
            scale_indicators.append(("cpu", "down", predicted_cpu))
        
        if predicted_memory > scale_up_threshold:
            scale_indicators.append(("memory", "up", predicted_memory))
        elif predicted_memory < scale_down_threshold:
            scale_indicators.append(("memory", "down", predicted_memory))
        
        if predicted_queue > 50:  # Queue length threshold
            scale_indicators.append(("queue", "up", predicted_queue))
        elif predicted_queue < 5:
            scale_indicators.append(("queue", "down", predicted_queue))
        
        # Make scaling decision
        up_votes = sum(1 for _, action, _ in scale_indicators if action == "up")
        down_votes = sum(1 for _, action, _ in scale_indicators if action == "down")
        
        if up_votes > down_votes and up_votes >= 2:
            action = ScalingAction.SCALE_UP
            confidence = min(1.0, up_votes / 3.0)
            recommended_instances = min(current_instances * 2, current_instances + 3)
            reason = f"predicted_high_load (cpu: {predicted_cpu:.2f}, mem: {predicted_memory:.2f})"
        elif down_votes > up_votes and down_votes >= 2:
            action = ScalingAction.SCALE_DOWN
            confidence = min(1.0, down_votes / 3.0)
            recommended_instances = max(1, current_instances - 1)
            reason = f"predicted_low_load (cpu: {predicted_cpu:.2f}, mem: {predicted_memory:.2f})"
        else:
            action = ScalingAction.MAINTAIN
            confidence = 0.5
            recommended_instances = current_instances
            reason = "balanced_predictions"
        
        return {
            "action": action,
            "recommended_instances": recommended_instances,
            "confidence": confidence,
            "reason": reason,
            "predictions": predictions,
            "indicators": scale_indicators
        }


class GlobalLoadBalancer:
    """Global load balancing across multiple regions and availability zones."""
    
    def __init__(self):
        self.regions: Dict[str, Dict[str, Any]] = {}
        self.global_routing_table: Dict[str, str] = {}
        self.health_check_interval = 30
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        self.traffic_distribution_strategy = "geographic_proximity"
        
    def add_region(self, region_id: str, endpoints: List[str], 
                   latitude: float, longitude: float,
                   capacity_weight: float = 1.0) -> None:
        """Add a region to global load balancing."""
        self.regions[region_id] = {
            "endpoints": endpoints,
            "latitude": latitude,
            "longitude": longitude,
            "capacity_weight": capacity_weight,
            "current_load": 0.0,
            "healthy_endpoints": set(endpoints),
            "last_health_check": datetime.now(),
            "total_requests": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0
        }
        
        logger.info(f"Added region {region_id} with {len(endpoints)} endpoints")
    
    def calculate_geographic_distance(self, lat1: float, lon1: float, 
                                    lat2: float, lon2: float) -> float:
        """Calculate geographic distance using Haversine formula."""
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in km
        r = 6371
        return c * r
    
    def select_optimal_region(self, client_lat: Optional[float] = None,
                            client_lon: Optional[float] = None,
                            client_ip: Optional[str] = None) -> Optional[str]:
        """Select optimal region for client request."""
        if not self.regions:
            return None
        
        healthy_regions = [
            region_id for region_id, region in self.regions.items()
            if region["healthy_endpoints"]
        ]
        
        if not healthy_regions:
            return None
        
        if len(healthy_regions) == 1:
            return healthy_regions[0]
        
        # Geographic proximity strategy
        if self.traffic_distribution_strategy == "geographic_proximity" and client_lat and client_lon:
            best_region = None
            min_distance = float('inf')
            
            for region_id in healthy_regions:
                region = self.regions[region_id]
                distance = self.calculate_geographic_distance(
                    client_lat, client_lon,
                    region["latitude"], region["longitude"]
                )
                
                # Factor in current load
                load_penalty = region["current_load"] * 100  # km equivalent
                adjusted_distance = distance + load_penalty
                
                if adjusted_distance < min_distance:
                    min_distance = adjusted_distance
                    best_region = region_id
            
            return best_region
        
        # Load-based strategy
        elif self.traffic_distribution_strategy == "load_balanced":
            # Weighted random selection based on available capacity
            weights = []
            for region_id in healthy_regions:
                region = self.regions[region_id]
                available_capacity = region["capacity_weight"] * (1.0 - region["current_load"])
                weights.append(max(0.1, available_capacity))  # Minimum weight
            
            if sum(weights) == 0:
                return healthy_regions[0]  # Fallback
            
            # Weighted random selection
            import random
            total_weight = sum(weights)
            random_value = random.uniform(0, total_weight)
            
            cumulative_weight = 0
            for i, region_id in enumerate(healthy_regions):
                cumulative_weight += weights[i]
                if random_value <= cumulative_weight:
                    return region_id
        
        # Fallback to first healthy region
        return healthy_regions[0]
    
    def update_region_metrics(self, region_id: str, load: float, 
                            response_time: float, error_rate: float) -> None:
        """Update region performance metrics."""
        if region_id not in self.regions:
            return
        
        region = self.regions[region_id]
        region["current_load"] = load
        region["avg_response_time"] = response_time
        region["error_rate"] = error_rate
        region["total_requests"] += 1
    
    async def perform_global_health_checks(self) -> Dict[str, Any]:
        """Perform health checks across all regions."""
        results = {}
        
        for region_id, region in self.regions.items():
            healthy_count = 0
            total_count = len(region["endpoints"])
            
            for endpoint in region["endpoints"]:
                try:
                    if AIOHTTP_AVAILABLE:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{endpoint}/health",
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as response:
                                if response.status < 400:
                                    healthy_count += 1
                                    region["healthy_endpoints"].add(endpoint)
                                else:
                                    region["healthy_endpoints"].discard(endpoint)
                    else:
                        # Assume healthy if no HTTP client available
                        healthy_count += 1
                        region["healthy_endpoints"].add(endpoint)
                        
                except Exception as e:
                    logger.warning(f"Health check failed for {endpoint}: {e}")
                    region["healthy_endpoints"].discard(endpoint)
            
            region["last_health_check"] = datetime.now()
            
            results[region_id] = {
                "healthy_endpoints": healthy_count,
                "total_endpoints": total_count,
                "health_ratio": healthy_count / total_count if total_count > 0 else 0,
                "current_load": region["current_load"],
                "avg_response_time": region["avg_response_time"]
            }
        
        return results
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global load balancer status."""
        total_regions = len(self.regions)
        healthy_regions = sum(1 for region in self.regions.values() if region["healthy_endpoints"])
        
        total_endpoints = sum(len(region["endpoints"]) for region in self.regions.values())
        healthy_endpoints = sum(len(region["healthy_endpoints"]) for region in self.regions.values())
        
        avg_load = sum(region["current_load"] for region in self.regions.values()) / max(total_regions, 1)
        avg_response_time = sum(region["avg_response_time"] for region in self.regions.values()) / max(total_regions, 1)
        
        return {
            "total_regions": total_regions,
            "healthy_regions": healthy_regions,
            "region_health_ratio": healthy_regions / max(total_regions, 1),
            "total_endpoints": total_endpoints,
            "healthy_endpoints": healthy_endpoints,
            "endpoint_health_ratio": healthy_endpoints / max(total_endpoints, 1),
            "global_average_load": avg_load,
            "global_average_response_time": avg_response_time,
            "traffic_distribution_strategy": self.traffic_distribution_strategy,
            "regions": {
                region_id: {
                    "endpoints": len(region["endpoints"]),
                    "healthy_endpoints": len(region["healthy_endpoints"]),
                    "current_load": region["current_load"],
                    "capacity_weight": region["capacity_weight"],
                    "avg_response_time": region["avg_response_time"],
                    "error_rate": region["error_rate"],
                    "location": (region["latitude"], region["longitude"])
                }
                for region_id, region in self.regions.items()
            }
        }


# Global instances
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()
predictive_scaler = PredictiveAutoScaler()
global_balancer = GlobalLoadBalancer()