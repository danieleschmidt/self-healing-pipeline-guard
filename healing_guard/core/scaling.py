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

import aioredis
import aiohttp
from kubernetes import client as k8s_client, config as k8s_config

logger = logging.getLogger(__name__)


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
        self._init_kubernetes()
        
        # Redis client for distributed coordination
        self.redis_client: Optional[aioredis.Redis] = None
        
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
    
    async def make_request(self, path: str, method: str = "GET", **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make a load-balanced request."""
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


# Global instances
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()