"""Advanced scaling and load balancing for the healing system.

Implements intelligent auto-scaling, load balancing, and resource optimization
with predictive analytics and adaptive algorithms.
"""

import asyncio
import logging
import time
import math
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque, defaultdict
import threading
import statistics

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASHING = "consistent_hashing"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerNode:
    """Represents a worker node in the system."""
    id: str
    capacity: int
    current_load: int = 0
    health_score: float = 1.0
    last_heartbeat: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    total_requests: int = 0
    weights: Dict[str, float] = field(default_factory=dict)
    
    @property
    def load_percentage(self) -> float:
        """Get current load as percentage of capacity."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Get error rate percentage."""
        return (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (
            self.health_score > 0.7 and
            self.error_rate < 5.0 and
            self.load_percentage < 90 and
            (not self.last_heartbeat or 
             (datetime.now() - self.last_heartbeat).seconds < 60)
        )
    
    def update_metrics(self, response_time: float, had_error: bool = False):
        """Update node metrics."""
        self.response_times.append(response_time)
        self.total_requests += 1
        if had_error:
            self.error_count += 1
        
        # Update health score based on recent performance
        recent_error_rate = sum(1 for _ in range(min(10, len(self.response_times)))) / min(10, len(self.response_times)) if self.response_times else 0
        self.health_score = max(0.1, 1.0 - (recent_error_rate * 0.5) - (self.load_percentage / 1000))


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=50))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=50))
    request_rate: deque = field(default_factory=lambda: deque(maxlen=50))
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    queue_depths: deque = field(default_factory=lambda: deque(maxlen=50))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def add_sample(self, cpu: float, memory: float, req_rate: float, resp_time: float, queue_depth: int, error_rate: float):
        """Add new metrics sample."""
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.request_rate.append(req_rate)
        self.response_times.append(resp_time)
        self.queue_depths.append(queue_depth)
        self.error_rates.append(error_rate)
    
    def get_trend(self, metric_name: str) -> Tuple[str, float]:
        \"\"\"Get trend analysis for a metric.\"\"\"
        metric_data = getattr(self, metric_name, None)
        if not metric_data or len(metric_data) < 10:
            return "stable", 0.0
        
        values = list(metric_data)
        mid_point = len(values) // 2
        early_avg = statistics.mean(values[:mid_point])
        late_avg = statistics.mean(values[mid_point:])
        
        if late_avg > early_avg * 1.1:
            return "increasing", (late_avg - early_avg) / early_avg
        elif late_avg < early_avg * 0.9:
            return "decreasing", (early_avg - late_avg) / early_avg
        else:
            return "stable", 0.0


class PredictiveScaler:
    """Predictive auto-scaling based on machine learning models."""
    
    def __init__(self):
        self.metrics = ScalingMetrics()
        self.scaling_history: List[Dict[str, Any]] = []
        self.prediction_window = 300  # 5 minutes
        self.scale_cooldown = 180  # 3 minutes
        self.last_scaling_action: Optional[datetime] = None
        
        # Scaling thresholds
        self.scale_up_threshold = {
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "response_time": 5.0,
            "queue_depth": 10,
            "error_rate": 2.0
        }
        
        self.scale_down_threshold = {
            "cpu_usage": 30.0,
            "memory_usage": 40.0,
            "response_time": 1.0,
            "queue_depth": 2,
            "error_rate": 0.5
        }
    
    def update_metrics(self, cpu: float, memory: float, req_rate: float, resp_time: float, queue_depth: int, error_rate: float):
        \"\"\"Update system metrics for scaling decisions.\"\"\"
        self.metrics.add_sample(cpu, memory, req_rate, resp_time, queue_depth, error_rate)
    
    def predict_scaling_need(self) -> Tuple[ScalingDirection, float, Dict[str, Any]]:
        \"\"\"Predict if scaling is needed based on current trends.\"\"\"
        if len(self.metrics.cpu_usage) < 10:
            return ScalingDirection.STABLE, 0.0, {"reason": "insufficient_data"}
        
        scaling_signals = []
        analysis = {}
        
        # Analyze each metric
        for metric_name, threshold_up in self.scale_up_threshold.items():
            threshold_down = self.scale_down_threshold[metric_name]
            trend, change_rate = self.metrics.get_trend(metric_name)
            
            current_values = getattr(self.metrics, metric_name)
            current_avg = statistics.mean(list(current_values)[-5:])  # Last 5 samples
            
            analysis[metric_name] = {
                "current_avg": current_avg,
                "trend": trend,
                "change_rate": change_rate,
                "threshold_up": threshold_up,
                "threshold_down": threshold_down
            }
            
            # Scale up signals
            if current_avg > threshold_up:
                if trend == "increasing":
                    scaling_signals.append(("up", metric_name, 0.8 + min(0.2, change_rate)))
                else:
                    scaling_signals.append(("up", metric_name, 0.6))
            
            # Scale down signals
            elif current_avg < threshold_down:
                if trend == "decreasing":
                    scaling_signals.append(("down", metric_name, 0.7 + min(0.3, change_rate)))
                else:
                    scaling_signals.append(("down", metric_name, 0.5))
        
        # Aggregate scaling signals
        if not scaling_signals:
            return ScalingDirection.STABLE, 0.0, {"analysis": analysis}
        
        # Count weighted votes
        up_weight = sum(weight for direction, _, weight in scaling_signals if direction == "up")
        down_weight = sum(weight for direction, _, weight in scaling_signals if direction == "down")
        
        if up_weight > down_weight and up_weight > 1.5:
            return ScalingDirection.UP, up_weight, {"analysis": analysis, "signals": scaling_signals}
        elif down_weight > up_weight and down_weight > 1.0:
            return ScalingDirection.DOWN, down_weight, {"analysis": analysis, "signals": scaling_signals}
        else:
            return ScalingDirection.STABLE, max(up_weight, down_weight), {"analysis": analysis, "signals": scaling_signals}
    
    def should_scale(self) -> bool:
        \"\"\"Check if scaling action should be taken based on cooldown.\"\"\"
        if not self.last_scaling_action:
            return True
        
        time_since_last = (datetime.now() - self.last_scaling_action).total_seconds()
        return time_since_last >= self.scale_cooldown
    
    def record_scaling_action(self, direction: ScalingDirection, reason: Dict[str, Any]):
        \"\"\"Record a scaling action.\"\"\"
        self.last_scaling_action = datetime.now()
        self.scaling_history.append({
            "timestamp": self.last_scaling_action,
            "direction": direction.value,
            "reason": reason
        })
        
        # Keep only recent history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]


class AdaptiveLoadBalancer:
    \"\"\"Adaptive load balancer with multiple strategies.\"\"\"
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.nodes: Dict[str, WorkerNode] = {}
        self.request_counter = 0
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.current_strategy = strategy
        self._lock = threading.Lock()
        
        # Adaptive strategy parameters
        self.strategy_evaluation_interval = 60  # seconds
        self.last_strategy_evaluation = time.time()
    
    def add_node(self, node: WorkerNode):
        \"\"\"Add a worker node to the pool.\"\"\"
        with self._lock:
            self.nodes[node.id] = node
            logger.info(f"Added worker node {node.id} with capacity {node.capacity}")
    
    def remove_node(self, node_id: str):
        \"\"\"Remove a worker node from the pool.\"\"\"
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed worker node {node_id}")
    
    def get_healthy_nodes(self) -> List[WorkerNode]:
        \"\"\"Get list of healthy nodes.\"\"\"
        return [node for node in self.nodes.values() if node.is_healthy]
    
    def select_node(self, request_context: Dict[str, Any] = None) -> Optional[WorkerNode]:
        \"\"\"Select optimal node based on current strategy.\"\"\"
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Adaptive strategy selection
        if self.current_strategy == LoadBalancingStrategy.ADAPTIVE:
            self._maybe_adapt_strategy()
        
        # Route based on current strategy
        if self.current_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        elif self.current_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        elif self.current_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_nodes)
        elif self.current_strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_nodes)
        elif self.current_strategy == LoadBalancingStrategy.CONSISTENT_HASHING:
            return self._consistent_hash_select(healthy_nodes, request_context)
        else:
            return self._least_connections_select(healthy_nodes)  # Default fallback
    
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        \"\"\"Round robin node selection.\"\"\"
        with self._lock:
            self.request_counter += 1
            index = self.request_counter % len(nodes)
            return nodes[index]
    
    def _weighted_round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        \"\"\"Weighted round robin based on node capacity.\"\"\"
        total_weight = sum(node.capacity for node in nodes)
        target_weight = (self.request_counter % total_weight) + 1
        
        current_weight = 0
        for node in nodes:
            current_weight += node.capacity
            if current_weight >= target_weight:
                self.request_counter += 1
                return node
        
        return nodes[0]  # Fallback
    
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        \"\"\"Select node with least active connections.\"\"\"
        return min(nodes, key=lambda n: n.current_load)
    
    def _least_response_time_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        \"\"\"Select node with lowest average response time.\"\"\"
        return min(nodes, key=lambda n: n.avg_response_time)
    
    def _consistent_hash_select(self, nodes: List[WorkerNode], context: Dict[str, Any] = None) -> WorkerNode:
        \"\"\"Consistent hashing for sticky sessions.\"\"\"
        if not context or "session_id" not in context:
            return self._least_connections_select(nodes)
        
        session_id = context["session_id"]
        hash_value = hash(session_id) % len(nodes)
        return nodes[hash_value]
    
    def _maybe_adapt_strategy(self):
        \"\"\"Evaluate and potentially change load balancing strategy.\"\"\"
        current_time = time.time()
        if current_time - self.last_strategy_evaluation < self.strategy_evaluation_interval:
            return
        
        self.last_strategy_evaluation = current_time
        
        # Evaluate performance of different strategies
        strategies_to_test = [
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
        ]
        
        best_strategy = self.current_strategy
        best_score = 0.0
        
        for strategy in strategies_to_test:
            perf_data = self.strategy_performance[strategy.value]
            if len(perf_data) >= 10:
                avg_response_time = statistics.mean(perf_data)
                score = 1.0 / (avg_response_time + 0.1)  # Lower response time = higher score
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        if best_strategy != self.current_strategy:
            logger.info(f"Adapting load balancing strategy from {self.current_strategy.value} to {best_strategy.value}")
            self.current_strategy = best_strategy
    
    def record_request_result(self, node_id: str, response_time: float, had_error: bool = False):
        \"\"\"Record the result of a request for analytics.\"\"\"
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.update_metrics(response_time, had_error)
            
            # Record strategy performance
            self.strategy_performance[self.current_strategy.value].append(response_time)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        \"\"\"Get load balancer statistics.\"\"\"
        healthy_nodes = self.get_healthy_nodes()
        total_capacity = sum(node.capacity for node in healthy_nodes)
        total_load = sum(node.current_load for node in healthy_nodes)
        
        node_stats = []
        for node in self.nodes.values():
            node_stats.append({
                "id": node.id,
                "capacity": node.capacity,
                "current_load": node.current_load,
                "load_percentage": node.load_percentage,
                "health_score": node.health_score,
                "avg_response_time": node.avg_response_time,
                "error_rate": node.error_rate,
                "is_healthy": node.is_healthy
            })
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len(healthy_nodes),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "system_load_percentage": (total_load / total_capacity) * 100 if total_capacity > 0 else 0,
            "current_strategy": self.current_strategy.value,
            "node_stats": node_stats,
            "request_counter": self.request_counter
        }


class ResourceOptimizer:
    \"\"\"Intelligent resource optimization and allocation.\"\"\"
    
    def __init__(self):
        self.resource_pools: Dict[str, Dict[str, Any]] = {}
        self.allocation_history: deque = deque(maxlen=1000)
        self.optimization_interval = 120  # seconds
        self.last_optimization = time.time()
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, multiprocessing.cpu_count() * 4))
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Resource usage tracking
        self.resource_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def create_resource_pool(self, pool_name: str, pool_type: str, initial_size: int, max_size: int, config: Dict[str, Any] = None):
        \"\"\"Create a new resource pool.\"\"\"
        self.resource_pools[pool_name] = {
            "type": pool_type,
            "current_size": initial_size,
            "max_size": max_size,
            "min_size": max(1, initial_size // 2),
            "active_resources": 0,
            "config": config or {},
            "created_at": datetime.now(),
            "last_resize": datetime.now()
        }
        
        logger.info(f"Created resource pool '{pool_name}' of type '{pool_type}' with size {initial_size}")
    
    def allocate_resource(self, pool_name: str, duration_estimate: float = None) -> Optional[str]:
        \"\"\"Allocate a resource from the pool.\"\"\"
        if pool_name not in self.resource_pools:
            return None
        
        pool = self.resource_pools[pool_name]
        
        if pool["active_resources"] >= pool["current_size"]:
            # Try to expand pool if possible
            if pool["current_size"] < pool["max_size"]:
                self._expand_pool(pool_name, 1)
            else:
                return None  # Pool exhausted
        
        resource_id = f"{pool_name}_{time.time()}"
        pool["active_resources"] += 1
        
        self.allocation_history.append({
            "pool_name": pool_name,
            "resource_id": resource_id,
            "allocated_at": datetime.now(),
            "estimated_duration": duration_estimate
        })
        
        return resource_id
    
    def release_resource(self, pool_name: str, resource_id: str, actual_duration: float = None):
        \"\"\"Release a resource back to the pool.\"\"\"
        if pool_name not in self.resource_pools:
            return
        
        pool = self.resource_pools[pool_name]
        pool["active_resources"] = max(0, pool["active_resources"] - 1)
        
        # Record usage metrics
        if actual_duration:
            self.resource_usage[pool_name].append(actual_duration)
        
        # Update allocation history
        for allocation in self.allocation_history:
            if allocation["resource_id"] == resource_id:
                allocation["released_at"] = datetime.now()
                if actual_duration:
                    allocation["actual_duration"] = actual_duration
                break
    
    def _expand_pool(self, pool_name: str, count: int):
        \"\"\"Expand resource pool size.\"\"\"
        pool = self.resource_pools[pool_name]
        new_size = min(pool["max_size"], pool["current_size"] + count)
        
        if new_size > pool["current_size"]:
            old_size = pool["current_size"]
            pool["current_size"] = new_size
            pool["last_resize"] = datetime.now()
            
            logger.info(f"Expanded pool '{pool_name}' from {old_size} to {new_size}")
    
    def _shrink_pool(self, pool_name: str, count: int):
        \"\"\"Shrink resource pool size.\"\"\"
        pool = self.resource_pools[pool_name]
        new_size = max(pool["min_size"], pool["current_size"] - count)
        
        if new_size < pool["current_size"] and pool["active_resources"] <= new_size:
            old_size = pool["current_size"]
            pool["current_size"] = new_size
            pool["last_resize"] = datetime.now()
            
            logger.info(f"Shrunk pool '{pool_name}' from {old_size} to {new_size}")
    
    def optimize_resources(self):
        \"\"\"Optimize resource allocation across all pools.\"\"\"
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        self.last_optimization = current_time
        
        for pool_name, pool in self.resource_pools.items():
            self._optimize_pool(pool_name, pool)
    
    def _optimize_pool(self, pool_name: str, pool: Dict[str, Any]):
        \"\"\"Optimize individual resource pool.\"\"\"
        usage_data = self.resource_usage.get(pool_name, deque())
        
        if len(usage_data) < 10:
            return  # Not enough data
        
        # Calculate utilization metrics
        current_utilization = pool["active_resources"] / pool["current_size"]
        recent_usage = list(usage_data)[-20:]  # Last 20 operations
        avg_duration = statistics.mean(recent_usage)
        usage_trend = self._calculate_usage_trend(usage_data)
        
        # Optimization decisions
        if current_utilization > 0.8 and usage_trend == "increasing":
            # High utilization and increasing trend - expand
            expand_count = max(1, int(pool["current_size"] * 0.2))
            self._expand_pool(pool_name, expand_count)
            
        elif current_utilization < 0.3 and usage_trend == "decreasing":
            # Low utilization and decreasing trend - shrink
            shrink_count = max(1, int(pool["current_size"] * 0.1))
            self._shrink_pool(pool_name, shrink_count)
        
        logger.debug(f"Pool '{pool_name}' optimization: utilization={current_utilization:.2f}, trend={usage_trend}")
    
    def _calculate_usage_trend(self, usage_data: deque) -> str:
        \"\"\"Calculate usage trend from historical data.\"\"\"
        if len(usage_data) < 20:
            return "stable"
        
        data = list(usage_data)
        mid_point = len(data) // 2
        early_avg = statistics.mean(data[:mid_point])
        late_avg = statistics.mean(data[mid_point:])
        
        if late_avg > early_avg * 1.2:
            return "increasing"
        elif late_avg < early_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        \"\"\"Get resource optimization statistics.\"\"\"
        pool_stats = {}
        
        for pool_name, pool in self.resource_pools.items():
            utilization = pool["active_resources"] / pool["current_size"] if pool["current_size"] > 0 else 0
            
            pool_stats[pool_name] = {
                "type": pool["type"],
                "current_size": pool["current_size"],
                "max_size": pool["max_size"],
                "min_size": pool["min_size"],
                "active_resources": pool["active_resources"],
                "utilization": utilization,
                "usage_samples": len(self.resource_usage.get(pool_name, [])),
                "last_resize": pool["last_resize"].isoformat()
            }
        
        return {
            "total_pools": len(self.resource_pools),
            "total_allocations": len(self.allocation_history),
            "pool_stats": pool_stats,
            "thread_pool_active": self.thread_pool._threads,
            "process_pool_active": len(self.process_pool._processes) if hasattr(self.process_pool, "_processes") else 0
        }


class AutoScalingOrchestrator:
    \"\"\"Main orchestrator for auto-scaling and load balancing.\"\"\"
    
    def __init__(self):
        self.predictive_scaler = PredictiveScaler()
        self.load_balancer = AdaptiveLoadBalancer()
        self.resource_optimizer = ResourceOptimizer()
        
        # Monitoring and control
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 30  # seconds
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable] = None
        self.scale_down_callback: Optional[Callable] = None
        
        # Initialize default worker nodes
        self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        \"\"\"Initialize default worker nodes.\"\"\"
        # Add default local node
        local_node = WorkerNode(
            id="local_primary",
            capacity=multiprocessing.cpu_count() * 2,
            current_load=0,
            health_score=1.0,
            last_heartbeat=datetime.now()
        )
        self.load_balancer.add_node(local_node)
        
        # Create default resource pools
        self.resource_optimizer.create_resource_pool(
            "healing_workers", "thread", 4, 16
        )
        self.resource_optimizer.create_resource_pool(
            "analysis_workers", "process", 2, 8
        )
    
    def start_monitoring(self):
        \"\"\"Start auto-scaling monitoring.\"\"\"
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        \"\"\"Stop auto-scaling monitoring.\"\"\"
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Stopped auto-scaling monitoring")
    
    def _monitoring_loop(self):
        \"\"\"Main monitoring loop for auto-scaling.\"\"\"
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._evaluate_scaling_decisions()
                self.resource_optimizer.optimize_resources()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        \"\"\"Collect current system metrics.\"\"\"
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Calculate request rate (simplified)
            current_time = time.time()
            request_rate = sum(1 for node in self.load_balancer.nodes.values()) * 10  # Approximate
            
            # Calculate average response time
            response_times = []
            for node in self.load_balancer.nodes.values():
                if node.response_times:
                    response_times.extend(list(node.response_times)[-5:])
            
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            
            # Calculate queue depth (simplified)
            queue_depth = sum(max(0, node.current_load - node.capacity // 2) for node in self.load_balancer.nodes.values())
            
            # Calculate error rate
            total_requests = sum(node.total_requests for node in self.load_balancer.nodes.values())
            total_errors = sum(node.error_count for node in self.load_balancer.nodes.values())
            error_rate = (total_errors / total_requests) * 100 if total_requests > 0 else 0
            
            # Update predictive scaler
            self.predictive_scaler.update_metrics(
                cpu_usage, memory_usage, request_rate, avg_response_time, queue_depth, error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _evaluate_scaling_decisions(self):
        \"\"\"Evaluate and execute scaling decisions.\"\"\"
        if not self.predictive_scaler.should_scale():
            return
        
        direction, confidence, analysis = self.predictive_scaler.predict_scaling_need()
        
        if direction == ScalingDirection.STABLE:
            return
        
        logger.info(f"Scaling decision: {direction.value} (confidence: {confidence:.2f})")
        
        # Execute scaling action
        if direction == ScalingDirection.UP and self.scale_up_callback:
            self.scale_up_callback(confidence, analysis)
        elif direction == ScalingDirection.DOWN and self.scale_down_callback:
            self.scale_down_callback(confidence, analysis)
        
        # Record the decision
        self.predictive_scaler.record_scaling_action(direction, analysis)
    
    def register_scaling_callbacks(self, scale_up: Callable, scale_down: Callable):
        \"\"\"Register callbacks for scaling actions.\"\"\"
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
        logger.info("Registered scaling callbacks")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive auto-scaling statistics.\"\"\"
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "predictive_scaling": {
                "metrics_samples": len(self.predictive_scaler.metrics.cpu_usage),
                "scaling_history": len(self.predictive_scaler.scaling_history),
                "last_scaling_action": self.predictive_scaler.last_scaling_action.isoformat() if self.predictive_scaler.last_scaling_action else None,
                "scale_cooldown_remaining": max(0, self.predictive_scaler.scale_cooldown - (
                    (datetime.now() - self.predictive_scaler.last_scaling_action).total_seconds()
                    if self.predictive_scaler.last_scaling_action else 0
                ))
            },
            "load_balancing": self.load_balancer.get_load_balancer_stats(),
            "resource_optimization": self.resource_optimizer.get_optimization_stats()
        }


# Global auto-scaling orchestrator instance
auto_scaler = AutoScalingOrchestrator()