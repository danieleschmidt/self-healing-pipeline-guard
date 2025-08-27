"""Distributed healing coordination system for multi-region deployments.

Implements cluster-wide healing coordination, leader election, and 
distributed state management for high-availability scenarios.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import hashlib
import random

from ..core.healing_engine import HealingEngine, HealingPlan, HealingResult, HealingStatus
from ..core.failure_detector import FailureEvent, FailureType
from ..monitoring.enhanced_monitoring import enhanced_monitoring
from ..tenancy.multi_tenant import tenant_manager

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Cluster node status."""
    ACTIVE = "active"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    JOINING = "joining"
    LEAVING = "leaving"


class LeaderStatus(Enum):
    """Leadership status."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class HealingPriority(Enum):
    """Healing operation priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class ClusterNode:
    """Represents a node in the healing cluster."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    region: str
    zone: str
    status: NodeStatus
    leader_status: LeaderStatus
    last_heartbeat: datetime
    load_average: float = 0.0
    healing_capacity: int = 10
    active_healings: int = 0
    total_healings_completed: int = 0
    node_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (
            self.status == NodeStatus.ACTIVE and
            datetime.now() - self.last_heartbeat < timedelta(seconds=30)
        )
    
    @property
    def available_capacity(self) -> int:
        """Get available healing capacity."""
        return max(0, self.healing_capacity - self.active_healings)
    
    @property
    def utilization(self) -> float:
        """Get current utilization percentage."""
        return (self.active_healings / self.healing_capacity) * 100 if self.healing_capacity > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DistributedHealingTask:
    """Distributed healing task with coordination metadata."""
    task_id: str
    failure_event: FailureEvent
    healing_plan: HealingPlan
    priority: HealingPriority
    assigned_node: Optional[str] = None
    backup_nodes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: HealingStatus = HealingStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    tenant_id: Optional[str] = None
    region_preference: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired."""
        max_age = timedelta(hours=1)
        return datetime.now() - self.created_at > max_age
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "failure_event": self.failure_event.to_dict(),
            "healing_plan": self.healing_plan.to_dict(),
            "priority": self.priority.value,
            "assigned_node": self.assigned_node,
            "backup_nodes": self.backup_nodes,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "tenant_id": self.tenant_id,
            "region_preference": self.region_preference
        }


class LeaderElection:
    """Implements distributed leader election using a simple consensus algorithm."""
    
    def __init__(self, node_id: str, cluster_nodes: Dict[str, ClusterNode]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.leader_id: Optional[str] = None
        self.election_timeout = 30.0  # seconds
        self.heartbeat_interval = 5.0  # seconds
        self.last_election = datetime.now()
        self.votes_received = 0
        self.votes_needed = 0
        self.lock = threading.RLock()
        
    def start_election(self) -> bool:
        """Start leader election process."""
        with self.lock:
            logger.info(f"Node {self.node_id} starting leader election")
            
            # Reset election state
            self.votes_received = 0
            self.last_election = datetime.now()
            
            # Count healthy nodes
            healthy_nodes = [
                node for node in self.cluster_nodes.values()
                if node.is_healthy
            ]
            self.votes_needed = len(healthy_nodes) // 2 + 1
            
            # Vote for self
            self.votes_received = 1
            
            # Request votes from other nodes (simplified - in real implementation would use network)
            for node in healthy_nodes:
                if node.node_id != self.node_id:
                    # Simulate voting based on node priority (node with lowest ID wins)
                    if self._should_vote_for(node.node_id):
                        self.votes_received += 1
            
            # Check if won election
            if self.votes_received >= self.votes_needed:
                self._become_leader()
                return True
            
            return False
    
    def _should_vote_for(self, candidate_id: str) -> bool:
        """Determine if should vote for candidate (simplified logic)."""
        # Vote for candidate with lowest ID (deterministic)
        return candidate_id > self.node_id
    
    def _become_leader(self):
        """Become cluster leader."""
        self.leader_id = self.node_id
        
        # Update node status
        if self.node_id in self.cluster_nodes:
            self.cluster_nodes[self.node_id].leader_status = LeaderStatus.LEADER
        
        logger.info(f"Node {self.node_id} became cluster leader")
    
    def is_leader(self) -> bool:
        """Check if this node is the current leader."""
        return self.leader_id == self.node_id
    
    def get_leader(self) -> Optional[str]:
        """Get current leader node ID."""
        # Check if leader is still healthy
        if self.leader_id and self.leader_id in self.cluster_nodes:
            leader_node = self.cluster_nodes[self.leader_id]
            if not leader_node.is_healthy:
                self.leader_id = None
        
        return self.leader_id
    
    def check_election_needed(self) -> bool:
        """Check if new election is needed."""
        current_leader = self.get_leader()
        
        # Need election if no leader or leader is unhealthy
        if not current_leader:
            return True
        
        # Need election if haven't heard from leader recently
        election_age = datetime.now() - self.last_election
        return election_age > timedelta(seconds=self.election_timeout)


class TaskScheduler:
    """Intelligent task scheduler for distributed healing operations."""
    
    def __init__(self, cluster_nodes: Dict[str, ClusterNode]):
        self.cluster_nodes = cluster_nodes
        self.pending_tasks: List[DistributedHealingTask] = []
        self.active_tasks: Dict[str, DistributedHealingTask] = {}
        self.completed_tasks: List[DistributedHealingTask] = []
        self.lock = threading.RLock()
        
        # Scheduling policies
        self.load_balancing_enabled = True
        self.region_affinity_enabled = True
        self.tenant_isolation_enabled = True
    
    def add_task(self, task: DistributedHealingTask):
        """Add a new healing task to the queue."""
        with self.lock:
            # Determine priority based on failure severity and type
            task.priority = self._calculate_priority(task.failure_event)
            
            # Insert task in priority order
            inserted = False
            for i, existing_task in enumerate(self.pending_tasks):
                if task.priority.value > existing_task.priority.value:
                    self.pending_tasks.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self.pending_tasks.append(task)
            
            logger.info(f"Added healing task {task.task_id} with priority {task.priority.value}")
    
    def _calculate_priority(self, failure_event: FailureEvent) -> HealingPriority:
        """Calculate task priority based on failure characteristics."""
        # Base priority on severity
        severity_priority = {
            "LOW": HealingPriority.LOW,
            "MEDIUM": HealingPriority.MEDIUM,
            "HIGH": HealingPriority.HIGH,
            "CRITICAL": HealingPriority.EMERGENCY
        }
        
        base_priority = severity_priority.get(
            failure_event.severity.value.upper(),
            HealingPriority.MEDIUM
        )
        
        # Upgrade priority for certain failure types
        if failure_event.failure_type in [FailureType.SECURITY_VIOLATION, FailureType.DATA_CORRUPTION]:
            return HealingPriority.EMERGENCY
        
        # Upgrade for production branches
        if failure_event.branch in ["main", "master", "production"]:
            if base_priority.value < HealingPriority.HIGH.value:
                return HealingPriority.HIGH
        
        return base_priority
    
    def get_next_task(self) -> Optional[DistributedHealingTask]:
        """Get the next highest priority task."""
        with self.lock:
            if self.pending_tasks:
                return self.pending_tasks.pop(0)
            return None
    
    def assign_task(
        self,
        task: DistributedHealingTask,
        preferred_region: Optional[str] = None
    ) -> Optional[str]:
        """Assign task to best available node."""
        with self.lock:
            available_nodes = [
                node for node in self.cluster_nodes.values()
                if node.is_healthy and node.available_capacity > 0
            ]
            
            if not available_nodes:
                return None
            
            # Apply scheduling policies
            best_node = self._select_best_node(task, available_nodes, preferred_region)
            
            if best_node:
                # Assign task
                task.assigned_node = best_node.node_id
                task.started_at = datetime.now()
                
                # Select backup nodes
                backup_candidates = [
                    node for node in available_nodes
                    if node.node_id != best_node.node_id and node.available_capacity > 0
                ]
                task.backup_nodes = [
                    node.node_id for node in backup_candidates[:2]  # Top 2 backups
                ]
                
                # Update node state
                best_node.active_healings += 1
                
                # Move to active tasks
                self.active_tasks[task.task_id] = task
                
                logger.info(f"Assigned task {task.task_id} to node {best_node.node_id}")
                return best_node.node_id
            
            return None
    
    def _select_best_node(
        self,
        task: DistributedHealingTask,
        available_nodes: List[ClusterNode],
        preferred_region: Optional[str] = None
    ) -> Optional[ClusterNode]:
        """Select the best node for task execution using multiple criteria."""
        
        # Score each node
        node_scores = {}
        
        for node in available_nodes:
            score = 100.0  # Base score
            
            # Load balancing - prefer less loaded nodes
            if self.load_balancing_enabled:
                utilization_penalty = node.utilization * 0.5
                score -= utilization_penalty
            
            # Region affinity - prefer nodes in same region
            if self.region_affinity_enabled:
                task_region = preferred_region or task.region_preference
                if task_region and node.region == task_region:
                    score += 25.0
                elif task_region and node.region != task_region:
                    score -= 10.0
            
            # Tenant isolation - prefer nodes already handling same tenant
            if self.tenant_isolation_enabled and task.tenant_id:
                # Check if node is already handling tasks for this tenant
                tenant_tasks = [
                    t for t in self.active_tasks.values()
                    if t.assigned_node == node.node_id and t.tenant_id == task.tenant_id
                ]
                if tenant_tasks:
                    score += 15.0  # Slight preference for tenant affinity
            
            # Capacity preference - prefer nodes with more available capacity
            capacity_bonus = (node.available_capacity / node.healing_capacity) * 10
            score += capacity_bonus
            
            # Historical performance - prefer nodes with good track record
            if node.total_healings_completed > 0:
                # Assume success rate is high (would calculate from history)
                performance_bonus = min(10.0, node.total_healings_completed * 0.1)
                score += performance_bonus
            
            node_scores[node.node_id] = score
        
        # Select node with highest score
        if node_scores:
            best_node_id = max(node_scores, key=node_scores.get)
            return next(node for node in available_nodes if node.node_id == best_node_id)
        
        return None
    
    def complete_task(
        self,
        task_id: str,
        result: HealingResult,
        success: bool = True
    ):
        """Mark task as completed and update node state."""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                task.completed_at = datetime.now()
                task.status = HealingStatus.SUCCESSFUL if success else HealingStatus.FAILED
                
                # Update assigned node
                if task.assigned_node and task.assigned_node in self.cluster_nodes:
                    node = self.cluster_nodes[task.assigned_node]
                    node.active_healings = max(0, node.active_healings - 1)
                    if success:
                        node.total_healings_completed += 1
                
                # Archive completed task
                self.completed_tasks.append(task)
                
                # Clean up old completed tasks
                self._cleanup_completed_tasks()
                
                logger.info(f"Completed task {task_id} on node {task.assigned_node}")
    
    def retry_failed_task(self, task_id: str) -> bool:
        """Retry a failed task if retries are available."""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = HealingStatus.PENDING
                    task.assigned_node = None
                    task.started_at = None
                    
                    # Move back to pending queue
                    del self.active_tasks[task_id]
                    self.add_task(task)
                    
                    logger.info(f"Retrying task {task_id} (attempt {task.retry_count}/{task.max_retries})")
                    return True
                else:
                    # Max retries exceeded
                    task.status = HealingStatus.FAILED
                    task.completed_at = datetime.now()
                    self.completed_tasks.append(task)
                    del self.active_tasks[task_id]
                    
                    logger.warning(f"Task {task_id} failed after {task.max_retries} retries")
                    return False
            
            return False
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.completed_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and task.completed_at > cutoff_time
        ]
        
        # Limit total completed tasks
        max_completed = 1000
        if len(self.completed_tasks) > max_completed:
            self.completed_tasks = self.completed_tasks[-max_completed:]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.lock:
            return {
                "pending_tasks": len(self.pending_tasks),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "pending_by_priority": {
                    priority.name: len([t for t in self.pending_tasks if t.priority == priority])
                    for priority in HealingPriority
                },
                "active_by_node": {
                    node_id: len([t for t in self.active_tasks.values() if t.assigned_node == node_id])
                    for node_id in self.cluster_nodes.keys()
                }
            }


class ClusterCoordinator:
    """Main distributed cluster coordination system."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        region: str = "us-east-1",
        zone: str = "us-east-1a"
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.region = region
        self.zone = zone
        
        # Cluster state
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        self.leader_election: Optional[LeaderElection] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        
        # Local healing engine
        self.healing_engine = HealingEngine()
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.coordination_task: Optional[asyncio.Task] = None
        self.election_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            "tasks_executed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "cluster_utilization": 0.0
        }
        
        self.lock = threading.RLock()
    
    async def start(
        self,
        initial_nodes: Optional[List[Dict[str, Any]]] = None
    ):
        """Start cluster coordination."""
        if self.running:
            return
        
        self.running = True
        
        # Initialize local node
        local_node = ClusterNode(
            node_id=self.node_id,
            hostname="localhost",  # Would get actual hostname
            ip_address="127.0.0.1",  # Would get actual IP
            port=8000,
            region=self.region,
            zone=self.zone,
            status=NodeStatus.JOINING,
            leader_status=LeaderStatus.FOLLOWER,
            last_heartbeat=datetime.now(),
            healing_capacity=10
        )
        
        with self.lock:
            self.cluster_nodes[self.node_id] = local_node
            
            # Add initial nodes if provided
            if initial_nodes:
                for node_data in initial_nodes:
                    if node_data["node_id"] != self.node_id:
                        node = ClusterNode(**node_data)
                        self.cluster_nodes[node.node_id] = node
            
            # Initialize leader election
            self.leader_election = LeaderElection(self.node_id, self.cluster_nodes)
            
            # Initialize task scheduler
            self.task_scheduler = TaskScheduler(self.cluster_nodes)
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        self.election_task = asyncio.create_task(self._election_loop())
        
        # Mark node as active
        local_node.status = NodeStatus.ACTIVE
        
        logger.info(f"Cluster coordinator started for node {self.node_id} in {self.region}")
    
    async def stop(self):
        """Stop cluster coordination."""
        if not self.running:
            return
        
        self.running = False
        
        # Mark node as leaving
        with self.lock:
            if self.node_id in self.cluster_nodes:
                self.cluster_nodes[self.node_id].status = NodeStatus.LEAVING
        
        # Cancel background tasks
        tasks = [self.heartbeat_task, self.coordination_task, self.election_task]
        for task in tasks:
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
        
        logger.info(f"Cluster coordinator stopped for node {self.node_id}")
    
    async def submit_healing_task(
        self,
        failure_event: FailureEvent,
        tenant_id: Optional[str] = None,
        region_preference: Optional[str] = None
    ) -> str:
        """Submit a healing task to the distributed cluster."""
        
        # Create healing plan
        healing_plan = await self.healing_engine.create_healing_plan(failure_event)
        
        # Create distributed task
        task = DistributedHealingTask(
            task_id=str(uuid.uuid4()),
            failure_event=failure_event,
            healing_plan=healing_plan,
            priority=HealingPriority.MEDIUM,  # Will be calculated by scheduler
            tenant_id=tenant_id,
            region_preference=region_preference
        )
        
        # Add to scheduler
        if self.task_scheduler:
            self.task_scheduler.add_task(task)
            
            logger.info(f"Submitted healing task {task.task_id} for failure {failure_event.id}")
            return task.task_id
        
        raise RuntimeError("Task scheduler not initialized")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats and update node status."""
        while self.running:
            try:
                with self.lock:
                    # Update local node heartbeat
                    if self.node_id in self.cluster_nodes:
                        node = self.cluster_nodes[self.node_id]
                        node.last_heartbeat = datetime.now()
                        
                        # Update load metrics
                        node.load_average = await self._calculate_load_average()
                        node.active_healings = len([
                            task for task in self.task_scheduler.active_tasks.values()
                            if task.assigned_node == self.node_id
                        ]) if self.task_scheduler else 0
                
                # Send heartbeat to other nodes (simplified - would use network)
                await self._send_heartbeats()
                
                # Check for failed nodes
                await self._check_node_health()
                
                await asyncio.sleep(5)  # 5-second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)
    
    async def _coordination_loop(self):
        """Main coordination loop for task processing."""
        while self.running:
            try:
                # Only leader processes the task queue
                if self.leader_election and self.leader_election.is_leader():
                    await self._process_task_queue()
                
                # All nodes can execute assigned tasks
                await self._execute_assigned_tasks()
                
                # Update metrics
                self._update_metrics()
                
                await asyncio.sleep(1)  # 1-second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(1)
    
    async def _election_loop(self):
        """Leader election management loop."""
        while self.running:
            try:
                if self.leader_election:
                    # Check if election is needed
                    if self.leader_election.check_election_needed():
                        logger.info("Starting leader election")
                        self.leader_election.start_election()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in election loop: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_load_average(self) -> float:
        """Calculate current node load average."""
        # Simplified load calculation
        if self.task_scheduler:
            active_tasks = len([
                task for task in self.task_scheduler.active_tasks.values()
                if task.assigned_node == self.node_id
            ])
            capacity = self.cluster_nodes[self.node_id].healing_capacity
            return (active_tasks / capacity) * 100 if capacity > 0 else 0
        return 0.0
    
    async def _send_heartbeats(self):
        """Send heartbeats to other cluster nodes."""
        # In a real implementation, would send network messages
        # For now, just simulate by updating timestamps
        pass
    
    async def _check_node_health(self):
        """Check health of all cluster nodes."""
        with self.lock:
            current_time = datetime.now()
            
            for node in self.cluster_nodes.values():
                if node.node_id != self.node_id:  # Skip self
                    # Check if node has missed heartbeats
                    time_since_heartbeat = current_time - node.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(seconds=30):
                        if node.status == NodeStatus.ACTIVE:
                            node.status = NodeStatus.FAILED
                            logger.warning(f"Node {node.node_id} marked as failed")
                            
                            # Reassign tasks from failed node
                            await self._reassign_tasks_from_failed_node(node.node_id)
    
    async def _reassign_tasks_from_failed_node(self, failed_node_id: str):
        """Reassign tasks from a failed node."""
        if not self.task_scheduler:
            return
        
        # Find tasks assigned to failed node
        failed_tasks = [
            task for task in self.task_scheduler.active_tasks.values()
            if task.assigned_node == failed_node_id
        ]
        
        for task in failed_tasks:
            logger.info(f"Reassigning task {task.task_id} from failed node {failed_node_id}")
            
            # Reset task assignment
            task.assigned_node = None
            task.started_at = None
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Move back to pending queue
                del self.task_scheduler.active_tasks[task.task_id]
                self.task_scheduler.add_task(task)
            else:
                # Mark as failed
                self.task_scheduler.complete_task(
                    task.task_id,
                    None,  # No result available
                    success=False
                )
    
    async def _process_task_queue(self):
        """Process pending tasks (leader only)."""
        if not self.task_scheduler:
            return
        
        # Try to assign pending tasks
        while True:
            task = self.task_scheduler.get_next_task()
            if not task:
                break
            
            # Try to assign to available node
            assigned_node = self.task_scheduler.assign_task(task)
            
            if not assigned_node:
                # No nodes available, put task back
                self.task_scheduler.pending_tasks.insert(0, task)
                break
    
    async def _execute_assigned_tasks(self):
        """Execute tasks assigned to this node."""
        if not self.task_scheduler:
            return
        
        # Find tasks assigned to this node
        my_tasks = [
            task for task in self.task_scheduler.active_tasks.values()
            if task.assigned_node == self.node_id and task.status == HealingStatus.PENDING
        ]
        
        for task in my_tasks:
            try:
                # Check tenant quota if applicable
                if task.tenant_id:
                    # Would check tenant resource quotas here
                    pass
                
                # Execute healing plan
                task.status = HealingStatus.IN_PROGRESS
                
                logger.info(f"Executing healing task {task.task_id}")
                result = await self.healing_engine.execute_healing_plan(task.healing_plan)
                
                # Complete task
                success = result.status == HealingStatus.SUCCESSFUL
                self.task_scheduler.complete_task(task.task_id, result, success)
                
                # Update metrics
                self.metrics["tasks_executed"] += 1
                if not success:
                    self.metrics["tasks_failed"] += 1
                
                # Update average execution time
                if task.execution_time:
                    current_avg = self.metrics["average_execution_time"]
                    total_tasks = self.metrics["tasks_executed"]
                    self.metrics["average_execution_time"] = (
                        (current_avg * (total_tasks - 1) + task.execution_time) / total_tasks
                    )
                
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}")
                
                # Retry or fail task
                if not self.task_scheduler.retry_failed_task(task.task_id):
                    self.metrics["tasks_failed"] += 1
    
    def _update_metrics(self):
        """Update cluster metrics."""
        with self.lock:
            if self.cluster_nodes:
                total_capacity = sum(node.healing_capacity for node in self.cluster_nodes.values())
                total_active = sum(node.active_healings for node in self.cluster_nodes.values())
                
                self.metrics["cluster_utilization"] = (
                    (total_active / total_capacity) * 100 if total_capacity > 0 else 0
                )
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self.lock:
            node_statuses = {}
            for node_id, node in self.cluster_nodes.items():
                node_statuses[node_id] = {
                    "status": node.status.value,
                    "leader_status": node.leader_status.value,
                    "region": node.region,
                    "zone": node.zone,
                    "utilization": node.utilization,
                    "available_capacity": node.available_capacity,
                    "last_heartbeat": node.last_heartbeat.isoformat(),
                    "is_healthy": node.is_healthy
                }
            
            queue_status = self.task_scheduler.get_queue_status() if self.task_scheduler else {}
            
            leader_id = self.leader_election.get_leader() if self.leader_election else None
            
            return {
                "cluster_id": f"healing-cluster-{self.region}",
                "total_nodes": len(self.cluster_nodes),
                "healthy_nodes": len([n for n in self.cluster_nodes.values() if n.is_healthy]),
                "leader_node": leader_id,
                "is_leader": leader_id == self.node_id,
                "nodes": node_statuses,
                "queue_status": queue_status,
                "metrics": self.metrics,
                "last_updated": datetime.now().isoformat()
            }
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about this node."""
        with self.lock:
            if self.node_id in self.cluster_nodes:
                return self.cluster_nodes[self.node_id].to_dict()
            return {}


# Global cluster coordinator instance
cluster_coordinator = ClusterCoordinator()