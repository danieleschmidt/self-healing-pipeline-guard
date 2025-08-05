"""Performance optimization and scaling utilities."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
from collections import defaultdict, deque
import functools
import threading
import multiprocessing

import numpy as np
from asyncio import Semaphore, Queue

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis."""
    function_name: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    last_execution: Optional[datetime] = None
    
    def add_execution(self, execution_time: float, had_error: bool = False):
        """Add a new execution measurement."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.recent_times.append(execution_time)
        self.last_execution = datetime.now()
        
        if had_error:
            self.error_count += 1
    
    def get_recent_avg(self, window: int = 10) -> float:
        """Get average time for recent executions."""
        if not self.recent_times:
            return 0.0
        
        recent = list(self.recent_times)[-window:]
        return sum(recent) / len(recent)
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile of recent execution times."""
        if not self.recent_times:
            return 0.0
        
        times = sorted(self.recent_times)
        index = int(len(times) * percentile / 100)
        return times[min(index, len(times) - 1)]


class PerformanceProfiler:
    """Advanced performance profiler for optimization insights."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        self.enabled = True
    
    def profile(self, name: Optional[str] = None):
        """Decorator to profile function performance."""
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                had_error = False
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    had_error = True
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_execution(profile_name, execution_time, had_error)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                had_error = False
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    had_error = True
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_execution(profile_name, execution_time, had_error)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _record_execution(self, name: str, execution_time: float, had_error: bool):
        """Record execution metrics."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetrics(name)
            
            self.metrics[name].add_execution(execution_time, had_error)
    
    def get_report(self, top_n: int = 20) -> Dict[str, Any]:
        """Get performance report."""
        with self._lock:
            sorted_metrics = sorted(
                self.metrics.values(),
                key=lambda m: m.total_time,
                reverse=True
            )[:top_n]
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_functions": len(self.metrics),
                "functions": []
            }
            
            for metric in sorted_metrics:
                report["functions"].append({
                    "name": metric.function_name,
                    "execution_count": metric.execution_count,
                    "total_time": metric.total_time,
                    "avg_time": metric.avg_time,
                    "min_time": metric.min_time,
                    "max_time": metric.max_time,
                    "recent_avg": metric.get_recent_avg(),
                    "p95_time": metric.get_percentile(95),
                    "p99_time": metric.get_percentile(99),
                    "error_rate": metric.error_count / metric.execution_count if metric.execution_count > 0 else 0,
                    "last_execution": metric.last_execution.isoformat() if metric.last_execution else None
                })
            
            return report
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False


class ResourcePool:
    """Generic resource pool for connection pooling and resource management."""
    
    def __init__(
        self,
        create_resource: Callable,
        destroy_resource: Optional[Callable] = None,
        validate_resource: Optional[Callable] = None,
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: int = 300,  # 5 minutes
        validation_interval: int = 60  # 1 minute
    ):
        self.create_resource = create_resource
        self.destroy_resource = destroy_resource
        self.validate_resource = validate_resource
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.validation_interval = validation_interval
        
        self._pool: Queue = Queue(maxsize=max_size)
        self._size = 0
        self._created = 0
        self._destroyed = 0
        self._active = 0
        self._lock = asyncio.Lock()
        self._last_validation = 0
        
        # Initialize minimum resources
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self):
        """Initialize the pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                resource = await self._create_resource_async()
                await self._pool.put(resource)
                self._size += 1
            except Exception as e:
                logger.error(f"Failed to initialize pool resource: {e}")
    
    async def _create_resource_async(self):
        """Create a resource asynchronously."""
        if asyncio.iscoroutinefunction(self.create_resource):
            return await self.create_resource()
        else:
            return self.create_resource()
    
    async def _destroy_resource_async(self, resource):
        """Destroy a resource asynchronously."""
        if self.destroy_resource:
            if asyncio.iscoroutinefunction(self.destroy_resource):
                await self.destroy_resource(resource)
            else:
                self.destroy_resource(resource)
    
    async def _validate_resource_async(self, resource) -> bool:
        """Validate a resource asynchronously."""
        if not self.validate_resource:
            return True
        
        try:
            if asyncio.iscoroutinefunction(self.validate_resource):
                return await self.validate_resource(resource)
            else:
                return self.validate_resource(resource)
        except Exception:
            return False
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a resource from the pool."""
        resource = None
        
        try:
            # Try to get from pool
            try:
                resource = await asyncio.wait_for(self._pool.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Pool is empty, create new resource if under max size
                async with self._lock:
                    if self._size < self.max_size:
                        resource = await self._create_resource_async()
                        self._size += 1
                        self._created += 1
                    else:
                        # Wait for resource to become available
                        resource = await self._pool.get()
            
            # Validate resource
            if not await self._validate_resource_async(resource):
                await self._destroy_resource_async(resource)
                self._size -= 1
                self._destroyed += 1
                # Create new resource
                resource = await self._create_resource_async()
                self._size += 1
                self._created += 1
            
            self._active += 1
            yield resource
            
        finally:
            if resource:
                self._active -= 1
                # Return resource to pool
                try:
                    await self._pool.put(resource)
                except asyncio.QueueFull:
                    # Pool is full, destroy resource
                    await self._destroy_resource_async(resource)
                    self._size -= 1
                    self._destroyed += 1
    
    async def close(self):
        """Close the pool and destroy all resources."""
        async with self._lock:
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    await self._destroy_resource_async(resource)
                    self._destroyed += 1
                except asyncio.QueueEmpty:
                    break
            
            self._size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "size": self._size,
            "active": self._active,
            "idle": self._size - self._active,
            "created": self._created,
            "destroyed": self._destroyed,
            "max_size": self.max_size,
            "min_size": self.min_size
        }


class BatchProcessor:
    """Efficient batch processing with automatic optimization."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        max_workers: int = 4
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        
        self._batches: defaultdict[str, List[Any]] = defaultdict(list)
        self._batch_timers: Dict[str, float] = {}
        self._processors: Dict[str, Callable] = {}
        self._results: Dict[str, Queue] = defaultdict(Queue)
        self._lock = asyncio.Lock()
        self._processing = False
        
        # Performance tracking
        self.stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "avg_batch_size": 0,
            "avg_processing_time": 0
        }
    
    def register_processor(self, batch_type: str, processor: Callable):
        """Register a batch processor function."""
        self._processors[batch_type] = processor
    
    async def add_item(self, batch_type: str, item: Any) -> Any:
        """Add an item to a batch and get the result."""
        if batch_type not in self._processors:
            raise ValueError(f"No processor registered for batch type: {batch_type}")
        
        result_queue = Queue()
        
        async with self._lock:
            self._batches[batch_type].append((item, result_queue))
            
            # Set timer if this is the first item in batch
            if len(self._batches[batch_type]) == 1:
                self._batch_timers[batch_type] = time.time()
            
            # Check if batch is ready for processing
            batch = self._batches[batch_type]
            if (len(batch) >= self.batch_size or
                time.time() - self._batch_timers[batch_type] >= self.max_wait_time):
                
                # Process batch
                asyncio.create_task(self._process_batch(batch_type))
        
        # Wait for result
        return await result_queue.get()
    
    async def _process_batch(self, batch_type: str):
        """Process a batch of items."""
        async with self._lock:
            if not self._batches[batch_type]:
                return
            
            batch = self._batches[batch_type].copy()
            self._batches[batch_type].clear()
            
            if batch_type in self._batch_timers:
                del self._batch_timers[batch_type]
        
        if not batch:
            return
        
        start_time = time.time()
        processor = self._processors[batch_type]
        
        try:
            # Extract items and result queues
            items = [item for item, _ in batch]
            result_queues = [queue for _, queue in batch]
            
            # Process batch
            if asyncio.iscoroutinefunction(processor):
                results = await processor(items)
            else:
                results = processor(items)
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results] * len(items)
            
            # Send results to waiting coroutines
            for result, result_queue in zip(results, result_queues):
                await result_queue.put(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["batches_processed"] += 1
            self.stats["items_processed"] += len(items)
            self.stats["avg_batch_size"] = (
                self.stats["items_processed"] / self.stats["batches_processed"]
            )
            
            # Update average processing time
            if self.stats["avg_processing_time"] == 0:
                self.stats["avg_processing_time"] = processing_time
            else:
                self.stats["avg_processing_time"] = (
                    self.stats["avg_processing_time"] * 0.9 + processing_time * 0.1
                )
            
        except Exception as e:
            logger.error(f"Batch processing error for {batch_type}: {e}")
            
            # Send error to all waiting coroutines
            for _, result_queue in batch:
                await result_queue.put(e)
    
    async def flush(self, batch_type: Optional[str] = None):
        """Force process all pending batches."""
        if batch_type:
            await self._process_batch(batch_type)
        else:
            tasks = []
            for bt in list(self._batches.keys()):
                if self._batches[bt]:
                    tasks.append(self._process_batch(bt))
            
            if tasks:
                await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            **self.stats,
            "pending_batches": {
                bt: len(batch) for bt, batch in self._batches.items()
            },
            "registered_processors": list(self._processors.keys())
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system performance."""
    
    def __init__(
        self,
        initial_rate: float = 100.0,  # requests per second
        min_rate: float = 1.0,
        max_rate: float = 1000.0,
        adjustment_factor: float = 0.1,
        response_time_threshold: float = 1.0,  # seconds
        error_rate_threshold: float = 0.05  # 5%
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_factor = adjustment_factor
        self.response_time_threshold = response_time_threshold
        self.error_rate_threshold = error_rate_threshold
        
        self._semaphore = Semaphore(int(initial_rate))
        self._request_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._total_requests = 0
        self._last_adjustment = time.time()
        self._adjustment_interval = 5.0  # seconds
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire permission to make a request."""
        start_time = time.time()
        
        await self._semaphore.acquire()
        
        try:
            yield
            
            # Record successful request
            response_time = time.time() - start_time
            self._request_times.append(response_time)
            self._total_requests += 1
            
        except Exception as e:
            # Record error
            self._error_count += 1
            self._total_requests += 1
            raise
        
        finally:
            self._semaphore.release()
            
            # Adaptive adjustment
            await self._adjust_rate()
    
    async def _adjust_rate(self):
        """Adjust rate based on performance metrics."""
        now = time.time()
        
        if now - self._last_adjustment < self._adjustment_interval:
            return
        
        if not self._request_times or self._total_requests == 0:
            return
        
        # Calculate metrics
        avg_response_time = sum(self._request_times) / len(self._request_times)
        error_rate = self._error_count / self._total_requests
        
        # Determine adjustment
        adjustment = 0
        
        if avg_response_time > self.response_time_threshold:
            # Slow responses, decrease rate
            adjustment = -self.adjustment_factor
        elif error_rate > self.error_rate_threshold:
            # High error rate, decrease rate
            adjustment = -self.adjustment_factor * 2
        else:
            # Good performance, try to increase rate
            adjustment = self.adjustment_factor * 0.5
        
        # Apply adjustment
        new_rate = self.current_rate * (1 + adjustment)
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))
        
        if abs(new_rate - self.current_rate) > 1:
            logger.info(
                f"Adjusting rate limit: {self.current_rate:.1f} -> {new_rate:.1f} "
                f"(avg_rt: {avg_response_time:.3f}s, err_rate: {error_rate:.3f})"
            )
            
            self.current_rate = new_rate
            
            # Update semaphore capacity
            new_capacity = int(new_rate)
            current_capacity = self._semaphore._value + (int(self.current_rate) - self._semaphore._value)
            
            # Create new semaphore with adjusted capacity
            self._semaphore = Semaphore(new_capacity)
        
        self._last_adjustment = now
        
        # Reset counters periodically
        if self._total_requests > 1000:
            self._error_count = int(self._error_count * 0.8)
            self._total_requests = int(self._total_requests * 0.8)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        if self._request_times and self._total_requests > 0:
            avg_response_time = sum(self._request_times) / len(self._request_times)
            error_rate = self._error_count / self._total_requests
        else:
            avg_response_time = 0
            error_rate = 0
        
        return {
            "current_rate": self.current_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "total_requests": self._total_requests,
            "error_count": self._error_count,
            "available_permits": self._semaphore._value
        }


class ConcurrencyLimiter:
    """Intelligent concurrency limiting with queue management."""
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: int = 100,
        timeout: float = 30.0,
        auto_adjust: bool = True
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self.auto_adjust = auto_adjust
        
        self._semaphore = Semaphore(max_concurrent)
        self._queue_size = 0
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._total_wait_time = 0.0
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire concurrency slot with queue management."""
        if self._queue_size >= self.max_queue_size:
            raise asyncio.QueueFull("Concurrency queue is full")
        
        wait_start = time.time()
        
        async with self._lock:
            self._queue_size += 1
        
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.timeout)
            
            wait_time = time.time() - wait_start
            self._total_wait_time += wait_time
            
            async with self._lock:
                self._queue_size -= 1
                self._active_tasks += 1
            
            try:
                yield
                self._completed_tasks += 1
            except Exception:
                self._failed_tasks += 1
                raise
            
        except asyncio.TimeoutError:
            async with self._lock:
                self._queue_size -= 1
            raise
        
        finally:
            async with self._lock:
                if self._active_tasks > 0:
                    self._active_tasks -= 1
            
            self._semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency limiter statistics."""
        total_tasks = self._completed_tasks + self._failed_tasks
        success_rate = self._completed_tasks / total_tasks if total_tasks > 0 else 0
        avg_wait_time = self._total_wait_time / total_tasks if total_tasks > 0 else 0
        
        return {
            "max_concurrent": self.max_concurrent,
            "active_tasks": self._active_tasks,
            "queue_size": self._queue_size,
            "max_queue_size": self.max_queue_size,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": success_rate,
            "avg_wait_time": avg_wait_time,
            "available_slots": self._semaphore._value
        }


# Global instances
profiler = PerformanceProfiler()
batch_processor = BatchProcessor()
rate_limiter = AdaptiveRateLimiter()
concurrency_limiter = ConcurrencyLimiter()