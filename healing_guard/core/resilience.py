"""Advanced resilience and reliability features for the healing system.

Implements circuit breakers, retry mechanisms, health monitoring,
and self-recovery capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import json

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failures detected, circuit open
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open to closed transition
    timeout: float = 30.0  # operation timeout


@dataclass
class HealthMetrics:
    """Health metrics tracking."""
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return (datetime.now() - self.uptime_start).total_seconds()


class CircuitBreaker:
    """Circuit breaker implementation for reliability."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Recent failures tracking
        self.recent_failures: deque = deque(maxlen=config.failure_threshold * 2)
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker {self.name} is OPEN. Next attempt at {self.next_attempt_time}"
                )
        
        try:
            # Execute with timeout
            start_time = time.time()
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            execution_time = time.time() - start_time
            await self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            await self._record_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.next_attempt_time is None:
            return True
        return datetime.now() >= self.next_attempt_time
    
    async def _record_success(self, execution_time: float):
        """Record successful operation."""
        self.success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.recent_failures.clear()
                logger.info(f"Circuit breaker {self.name} moved to CLOSED")
        
        logger.debug(f"Circuit breaker {self.name}: Success recorded ({execution_time:.3f}s)")
    
    async def _record_failure(self, exception: Exception):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.recent_failures.append({
            "timestamp": self.last_failure_time,
            "exception": str(exception),
            "type": type(exception).__name__
        })
        
        if self.state == CircuitState.HALF_OPEN:
            # Go back to OPEN immediately on any failure in HALF_OPEN
            self.state = CircuitState.OPEN
            self._set_next_attempt_time()
            logger.warning(f"Circuit breaker {self.name} moved back to OPEN after failure in HALF_OPEN")
            
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._set_next_attempt_time()
            logger.warning(f"Circuit breaker {self.name} moved to OPEN after {self.failure_count} failures")
        
        logger.error(f"Circuit breaker {self.name}: Failure recorded - {exception}")
    
    def _set_next_attempt_time(self):
        """Set next attempt time based on recovery timeout."""
        self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            "recent_failures": list(self.recent_failures)
        }


class RetryPolicy:
    """Advanced retry policy with backoff strategies."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry policy."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_attempts} attempts failed")
        
        raise last_exception


class HealthMonitor:
    """Health monitoring and alerting system."""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds = {
            "success_rate": 0.95,
            "avg_response_time": 5.0,  # seconds
            "failure_rate": 0.05
        }
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker for monitoring."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        self.metrics[name] = HealthMetrics()
        logger.info(f"Registered circuit breaker: {name}")
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def execute_with_monitoring(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with full monitoring."""
        if service_name not in self.circuit_breakers:
            # Create default circuit breaker if not exists
            self.register_circuit_breaker(service_name, CircuitBreakerConfig())
        
        circuit_breaker = self.circuit_breakers[service_name]
        metrics = self.metrics[service_name]
        
        start_time = time.time()
        
        try:
            result = await circuit_breaker.call(func, *args, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            metrics.success_count += 1
            metrics.total_requests += 1
            metrics.last_success = datetime.now()
            
            # Update rolling average
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = execution_time
            else:
                metrics.avg_response_time = (metrics.avg_response_time + execution_time) / 2
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            metrics.failure_count += 1
            metrics.total_requests += 1
            metrics.last_failure = datetime.now()
            
            # Check if alert thresholds are breached
            await self._check_health_alerts(service_name)
            
            raise
    
    async def _check_health_alerts(self, service_name: str):
        """Check if health alerts should be triggered."""
        metrics = self.metrics[service_name]
        
        success_rate = metrics.success_rate
        if success_rate < self.alert_thresholds["success_rate"]:
            await self._trigger_alert(
                service_name, 
                "low_success_rate", 
                f"Success rate {success_rate:.2%} below threshold {self.alert_thresholds['success_rate']:.2%}"
            )
        
        if metrics.avg_response_time > self.alert_thresholds["avg_response_time"]:
            await self._trigger_alert(
                service_name,
                "high_response_time",
                f"Average response time {metrics.avg_response_time:.2f}s above threshold {self.alert_thresholds['avg_response_time']}s"
            )
    
    async def _trigger_alert(self, service_name: str, alert_type: str, message: str):
        """Trigger health alert."""
        alert = {
            "service": service_name,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "severity": "warning"
        }
        
        logger.warning(f"Health alert: {json.dumps(alert)}")
        
        # Here you would integrate with your alerting system (Slack, PagerDuty, etc.)
        # For now, we'll just log the alert
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = await check_func()
                check_time = time.time() - start_time
                
                results[name] = {
                    "healthy": is_healthy,
                    "check_time": check_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        circuit_states = {name: cb.get_state() for name, cb in self.circuit_breakers.items()}
        
        metrics_summary = {}
        for name, metrics in self.metrics.items():
            metrics_summary[name] = {
                "success_rate": metrics.success_rate,
                "total_requests": metrics.total_requests,
                "avg_response_time": metrics.avg_response_time,
                "uptime_seconds": metrics.uptime_seconds,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None
            }
        
        return {
            "circuit_breakers": circuit_states,
            "metrics": metrics_summary,
            "health_checks": len(self.health_checks),
            "timestamp": datetime.now().isoformat()
        }


class SelfHealingManager:
    """Self-healing capabilities manager."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.healing_strategies: Dict[str, Callable] = {}
        self.auto_healing_enabled = True
        self.healing_history: List[Dict[str, Any]] = []
    
    def register_healing_strategy(self, problem_type: str, strategy: Callable):
        """Register a self-healing strategy for a problem type."""
        self.healing_strategies[problem_type] = strategy
        logger.info(f"Registered healing strategy for: {problem_type}")
    
    async def attempt_self_healing(self, service_name: str, problem_type: str) -> bool:
        """Attempt to self-heal a detected problem."""
        if not self.auto_healing_enabled:
            logger.info("Auto-healing is disabled")
            return False
        
        if problem_type not in self.healing_strategies:
            logger.warning(f"No healing strategy available for problem type: {problem_type}")
            return False
        
        healing_start = datetime.now()
        strategy = self.healing_strategies[problem_type]
        
        try:
            logger.info(f"Attempting self-healing for {service_name} problem: {problem_type}")
            
            success = await strategy(service_name)
            
            healing_record = {
                "service": service_name,
                "problem_type": problem_type,
                "start_time": healing_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "success": success,
                "strategy": strategy.__name__
            }
            
            self.healing_history.append(healing_record)
            
            if success:
                logger.info(f"Self-healing succeeded for {service_name}")
            else:
                logger.warning(f"Self-healing failed for {service_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Self-healing strategy failed: {e}")
            
            healing_record = {
                "service": service_name,
                "problem_type": problem_type,
                "start_time": healing_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "strategy": strategy.__name__
            }
            
            self.healing_history.append(healing_record)
            return False
    
    def get_healing_history(self) -> List[Dict[str, Any]]:
        """Get self-healing history."""
        return self.healing_history[-100:]  # Return last 100 healing attempts


# Exception classes
class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass


class ResilienceManager:
    """Main resilience manager coordinating all reliability features."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.self_healing = SelfHealingManager(self.health_monitor)
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Register default healing strategies
        self._register_default_healing_strategies()
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        async def memory_check():
            """Check memory usage."""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 85.0  # Alert if memory > 85%
            except ImportError:
                return True  # Skip if psutil not available
        
        async def disk_check():
            """Check disk space."""
            try:
                import psutil
                disk_usage = psutil.disk_usage('/').percent
                return disk_usage < 90.0  # Alert if disk > 90%
            except ImportError:
                return True  # Skip if psutil not available
        
        async def api_health_check():
            """Check API responsiveness."""
            try:
                # Simple health check - verify we can create basic objects
                from ..core.healing_engine import HealingEngine
                engine = HealingEngine()
                return True
            except Exception:
                return False
        
        self.health_monitor.register_health_check("memory", memory_check)
        self.health_monitor.register_health_check("disk", disk_check)
        self.health_monitor.register_health_check("api", api_health_check)
    
    def _register_default_healing_strategies(self):
        """Register default self-healing strategies."""
        
        async def restart_service_strategy(service_name: str) -> bool:
            """Strategy to restart a failing service."""
            logger.info(f"Attempting to restart service: {service_name}")
            # In a real implementation, this would restart the actual service
            await asyncio.sleep(1.0)  # Simulate restart time
            return True
        
        async def clear_cache_strategy(service_name: str) -> bool:
            """Strategy to clear caches."""
            logger.info(f"Attempting to clear caches for service: {service_name}")
            # In a real implementation, this would clear relevant caches
            await asyncio.sleep(0.5)
            return True
        
        async def scale_resources_strategy(service_name: str) -> bool:
            """Strategy to scale resources."""
            logger.info(f"Attempting to scale resources for service: {service_name}")
            # In a real implementation, this would scale compute resources
            await asyncio.sleep(2.0)
            return True
        
        self.self_healing.register_healing_strategy("service_failure", restart_service_strategy)
        self.self_healing.register_healing_strategy("cache_corruption", clear_cache_strategy)
        self.self_healing.register_healing_strategy("resource_exhaustion", scale_resources_strategy)
    
    def create_retry_policy(self, name: str, **kwargs) -> RetryPolicy:
        """Create and register a retry policy."""
        policy = RetryPolicy(**kwargs)
        self.retry_policies[name] = policy
        return policy
    
    def get_retry_policy(self, name: str) -> Optional[RetryPolicy]:
        """Get a registered retry policy."""
        return self.retry_policies.get(name)
    
    async def execute_with_resilience(
        self,
        service_name: str,
        func: Callable,
        retry_policy_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full resilience features."""
        
        # Wrap with retry policy if specified
        if retry_policy_name and retry_policy_name in self.retry_policies:
            retry_policy = self.retry_policies[retry_policy_name]
            
            async def retry_wrapped():
                return await self.health_monitor.execute_with_monitoring(
                    service_name, func, *args, **kwargs
                )
            
            return await retry_policy.execute(retry_wrapped)
        else:
            return await self.health_monitor.execute_with_monitoring(
                service_name, func, *args, **kwargs
            )
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        health_checks = await self.health_monitor.run_health_checks()
        health_summary = self.health_monitor.get_health_summary()
        healing_history = self.self_healing.get_healing_history()
        
        return {
            "health_checks": health_checks,
            "health_summary": health_summary,
            "healing_history": healing_history,
            "retry_policies": list(self.retry_policies.keys()),
            "auto_healing_enabled": self.self_healing.auto_healing_enabled,
            "timestamp": datetime.now().isoformat()
        }


# Global resilience manager instance
resilience_manager = ResilienceManager()