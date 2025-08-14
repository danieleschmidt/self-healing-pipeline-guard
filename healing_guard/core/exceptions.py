"""Enhanced exception handling for the healing system.

Provides specialized exceptions with detailed context and recovery information.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


class HealingSystemException(Exception):
    """Base exception for healing system."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, recoverable: bool = True):
        super().__init__(message)
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()


class CircuitBreakerOpenException(HealingSystemException):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str = None, next_attempt_time: datetime = None):
        super().__init__(message, recoverable=True)
        self.circuit_name = circuit_name
        self.next_attempt_time = next_attempt_time


class RetryExhaustedException(HealingSystemException):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, attempts: int, last_exception: Exception, metadata: List[Dict] = None):
        super().__init__(message, recoverable=False)
        self.attempts = attempts
        self.last_exception = last_exception
        self.metadata = metadata or []


class HealingPlanException(HealingSystemException):
    """Exception raised during healing plan creation or execution."""
    
    def __init__(self, message: str, plan_id: str = None, stage: str = None, action_id: str = None):
        super().__init__(message)
        self.plan_id = plan_id
        self.stage = stage
        self.action_id = action_id


class FailureDetectionException(HealingSystemException):
    """Exception raised during failure detection and analysis."""
    
    def __init__(self, message: str, job_id: str = None, analysis_stage: str = None):
        super().__init__(message)
        self.job_id = job_id
        self.analysis_stage = analysis_stage


class QuantumPlannerException(HealingSystemException):
    """Exception raised during quantum-inspired task planning."""
    
    def __init__(self, message: str, optimization_phase: str = None, task_count: int = None):
        super().__init__(message)
        self.optimization_phase = optimization_phase
        self.task_count = task_count


class ResilienceException(HealingSystemException):
    """Exception raised by resilience mechanisms."""
    
    def __init__(self, message: str, mechanism: str = None, service_name: str = None):
        super().__init__(message)
        self.mechanism = mechanism
        self.service_name = service_name


class ValidationException(HealingSystemException):
    """Exception raised during input validation."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, recoverable=False)
        self.field = field
        self.value = value


class ConfigurationException(HealingSystemException):
    """Exception raised due to configuration issues."""
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None):
        super().__init__(message, recoverable=False)
        self.config_key = config_key
        self.config_value = config_value


class SecurityException(HealingSystemException):
    """Exception raised for security-related issues."""
    
    def __init__(self, message: str, security_check: str = None, severity: str = "medium"):
        super().__init__(message, recoverable=False)
        self.security_check = security_check
        self.severity = severity


def handle_exception_with_context(exception: Exception, context: Dict[str, Any] = None) -> HealingSystemException:
    """Convert generic exceptions to healing system exceptions with context."""
    
    context = context or {}
    
    # Map common exceptions to healing system exceptions
    if isinstance(exception, asyncio.TimeoutError):
        return ResilienceException(
            f"Operation timed out: {str(exception)}",
            mechanism="timeout",
            service_name=context.get("service_name")
        )
    elif isinstance(exception, ConnectionError):
        return ResilienceException(
            f"Connection error: {str(exception)}",
            mechanism="connection",
            service_name=context.get("service_name")
        )
    elif isinstance(exception, ValueError):
        return ValidationException(
            f"Validation error: {str(exception)}",
            field=context.get("field"),
            value=context.get("value")
        )
    elif isinstance(exception, KeyError):
        return ConfigurationException(
            f"Configuration key missing: {str(exception)}",
            config_key=str(exception).strip("'\""),
            config_value=None
        )
    else:
        # Generic healing system exception
        return HealingSystemException(
            f"Unexpected error: {str(exception)}",
            context=context
        )