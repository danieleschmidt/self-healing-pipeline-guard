"""Custom exceptions for the Healing Guard system with sentiment analysis support."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


class HealingGuardException(Exception):
    """Base exception for all Healing Guard errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now()
        
        # Log the exception
        logger = logging.getLogger(__name__)
        logger.error(f"{self.error_code}: {message}", extra={"details": self.details})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SentimentAnalysisException(HealingGuardException):
    """Exception raised when sentiment analysis fails."""
    
    def __init__(
        self, 
        message: str = "Sentiment analysis failed", 
        text_preview: str = None,
        analysis_stage: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if text_preview:
            details['text_preview'] = text_preview[:100] + "..." if len(text_preview) > 100 else text_preview
        if analysis_stage:
            details['analysis_stage'] = analysis_stage
        
        super().__init__(message, "SENTIMENT_ANALYSIS_ERROR", details)


class SentimentValidationException(HealingGuardException):
    """Exception raised when sentiment analysis input validation fails."""
    
    def __init__(
        self, 
        message: str = "Sentiment analysis validation failed",
        validation_errors: List[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if validation_errors:
            details['validation_errors'] = validation_errors
        
        super().__init__(message, "SENTIMENT_VALIDATION_ERROR", details)


class HealingEngineException(HealingGuardException):
    """Exception raised when healing engine operations fail."""
    
    def __init__(
        self,
        message: str = "Healing engine operation failed",
        failure_event_id: str = None,
        healing_plan_id: str = None,
        strategy: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if failure_event_id:
            details['failure_event_id'] = failure_event_id
        if healing_plan_id:
            details['healing_plan_id'] = healing_plan_id
        if strategy:
            details['strategy'] = strategy
        
        super().__init__(message, "HEALING_ENGINE_ERROR", details)


class HealingActionException(HealingGuardException):
    """Exception raised when individual healing actions fail."""
    
    def __init__(
        self,
        message: str = "Healing action failed",
        action_id: str = None,
        strategy: str = None,
        rollback_available: bool = False,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if action_id:
            details['action_id'] = action_id
        if strategy:
            details['strategy'] = strategy
        details['rollback_available'] = rollback_available
        
        super().__init__(message, "HEALING_ACTION_ERROR", details)


class QuantumPlannerException(HealingGuardException):
    """Exception raised when quantum planner operations fail."""
    
    def __init__(
        self,
        message: str = "Quantum planner operation failed",
        optimization_stage: str = None,
        task_count: int = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if optimization_stage:
            details['optimization_stage'] = optimization_stage
        if task_count is not None:
            details['task_count'] = task_count
        
        super().__init__(message, "QUANTUM_PLANNER_ERROR", details)


class FailureDetectionException(HealingGuardException):
    """Exception raised when failure detection fails."""
    
    def __init__(
        self,
        message: str = "Failure detection failed",
        job_id: str = None,
        repository: str = None,
        detection_stage: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if job_id:
            details['job_id'] = job_id
        if repository:
            details['repository'] = repository
        if detection_stage:
            details['detection_stage'] = detection_stage
        
        super().__init__(message, "FAILURE_DETECTION_ERROR", details)


class ConfigurationException(HealingGuardException):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: str = None,
        expected_type: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if expected_type:
            details['expected_type'] = expected_type
        
        super().__init__(message, "CONFIGURATION_ERROR", details)


class RateLimitException(HealingGuardException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: int = None,
        window_seconds: int = None,
        retry_after: int = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if limit is not None:
            details['limit'] = limit
        if window_seconds is not None:
            details['window_seconds'] = window_seconds
        if retry_after is not None:
            details['retry_after'] = retry_after
        
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class SecurityException(HealingGuardException):
    """Exception raised for security-related issues."""
    
    def __init__(
        self,
        message: str = "Security violation",
        security_check: str = None,
        severity: str = "medium",
        **kwargs
    ):
        details = kwargs.get('details', {})
        if security_check:
            details['security_check'] = security_check
        details['severity'] = severity
        
        super().__init__(message, "SECURITY_ERROR", details)


class ResourceException(HealingGuardException):
    """Exception raised when resource constraints are exceeded."""
    
    def __init__(
        self,
        message: str = "Resource constraint exceeded",
        resource_type: str = None,
        current_usage: float = None,
        limit: float = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if current_usage is not None:
            details['current_usage'] = current_usage
        if limit is not None:
            details['limit'] = limit
        
        super().__init__(message, "RESOURCE_ERROR", details)


class IntegrationException(HealingGuardException):
    """Exception raised when external integrations fail."""
    
    def __init__(
        self,
        message: str = "External integration failed",
        integration_name: str = None,
        endpoint: str = None,
        status_code: int = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if integration_name:
            details['integration_name'] = integration_name
        if endpoint:
            details['endpoint'] = endpoint
        if status_code is not None:
            details['status_code'] = status_code
        
        super().__init__(message, "INTEGRATION_ERROR", details)


class RetryableException(HealingGuardException):
    """Exception that indicates the operation can be retried."""
    
    def __init__(
        self,
        message: str = "Operation failed but can be retried",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['max_retries'] = max_retries
        details['retry_delay'] = retry_delay
        details['retryable'] = True
        
        super().__init__(message, "RETRYABLE_ERROR", details)


class NonRetryableException(HealingGuardException):
    """Exception that indicates the operation should not be retried."""
    
    def __init__(
        self,
        message: str = "Operation failed and should not be retried",
        reason: str = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['retryable'] = False
        if reason:
            details['reason'] = reason
        
        super().__init__(message, "NON_RETRYABLE_ERROR", details)


# Exception handling utilities

def handle_sentiment_errors(func):
    """Decorator to handle sentiment analysis errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert generic exceptions to sentiment-specific ones
            if "sentiment" in str(e).lower() or "analysis" in str(e).lower():
                raise SentimentAnalysisException(
                    message=f"Sentiment analysis failed: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                )
            else:
                raise HealingGuardException(
                    message=f"Unexpected error in {func.__name__}: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                )
    return wrapper


def handle_healing_errors(func):
    """Decorator to handle healing engine errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert generic exceptions to healing-specific ones
            if any(keyword in str(e).lower() for keyword in ["healing", "action", "plan"]):
                raise HealingEngineException(
                    message=f"Healing operation failed: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                )
            else:
                raise HealingGuardException(
                    message=f"Unexpected error in {func.__name__}: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                )
    return wrapper


def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exception, RetryableException):
        return True
    elif isinstance(exception, NonRetryableException):
        return False
    
    # Check for common retryable error patterns
    retryable_patterns = [
        "timeout", "connection", "network", "temporary", 
        "rate limit", "service unavailable", "502", "503", "504"
    ]
    
    error_message = str(exception).lower()
    return any(pattern in error_message for pattern in retryable_patterns)


def get_retry_delay(exception: Exception, attempt: int) -> float:
    """Calculate retry delay for retryable exceptions."""
    if isinstance(exception, (RetryableException, RateLimitException)):
        base_delay = exception.details.get('retry_delay', 1.0)
        retry_after = exception.details.get('retry_after')
        if retry_after:
            return float(retry_after)
    else:
        base_delay = 1.0
    
    # Exponential backoff with jitter
    import random
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0.1, 0.3) * delay
    return delay + jitter


class ExceptionCollector:
    """Utility class to collect and analyze exceptions."""
    
    def __init__(self):
        self.exceptions: List[HealingGuardException] = []
    
    def add(self, exception: Exception):
        """Add an exception to the collection."""
        if isinstance(exception, HealingGuardException):
            self.exceptions.append(exception)
        else:
            # Convert generic exceptions
            healing_exception = HealingGuardException(
                message=str(exception),
                details={"original_type": type(exception).__name__}
            )
            self.exceptions.append(healing_exception)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected exceptions."""
        if not self.exceptions:
            return {"total_exceptions": 0}
        
        error_codes = [exc.error_code for exc in self.exceptions]
        error_code_counts = {code: error_codes.count(code) for code in set(error_codes)}
        
        retryable_count = sum(
            1 for exc in self.exceptions 
            if exc.details.get('retryable', is_retryable_error(exc))
        )
        
        return {
            "total_exceptions": len(self.exceptions),
            "error_code_distribution": error_code_counts,
            "retryable_exceptions": retryable_count,
            "non_retryable_exceptions": len(self.exceptions) - retryable_count,
            "most_recent": self.exceptions[-1].to_dict() if self.exceptions else None,
            "time_range": {
                "first": self.exceptions[0].timestamp.isoformat(),
                "last": self.exceptions[-1].timestamp.isoformat()
            } if len(self.exceptions) > 1 else None
        }
    
    def has_critical_errors(self) -> bool:
        """Check if collection contains critical errors."""
        critical_codes = {
            "SECURITY_ERROR", "CONFIGURATION_ERROR", 
            "NON_RETRYABLE_ERROR", "RESOURCE_ERROR"
        }
        return any(exc.error_code in critical_codes for exc in self.exceptions)