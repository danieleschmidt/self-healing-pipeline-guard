"""Custom exception classes for the API."""

from typing import Any, Dict, Optional


class HealingGuardException(Exception):
    """Base exception for Healing Guard errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationException(HealingGuardException):
    """Raised when input validation fails."""
    pass


class AuthenticationException(HealingGuardException):
    """Raised when authentication fails."""
    pass


class AuthorizationException(HealingGuardException):
    """Raised when authorization fails."""
    pass


class ResourceNotFoundException(HealingGuardException):
    """Raised when a requested resource is not found."""
    pass


class ServiceUnavailableException(HealingGuardException):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, message: str, retry_after: int = 60, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after


class RateLimitExceededException(HealingGuardException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: int = 60, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after


class QuantumPlannerException(HealingGuardException):
    """Raised when quantum planner operations fail."""
    pass


class HealingEngineException(HealingGuardException):
    """Raised when healing engine operations fail."""
    pass


class FailureDetectionException(HealingGuardException):
    """Raised when failure detection operations fail."""
    pass