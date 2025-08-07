"""Structured logging configuration for Healing Guard with sentiment analysis support."""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps
import threading
import uuid

from ..core.config import settings

# Thread-local storage for request context
_context_storage = threading.local()


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self):
        super().__init__()
        self.service_name = "healing-guard"
        self.version = "1.0.0"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "version": self.version,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add thread information
        log_entry["thread_id"] = record.thread
        log_entry["thread_name"] = record.threadName
        
        # Add process information
        log_entry["process_id"] = record.process
        
        # Add request context if available
        context = getattr(_context_storage, 'context', None)
        if context:
            log_entry["request_context"] = context
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 
                'processName', 'process', 'exc_info', 'exc_text', 'stack_info',
                'getMessage'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class SentimentAnalysisLogger:
    """Specialized logger for sentiment analysis operations."""
    
    def __init__(self, logger_name: str = "sentiment_analysis"):
        self.logger = logging.getLogger(logger_name)
    
    def log_analysis_start(
        self,
        text_preview: str,
        context: Optional[Dict[str, Any]] = None,
        analysis_type: str = "single"
    ):
        """Log the start of sentiment analysis."""
        self.logger.info(
            "Starting sentiment analysis",
            extra={
                "event_type": "sentiment_analysis_start",
                "analysis_type": analysis_type,
                "text_length": len(text_preview),
                "text_preview": text_preview[:100] + "..." if len(text_preview) > 100 else text_preview,
                "context": context
            }
        )
    
    def log_analysis_complete(
        self,
        analysis_id: str,
        sentiment_label: str,
        confidence: float,
        processing_time_ms: float,
        text_length: int,
        urgency_score: float = None,
        is_urgent: bool = False,
        is_frustrated: bool = False
    ):
        """Log successful completion of sentiment analysis."""
        extra_data = {
            "event_type": "sentiment_analysis_complete",
            "analysis_id": analysis_id,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "processing_time_ms": processing_time_ms,
            "text_length": text_length,
            "success": True
        }
        
        if urgency_score is not None:
            extra_data.update({
                "urgency_score": urgency_score,
                "is_urgent": is_urgent,
                "is_frustrated": is_frustrated
            })
        
        # Use different log levels based on sentiment urgency
        if is_urgent:
            self.logger.warning("URGENT sentiment detected", extra=extra_data)
        elif is_frustrated:
            self.logger.warning("FRUSTRATED sentiment detected", extra=extra_data)
        else:
            self.logger.info("Sentiment analysis completed", extra=extra_data)
    
    def log_analysis_error(
        self,
        text_preview: str,
        error_message: str,
        error_type: str = "unknown",
        processing_time_ms: float = None
    ):
        """Log sentiment analysis errors."""
        self.logger.error(
            "Sentiment analysis failed",
            extra={
                "event_type": "sentiment_analysis_error",
                "error_type": error_type,
                "error_message": error_message,
                "text_preview": text_preview[:100] + "..." if len(text_preview) > 100 else text_preview,
                "text_length": len(text_preview),
                "processing_time_ms": processing_time_ms,
                "success": False
            }
        )
    
    def log_batch_analysis(
        self,
        batch_size: int,
        total_processing_time_ms: float,
        successful_analyses: int,
        failed_analyses: int,
        sentiment_distribution: Dict[str, int] = None
    ):
        """Log batch sentiment analysis results."""
        self.logger.info(
            "Batch sentiment analysis completed",
            extra={
                "event_type": "batch_sentiment_analysis",
                "batch_size": batch_size,
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "total_processing_time_ms": total_processing_time_ms,
                "average_processing_time_ms": total_processing_time_ms / batch_size if batch_size > 0 else 0,
                "sentiment_distribution": sentiment_distribution or {},
                "success_rate": successful_analyses / batch_size if batch_size > 0 else 0
            }
        )
    
    def log_concerning_pattern(
        self,
        pattern_type: str,
        urgency_count: int,
        frustrated_count: int,
        negative_count: int,
        total_analyses: int,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log concerning sentiment patterns."""
        self.logger.warning(
            "Concerning sentiment pattern detected",
            extra={
                "event_type": "concerning_sentiment_pattern",
                "pattern_type": pattern_type,
                "urgency_count": urgency_count,
                "frustrated_count": frustrated_count,
                "negative_count": negative_count,
                "total_analyses": total_analyses,
                "urgency_rate": urgency_count / total_analyses if total_analyses > 0 else 0,
                "frustration_rate": frustrated_count / total_analyses if total_analyses > 0 else 0,
                "negative_rate": negative_count / total_analyses if total_analyses > 0 else 0,
                "context": context
            }
        )


class HealingEngineLogger:
    """Specialized logger for healing engine operations."""
    
    def __init__(self, logger_name: str = "healing_engine"):
        self.logger = logging.getLogger(logger_name)
    
    def log_healing_plan_created(
        self,
        plan_id: str,
        failure_event_id: str,
        failure_type: str,
        action_count: int,
        estimated_time: float,
        success_probability: float,
        priority: int,
        sentiment_enhanced: bool = False,
        sentiment_urgency: float = None
    ):
        """Log creation of healing plan."""
        extra_data = {
            "event_type": "healing_plan_created",
            "plan_id": plan_id,
            "failure_event_id": failure_event_id,
            "failure_type": failure_type,
            "action_count": action_count,
            "estimated_time_minutes": estimated_time,
            "success_probability": success_probability,
            "priority": priority,
            "sentiment_enhanced": sentiment_enhanced
        }
        
        if sentiment_urgency is not None:
            extra_data["sentiment_urgency"] = sentiment_urgency
        
        self.logger.info("Healing plan created", extra=extra_data)
    
    def log_sentiment_priority_adjustment(
        self,
        plan_id: str,
        original_priority: int,
        adjusted_priority: int,
        sentiment_label: str,
        urgency_score: float,
        adjustment_reason: str
    ):
        """Log sentiment-based priority adjustments."""
        self.logger.info(
            "Healing priority adjusted based on sentiment",
            extra={
                "event_type": "sentiment_priority_adjustment",
                "plan_id": plan_id,
                "original_priority": original_priority,
                "adjusted_priority": adjusted_priority,
                "priority_change": original_priority - adjusted_priority,
                "sentiment_label": sentiment_label,
                "urgency_score": urgency_score,
                "adjustment_reason": adjustment_reason
            }
        )
    
    def log_healing_execution_start(
        self,
        healing_id: str,
        plan_id: str,
        action_count: int
    ):
        """Log start of healing execution."""
        self.logger.info(
            "Healing execution started",
            extra={
                "event_type": "healing_execution_start",
                "healing_id": healing_id,
                "plan_id": plan_id,
                "action_count": action_count
            }
        )
    
    def log_healing_action_executed(
        self,
        healing_id: str,
        action_id: str,
        strategy: str,
        success: bool,
        duration: float,
        error_message: Optional[str] = None
    ):
        """Log individual healing action execution."""
        log_level = logging.INFO if success else logging.ERROR
        
        self.logger.log(
            log_level,
            f"Healing action {'completed' if success else 'failed'}",
            extra={
                "event_type": "healing_action_executed",
                "healing_id": healing_id,
                "action_id": action_id,
                "strategy": strategy,
                "success": success,
                "duration_seconds": duration,
                "error_message": error_message
            }
        )
    
    def log_healing_execution_complete(
        self,
        healing_id: str,
        plan_id: str,
        success: bool,
        total_duration: float,
        successful_actions: int,
        failed_actions: int,
        rollback_performed: bool = False
    ):
        """Log completion of healing execution."""
        log_level = logging.INFO if success else logging.ERROR
        
        self.logger.log(
            log_level,
            f"Healing execution {'completed successfully' if success else 'failed'}",
            extra={
                "event_type": "healing_execution_complete",
                "healing_id": healing_id,
                "plan_id": plan_id,
                "success": success,
                "total_duration_minutes": total_duration,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "rollback_performed": rollback_performed,
                "success_rate": successful_actions / (successful_actions + failed_actions) if (successful_actions + failed_actions) > 0 else 0
            }
        )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger_name: str = "security"):
        self.logger = logging.getLogger(logger_name)
    
    def log_security_violation(
        self,
        violation_type: str,
        severity: str,
        details: Dict[str, Any],
        blocked: bool = True
    ):
        """Log security violations."""
        self.logger.warning(
            f"Security violation detected: {violation_type}",
            extra={
                "event_type": "security_violation",
                "violation_type": violation_type,
                "severity": severity,
                "blocked": blocked,
                "details": details
            }
        )
    
    def log_input_validation_failure(
        self,
        validation_type: str,
        errors: List[str],
        security_flags: List[str],
        input_preview: str = None
    ):
        """Log input validation failures."""
        self.logger.warning(
            "Input validation failed",
            extra={
                "event_type": "input_validation_failure",
                "validation_type": validation_type,
                "error_count": len(errors),
                "errors": errors,
                "security_flags": security_flags,
                "security_flag_count": len(security_flags),
                "input_preview": input_preview[:100] + "..." if input_preview and len(input_preview) > 100 else input_preview
            }
        )
    
    def log_rate_limit_exceeded(
        self,
        client_id: str,
        endpoint: str,
        limit: int,
        window_seconds: int,
        current_requests: int
    ):
        """Log rate limit violations."""
        self.logger.warning(
            "Rate limit exceeded",
            extra={
                "event_type": "rate_limit_exceeded",
                "client_id": client_id,
                "endpoint": endpoint,
                "limit": limit,
                "window_seconds": window_seconds,
                "current_requests": current_requests,
                "excess_requests": current_requests - limit
            }
        )


class PerformanceLogger:
    """Specialized logger for performance monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_slow_operation(
        self,
        operation_name: str,
        duration_ms: float,
        threshold_ms: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log slow operations."""
        self.logger.warning(
            f"Slow operation detected: {operation_name}",
            extra={
                "event_type": "slow_operation",
                "operation_name": operation_name,
                "duration_ms": duration_ms,
                "threshold_ms": threshold_ms,
                "slowness_factor": duration_ms / threshold_ms,
                "details": details or {}
            }
        )
    
    def log_resource_usage(
        self,
        resource_type: str,
        current_usage: float,
        limit: float,
        usage_percentage: float
    ):
        """Log high resource usage."""
        log_level = logging.WARNING if usage_percentage > 80 else logging.INFO
        
        self.logger.log(
            log_level,
            f"Resource usage: {resource_type} at {usage_percentage:.1f}%",
            extra={
                "event_type": "resource_usage",
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "usage_percentage": usage_percentage,
                "available": limit - current_usage
            }
        )


# Context management
@contextmanager
def logging_context(**context):
    """Context manager for adding context to all logs within the block."""
    old_context = getattr(_context_storage, 'context', {})
    _context_storage.context = {**old_context, **context}
    try:
        yield
    finally:
        _context_storage.context = old_context


def add_logging_context(**context):
    """Add context that will be included in all subsequent logs."""
    current_context = getattr(_context_storage, 'context', {})
    _context_storage.context = {**current_context, **context}


def clear_logging_context():
    """Clear all logging context."""
    _context_storage.context = {}


def get_logging_context() -> Dict[str, Any]:
    """Get current logging context."""
    return getattr(_context_storage, 'context', {}).copy()


# Decorators for automatic logging
def log_operation(
    operation_name: str = None,
    log_args: bool = False,
    log_result: bool = False,
    performance_threshold_ms: float = 1000.0
):
    """Decorator to automatically log operation execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = str(uuid.uuid4())[:8]
            
            logger = logging.getLogger(func.__module__)
            
            start_time = datetime.now()
            
            with logging_context(operation_id=operation_id):
                try:
                    # Log operation start
                    extra_data = {
                        "event_type": "operation_start",
                        "operation_name": op_name,
                        "operation_id": operation_id
                    }
                    
                    if log_args:
                        extra_data["args"] = str(args)[:500]  # Truncate for safety
                        extra_data["kwargs"] = {k: str(v)[:500] for k, v in kwargs.items()}
                    
                    logger.info(f"Starting operation: {op_name}", extra=extra_data)
                    
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    # Log operation completion
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    
                    extra_data = {
                        "event_type": "operation_complete",
                        "operation_name": op_name,
                        "operation_id": operation_id,
                        "duration_ms": duration,
                        "success": True
                    }
                    
                    if log_result:
                        extra_data["result"] = str(result)[:500]  # Truncate for safety
                    
                    log_level = logging.WARNING if duration > performance_threshold_ms else logging.INFO
                    logger.log(log_level, f"Operation completed: {op_name}", extra=extra_data)
                    
                    return result
                    
                except Exception as e:
                    # Log operation failure
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    
                    logger.error(
                        f"Operation failed: {op_name}",
                        extra={
                            "event_type": "operation_failed",
                            "operation_name": op_name,
                            "operation_id": operation_id,
                            "duration_ms": duration,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "success": False
                        },
                        exc_info=True
                    )
                    raise
        
        return wrapper
    return decorator


def configure_logging():
    """Configure structured logging for the entire application."""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.monitoring.log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with structured formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Configure specific loggers
    logging.getLogger('uvicorn.access').handlers = []
    
    logging.info(
        "Structured logging configured",
        extra={
            "log_level": settings.monitoring.log_level,
            "environment": settings.environment
        }
    )


# Pre-configured logger instances
sentiment_logger = SentimentAnalysisLogger()
healing_logger = HealingEngineLogger()
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()