"""Input validation and sanitization for sentiment analysis and healing operations."""

import re
import html
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from urllib.parse import urlparse
from dataclasses import dataclass

from ..core.exceptions import (
    SentimentValidationException,
    SecurityException,
    HealingGuardException
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Any
    errors: List[str]
    warnings: List[str]
    security_flags: List[str]
    
    def has_security_issues(self) -> bool:
        """Check if validation found security issues."""
        return len(self.security_flags) > 0


class InputValidator:
    """Comprehensive input validator with security checks."""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for validation."""
        # Security patterns
        self.sql_injection_pattern = re.compile(
            r'(union\s+select|insert\s+into|delete\s+from|drop\s+table|'
            r'exec\s*\(|script|javascript:|vbscript:|onload=|onerror=)',
            re.IGNORECASE
        )
        
        self.xss_pattern = re.compile(
            r'(<script[^>]*>.*?</script>|<iframe[^>]*>.*?</iframe>|'
            r'javascript:|vbscript:|onload=|onerror=|alert\(|eval\()',
            re.IGNORECASE | re.DOTALL
        )
        
        self.command_injection_pattern = re.compile(
            r'(;|\||&|\$\(|`|exec|system|eval|rm\s+-rf|wget|curl\s)',
            re.IGNORECASE
        )
        
        # Content patterns
        self.excessive_whitespace = re.compile(r'\s{10,}')
        self.excessive_repeating_chars = re.compile(r'(.)\1{50,}')
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Sentiment-specific patterns
        self.log_metadata_pattern = re.compile(r'\[(ERROR|WARN|INFO|DEBUG)\]|\d{4}-\d{2}-\d{2}')
        self.stack_trace_pattern = re.compile(r'at\s+[\w.]+\([^)]+\)|Traceback|Exception in thread')
    
    def validate_sentiment_text(
        self, 
        text: str, 
        max_length: int = 10000,
        allow_html: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate text input for sentiment analysis."""
        errors = []
        warnings = []
        security_flags = []
        
        # Basic validation
        if not isinstance(text, str):
            errors.append("Input must be a string")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        original_text = text
        
        # Length validation
        if len(text) > max_length:
            errors.append(f"Text exceeds maximum length of {max_length} characters")
            text = text[:max_length]
            warnings.append(f"Text truncated to {max_length} characters")
        
        if len(text.strip()) == 0:
            errors.append("Text cannot be empty or whitespace only")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        # Security checks
        security_result = self._check_security_patterns(text)
        security_flags.extend(security_result['flags'])
        
        if security_result['high_risk']:
            errors.append("Input contains potentially malicious content")
            logger.warning(f"High-risk input detected: {security_result['flags']}")
        
        # Content sanitization
        sanitized_text = self._sanitize_text(text, allow_html)
        
        # Content quality checks
        quality_result = self._check_content_quality(sanitized_text)
        warnings.extend(quality_result['warnings'])
        
        # Context-specific validation
        if context:
            context_result = self._validate_context(sanitized_text, context)
            warnings.extend(context_result['warnings'])
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_text if is_valid else None,
            errors=errors,
            warnings=warnings,
            security_flags=security_flags
        )
    
    def validate_context_dict(
        self, 
        context: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        max_depth: int = 5
    ) -> ValidationResult:
        """Validate context dictionary for sentiment analysis."""
        errors = []
        warnings = []
        security_flags = []
        
        if not isinstance(context, dict):
            errors.append("Context must be a dictionary")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        # Check depth to prevent deeply nested attacks
        if self._get_dict_depth(context) > max_depth:
            errors.append(f"Context dictionary exceeds maximum depth of {max_depth}")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        # Validate required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in context]
            if missing_fields:
                errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate and sanitize context values
        sanitized_context = {}
        for key, value in context.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                warnings.append(f"Invalid context key: {key}")
                continue
            
            # Validate and sanitize value
            if isinstance(value, str):
                if len(value) > 1000:
                    warnings.append(f"Context value for '{key}' truncated")
                    value = value[:1000]
                
                # Check for security issues in string values
                security_result = self._check_security_patterns(value)
                security_flags.extend(security_result['flags'])
                value = self._sanitize_text(value, allow_html=False)
            
            elif isinstance(value, (int, float, bool)):
                # Numeric and boolean values are safe
                pass
            elif isinstance(value, (list, dict)):
                # Recursively validate nested structures (with depth limit)
                if self._get_dict_depth({key: value}) > max_depth - 1:
                    warnings.append(f"Nested structure in '{key}' too deep, skipping")
                    continue
            else:
                warnings.append(f"Unsupported value type for '{key}': {type(value)}")
                continue
            
            sanitized_context[key] = value
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_context if is_valid else None,
            errors=errors,
            warnings=warnings,
            security_flags=security_flags
        )
    
    def validate_batch_texts(
        self, 
        texts: List[str],
        max_batch_size: int = 100,
        max_text_length: int = 10000
    ) -> ValidationResult:
        """Validate batch of texts for sentiment analysis."""
        errors = []
        warnings = []
        security_flags = []
        
        if not isinstance(texts, list):
            errors.append("Texts must be a list")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        if len(texts) == 0:
            errors.append("Texts list cannot be empty")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        if len(texts) > max_batch_size:
            errors.append(f"Batch size exceeds maximum of {max_batch_size}")
            return ValidationResult(False, None, errors, warnings, security_flags)
        
        sanitized_texts = []
        for i, text in enumerate(texts):
            text_result = self.validate_sentiment_text(
                text, 
                max_length=max_text_length,
                context={'batch_index': i}
            )
            
            if not text_result.is_valid:
                errors.extend([f"Text {i}: {error}" for error in text_result.errors])
            else:
                sanitized_texts.append(text_result.sanitized_input)
            
            warnings.extend([f"Text {i}: {warning}" for warning in text_result.warnings])
            security_flags.extend(text_result.security_flags)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_texts if is_valid else None,
            errors=errors,
            warnings=warnings,
            security_flags=security_flags
        )
    
    def validate_pipeline_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate pipeline event data."""
        errors = []
        warnings = []
        security_flags = []
        
        # Validate event type
        allowed_event_types = {
            'pipeline_failure', 'test_failure', 'build_error', 'deployment_error',
            'commit_message', 'pr_comment', 'issue_comment', 'code_review',
            'merge_request', 'release_notes', 'incident_report'
        }
        
        if event_type not in allowed_event_types:
            errors.append(f"Invalid event_type. Must be one of: {', '.join(allowed_event_types)}")
        
        # Validate message
        message_result = self.validate_sentiment_text(message, max_length=10000)
        errors.extend(message_result.errors)
        warnings.extend(message_result.warnings)
        security_flags.extend(message_result.security_flags)
        
        # Validate metadata
        sanitized_metadata = None
        if metadata:
            metadata_result = self.validate_context_dict(metadata)
            errors.extend(metadata_result.errors)
            warnings.extend(metadata_result.warnings)
            security_flags.extend(metadata_result.security_flags)
            sanitized_metadata = metadata_result.sanitized_input
        
        is_valid = len(errors) == 0
        sanitized_input = {
            'event_type': event_type,
            'message': message_result.sanitized_input,
            'metadata': sanitized_metadata
        } if is_valid else None
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input,
            errors=errors,
            warnings=warnings,
            security_flags=security_flags
        )
    
    def _check_security_patterns(self, text: str) -> Dict[str, Any]:
        """Check text against security patterns."""
        flags = []
        high_risk = False
        
        # SQL injection check
        if self.sql_injection_pattern.search(text):
            flags.append("potential_sql_injection")
            high_risk = True
        
        # XSS check
        if self.xss_pattern.search(text):
            flags.append("potential_xss")
            high_risk = True
        
        # Command injection check
        if self.command_injection_pattern.search(text):
            flags.append("potential_command_injection")
            high_risk = True
        
        # Check for suspicious URLs
        urls = self.url_pattern.findall(text)
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.hostname and any(
                    suspicious in parsed.hostname.lower() 
                    for suspicious in ['localhost', '127.0.0.1', '0.0.0.0']
                ):
                    flags.append("suspicious_url")
            except Exception:
                flags.append("malformed_url")
        
        return {
            'flags': flags,
            'high_risk': high_risk,
            'url_count': len(urls)
        }
    
    def _sanitize_text(self, text: str, allow_html: bool = False) -> str:
        """Sanitize text input."""
        # HTML escape if HTML not allowed
        if not allow_html:
            text = html.escape(text)
        
        # Remove excessive whitespace
        text = self.excessive_whitespace.sub(' ', text)
        
        # Limit excessive repeating characters (but preserve some for sentiment analysis)
        text = self.excessive_repeating_chars.sub(r'\1\1\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _check_content_quality(self, text: str) -> Dict[str, Any]:
        """Check content quality and provide warnings."""
        warnings = []
        
        # Check for very short text
        if len(text.strip()) < 3:
            warnings.append("Text is very short, sentiment analysis may be unreliable")
        
        # Check for very repetitive content
        words = text.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            warnings.append("Text appears highly repetitive")
        
        # Check for log-like content
        if self.log_metadata_pattern.search(text):
            warnings.append("Text appears to contain log metadata")
        
        # Check for stack traces
        if self.stack_trace_pattern.search(text):
            warnings.append("Text contains stack trace information")
        
        # Check for excessive punctuation
        punct_count = len(re.findall(r'[!?]{3,}', text))
        if punct_count > 5:
            warnings.append("Text contains excessive punctuation")
        
        return {'warnings': warnings}
    
    def _validate_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform context-specific validation."""
        warnings = []
        
        # Check consistency between text and context
        event_type = context.get('event_type')
        if event_type == 'pipeline_failure':
            # Should contain failure-related keywords for failures
            failure_keywords = {'failed', 'error', 'broken', 'crashed', 'timeout'}
            text_lower = text.lower()
            if not any(keyword in text_lower for keyword in failure_keywords):
                warnings.append("Text doesn't seem to match pipeline_failure event type")
        
        elif event_type in ['commit_message', 'pr_comment']:
            # Should be reasonably structured for commit messages
            if len(text) < 10:
                warnings.append("Text seems too short for a meaningful commit message")
        
        # Check for environment context consistency
        is_production = context.get('environment') == 'production'
        if is_production and 'test' in text.lower():
            warnings.append("Production event contains test-related content")
        
        return {'warnings': warnings}
    
    def _get_dict_depth(self, d: Dict) -> int:
        """Calculate the maximum depth of nested dictionaries."""
        if not isinstance(d, dict) or not d:
            return 0
        return 1 + max(self._get_dict_depth(v) if isinstance(v, dict) else 0 for v in d.values())


# Global validator instance
input_validator = InputValidator()


# Validation decorators
def validate_sentiment_input(max_length: int = 10000, allow_html: bool = False):
    """Decorator to validate sentiment analysis input."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Find text argument
            text = None
            if args and isinstance(args[0], str):
                text = args[0]
            elif 'text' in kwargs:
                text = kwargs['text']
            
            if text is not None:
                validation_result = input_validator.validate_sentiment_text(
                    text, max_length, allow_html
                )
                
                if not validation_result.is_valid:
                    raise SentimentValidationException(
                        "Input validation failed",
                        validation_errors=validation_result.errors,
                        details={
                            'warnings': validation_result.warnings,
                            'security_flags': validation_result.security_flags
                        }
                    )
                
                if validation_result.has_security_issues():
                    logger.warning(f"Security flags detected: {validation_result.security_flags}")
                
                # Replace original text with sanitized version
                if args and isinstance(args[0], str):
                    args = (validation_result.sanitized_input,) + args[1:]
                elif 'text' in kwargs:
                    kwargs['text'] = validation_result.sanitized_input
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_context_input(required_fields: Optional[List[str]] = None):
    """Decorator to validate context input."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Find context argument
            context = kwargs.get('context')
            
            if context is not None:
                validation_result = input_validator.validate_context_dict(
                    context, required_fields
                )
                
                if not validation_result.is_valid:
                    raise SentimentValidationException(
                        "Context validation failed",
                        validation_errors=validation_result.errors,
                        details={
                            'warnings': validation_result.warnings,
                            'security_flags': validation_result.security_flags
                        }
                    )
                
                kwargs['context'] = validation_result.sanitized_input
            
            return func(*args, **kwargs)
        return wrapper
    return decorator