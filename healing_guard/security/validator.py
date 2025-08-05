"""Input validation and security validation utilities."""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

import bleach
from pydantic import BaseModel, ValidationError, validator

logger = logging.getLogger(__name__)


class ValidationRule(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    LENGTH = "length"
    PATTERN = "pattern"
    RANGE = "range"
    CUSTOM = "custom"
    SANITIZE = "sanitize"
    SECURITY = "security"


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "sanitized_value": self.sanitized_value
        }


class InputValidator:
    """Comprehensive input validation with security focus."""
    
    def __init__(self):
        self.security_patterns = {
            "sql_injection": [
                r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)",
                r"(?i)(OR|AND)\s+\d+\s*=\s*\d+",
                r"['\"];?\s*(OR|AND|SELECT|INSERT|UPDATE|DELETE)",
                r"(?i)(\bor\b|\band\b)\s+['\"]?[^'\"]*['\"]?\s*=\s*['\"]?[^'\"]*['\"]?"
            ],
            "xss": [
                r"<\s*script[^>]*>",
                r"javascript\s*:",
                r"on\w+\s*=",
                r"<\s*iframe[^>]*>",
                r"<\s*object[^>]*>",
                r"<\s*embed[^>]*>"
            ],
            "command_injection": [
                r"[;&|`$(){}[\]\\]",
                r"(?i)(exec|eval|system|shell_exec|passthru)",
                r"\|\s*(cat|ls|pwd|whoami|id|ps|netstat)",
                r"['\"]?\s*;\s*['\"]?"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
                r"~",
                r"/etc/passwd",
                r"/proc/",
                r"\\windows\\",
                r"\\system32\\"
            ],
            "ldap_injection": [
                r"[()&|!]",
                r"\*",
                r"(?i)(objectclass|cn|uid|ou|dc)="
            ]
        }
        
        self.common_rules = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "url": r"^https?://[^\s/$.?#].[^\s]*$",
            "ipv4": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            "alphanumeric": r"^[a-zA-Z0-9]+$",
            "safe_filename": r"^[a-zA-Z0-9._-]+$",
            "version": r"^\d+\.\d+\.\d+([.-][a-zA-Z0-9]+)*$"
        }
    
    def validate_string(
        self,
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_empty: bool = False,
        sanitize: bool = False
    ) -> ValidationResult:
        """Validate string input with security checks."""
        errors = []
        warnings = []
        sanitized_value = value
        
        # Basic checks
        if not isinstance(value, str):
            errors.append("Value must be a string")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        if not allow_empty and not value.strip():
            errors.append("Value cannot be empty")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Length checks
        if min_length is not None and len(value) < min_length:
            errors.append(f"Value must be at least {min_length} characters long")
        
        if max_length is not None and len(value) > max_length:
            errors.append(f"Value must be at most {max_length} characters long")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            errors.append("Value does not match required pattern")
        
        # Security checks
        security_issues = self._check_security_patterns(value)
        if security_issues:
            for issue in security_issues:
                errors.append(f"Security issue detected: {issue}")
        
        # Sanitization
        if sanitize:
            sanitized_value = self._sanitize_string(value)
            if sanitized_value != value:
                warnings.append("Input was sanitized for security")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_value
        )
    
    def validate_integer(
        self,
        value: Union[int, str],
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """Validate integer input."""
        errors = []
        warnings = []
        
        # Type conversion
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append("Value must be a valid integer")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Range checks
        if min_value is not None and int_value < min_value:
            errors.append(f"Value must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            errors.append(f"Value must be at most {max_value}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=int_value
        )
    
    def validate_email(self, email: str) -> ValidationResult:
        """Validate email address."""
        return self.validate_string(
            email,
            min_length=5,
            max_length=255,
            pattern=self.common_rules["email"],
            sanitize=True
        )
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL."""
        result = self.validate_string(
            url,
            min_length=7,
            max_length=2048,
            pattern=self.common_rules["url"]
        )
        
        # Additional URL security checks
        if result.valid:
            dangerous_schemes = ["javascript:", "data:", "vbscript:", "file:"]
            lower_url = url.lower()
            
            for scheme in dangerous_schemes:
                if lower_url.startswith(scheme):
                    result.errors.append(f"Dangerous URL scheme detected: {scheme}")
                    result.valid = False
                    break
        
        return result
    
    def validate_filename(self, filename: str) -> ValidationResult:
        """Validate filename for security."""
        result = self.validate_string(
            filename,
            min_length=1,
            max_length=255,
            pattern=self.common_rules["safe_filename"]
        )
        
        # Additional filename security checks
        if result.valid:
            dangerous_names = [
                "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4",
                "lpt1", "lpt2", "lpt3", "lpt4", "..", "."
            ]
            
            if filename.lower() in dangerous_names:
                result.errors.append("Dangerous filename detected")
                result.valid = False
        
        return result
    
    def validate_json(self, json_str: str, max_size: int = 1024 * 1024) -> ValidationResult:
        """Validate JSON string."""
        errors = []
        warnings = []
        
        # Size check
        if len(json_str) > max_size:
            errors.append(f"JSON size exceeds maximum allowed size of {max_size} bytes")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Parse JSON
        try:
            import json
            parsed_json = json.loads(json_str)
            
            # Check for dangerous patterns in JSON
            json_str_lower = json_str.lower()
            if any(pattern in json_str_lower for pattern in ["<script", "javascript:", "eval("]):
                warnings.append("Potentially dangerous content detected in JSON")
            
            return ValidationResult(
                valid=True,
                errors=errors,
                warnings=warnings,
                sanitized_value=parsed_json
            )
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
    
    def _check_security_patterns(self, value: str) -> List[str]:
        """Check for security patterns in input."""
        issues = []
        value_lower = value.lower()
        
        for attack_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value_lower):
                    issues.append(f"{attack_type.replace('_', ' ').title()} pattern detected")
                    break  # Only report each attack type once
        
        return issues
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        # Remove potential XSS content
        sanitized = bleach.clean(
            value,
            tags=[],  # Remove all HTML tags
            attributes={},  # Remove all attributes
            strip=True
        )
        
        # Remove potential SQL injection characters
        sanitized = re.sub(r"['\";\\]", "", sanitized)
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        
        return sanitized.strip()
    
    def validate_dict(
        self,
        data: dict,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable]] = None,
        max_fields: int = 100
    ) -> ValidationResult:
        """Validate dictionary data structure."""
        errors = []
        warnings = []
        sanitized_value = {}
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Check field count
        if len(data) > max_fields:
            errors.append(f"Too many fields. Maximum allowed: {max_fields}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Check required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate individual fields
        if field_validators:
            for field_name, validator_func in field_validators.items():
                if field_name in data:
                    try:
                        field_result = validator_func(data[field_name])
                        if hasattr(field_result, 'valid') and not field_result.valid:
                            errors.extend([f"{field_name}: {error}" for error in field_result.errors])
                        else:
                            sanitized_value[field_name] = getattr(field_result, 'sanitized_value', data[field_name])
                    except Exception as e:
                        errors.append(f"{field_name}: Validation error - {str(e)}")
                else:
                    sanitized_value[field_name] = data[field_name]
        else:
            sanitized_value = data.copy()
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_value
        )


class SecurityValidator:
    """Security-focused validation for API inputs and data."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: set = set()
        
    def validate_api_input(self, data: Dict[str, Any], endpoint: str) -> ValidationResult:
        """Validate API input with endpoint-specific rules."""
        errors = []
        warnings = []
        sanitized_data = {}
        
        # General API input validation rules
        api_rules = {
            # Task creation endpoint
            "/api/v1/tasks": {
                "required_fields": ["name", "priority", "estimated_duration"],
                "field_validators": {
                    "name": lambda x: self.input_validator.validate_string(x, min_length=1, max_length=200, sanitize=True),
                    "priority": lambda x: self.input_validator.validate_integer(x, min_value=1, max_value=4),
                    "estimated_duration": lambda x: self.input_validator.validate_integer(x, min_value=1, max_value=10080)
                }
            },
            # Failure analysis endpoint
            "/api/v1/failures/analyze": {
                "required_fields": ["job_id", "repository", "branch", "commit_sha", "logs"],
                "field_validators": {
                    "job_id": lambda x: self.input_validator.validate_string(x, min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$"),
                    "repository": lambda x: self.input_validator.validate_string(x, min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9._/-]+$"),
                    "branch": lambda x: self.input_validator.validate_string(x, min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._/-]+$"),
                    "commit_sha": lambda x: self.input_validator.validate_string(x, min_length=7, max_length=40, pattern=r"^[a-fA-F0-9]+$"),
                    "logs": lambda x: self.input_validator.validate_string(x, min_length=1, max_length=1024*1024, sanitize=True)  # 1MB max
                }
            }
        }
        
        # Apply endpoint-specific validation
        if endpoint in api_rules:
            rule = api_rules[endpoint]
            
            result = self.input_validator.validate_dict(
                data,
                required_fields=rule.get("required_fields"),
                field_validators=rule.get("field_validators"),
                max_fields=50
            )
            
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            if result.sanitized_value:
                sanitized_data.update(result.sanitized_value)
        
        # Generic security checks
        self._check_input_size(data, errors)
        self._check_nested_depth(data, errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_data if sanitized_data else data
        )
    
    def _check_input_size(self, data: Any, errors: List[str], max_size: int = 10 * 1024 * 1024):
        """Check total input size."""
        try:
            import sys
            size = sys.getsizeof(str(data))
            if size > max_size:
                errors.append(f"Input size ({size} bytes) exceeds maximum allowed size ({max_size} bytes)")
        except Exception:
            pass  # Size check failed, but don't fail validation for this
    
    def _check_nested_depth(self, data: Any, errors: List[str], max_depth: int = 10, current_depth: int = 0):
        """Check nesting depth to prevent stack overflow attacks."""
        if current_depth > max_depth:
            errors.append(f"Input nesting depth ({current_depth}) exceeds maximum allowed depth ({max_depth})")
            return
        
        if isinstance(data, dict):
            for value in data.values():
                self._check_nested_depth(value, errors, max_depth, current_depth + 1)
        elif isinstance(data, list):
            for item in data:
                self._check_nested_depth(item, errors, max_depth, current_depth + 1)
    
    def check_rate_limit(self, client_id: str, endpoint: str, limit: int = 100, window: int = 60) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = datetime.now().timestamp()
        key = f"{client_id}:{endpoint}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "requests": [],
                "blocked_until": 0
            }
        
        client_data = self.rate_limits[key]
        
        # Check if client is currently blocked
        if current_time < client_data["blocked_until"]:
            return False
        
        # Clean old requests
        client_data["requests"] = [
            req_time for req_time in client_data["requests"]
            if current_time - req_time < window
        ]
        
        # Check limit
        if len(client_data["requests"]) >= limit:
            # Block client for the window duration
            client_data["blocked_until"] = current_time + window
            logger.warning(f"Rate limit exceeded for client {client_id} on endpoint {endpoint}")
            return False
        
        # Record request
        client_data["requests"].append(current_time)
        return True
    
    def block_ip(self, ip_address: str, duration: int = 3600):
        """Block an IP address for a specified duration."""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address}")
        
        # Schedule unblock (in a real implementation, this would use a task scheduler)
        # For now, we'll rely on periodic cleanup
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def validate_file_upload(
        self,
        filename: str,
        content: bytes,
        allowed_extensions: Optional[List[str]] = None,
        max_size: int = 10 * 1024 * 1024  # 10MB
    ) -> ValidationResult:
        """Validate file upload for security."""
        errors = []
        warnings = []
        
        # Validate filename
        filename_result = self.input_validator.validate_filename(filename)
        if not filename_result.valid:
            errors.extend(filename_result.errors)
        
        # Check file extension
        if allowed_extensions:
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if file_ext not in [ext.lower() for ext in allowed_extensions]:
                errors.append(f"File extension '.{file_ext}' not allowed")
        
        # Check file size
        if len(content) > max_size:
            errors.append(f"File size ({len(content)} bytes) exceeds maximum allowed size ({max_size} bytes)")
        
        # Check for embedded executables (basic check)
        if self._has_executable_content(content):
            errors.append("File contains potentially dangerous executable content")
        
        # Check file magic numbers
        if not self._validate_file_magic(content, filename):
            warnings.append("File content may not match file extension")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _has_executable_content(self, content: bytes) -> bool:
        """Check for executable content in file."""
        # Check for common executable signatures
        executable_signatures = [
            b'\x4d\x5a',  # Windows PE
            b'\x7f\x45\x4c\x46',  # ELF (Linux)
            b'\xfe\xed\xfa\xce',  # Mach-O (macOS)
            b'\xfe\xed\xfa\xcf',  # Mach-O (macOS)
            b'PK\x03\x04',  # ZIP (could contain executables)
        ]
        
        for signature in executable_signatures:
            if content.startswith(signature):
                return True
        
        return False
    
    def _validate_file_magic(self, content: bytes, filename: str) -> bool:
        """Validate file magic number matches extension."""
        if not content:
            return True
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Basic magic number validation
        magic_mappings = {
            'jpg': [b'\xff\xd8\xff'],
            'jpeg': [b'\xff\xd8\xff'],
            'png': [b'\x89\x50\x4e\x47'],
            'gif': [b'\x47\x49\x46\x38'],
            'pdf': [b'\x25\x50\x44\x46'],
            'txt': [],  # Text files don't have specific magic numbers
            'json': [],
            'yaml': [],
            'yml': []
        }
        
        if file_ext in magic_mappings:
            expected_magics = magic_mappings[file_ext]
            if expected_magics:  # Only check if we have expected magic numbers
                return any(content.startswith(magic) for magic in expected_magics)
        
        return True  # Don't fail validation if we don't know the magic number


# Global validator instances
input_validator = InputValidator()
security_validator = SecurityValidator()