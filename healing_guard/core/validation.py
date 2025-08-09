"""Comprehensive input validation and sanitization for the Healing Guard."""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error with detailed information."""
    
    def __init__(self, message: str, field: str = None, code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.field = field
        self.code = code
        self.details = details or {}


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    sanitized_data: Dict[str, Any]
    
    def add_error(self, field: str, message: str, code: str = None):
        """Add validation error."""
        self.is_valid = False
        self.errors.append({
            "field": field,
            "message": message,
            "code": code,
            "severity": ValidationSeverity.ERROR.value
        })
    
    def add_warning(self, field: str, message: str, code: str = None):
        """Add validation warning."""
        self.warnings.append({
            "field": field,
            "message": message,
            "code": code,
            "severity": ValidationSeverity.WARNING.value
        })


class BaseValidator:
    """Base validator class."""
    
    def __init__(self, required: bool = True, allow_none: bool = False):
        self.required = required
        self.allow_none = allow_none
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate value and return result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            sanitized_data={}
        )
        
        # Check if value is None
        if value is None:
            if self.required and not self.allow_none:
                result.add_error(field_name, "Field is required", "REQUIRED")
                return result
            elif self.allow_none:
                result.sanitized_data[field_name] = None
                return result
        
        # Perform specific validation
        try:
            sanitized_value = self._validate_value(value, field_name, result)
            result.sanitized_data[field_name] = sanitized_value
        except ValidationError as e:
            result.add_error(e.field or field_name, str(e), e.code)
        except Exception as e:
            result.add_error(field_name, f"Validation failed: {str(e)}", "VALIDATION_FAILED")
        
        return result
    
    def _validate_value(self, value: Any, field_name: str, result: ValidationResult) -> Any:
        """Override in subclasses for specific validation."""
        return value


class StringValidator(BaseValidator):
    """String validation with various rules."""
    
    def __init__(
        self,
        min_length: int = 0,
        max_length: int = None,
        pattern: str = None,
        allowed_chars: str = None,
        forbidden_chars: str = None,
        strip_whitespace: bool = True,
        normalize_unicode: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.forbidden_chars = set(forbidden_chars) if forbidden_chars else None
        self.strip_whitespace = strip_whitespace
        self.normalize_unicode = normalize_unicode
    
    def _validate_value(self, value: Any, field_name: str, result: ValidationResult) -> str:
        """Validate and sanitize string value."""
        # Convert to string
        if not isinstance(value, str):
            value = str(value)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            value = value.strip()
        
        # Normalize Unicode if requested
        if self.normalize_unicode:
            import unicodedata
            value = unicodedata.normalize('NFKC', value)
        
        # Check length constraints
        if len(value) < self.min_length:
            raise ValidationError(
                f"String too short (minimum {self.min_length} characters)",
                field_name,
                "MIN_LENGTH"
            )
        
        if self.max_length and len(value) > self.max_length:
            raise ValidationError(
                f"String too long (maximum {self.max_length} characters)",
                field_name,
                "MAX_LENGTH"
            )
        
        # Check pattern
        if self.pattern and not self.pattern.match(value):
            raise ValidationError(
                f"String does not match required pattern",
                field_name,
                "PATTERN_MISMATCH"
            )
        
        # Check allowed characters
        if self.allowed_chars:
            invalid_chars = set(value) - self.allowed_chars
            if invalid_chars:
                raise ValidationError(
                    f"Contains forbidden characters: {', '.join(sorted(invalid_chars))}",
                    field_name,
                    "FORBIDDEN_CHARS"
                )
        
        # Check forbidden characters
        if self.forbidden_chars:
            found_forbidden = set(value) & self.forbidden_chars
            if found_forbidden:
                raise ValidationError(
                    f"Contains forbidden characters: {', '.join(sorted(found_forbidden))}",
                    field_name,
                    "FORBIDDEN_CHARS"
                )
        
        return value


class NumberValidator(BaseValidator):
    """Numeric validation with range checks."""
    
    def __init__(
        self,
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
        numeric_type: type = float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.numeric_type = numeric_type
    
    def _validate_value(self, value: Any, field_name: str, result: ValidationResult) -> Union[int, float]:
        """Validate and convert numeric value."""
        # Convert to numeric type
        try:
            if self.numeric_type == int:
                if isinstance(value, float) and not value.is_integer():
                    raise ValidationError(
                        "Value must be an integer",
                        field_name,
                        "NOT_INTEGER"
                    )
                value = int(value)
            else:
                value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                f"Cannot convert to {self.numeric_type.__name__}",
                field_name,
                "INVALID_NUMBER"
            )
        
        # Check range
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Value too small (minimum {self.min_value})",
                field_name,
                "MIN_VALUE"
            )
        
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Value too large (maximum {self.max_value})",
                field_name,
                "MAX_VALUE"
            )
        
        return value


class ListValidator(BaseValidator):
    """List validation with element validation."""
    
    def __init__(
        self,
        min_items: int = 0,
        max_items: int = None,
        element_validator: BaseValidator = None,
        unique_items: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_items = min_items
        self.max_items = max_items
        self.element_validator = element_validator
        self.unique_items = unique_items
    
    def _validate_value(self, value: Any, field_name: str, result: ValidationResult) -> List[Any]:
        """Validate and sanitize list value."""
        # Ensure it's a list
        if not isinstance(value, list):
            try:
                value = list(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    "Value must be a list",
                    field_name,
                    "NOT_LIST"
                )
        
        # Check length constraints
        if len(value) < self.min_items:
            raise ValidationError(
                f"List too short (minimum {self.min_items} items)",
                field_name,
                "MIN_ITEMS"
            )
        
        if self.max_items and len(value) > self.max_items:
            raise ValidationError(
                f"List too long (maximum {self.max_items} items)",
                field_name,
                "MAX_ITEMS"
            )
        
        # Validate each element
        sanitized_items = []
        if self.element_validator:
            for i, item in enumerate(value):
                element_result = self.element_validator.validate(item, f"{field_name}[{i}]")
                if not element_result.is_valid:
                    # Propagate element errors
                    for error in element_result.errors:
                        result.errors.append(error)
                    result.is_valid = False
                else:
                    sanitized_items.append(element_result.sanitized_data.get(f"{field_name}[{i}]", item))
        else:
            sanitized_items = value
        
        # Check uniqueness
        if self.unique_items:
            seen = set()
            unique_items = []
            for item in sanitized_items:
                # Use string representation for hashability
                item_key = str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                if item_key not in seen:
                    seen.add(item_key)
                    unique_items.append(item)
                else:
                    result.add_warning(field_name, f"Duplicate item removed: {item}", "DUPLICATE_ITEM")
            sanitized_items = unique_items
        
        return sanitized_items


class DictValidator(BaseValidator):
    """Dictionary validation with field-specific validators."""
    
    def __init__(
        self,
        field_validators: Dict[str, BaseValidator] = None,
        allow_extra_fields: bool = True,
        required_fields: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.field_validators = field_validators or {}
        self.allow_extra_fields = allow_extra_fields
        self.required_fields = set(required_fields or [])
    
    def _validate_value(self, value: Any, field_name: str, result: ValidationResult) -> Dict[str, Any]:
        """Validate and sanitize dictionary value."""
        # Ensure it's a dictionary
        if not isinstance(value, dict):
            raise ValidationError(
                "Value must be a dictionary",
                field_name,
                "NOT_DICT"
            )
        
        sanitized_dict = {}
        
        # Check required fields
        missing_fields = self.required_fields - set(value.keys())
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(sorted(missing_fields))}",
                field_name,
                "MISSING_REQUIRED_FIELDS"
            )
        
        # Validate each field
        for key, val in value.items():
            if key in self.field_validators:
                field_result = self.field_validators[key].validate(val, key)
                if not field_result.is_valid:
                    # Propagate field errors
                    for error in field_result.errors:
                        result.errors.append(error)
                    result.is_valid = False
                else:
                    sanitized_dict[key] = field_result.sanitized_data.get(key, val)
            elif self.allow_extra_fields:
                sanitized_dict[key] = val
            else:
                result.add_warning(field_name, f"Extra field ignored: {key}", "EXTRA_FIELD")
        
        return sanitized_dict


class SecurityValidator:
    """Security-focused validation utilities."""
    
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """Sanitize input to prevent SQL injection."""
        if not isinstance(value, str):
            value = str(value)
        
        # Convert to lowercase for case-insensitive matching
        value_lower = value.lower()
        
        # Remove dangerous SQL keywords and characters
        dangerous_patterns = [
            "drop", "table", "delete", "update", "insert", "select",
            "union", "exec", "execute", "script", "'", '"', ";", 
            "--", "/*", "*/", "xp_", "sp_", "<script", "</script"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in value_lower:
                value = value.replace(pattern, "")
                value = value.replace(pattern.upper(), "")
                value = value.replace(pattern.capitalize(), "")
        
        return value.strip()
    
    @staticmethod
    def sanitize_shell_input(value: str) -> str:
        """Sanitize input to prevent shell injection."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove dangerous shell characters and commands
        dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "{", "}", "[", "]", "<", ">", "\\"]
        dangerous_commands = ["rm", "del", "format", "shutdown", "reboot", "kill", "chmod", "chown"]
        
        for char in dangerous_chars:
            value = value.replace(char, "")
        
        # Remove dangerous commands (case-insensitive)
        value_lower = value.lower()
        for cmd in dangerous_commands:
            if cmd in value_lower:
                value = value.replace(cmd, "")
                value = value.replace(cmd.upper(), "")
                value = value.replace(cmd.capitalize(), "")
        
        return value.strip()
    
    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate and sanitize file paths."""
        if not isinstance(path, str):
            raise ValidationError("Path must be a string", code="INVALID_PATH")
        
        # Remove path traversal attempts
        if ".." in path or path.startswith("/"):
            raise ValidationError("Path traversal not allowed", code="PATH_TRAVERSAL")
        
        # Check for dangerous characters
        dangerous_chars = ["|", "&", ";", "$", "`", "<", ">"]
        for char in dangerous_chars:
            if char in path:
                raise ValidationError(f"Dangerous character in path: {char}", code="DANGEROUS_CHAR")
        
        return path.strip()
    
    @staticmethod
    def validate_ci_platform(platform: str) -> str:
        """Validate CI platform names."""
        allowed_platforms = [
            "github", "gitlab", "jenkins", "circleci", "azure", "buildkite",
            "travis", "bitbucket", "drone", "teamcity", "bamboo", "generic"
        ]
        
        platform = platform.lower().strip()
        if platform not in allowed_platforms:
            raise ValidationError(
                f"Unsupported CI platform: {platform}",
                code="UNSUPPORTED_PLATFORM"
            )
        
        return platform


class FailureAnalysisValidator:
    """Specialized validators for failure analysis data."""
    
    @staticmethod
    def validate_job_id(job_id: str) -> str:
        """Validate CI job ID format."""
        validator = StringValidator(
            min_length=1,
            max_length=100,
            pattern=r'^[a-zA-Z0-9_\-\.]+$',
            strip_whitespace=True
        )
        result = validator.validate(job_id, "job_id")
        if not result.is_valid:
            raise ValidationError(
                result.errors[0]["message"],
                "job_id",
                result.errors[0]["code"]
            )
        return result.sanitized_data["job_id"]
    
    @staticmethod
    def validate_repository_name(repo: str) -> str:
        """Validate repository name format."""
        validator = StringValidator(
            min_length=1,
            max_length=200,
            pattern=r'^[a-zA-Z0-9_\-\.\/]+$',
            strip_whitespace=True
        )
        result = validator.validate(repo, "repository")
        if not result.is_valid:
            raise ValidationError(
                result.errors[0]["message"],
                "repository", 
                result.errors[0]["code"]
            )
        return result.sanitized_data["repository"]
    
    @staticmethod
    def validate_commit_sha(sha: str) -> str:
        """Validate Git commit SHA format."""
        validator = StringValidator(
            min_length=7,
            max_length=40,
            pattern=r'^[a-f0-9]+$',
            strip_whitespace=True
        )
        result = validator.validate(sha.lower(), "commit_sha")
        if not result.is_valid:
            raise ValidationError(
                result.errors[0]["message"],
                "commit_sha",
                result.errors[0]["code"]
            )
        return result.sanitized_data["commit_sha"]
    
    @staticmethod
    def validate_logs(logs: str) -> str:
        """Validate and sanitize log content."""
        if not isinstance(logs, str):
            logs = str(logs)
        
        # Check length
        if len(logs) > 1_000_000:  # 1MB limit
            raise ValidationError(
                "Log content too large (max 1MB)",
                "logs",
                "LOGS_TOO_LARGE"
            )
        
        # Remove potentially dangerous content
        logs = SecurityValidator.sanitize_shell_input(logs)
        
        return logs.strip()


class TaskValidation:
    """Specialized validators for quantum planner tasks."""
    
    @staticmethod
    def validate_task_creation_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task creation request."""
        schema = DictValidator(
            field_validators={
                "name": StringValidator(min_length=1, max_length=200),
                "priority": NumberValidator(min_value=1, max_value=4, numeric_type=int),
                "estimated_duration": NumberValidator(min_value=0.1, numeric_type=float),
                "dependencies": ListValidator(
                    element_validator=StringValidator(min_length=1, max_length=100),
                    unique_items=True
                ),
                "resources_required": DictValidator(
                    field_validators={
                        "cpu": NumberValidator(min_value=0.1, numeric_type=float),
                        "memory": NumberValidator(min_value=0.1, numeric_type=float)
                    },
                    allow_extra_fields=True
                ),
                "failure_probability": NumberValidator(min_value=0.0, max_value=1.0, numeric_type=float),
                "max_retries": NumberValidator(min_value=0, max_value=10, numeric_type=int),
                "metadata": DictValidator(allow_extra_fields=True)
            },
            required_fields=["name", "priority", "estimated_duration"]
        )
        
        result = schema.validate(data, "task")
        if not result.is_valid:
            error_messages = [error["message"] for error in result.errors]
            raise ValidationError(
                f"Task validation failed: {'; '.join(error_messages)}",
                "task",
                "TASK_VALIDATION_FAILED"
            )
        
        return result.sanitized_data["task"]


def validate_and_sanitize(data: Dict[str, Any], validators: Dict[str, BaseValidator]) -> Dict[str, Any]:
    """Validate and sanitize data using provided validators."""
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        sanitized_data={}
    )
    
    for field, validator in validators.items():
        if field in data:
            field_result = validator.validate(data[field], field)
            
            # Merge results
            result.errors.extend(field_result.errors)
            result.warnings.extend(field_result.warnings)
            result.sanitized_data.update(field_result.sanitized_data)
            
            if not field_result.is_valid:
                result.is_valid = False
        elif validator.required:
            result.add_error(field, "Required field missing", "REQUIRED")
    
    if not result.is_valid:
        error_messages = [error["message"] for error in result.errors]
        raise ValidationError(
            f"Validation failed: {'; '.join(error_messages)}",
            details={"errors": result.errors, "warnings": result.warnings}
        )
    
    return result.sanitized_data