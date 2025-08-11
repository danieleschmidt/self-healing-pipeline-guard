"""Security module for the Healing Guard system."""

try:
    from .scanner import SecurityScanner, VulnerabilityScanner
    from .validator import InputValidator, SecurityValidator
    from .audit import SecurityAuditor, AuditLogger
    from .advanced_validator import AdvancedSecurityValidator, security_validator
    
    __all__ = [
        "SecurityScanner",
        "VulnerabilityScanner", 
        "InputValidator",
        "SecurityValidator",
        "SecurityAuditor",
        "AuditLogger",
        "AdvancedSecurityValidator",
        "security_validator"
    ]
except ImportError as e:
    # Fallback when optional dependencies are missing
    from .advanced_validator import AdvancedSecurityValidator, security_validator
    
    __all__ = [
        "AdvancedSecurityValidator", 
        "security_validator"
    ]