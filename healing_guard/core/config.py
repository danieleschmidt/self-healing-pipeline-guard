"""Configuration management for the Self-Healing Pipeline Guard.

Handles settings, environment variables, and configuration validation
with security best practices and multi-environment support.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/healing_guard")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"

@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "10"))
    timeout: int = int(os.getenv("REDIS_TIMEOUT", "30"))
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))

@dataclass
class QuantumPlannerConfig:
    """Quantum planner configuration."""
    max_parallel_tasks: int = int(os.getenv("QUANTUM_MAX_PARALLEL_TASKS", "4"))
    optimization_iterations: int = int(os.getenv("QUANTUM_OPTIMIZATION_ITERATIONS", "1000"))
    temperature_schedule: str = os.getenv("QUANTUM_TEMP_SCHEDULE", "exponential")
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        "cpu": float(os.getenv("QUANTUM_CPU_LIMIT", "8.0")),
        "memory": float(os.getenv("QUANTUM_MEMORY_LIMIT", "16.0"))
    })

@dataclass
class HealingEngineConfig:
    """Healing engine configuration."""
    max_concurrent_healings: int = int(os.getenv("HEALING_MAX_CONCURRENT", "3"))
    healing_timeout_minutes: int = int(os.getenv("HEALING_TIMEOUT", "30"))
    max_retry_attempts: int = int(os.getenv("HEALING_MAX_RETRIES", "3"))
    default_resource_multiplier: float = float(os.getenv("HEALING_RESOURCE_MULTIPLIER", "2.0"))

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    cors_origins: List[str] = field(default_factory=lambda: 
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    )

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    jaeger_endpoint: Optional[str] = os.getenv("JAEGER_ENDPOINT")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))

@dataclass
class IntegrationConfig:
    """External integration configuration."""
    github_app_id: Optional[str] = os.getenv("GITHUB_APP_ID")
    github_private_key_path: Optional[str] = os.getenv("GITHUB_PRIVATE_KEY_PATH")
    gitlab_token: Optional[str] = os.getenv("GITLAB_TOKEN")
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    jira_url: Optional[str] = os.getenv("JIRA_URL")
    jira_token: Optional[str] = os.getenv("JIRA_TOKEN")

@dataclass
class MLConfig:
    """Machine learning configuration."""
    model_path: str = os.getenv("ML_MODEL_PATH", "/app/models/failure_classifier.pkl")
    confidence_threshold: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.75"))
    retrain_interval_hours: int = int(os.getenv("ML_RETRAIN_INTERVAL", "24"))
    feature_engineering_enabled: bool = os.getenv("ML_FEATURE_ENGINEERING", "true").lower() == "true"
    model_type: str = os.getenv("ML_MODEL_TYPE", "gradient_boost")

@dataclass
class NotificationConfig:
    """Notification configuration."""
    slack_webhook: Optional[str] = os.getenv("SLACK_WEBHOOK")
    slack_channel: str = os.getenv("SLACK_CHANNEL", "#ci-alerts")
    email_enabled: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    smtp_host: Optional[str] = os.getenv("SMTP_HOST")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    pagerduty_token: Optional[str] = os.getenv("PAGERDUTY_TOKEN")

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    caching_enabled: bool = os.getenv("CACHING_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "3600"))
    batch_processing_enabled: bool = os.getenv("BATCH_PROCESSING", "true").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    async_workers: int = int(os.getenv("ASYNC_WORKERS", "4"))
    connection_pool_size: int = int(os.getenv("CONNECTION_POOL_SIZE", "20"))

class Settings:
    """Main settings class for the Self-Healing Pipeline Guard."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # Initialize configurations
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.quantum_planner = QuantumPlannerConfig()
        self.healing_engine = HealingEngineConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.integrations = IntegrationConfig()
        self.ml = MLConfig()
        self.notifications = NotificationConfig()
        self.performance = PerformanceConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate database URL
        try:
            parsed_url = urlparse(self.database.url)
            if not parsed_url.scheme or not parsed_url.netloc:
                errors.append("Invalid database URL format")
        except Exception:
            errors.append("Invalid database URL")
        
        # Validate Redis URL
        try:
            parsed_url = urlparse(self.redis.url)
            if not parsed_url.scheme or not parsed_url.netloc:
                errors.append("Invalid Redis URL format")
        except Exception:
            errors.append("Invalid Redis URL")
        
        # Validate ML configuration
        if self.ml.confidence_threshold < 0 or self.ml.confidence_threshold > 1:
            errors.append("ML confidence threshold must be between 0 and 1")
        
        # Validate security configuration
        if self.security.secret_key == "dev-secret-key-change-in-production" and self.environment == "production":
            errors.append("Secret key must be changed in production")
        
        if errors and self.environment == "production":
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif errors:
            logger.warning(f"Configuration warnings: {errors}")
        
        logger.info("Configuration validation passed")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (excluding sensitive data)."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "database": {
                "url": self._mask_url(self.database.url),
                "pool_size": self.database.pool_size,
                "echo": self.database.echo
            },
            "redis": {
                "url": self._mask_url(self.redis.url),
                "ssl": self.redis.ssl,
                "pool_size": self.redis.pool_size
            },
            "ml": {
                "model_path": self.ml.model_path,
                "confidence_threshold": self.ml.confidence_threshold,
                "model_type": self.ml.model_type
            },
            "healing": {
                "max_concurrent_healings": self.healing_engine.max_concurrent_healings,
                "healing_timeout_minutes": self.healing_engine.healing_timeout_minutes
            },
            "performance": {
                "caching_enabled": self.performance.caching_enabled,
                "batch_processing_enabled": self.performance.batch_processing_enabled,
                "async_workers": self.performance.async_workers
            }
        }
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of URLs."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                masked = parsed._replace(
                    netloc=parsed.netloc.replace(parsed.password, "***")
                )
                return masked.geturl()
            return url
        except Exception:
            return "***"

# Global settings instance
settings = Settings()