"""Configuration management for the Healing Guard system."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/healing_guard")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"

@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
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
class Settings:
    """Main application settings."""
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    quantum_planner: QuantumPlannerConfig = field(default_factory=QuantumPlannerConfig)
    healing_engine: HealingEngineConfig = field(default_factory=HealingEngineConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.environment == "production":
            self._validate_production_config()
    
    def _validate_production_config(self):
        """Validate production configuration."""
        required_vars = [
            ("DATABASE_URL", self.database.url),
            ("REDIS_URL", self.redis.url),
            ("SECRET_KEY", self.security.secret_key)
        ]
        
        for var_name, value in required_vars:
            if not value or value.startswith("dev-"):
                raise ValueError(f"Production environment requires proper {var_name}")

# Global settings instance
settings = Settings()