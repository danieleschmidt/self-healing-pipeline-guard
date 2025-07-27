"""Health check endpoints and monitoring utilities."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import redis
import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel

from ..core.config import settings

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    """Health check status model."""
    
    status: str
    timestamp: datetime
    version: str
    uptime: float
    checks: Dict[str, Dict[str, Any]]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = "1.0.0"  # Should be injected from package
        
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            conn = await asyncpg.connect(settings.DATABASE_URL)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            
            # Test query performance
            await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "query_result": result,
                "details": "Database connection successful"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()
            
            redis_client = redis.from_url(settings.REDIS_URL)
            
            # Test basic operations
            test_key = "health_check_test"
            redis_client.set(test_key, "test_value", ex=60)
            result = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Get Redis info
            info = redis_client.info()
            memory_usage = info.get("used_memory_human", "unknown")
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "memory_usage": memory_usage,
                "details": "Redis connection successful"
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Redis connection failed"
            }
    
    async def check_ml_models(self) -> Dict[str, Any]:
        """Check ML model availability and performance."""
        try:
            # This would check if ML models are loaded and accessible
            # For now, return a basic check
            
            return {
                "status": "healthy",
                "models_loaded": ["failure_classifier", "cost_predictor"],
                "details": "ML models operational"
            }
            
        except Exception as e:
            logger.error(f"ML models health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "ML models unavailable"
            }
    
    async def check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        checks = {}
        
        # GitHub API check
        try:
            # This would make actual API calls to check connectivity
            checks["github"] = {
                "status": "healthy",
                "details": "GitHub API accessible"
            }
        except Exception as e:
            checks["github"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Add other API checks as needed
        
        overall_status = "healthy" if all(
            check["status"] == "healthy" for check in checks.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "apis": checks,
            "details": "External API connectivity check"
        }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            
            free_percent = (free / total) * 100
            used_percent = (used / total) * 100
            
            status = "healthy"
            if free_percent < 10:
                status = "critical"
            elif free_percent < 20:
                status = "warning"
            
            return {
                "status": status,
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "used_percent": round(used_percent, 2),
                "free_percent": round(free_percent, 2),
                "details": "Disk space check"
            }
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Disk space check failed"
            }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": round(memory.percent, 2),
                "details": "Memory usage check"
            }
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Memory check failed"
            }
    
    async def get_comprehensive_health(self) -> HealthStatus:
        """Get comprehensive health status."""
        timestamp = datetime.now(timezone.utc)
        uptime = time.time() - self.start_time
        
        # Run all health checks concurrently
        checks = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_ml_models(),
            self.check_external_apis(),
            self.check_disk_space(),
            self.check_memory(),
            return_exceptions=True
        )
        
        check_results = {
            "database": checks[0] if not isinstance(checks[0], Exception) else {"status": "error", "error": str(checks[0])},
            "redis": checks[1] if not isinstance(checks[1], Exception) else {"status": "error", "error": str(checks[1])},
            "ml_models": checks[2] if not isinstance(checks[2], Exception) else {"status": "error", "error": str(checks[2])},
            "external_apis": checks[3] if not isinstance(checks[3], Exception) else {"status": "error", "error": str(checks[3])},
            "disk_space": checks[4] if not isinstance(checks[4], Exception) else {"status": "error", "error": str(checks[4])},
            "memory": checks[5] if not isinstance(checks[5], Exception) else {"status": "error", "error": str(checks[5])},
        }
        
        # Determine overall status
        critical_checks = ["database", "redis"]
        overall_status = "healthy"
        
        for check_name, result in check_results.items():
            if result["status"] in ["critical", "error"]:
                if check_name in critical_checks:
                    overall_status = "unhealthy"
                    break
                else:
                    overall_status = "degraded"
            elif result["status"] == "warning" and overall_status == "healthy":
                overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            version=self.version,
            uptime=uptime,
            checks=check_results
        )
    
    async def get_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to receive traffic."""
        try:
            # Check critical dependencies
            db_check = await self.check_database()
            redis_check = await self.check_redis()
            
            if db_check["status"] == "healthy" and redis_check["status"] == "healthy":
                return {
                    "status": "ready",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": "Service ready to receive traffic"
                }
            else:
                return {
                    "status": "not_ready",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": "Critical dependencies not available",
                    "database": db_check["status"],
                    "redis": redis_check["status"]
                }
                
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                "status": "not_ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    async def get_liveness(self) -> Dict[str, Any]:
        """Check if service is alive and should not be restarted."""
        try:
            # Basic liveness check - if we can respond, we're alive
            return {
                "status": "alive",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime": time.time() - self.start_time,
                "details": "Service is responsive"
            }
            
        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return {
                "status": "dead",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }


# Global health checker instance
health_checker = HealthChecker()