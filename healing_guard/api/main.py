"""FastAPI application factory and configuration."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..monitoring.health import health_checker
from ..monitoring.metrics import metrics_collector
from .routes import router
from .middleware import SecurityMiddleware, RateLimitMiddleware
from .exceptions import (
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    ResourceNotFoundException,
    ServiceUnavailableException
)

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Counter('http_active_connections', 'Active HTTP connections')


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            # Add response headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            logger.error(f"Request failed: {e}")
            raise
            
        finally:
            ACTIVE_CONNECTIONS.dec()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Healing Guard API")
    
    # Startup tasks
    try:
        # Initialize monitoring
        await metrics_collector.initialize()
        logger.info("Metrics collector initialized")
        
        # Perform health checks
        health_status = await health_checker.get_comprehensive_health()
        if health_status.status == "unhealthy":
            logger.warning("Some health checks failed during startup")
            
        logger.info("Healing Guard API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
        
    yield
    
    # Shutdown tasks
    logger.info("Shutting down Healing Guard API")
    await metrics_collector.cleanup()
    logger.info("Healing Guard API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Self-Healing Pipeline Guard",
        description="AI-powered CI/CD guardian with quantum-inspired task planning",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters - last added is first executed)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(SecurityMiddleware)
    
    # Exception handlers
    
    @app.exception_handler(ValidationException)
    async def validation_exception_handler(request: Request, exc: ValidationException):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation Error",
                "message": str(exc),
                "details": exc.details if hasattr(exc, 'details') else None
            }
        )
    
    @app.exception_handler(AuthenticationException)
    async def authentication_exception_handler(request: Request, exc: AuthenticationException):
        return JSONResponse(
            status_code=401,
            content={
                "error": "Authentication Required",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(AuthorizationException)
    async def authorization_exception_handler(request: Request, exc: AuthorizationException):
        return JSONResponse(
            status_code=403,
            content={
                "error": "Access Forbidden",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(ResourceNotFoundException)
    async def not_found_exception_handler(request: Request, exc: ResourceNotFoundException):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Resource Not Found",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(ServiceUnavailableException)
    async def service_unavailable_exception_handler(request: Request, exc: ServiceUnavailableException):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": str(exc),
                "retry_after": getattr(exc, 'retry_after', 60)
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": exc.detail
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        if settings.debug:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": str(exc),
                    "type": type(exc).__name__
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred"
                }
            )
    
    # Health check endpoints
    
    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe endpoint."""
        result = await health_checker.get_liveness()
        status_code = 200 if result["status"] == "alive" else 503
        return Response(content="OK" if status_code == 200 else "NOT OK", status_code=status_code)
    
    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe endpoint."""
        result = await health_checker.get_readiness()
        status_code = 200 if result["status"] == "ready" else 503
        return Response(content="READY" if status_code == 200 else "NOT READY", status_code=status_code)
    
    @app.get("/health")
    async def health_check():
        """Comprehensive health check endpoint."""
        health_status = await health_checker.get_comprehensive_health()
        status_code = 200 if health_status.status == "healthy" else 503
        return JSONResponse(content=health_status.dict(), status_code=status_code)
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Self-Healing Pipeline Guard",
            "version": "1.0.0",
            "description": "AI-powered CI/CD guardian with quantum-inspired task planning",
            "status": "operational",
            "docs_url": "/docs" if settings.debug else None,
            "health_url": "/health",
            "metrics_url": "/metrics"
        }
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Include sentiment analysis routes
    from .sentiment_routes import router as sentiment_router
    app.include_router(sentiment_router, prefix="/api/v1")
    
    return app