"""Custom middleware for the API."""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from .exceptions import RateLimitExceededException, AuthenticationException

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and request processing."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add security headers
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_counts: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Remove old request records."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        cutoff_time = current_time - 60  # 1 minute ago
        
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                req_time for req_time in self.request_counts[ip]
                if req_time > cutoff_time
            ]
            
            if not self.request_counts[ip]:
                del self.request_counts[ip]
                
        self.last_cleanup = current_time
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)
            
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        # Check rate limit
        recent_requests = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > current_time - 60  # Last minute
        ]
        
        if len(recent_requests) >= self.requests_per_minute:
            raise RateLimitExceededException(
                f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                retry_after=60
            )
        
        # Record this request
        self.request_counts[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(recent_requests) - 1)
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[{request_id}] from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"[{request_id}] {response.status_code} in {duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"[{request_id}] {type(e).__name__}: {e} in {duration:.3f}s"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            return Response(
                content='{"error": "Request timeout"}',
                status_code=504,
                media_type="application/json"
            )
        except ConnectionError:
            logger.error("Connection error")
            return Response(
                content='{"error": "Connection error"}',
                status_code=502,
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
            if settings.debug:
                return Response(
                    content=f'{{"error": "Internal server error", "details": "{str(e)}"}}',
                    status_code=500,
                    media_type="application/json"
                )
            else:
                return Response(
                    content='{"error": "Internal server error"}',
                    status_code=500,
                    media_type="application/json"
                )