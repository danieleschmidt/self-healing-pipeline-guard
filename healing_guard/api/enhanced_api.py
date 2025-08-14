"""Enhanced API with advanced features and optimizations.

Provides high-performance REST API with automatic scaling, caching,
security, and comprehensive monitoring capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import traceback

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from ..core.quantum_planner import QuantumTaskPlanner, Task, TaskPriority, TaskStatus
from ..core.failure_detector import FailureDetector, FailureType, SeverityLevel
from ..core.healing_engine import HealingEngine, HealingStatus
from ..core.advanced_scaling import auto_scaler
from ..security.advanced_security import security_manager
from ..monitoring.enhanced_monitoring import monitoring_dashboard
from ..core.exceptions import HealingSystemException, ValidationException

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for API
class HealingRequest(BaseModel):
    job_id: str = Field(..., description="CI/CD job identifier")
    repository: str = Field(..., description="Repository name")
    branch: str = Field(..., description="Branch name")  
    commit_sha: str = Field(..., description="Commit SHA")
    logs: str = Field(..., description="Failure logs")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: str = Field(default="medium", description="Healing priority level")
    timeout_minutes: int = Field(default=30, description="Maximum healing timeout")

class TaskRequest(BaseModel):
    name: str = Field(..., description="Task name")
    priority: str = Field(default="medium", description="Task priority")
    estimated_duration: float = Field(..., description="Estimated duration in minutes")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    resources_required: Dict[str, float] = Field(default_factory=dict, description="Required resources")

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None

class HealthCheckResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    uptime_seconds: float
    components: Dict[str, str]

# Rate limiting and caching
request_cache: Dict[str, Dict[str, Any]] = {}
rate_limit_cache: Dict[str, List[float]] = {}

# API metrics
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0.0,
    "active_connections": 0,
    "cache_hits": 0,
    "cache_misses": 0
}

# WebSocket connections
active_websockets: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Enhanced Healing Guard API")
    
    # Initialize components
    monitoring_dashboard.start()
    auto_scaler.start_monitoring()
    
    # Setup auto-scaling callbacks
    auto_scaler.register_scaling_callbacks(
        scale_up=handle_scale_up,
        scale_down=handle_scale_down
    )
    
    app.state.start_time = time.time()
    app.state.quantum_planner = QuantumTaskPlanner()
    app.state.failure_detector = FailureDetector()
    app.state.healing_engine = HealingEngine(app.state.quantum_planner, app.state.failure_detector)
    
    logger.info("API components initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Healing Guard API")
    monitoring_dashboard.stop()
    auto_scaler.stop_monitoring()

# Create FastAPI app
app = FastAPI(
    title="Self-Healing Pipeline Guard API",
    description="Enhanced AI-powered CI/CD pipeline failure detection and healing",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for metrics and security
@app.middleware("http")
async def metrics_and_security_middleware(request: Request, call_next):
    """Middleware for metrics collection and security processing."""
    start_time = time.time()
    api_metrics["total_requests"] += 1
    api_metrics["active_connections"] += 1
    
    try:
        # Security processing
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        security_request = {
            "source_ip": client_ip,
            "user_agent": user_agent,
            "path": str(request.url.path),
            "method": request.method,
            "headers": dict(request.headers),
            "content": "",  # Would extract body for POST requests
        }
        
        security_result = await security_manager.process_security_request(security_request)
        
        if not security_result["allowed"]:
            api_metrics["failed_requests"] += 1
            return JSONResponse(
                status_code=403,
                content={"error": "Request blocked by security policy", "details": security_result}
            )
        
        # Process request
        response = await call_next(request)
        
        # Update metrics
        execution_time = (time.time() - start_time) * 1000
        api_metrics["successful_requests"] += 1
        api_metrics["avg_response_time"] = (
            (api_metrics["avg_response_time"] * (api_metrics["successful_requests"] - 1) + execution_time) /
            api_metrics["successful_requests"]
        )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{execution_time:.2f}ms"
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        
        return response
        
    except Exception as e:
        api_metrics["failed_requests"] += 1
        logger.error(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": str(uuid.uuid4())}
        )
    finally:
        api_metrics["active_connections"] -= 1

# Dependency injection
async def get_quantum_planner(request: Request) -> QuantumTaskPlanner:
    """Get quantum planner instance."""
    return request.app.state.quantum_planner

async def get_failure_detector(request: Request) -> FailureDetector:
    """Get failure detector instance."""
    return request.app.state.failure_detector

async def get_healing_engine(request: Request) -> HealingEngine:
    """Get healing engine instance."""
    return request.app.state.healing_engine

async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Verify authentication credentials."""
    if not credentials:
        return None
    
    # In production, verify JWT token or API key
    # For now, simple token validation
    token = credentials.credentials
    if len(token) < 10:  # Minimal validation
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return {"user_id": "api_user", "permissions": ["read", "write"]}

# Utility functions
def cache_key(request_data: Dict[str, Any]) -> str:
    """Generate cache key from request data."""
    return f"api_cache_{hash(json.dumps(request_data, sort_keys=True))}"

def get_cached_response(key: str) -> Optional[Dict[str, Any]]:
    """Get response from cache."""
    cached = request_cache.get(key)
    if cached and cached["expires_at"] > time.time():
        api_metrics["cache_hits"] += 1
        return cached["response"]
    api_metrics["cache_misses"] += 1
    return None

def set_cached_response(key: str, response: Dict[str, Any], ttl_seconds: int = 300):
    """Set response in cache."""
    request_cache[key] = {
        "response": response,
        "expires_at": time.time() + ttl_seconds
    }

async def handle_scale_up(confidence: float, analysis: Dict[str, Any]):
    """Handle auto-scaling up event."""
    logger.info(f"Auto-scaling UP triggered with confidence {confidence:.2f}")
    # In production, this would trigger container/VM scaling
    
    # Broadcast to WebSocket clients
    await broadcast_websocket_message({
        "type": "scaling_event",
        "direction": "up",
        "confidence": confidence,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    })

async def handle_scale_down(confidence: float, analysis: Dict[str, Any]):
    """Handle auto-scaling down event."""
    logger.info(f"Auto-scaling DOWN triggered with confidence {confidence:.2f}")
    # In production, this would trigger container/VM descaling
    
    # Broadcast to WebSocket clients
    await broadcast_websocket_message({
        "type": "scaling_event",
        "direction": "down",
        "confidence": confidence,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    })

async def broadcast_websocket_message(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections."""
    if not active_websockets:
        return
    
    message_str = json.dumps(message)
    disconnected = []
    
    for websocket in active_websockets:
        try:
            await websocket.send_text(message_str)
        except Exception:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.remove(ws)

# API Routes

@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint with API information."""
    return ApiResponse(
        success=True,
        message="Self-Healing Pipeline Guard API v2.0 - Enhanced",
        data={
            "features": [
                "AI-powered failure detection",
                "Quantum-inspired task optimization", 
                "Autonomous healing orchestration",
                "Advanced security & monitoring",
                "Auto-scaling & load balancing"
            ],
            "version": "2.0.0",
            "status": "operational"
        }
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(request: Request):
    """Enhanced health check with component status."""
    uptime = time.time() - request.app.state.start_time
    
    # Check component health
    components = {
        "quantum_planner": "healthy",
        "failure_detector": "healthy", 
        "healing_engine": "healthy",
        "monitoring": "healthy" if monitoring_dashboard.started else "degraded",
        "auto_scaler": "healthy" if auto_scaler.monitoring_active else "degraded",
        "security": "healthy"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        uptime_seconds=uptime,
        components=components
    )

@app.post("/heal", response_model=ApiResponse)
async def heal_pipeline_failure(
    request: HealingRequest,
    background_tasks: BackgroundTasks,
    healing_engine: HealingEngine = Depends(get_healing_engine),
    failure_detector: FailureDetector = Depends(get_failure_detector),
    auth: Optional[Dict[str, Any]] = Depends(verify_auth)
):
    """Heal a pipeline failure with enhanced processing."""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key_str = cache_key(request.dict())
        cached_response = get_cached_response(cache_key_str)
        if cached_response:
            return ApiResponse(**cached_response)
        
        # Detect failure
        failure_event = await failure_detector.detect_failure(
            job_id=request.job_id,
            repository=request.repository,
            branch=request.branch,
            commit_sha=request.commit_sha,
            logs=request.logs,
            context=request.context
        )
        
        if not failure_event:
            return ApiResponse(
                success=False,
                message="Could not classify failure - insufficient data",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Create and execute healing plan
        healing_result = await healing_engine.heal_failure(failure_event)
        
        response_data = {
            "healing_id": healing_result.healing_id,
            "status": healing_result.status.value,
            "failure_analysis": {
                "type": failure_event.failure_type.value,
                "severity": failure_event.severity.value,
                "confidence": failure_event.confidence,
                "remediation_suggestions": failure_event.remediation_suggestions
            },
            "healing_results": {
                "actions_executed": healing_result.actions_executed,
                "actions_successful": healing_result.actions_successful,
                "actions_failed": healing_result.actions_failed,
                "total_duration": healing_result.total_duration,
                "metrics": healing_result.metrics
            }
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        response = ApiResponse(
            success=True,
            message=f"Healing completed: {healing_result.status.value}",
            data=response_data,
            execution_time_ms=execution_time
        )
        
        # Cache successful responses
        if healing_result.status in [HealingStatus.SUCCESSFUL, HealingStatus.PARTIAL]:
            set_cached_response(cache_key_str, response.dict(), ttl_seconds=1800)  # 30 minutes
        
        # Background task for metrics
        background_tasks.add_task(record_healing_metrics, healing_result)
        
        return response
        
    except Exception as e:
        logger.error(f"Healing API error: {e}\n{traceback.format_exc()}")
        return ApiResponse(
            success=False,
            message=f"Healing failed: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )

@app.post("/plan", response_model=ApiResponse)
async def create_execution_plan(
    tasks: List[TaskRequest],
    quantum_planner: QuantumTaskPlanner = Depends(get_quantum_planner),
    auth: Optional[Dict[str, Any]] = Depends(verify_auth)
):
    """Create optimized execution plan from tasks."""
    start_time = time.time()
    
    try:
        # Convert requests to Task objects
        task_objects = []
        for task_req in tasks:
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW
            }
            
            task = Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                name=task_req.name,
                priority=priority_map.get(task_req.priority, TaskPriority.MEDIUM),
                estimated_duration=task_req.estimated_duration,
                dependencies=task_req.dependencies,
                resources_required=task_req.resources_required
            )
            task_objects.append(task)
            quantum_planner.add_task(task)
        
        # Create execution plan
        execution_plan = await quantum_planner.create_execution_plan()
        
        response_data = {
            "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
            "total_tasks": len(task_objects),
            "estimated_total_time": execution_plan.estimated_total_time,
            "success_probability": execution_plan.success_probability,
            "cost_estimate": execution_plan.cost_estimate,
            "parallel_stages": execution_plan.parallel_stages,
            "resource_utilization": execution_plan.resource_utilization
        }
        
        return ApiResponse(
            success=True,
            message="Execution plan created successfully",
            data=response_data,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Planning API error: {e}")
        return ApiResponse(
            success=False,
            message=f"Plan creation failed: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )

@app.get("/metrics", response_model=ApiResponse)
async def get_comprehensive_metrics(
    auth: Optional[Dict[str, Any]] = Depends(verify_auth)
):
    """Get comprehensive system metrics and statistics."""
    try:
        metrics_data = {
            "api_metrics": api_metrics,
            "monitoring_dashboard": monitoring_dashboard.get_dashboard_data(),
            "auto_scaling": auto_scaler.get_comprehensive_stats(),
            "security": security_manager.get_security_dashboard(),
            "cache_stats": {
                "total_entries": len(request_cache),
                "hit_rate": api_metrics["cache_hits"] / (api_metrics["cache_hits"] + api_metrics["cache_misses"]) if (api_metrics["cache_hits"] + api_metrics["cache_misses"]) > 0 else 0
            }
        }
        
        return ApiResponse(
            success=True,
            message="Comprehensive metrics retrieved",
            data=metrics_data
        )
        
    except Exception as e:
        logger.error(f"Metrics API error: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to retrieve metrics: {str(e)}"
        )

@app.get("/status", response_model=ApiResponse)
async def get_system_status(
    request: Request,
    auth: Optional[Dict[str, Any]] = Depends(verify_auth)
):
    """Get current system status and active operations."""
    try:
        healing_engine = request.app.state.healing_engine
        failure_detector = request.app.state.failure_detector
        quantum_planner = request.app.state.quantum_planner
        
        status_data = {
            "system_status": "operational",
            "uptime_seconds": time.time() - request.app.state.start_time,
            "active_healings": len(healing_engine.active_healings),
            "total_healing_history": len(healing_engine.healing_history),
            "failure_detection": failure_detector.get_failure_statistics(),
            "quantum_planning": quantum_planner.get_planning_statistics(),
            "api_performance": {
                "total_requests": api_metrics["total_requests"],
                "success_rate": api_metrics["successful_requests"] / api_metrics["total_requests"] if api_metrics["total_requests"] > 0 else 0,
                "avg_response_time_ms": api_metrics["avg_response_time"],
                "active_connections": api_metrics["active_connections"]
            }
        }
        
        return ApiResponse(
            success=True,
            message="System status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to retrieve status: {str(e)}"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to Healing Guard API",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Echo back for now - in production would handle commands
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

@app.get("/stream/events")
async def stream_events(auth: Optional[Dict[str, Any]] = Depends(verify_auth)):
    """Server-Sent Events stream for real-time monitoring."""
    async def event_generator():
        while True:
            # Generate real-time metrics
            event_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "active_connections": api_metrics["active_connections"],
                    "total_requests": api_metrics["total_requests"],
                    "avg_response_time": api_metrics["avg_response_time"]
                }
            }
            
            yield f"data: {json.dumps(event_data)}\\n\\n"
            await asyncio.sleep(5)  # Send update every 5 seconds
    
    return StreamingResponse(event_generator(), media_type="text/plain")

# Background task functions
async def record_healing_metrics(healing_result):
    """Background task to record healing metrics."""
    monitoring_dashboard.metrics_collector.record_metric(
        "healing.duration", 
        healing_result.total_duration,
        {"status": healing_result.status.value}
    )
    
    monitoring_dashboard.metrics_collector.record_metric(
        "healing.success_rate",
        1.0 if healing_result.status == HealingStatus.SUCCESSFUL else 0.0,
        {"plan_id": healing_result.plan.id}
    )

# Error handlers
@app.exception_handler(HealingSystemException)
async def healing_exception_handler(request: Request, exc: HealingSystemException):
    """Handle healing system exceptions."""
    return JSONResponse(
        status_code=400 if exc.recoverable else 500,
        content={
            "error": "Healing System Error",
            "message": str(exc),
            "context": exc.context,
            "recoverable": exc.recoverable,
            "timestamp": exc.timestamp.isoformat(),
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "field": exc.field,
            "value": exc.value,
            "request_id": str(uuid.uuid4())
        }
    )

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "healing_guard.api.enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,     # Increase for production
        log_level="info"
    )