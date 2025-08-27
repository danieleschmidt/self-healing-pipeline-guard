"""Real-time dashboard API with WebSocket support for live monitoring.

Provides real-time system health, metrics, and healing operation updates
through WebSocket connections and REST endpoints.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ..monitoring.enhanced_monitoring import enhanced_monitoring, AlertSeverity, ComponentStatus
from ..core.healing_engine import HealingEngine, HealingStatus
from ..core.failure_detector import FailureDetector, FailureType, SeverityLevel

logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Dashboard theme options."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class MetricType(Enum):
    """Real-time metric types."""
    HEALING_SUCCESS_RATE = "healing_success_rate"
    SYSTEM_HEALTH_SCORE = "system_health_score"
    ACTIVE_HEALINGS = "active_healings"
    FAILURES_DETECTED = "failures_detected"
    COST_SAVINGS = "cost_savings"
    RESPONSE_TIME = "response_time"


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    theme: DashboardTheme = DashboardTheme.DARK
    refresh_interval: int = 5  # seconds
    show_debug_info: bool = False
    max_chart_points: int = 100
    alert_notifications: bool = True
    sound_alerts: bool = False
    auto_refresh: bool = True


@dataclass
class RealTimeUpdate:
    """Real-time update message."""
    timestamp: datetime
    update_type: str
    data: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "update_type": self.update_type,
            "data": self.data,
            "message_id": self.message_id
        }


@dataclass
class SystemOverview:
    """System overview data for dashboard."""
    health_score: float
    overall_status: ComponentStatus
    active_healings: int
    total_failures_24h: int
    success_rate_24h: float
    cost_savings_24h: float
    uptime_hours: float
    active_alerts: int
    last_updated: datetime


@dataclass
class ChartDataPoint:
    """Chart data point."""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "label": self.label
        }


class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.update_queue: asyncio.Queue = asyncio.Queue()
        
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept a WebSocket connection."""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
            
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "last_ping": datetime.now(),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket client {client_id} connected")
        return client_id
    
    def disconnect(self, client_id: str):
        """Disconnect a client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], subscription_filter: str = None):
        """Broadcast message to all connected clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                # Check subscription filter
                if subscription_filter:
                    subscriptions = self.connection_metadata[client_id].get("subscriptions", set())
                    if subscription_filter not in subscriptions:
                        continue
                
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe_client(self, client_id: str, subscription_type: str):
        """Subscribe client to specific update types."""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].add(subscription_type)
    
    def unsubscribe_client(self, client_id: str, subscription_type: str):
        """Unsubscribe client from specific update types."""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].discard(subscription_type)


class RealTimeDashboard:
    """Real-time dashboard with live monitoring capabilities."""
    
    def __init__(
        self,
        healing_engine: Optional[HealingEngine] = None,
        failure_detector: Optional[FailureDetector] = None
    ):
        self.healing_engine = healing_engine or HealingEngine()
        self.failure_detector = failure_detector or FailureDetector()
        self.connection_manager = WebSocketConnectionManager()
        
        # Dashboard state
        self.config = DashboardConfig()
        self.chart_data: Dict[str, List[ChartDataPoint]] = {
            metric.value: [] for metric in MetricType
        }
        self.last_system_overview: Optional[SystemOverview] = None
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Initialize monitoring callbacks
        self._setup_monitoring_callbacks()
        
    def _setup_monitoring_callbacks(self):
        """Setup callbacks for real-time monitoring updates."""
        def on_alert_fired(alert):
            """Handle alert notifications."""
            asyncio.create_task(self._handle_alert_update(alert))
        
        # Register callback with monitoring system
        enhanced_monitoring.add_alert_callback(on_alert_fired)
        
    async def start(self):
        """Start the real-time dashboard."""
        if self.running:
            return
            
        self.running = True
        
        # Start background tasks
        self.update_task = asyncio.create_task(self._update_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Start monitoring if not already started
        await enhanced_monitoring.start_monitoring()
        
        logger.info("Real-time dashboard started")
    
    async def stop(self):
        """Stop the real-time dashboard."""
        self.running = False
        
        # Cancel background tasks
        if self.update_task:
            self.update_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all WebSocket connections
        for client_id in list(self.connection_manager.active_connections.keys()):
            self.connection_manager.disconnect(client_id)
        
        logger.info("Real-time dashboard stopped")
    
    async def _update_loop(self):
        """Main update loop for real-time data."""
        while self.running:
            try:
                # Collect current system data
                await self._collect_real_time_data()
                
                # Send updates to connected clients
                await self._broadcast_updates()
                
                # Wait for next update
                await asyncio.sleep(self.config.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_loop(self):
        """Cleanup old data and maintain performance."""
        while self.running:
            try:
                # Clean up old chart data
                self._cleanup_chart_data()
                
                # Clean up old connection metadata
                self._cleanup_connections()
                
                # Wait for next cleanup
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def _cleanup_chart_data(self):
        """Remove old chart data points."""
        cutoff_time = datetime.now() - timedelta(hours=2)
        
        for metric_type in self.chart_data:
            self.chart_data[metric_type] = [
                point for point in self.chart_data[metric_type]
                if point.timestamp > cutoff_time
            ]
            
            # Limit number of points
            if len(self.chart_data[metric_type]) > self.config.max_chart_points:
                self.chart_data[metric_type] = self.chart_data[metric_type][-self.config.max_chart_points:]
    
    def _cleanup_connections(self):
        """Clean up stale connections."""
        cutoff_time = datetime.now() - timedelta(minutes=30)
        
        stale_connections = [
            client_id for client_id, metadata in self.connection_manager.connection_metadata.items()
            if metadata.get("last_ping", datetime.now()) < cutoff_time
        ]
        
        for client_id in stale_connections:
            self.connection_manager.disconnect(client_id)
    
    async def _collect_real_time_data(self):
        """Collect real-time system data."""
        try:
            # Get system health
            system_health = enhanced_monitoring.get_system_health()
            real_time_metrics = enhanced_monitoring.get_real_time_metrics()
            
            # Create system overview
            current_overview = SystemOverview(
                health_score=system_health.health_score,
                overall_status=system_health.overall_status,
                active_healings=real_time_metrics.active_healings,
                total_failures_24h=real_time_metrics.failures_detected_24h,
                success_rate_24h=real_time_metrics.healing_success_rate,
                cost_savings_24h=real_time_metrics.cost_savings_usd,
                uptime_hours=system_health.uptime_seconds / 3600,
                active_alerts=len(system_health.active_alerts),
                last_updated=datetime.now()
            )
            
            # Update chart data
            timestamp = datetime.now()
            self.chart_data[MetricType.HEALING_SUCCESS_RATE.value].append(
                ChartDataPoint(timestamp, current_overview.success_rate_24h)
            )
            self.chart_data[MetricType.SYSTEM_HEALTH_SCORE.value].append(
                ChartDataPoint(timestamp, current_overview.health_score)
            )
            self.chart_data[MetricType.ACTIVE_HEALINGS.value].append(
                ChartDataPoint(timestamp, current_overview.active_healings)
            )
            self.chart_data[MetricType.FAILURES_DETECTED.value].append(
                ChartDataPoint(timestamp, current_overview.total_failures_24h)
            )
            self.chart_data[MetricType.COST_SAVINGS.value].append(
                ChartDataPoint(timestamp, current_overview.cost_savings_24h)
            )
            
            self.last_system_overview = current_overview
            
        except Exception as e:
            logger.error(f"Error collecting real-time data: {e}")
    
    async def _broadcast_updates(self):
        """Broadcast updates to connected clients."""
        if not self.last_system_overview:
            return
        
        # System overview update
        overview_update = RealTimeUpdate(
            timestamp=datetime.now(),
            update_type="system_overview",
            data=asdict(self.last_system_overview)
        )
        
        await self.connection_manager.broadcast(
            overview_update.to_dict(),
            subscription_filter="system_overview"
        )
        
        # Chart data updates
        for metric_type, data_points in self.chart_data.items():
            if not data_points:
                continue
            
            # Send recent data points (last 5 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            recent_points = [
                point.to_dict() for point in data_points
                if point.timestamp >= recent_cutoff
            ]
            
            if recent_points:
                chart_update = RealTimeUpdate(
                    timestamp=datetime.now(),
                    update_type="chart_data",
                    data={
                        "metric_type": metric_type,
                        "points": recent_points
                    }
                )
                
                await self.connection_manager.broadcast(
                    chart_update.to_dict(),
                    subscription_filter="chart_data"
                )
    
    async def _handle_alert_update(self, alert):
        """Handle alert notifications."""
        alert_update = RealTimeUpdate(
            timestamp=datetime.now(),
            update_type="alert_notification",
            data={
                "alert_id": alert.id,
                "message": alert.message,
                "severity": alert.severity.value,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat()
            }
        )
        
        await self.connection_manager.broadcast(
            alert_update.to_dict(),
            subscription_filter="alert_notifications"
        )
    
    def get_system_overview(self) -> Optional[Dict[str, Any]]:
        """Get current system overview."""
        if self.last_system_overview:
            return asdict(self.last_system_overview)
        return None
    
    def get_chart_data(self, metric_type: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get chart data for specific metric."""
        if metric_type not in self.chart_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            point.to_dict() for point in self.chart_data[metric_type]
            if point.timestamp >= cutoff_time
        ]
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "active_connections": len(self.connection_manager.active_connections),
            "total_subscriptions": sum(
                len(metadata.get("subscriptions", set()))
                for metadata in self.connection_manager.connection_metadata.values()
            ),
            "connection_details": [
                {
                    "client_id": client_id,
                    "connected_at": metadata["connected_at"].isoformat(),
                    "subscriptions": list(metadata.get("subscriptions", set()))
                }
                for client_id, metadata in self.connection_manager.connection_metadata.items()
            ]
        }


# Pydantic models for API
class DashboardConfigModel(BaseModel):
    """Dashboard configuration model."""
    theme: str = "dark"
    refresh_interval: int = 5
    show_debug_info: bool = False
    max_chart_points: int = 100
    alert_notifications: bool = True
    sound_alerts: bool = False
    auto_refresh: bool = True


class ChartDataRequest(BaseModel):
    """Chart data request model."""
    metric_type: str
    hours: int = 1


# Global dashboard instance
dashboard = RealTimeDashboard()


# FastAPI application
app = FastAPI(title="Healing Guard Real-Time Dashboard API")


@app.on_event("startup")
async def startup_event():
    """Start dashboard on application startup."""
    await dashboard.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop dashboard on application shutdown."""
    await dashboard.stop()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    client_id = await dashboard.connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription requests
            if message.get("type") == "subscribe":
                subscription_type = message.get("subscription")
                if subscription_type:
                    dashboard.connection_manager.subscribe_client(client_id, subscription_type)
                    
            elif message.get("type") == "unsubscribe":
                subscription_type = message.get("subscription")
                if subscription_type:
                    dashboard.connection_manager.unsubscribe_client(client_id, subscription_type)
                    
            elif message.get("type") == "ping":
                # Update last ping time
                dashboard.connection_manager.connection_metadata[client_id]["last_ping"] = datetime.now()
                await dashboard.connection_manager.send_to_client(client_id, {"type": "pong"})
                
    except WebSocketDisconnect:
        dashboard.connection_manager.disconnect(client_id)


@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Get current system overview."""
    overview = dashboard.get_system_overview()
    if not overview:
        raise HTTPException(status_code=503, detail="Dashboard not ready")
    return overview


@app.post("/api/dashboard/chart-data")
async def get_chart_data(request: ChartDataRequest):
    """Get chart data for specific metric."""
    data = dashboard.get_chart_data(request.metric_type, request.hours)
    return {"metric_type": request.metric_type, "data": data}


@app.get("/api/dashboard/metrics")
async def get_available_metrics():
    """Get list of available metrics."""
    return {
        "metrics": [metric.value for metric in MetricType],
        "descriptions": {
            MetricType.HEALING_SUCCESS_RATE.value: "Success rate of healing operations",
            MetricType.SYSTEM_HEALTH_SCORE.value: "Overall system health score (0-100)",
            MetricType.ACTIVE_HEALINGS.value: "Number of currently active healing processes",
            MetricType.FAILURES_DETECTED.value: "Total failures detected in last 24 hours",
            MetricType.COST_SAVINGS.value: "Cost savings from automated healing (USD)",
            MetricType.RESPONSE_TIME.value: "Average system response time"
        }
    }


@app.get("/api/dashboard/config")
async def get_dashboard_config():
    """Get current dashboard configuration."""
    return asdict(dashboard.config)


@app.put("/api/dashboard/config")
async def update_dashboard_config(config: DashboardConfigModel):
    """Update dashboard configuration."""
    dashboard.config.theme = DashboardTheme(config.theme)
    dashboard.config.refresh_interval = config.refresh_interval
    dashboard.config.show_debug_info = config.show_debug_info
    dashboard.config.max_chart_points = config.max_chart_points
    dashboard.config.alert_notifications = config.alert_notifications
    dashboard.config.sound_alerts = config.sound_alerts
    dashboard.config.auto_refresh = config.auto_refresh
    
    return {"message": "Configuration updated successfully"}


@app.get("/api/dashboard/connections")
async def get_connection_stats():
    """Get WebSocket connection statistics."""
    return dashboard.get_connection_stats()


@app.get("/api/dashboard/health")
async def get_dashboard_health():
    """Get dashboard service health."""
    return {
        "status": "healthy" if dashboard.running else "stopped",
        "uptime_seconds": time.time() - (dashboard.start_time if hasattr(dashboard, 'start_time') else time.time()),
        "active_connections": len(dashboard.connection_manager.active_connections),
        "last_update": dashboard.last_system_overview.last_updated.isoformat() if dashboard.last_system_overview else None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")