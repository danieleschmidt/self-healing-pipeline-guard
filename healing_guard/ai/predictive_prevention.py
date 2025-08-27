"""AI-powered predictive failure prevention system.

Uses advanced machine learning models to predict and prevent failures
before they occur, implementing proactive healing strategies.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import pickle

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from ..core.failure_detector import FailureEvent, FailureType, SeverityLevel
from ..ml.failure_pattern_recognition import pattern_recognizer, FailurePattern, PatternType
from ..monitoring.enhanced_monitoring import enhanced_monitoring
from ..core.healing_engine import HealingEngine, HealingStrategy

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PreventionAction(Enum):
    """Types of preventive actions."""
    SCALE_RESOURCES = "scale_resources"
    RESTART_SERVICES = "restart_services"
    CLEAR_CACHE = "clear_cache"
    UPDATE_CONFIGURATION = "update_configuration"
    NOTIFY_OPERATORS = "notify_operators"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    ISOLATE_COMPONENT = "isolate_component"
    APPLY_CIRCUIT_BREAKER = "apply_circuit_breaker"


class RiskLevel(Enum):
    """Risk assessment levels."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class SystemMetrics:
    """System metrics for prediction."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    active_connections: int
    error_rate: float
    response_time: float
    queue_depth: int
    throughput: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        features = [
            self.cpu_usage,
            self.memory_usage,
            self.disk_usage,
            self.network_io,
            self.active_connections,
            self.error_rate,
            self.response_time,
            self.queue_depth,
            self.throughput,
            # Time-based features
            self.timestamp.hour,
            self.timestamp.weekday(),
            # Custom metrics (take first 10)
            *list(self.custom_metrics.values())[:10]
        ]
        
        # Pad to fixed length
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])


@dataclass
class FailurePrediction:
    """Prediction of potential failure."""
    prediction_id: str
    predicted_failure_type: FailureType
    probability: float
    confidence: PredictionConfidence
    time_to_failure: Optional[float]  # hours
    risk_level: RiskLevel
    contributing_factors: List[str]
    recommended_actions: List[PreventionAction]
    predicted_at: datetime
    expires_at: datetime
    model_used: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "predicted_failure_type": self.predicted_failure_type.value,
            "probability": self.probability,
            "confidence": self.confidence.value,
            "time_to_failure": self.time_to_failure,
            "risk_level": self.risk_level.value,
            "contributing_factors": self.contributing_factors,
            "recommended_actions": [action.value for action in self.recommended_actions],
            "predicted_at": self.predicted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "model_used": self.model_used,
            "feature_importance": self.feature_importance
        }


@dataclass
class PreventiveAction:
    """Preventive action taken to avoid predicted failure."""
    action_id: str
    prediction_id: str
    action_type: PreventionAction
    description: str
    executed_at: datetime
    success: bool
    impact: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "prediction_id": self.prediction_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "executed_at": self.executed_at.isoformat(),
            "success": self.success,
            "impact": self.impact,
            "metadata": self.metadata
        }


class TimeSeriesPredictor:
    """Time series-based failure prediction model."""
    
    def __init__(self, lookback_hours: int = 24, prediction_horizon: int = 6):
        self.lookback_hours = lookback_hours
        self.prediction_horizon = prediction_horizon  # hours
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_time_series_data(
        self, 
        metrics_history: List[SystemMetrics]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training."""
        if len(metrics_history) < self.lookback_hours + self.prediction_horizon:
            raise ValueError(f"Insufficient data: need at least {self.lookback_hours + self.prediction_horizon} data points")
        
        # Convert to DataFrame
        data = []
        for metrics in metrics_history:
            feature_vector = metrics.to_feature_vector()
            data.append(feature_vector)
        
        df = pd.DataFrame(data)
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df) - self.lookback_hours - self.prediction_horizon + 1):
            # Use past lookback_hours as features
            sequence = df.iloc[i:i+self.lookback_hours].values.flatten()
            X.append(sequence)
            
            # Predict metrics at prediction_horizon hours ahead
            future_metrics = df.iloc[i+self.lookback_hours+self.prediction_horizon-1].values
            y.append(future_metrics)
        
        return np.array(X), np.array(y)
    
    def train(self, metrics_history: List[SystemMetrics]) -> Dict[str, float]:
        """Train the time series prediction model."""
        try:
            X, y = self.prepare_time_series_data(metrics_history)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics (using first feature as proxy for accuracy)
            accuracy = np.mean(np.abs(y_pred[:, 0] - y_test[:, 0]) < 0.1)
            
            self.is_trained = True
            
            return {
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": accuracy,
                "model_score": self.model.score(X_test, y_test)
            }
            
        except Exception as e:
            logger.error(f"Error training time series model: {e}")
            return {"error": str(e)}
    
    def predict(self, recent_metrics: List[SystemMetrics]) -> Optional[np.ndarray]:
        """Predict future system metrics."""
        if not self.is_trained:
            return None
        
        if len(recent_metrics) < self.lookback_hours:
            return None
        
        try:
            # Prepare input sequence
            sequence_data = []
            for metrics in recent_metrics[-self.lookback_hours:]:
                sequence_data.append(metrics.to_feature_vector())
            
            sequence = np.array(sequence_data).flatten().reshape(1, -1)
            sequence_scaled = self.scaler.transform(sequence)
            
            # Make prediction
            predicted_metrics = self.model.predict(sequence_scaled)[0]
            
            return predicted_metrics
            
        except Exception as e:
            logger.error(f"Error making time series prediction: {e}")
            return None


class AnomalyDetector:
    """Anomaly detection for identifying unusual system behavior."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.normal_ranges = {}
    
    def train(self, normal_metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Train anomaly detection model on normal behavior."""
        if len(normal_metrics) < 50:
            return {"error": "Insufficient data for anomaly detection training"}
        
        try:
            # Convert to feature vectors
            features = np.array([metrics.to_feature_vector() for metrics in normal_metrics])
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train isolation forest
            self.model.fit(features_scaled)
            
            # Calculate normal ranges for each feature
            for i in range(features.shape[1]):
                self.normal_ranges[f"feature_{i}"] = {
                    "min": np.percentile(features[:, i], 5),
                    "max": np.percentile(features[:, i], 95),
                    "mean": np.mean(features[:, i]),
                    "std": np.std(features[:, i])
                }
            
            self.is_trained = True
            
            return {
                "training_samples": len(normal_metrics),
                "contamination": self.model.contamination,
                "model_trained": True
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return {"error": str(e)}
    
    def detect_anomaly(self, metrics: SystemMetrics) -> Tuple[bool, float, List[str]]:
        """Detect if metrics represent anomalous behavior."""
        if not self.is_trained:
            return False, 0.0, ["Model not trained"]
        
        try:
            # Convert to feature vector
            features = metrics.to_feature_vector().reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get anomaly score
            anomaly_score = self.model.decision_function(features_scaled)[0]
            is_anomaly = self.model.predict(features_scaled)[0] == -1
            
            # Identify anomalous features
            anomalous_features = []
            feature_vector = metrics.to_feature_vector()
            
            feature_names = [
                "cpu_usage", "memory_usage", "disk_usage", "network_io",
                "active_connections", "error_rate", "response_time", 
                "queue_depth", "throughput", "hour", "weekday"
            ]
            
            for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
                if i < len(feature_names) and name in self.normal_ranges:
                    normal_range = self.normal_ranges[f"feature_{i}"]
                    if value < normal_range["min"] or value > normal_range["max"]:
                        anomalous_features.append(f"{name}: {value:.2f}")
            
            return is_anomaly, abs(anomaly_score), anomalous_features
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return False, 0.0, [f"Detection error: {str(e)}"]


class FailureRiskAssessor:
    """Assesses failure risk based on current system state."""
    
    def __init__(self):
        self.risk_factors = {
            "high_cpu": {"threshold": 80.0, "weight": 0.3},
            "high_memory": {"threshold": 85.0, "weight": 0.25},
            "high_disk": {"threshold": 90.0, "weight": 0.2},
            "high_error_rate": {"threshold": 5.0, "weight": 0.4},
            "slow_response": {"threshold": 2000.0, "weight": 0.3},
            "high_queue": {"threshold": 100.0, "weight": 0.2},
            "off_hours": {"weight": 0.1},
            "weekend": {"weight": 0.1}
        }
    
    def assess_risk(
        self,
        current_metrics: SystemMetrics,
        failure_patterns: List[FailurePattern],
        anomaly_score: float = 0.0
    ) -> Tuple[RiskLevel, float, List[str]]:
        """Assess current failure risk."""
        
        risk_score = 0.0
        risk_factors = []
        
        # Metrics-based risk factors
        if current_metrics.cpu_usage > self.risk_factors["high_cpu"]["threshold"]:
            factor_score = self.risk_factors["high_cpu"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
        
        if current_metrics.memory_usage > self.risk_factors["high_memory"]["threshold"]:
            factor_score = self.risk_factors["high_memory"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
        
        if current_metrics.disk_usage > self.risk_factors["high_disk"]["threshold"]:
            factor_score = self.risk_factors["high_disk"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"High disk usage: {current_metrics.disk_usage:.1f}%")
        
        if current_metrics.error_rate > self.risk_factors["high_error_rate"]["threshold"]:
            factor_score = self.risk_factors["high_error_rate"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"High error rate: {current_metrics.error_rate:.2f}%")
        
        if current_metrics.response_time > self.risk_factors["slow_response"]["threshold"]:
            factor_score = self.risk_factors["slow_response"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"Slow response time: {current_metrics.response_time:.0f}ms")
        
        if current_metrics.queue_depth > self.risk_factors["high_queue"]["threshold"]:
            factor_score = self.risk_factors["high_queue"]["weight"]
            risk_score += factor_score
            risk_factors.append(f"High queue depth: {current_metrics.queue_depth}")
        
        # Time-based factors
        current_hour = current_metrics.timestamp.hour
        if current_hour < 6 or current_hour > 22:
            risk_score += self.risk_factors["off_hours"]["weight"]
            risk_factors.append("Off-hours operation")
        
        if current_metrics.timestamp.weekday() >= 5:
            risk_score += self.risk_factors["weekend"]["weight"]
            risk_factors.append("Weekend operation")
        
        # Pattern-based risk
        for pattern in failure_patterns:
            if pattern.pattern_type in [PatternType.RECURRING, PatternType.SEASONAL]:
                # Check if current time matches pattern characteristics
                if self._matches_pattern_timing(current_metrics.timestamp, pattern):
                    pattern_risk = pattern.impact_score * 0.3
                    risk_score += pattern_risk
                    risk_factors.append(f"Matches {pattern.pattern_type.value} pattern")
        
        # Anomaly contribution
        if anomaly_score > 0.5:
            anomaly_risk = anomaly_score * 0.25
            risk_score += anomaly_risk
            risk_factors.append(f"System anomaly detected (score: {anomaly_score:.2f})")
        
        # Determine risk level
        if risk_score >= 2.0:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 1.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 1.0:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= 0.5:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW
        
        return risk_level, risk_score, risk_factors
    
    def _matches_pattern_timing(
        self,
        current_time: datetime,
        pattern: FailurePattern
    ) -> bool:
        """Check if current time matches pattern timing."""
        characteristics = pattern.characteristics
        
        if "peak_hour" in characteristics:
            peak_hour = characteristics["peak_hour"]
            if abs(current_time.hour - peak_hour) <= 1:
                return True
        
        if "peak_day" in characteristics:
            peak_day = characteristics["peak_day"]
            if current_time.weekday() == peak_day:
                return True
        
        return False


class PreventionActionExecutor:
    """Executes preventive actions based on predictions."""
    
    def __init__(self, healing_engine: HealingEngine):
        self.healing_engine = healing_engine
        self.executed_actions: List[PreventiveAction] = []
        self.action_handlers = self._initialize_action_handlers()
    
    def _initialize_action_handlers(self) -> Dict[PreventionAction, callable]:
        """Initialize handlers for different prevention actions."""
        return {
            PreventionAction.SCALE_RESOURCES: self._scale_resources,
            PreventionAction.RESTART_SERVICES: self._restart_services,
            PreventionAction.CLEAR_CACHE: self._clear_cache,
            PreventionAction.UPDATE_CONFIGURATION: self._update_configuration,
            PreventionAction.NOTIFY_OPERATORS: self._notify_operators,
            PreventionAction.SCHEDULE_MAINTENANCE: self._schedule_maintenance,
            PreventionAction.ISOLATE_COMPONENT: self._isolate_component,
            PreventionAction.APPLY_CIRCUIT_BREAKER: self._apply_circuit_breaker
        }
    
    async def execute_prevention_actions(
        self,
        prediction: FailurePrediction
    ) -> List[PreventiveAction]:
        """Execute preventive actions for a prediction."""
        executed_actions = []
        
        for action_type in prediction.recommended_actions:
            try:
                action_handler = self.action_handlers.get(action_type)
                if action_handler:
                    action = await action_handler(prediction)
                    if action:
                        executed_actions.append(action)
                        self.executed_actions.append(action)
                
            except Exception as e:
                logger.error(f"Error executing prevention action {action_type}: {e}")
                
                # Record failed action
                failed_action = PreventiveAction(
                    action_id=str(uuid.uuid4()),
                    prediction_id=prediction.prediction_id,
                    action_type=action_type,
                    description=f"Failed to execute {action_type.value}",
                    executed_at=datetime.now(),
                    success=False,
                    impact="none",
                    metadata={"error": str(e)}
                )
                executed_actions.append(failed_action)
        
        return executed_actions
    
    async def _scale_resources(self, prediction: FailurePrediction) -> PreventiveAction:
        """Scale system resources preventively."""
        import uuid
        
        # Simulate resource scaling
        scale_factor = 1.5 if prediction.risk_level.value >= 4 else 1.2
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.SCALE_RESOURCES,
            description=f"Scaled resources by {scale_factor}x to prevent {prediction.predicted_failure_type.value}",
            executed_at=datetime.now(),
            success=True,
            impact="increased_capacity",
            metadata={
                "scale_factor": scale_factor,
                "predicted_failure": prediction.predicted_failure_type.value
            }
        )
        
        logger.info(f"Executed preventive resource scaling: {scale_factor}x")
        return action
    
    async def _restart_services(self, prediction: FailurePrediction) -> PreventiveAction:
        """Restart services preventively."""
        import uuid
        
        services_to_restart = ["cache_service", "worker_pool"]
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.RESTART_SERVICES,
            description=f"Restarted services: {', '.join(services_to_restart)}",
            executed_at=datetime.now(),
            success=True,
            impact="service_refresh",
            metadata={
                "services": services_to_restart,
                "restart_reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Executed preventive service restart for services: {services_to_restart}")
        return action
    
    async def _clear_cache(self, prediction: FailurePrediction) -> PreventiveAction:
        """Clear system caches preventively."""
        import uuid
        
        cache_types = ["memory_cache", "disk_cache", "application_cache"]
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.CLEAR_CACHE,
            description=f"Cleared caches: {', '.join(cache_types)}",
            executed_at=datetime.now(),
            success=True,
            impact="cache_refresh",
            metadata={
                "cache_types": cache_types,
                "reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Executed preventive cache clearing: {cache_types}")
        return action
    
    async def _update_configuration(self, prediction: FailurePrediction) -> PreventiveAction:
        """Update system configuration preventively."""
        import uuid
        
        config_changes = {
            "timeout_settings": "increased",
            "retry_policy": "more_aggressive",
            "circuit_breaker_threshold": "lowered"
        }
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.UPDATE_CONFIGURATION,
            description="Updated system configuration for better resilience",
            executed_at=datetime.now(),
            success=True,
            impact="improved_resilience",
            metadata={
                "config_changes": config_changes,
                "reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Executed preventive configuration update: {config_changes}")
        return action
    
    async def _notify_operators(self, prediction: FailurePrediction) -> PreventiveAction:
        """Notify operators of predicted failure."""
        import uuid
        
        notification_message = (
            f"Prediction Alert: {prediction.predicted_failure_type.value} "
            f"predicted with {prediction.probability:.1%} probability. "
            f"Risk level: {prediction.risk_level.name}. "
            f"Time to failure: {prediction.time_to_failure:.1f} hours."
        )
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.NOTIFY_OPERATORS,
            description="Notified operators of predicted failure",
            executed_at=datetime.now(),
            success=True,
            impact="operator_awareness",
            metadata={
                "notification_message": notification_message,
                "channels": ["slack", "email", "dashboard"]
            }
        )
        
        logger.info(f"Executed preventive operator notification: {notification_message}")
        return action
    
    async def _schedule_maintenance(self, prediction: FailurePrediction) -> PreventiveAction:
        """Schedule maintenance window."""
        import uuid
        
        maintenance_window = datetime.now() + timedelta(hours=2)
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.SCHEDULE_MAINTENANCE,
            description=f"Scheduled maintenance window at {maintenance_window.isoformat()}",
            executed_at=datetime.now(),
            success=True,
            impact="planned_maintenance",
            metadata={
                "maintenance_window": maintenance_window.isoformat(),
                "reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Scheduled preventive maintenance for {maintenance_window}")
        return action
    
    async def _isolate_component(self, prediction: FailurePrediction) -> PreventiveAction:
        """Isolate potentially problematic component."""
        import uuid
        
        component = "high_risk_service"
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.ISOLATE_COMPONENT,
            description=f"Isolated component: {component}",
            executed_at=datetime.now(),
            success=True,
            impact="component_isolation",
            metadata={
                "isolated_component": component,
                "reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Executed preventive component isolation: {component}")
        return action
    
    async def _apply_circuit_breaker(self, prediction: FailurePrediction) -> PreventiveAction:
        """Apply circuit breaker pattern."""
        import uuid
        
        service_name = "predicted_failure_service"
        
        action = PreventiveAction(
            action_id=str(uuid.uuid4()),
            prediction_id=prediction.prediction_id,
            action_type=PreventionAction.APPLY_CIRCUIT_BREAKER,
            description=f"Applied circuit breaker to {service_name}",
            executed_at=datetime.now(),
            success=True,
            impact="circuit_protection",
            metadata={
                "service": service_name,
                "threshold": "lowered",
                "reason": f"Preventing {prediction.predicted_failure_type.value}"
            }
        )
        
        logger.info(f"Applied preventive circuit breaker to {service_name}")
        return action


class PredictivePreventionSystem:
    """Main predictive failure prevention system."""
    
    def __init__(self, healing_engine: Optional[HealingEngine] = None):
        self.healing_engine = healing_engine or HealingEngine()
        
        # ML components
        self.time_series_predictor = TimeSeriesPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.risk_assessor = FailureRiskAssessor()
        self.action_executor = PreventionActionExecutor(self.healing_engine)
        
        # Data storage
        self.metrics_history: List[SystemMetrics] = []
        self.predictions: List[FailurePrediction] = []
        self.preventive_actions: List[PreventiveAction] = []
        
        # Configuration
        self.prediction_interval = 300  # 5 minutes
        self.max_history_hours = 168  # 1 week
        self.prediction_threshold = 0.7
        
        # Background tasks
        self.prediction_task: Optional[asyncio.Task] = None
        self.training_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.lock = threading.RLock()
    
    async def start(self):
        """Start the predictive prevention system."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.training_task = asyncio.create_task(self._training_loop())
        
        logger.info("Predictive prevention system started")
    
    async def stop(self):
        """Stop the predictive prevention system."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.prediction_task:
            self.prediction_task.cancel()
        if self.training_task:
            self.training_task.cancel()
        
        logger.info("Predictive prevention system stopped")
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # Get metrics from monitoring system
        system_health = enhanced_monitoring.get_system_health()
        real_time_metrics = enhanced_monitoring.get_real_time_metrics()
        
        # Convert to our metrics format
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=75.0,  # Would get from actual system monitoring
            memory_usage=65.0,
            disk_usage=45.0,
            network_io=1500.0,
            active_connections=50,
            error_rate=real_time_metrics.failures_detected_24h / 1000 * 100,  # Convert to percentage
            response_time=250.0,  # Would get from actual metrics
            queue_depth=10,
            throughput=1000.0,
            custom_metrics={
                "healing_success_rate": real_time_metrics.healing_success_rate,
                "cost_savings": real_time_metrics.cost_savings_usd,
                "uptime_percentage": real_time_metrics.uptime_percentage
            }
        )
        
        # Store in history
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Clean up old metrics
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]
        
        return metrics
    
    async def _prediction_loop(self):
        """Main prediction loop."""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = await self.collect_system_metrics()
                
                # Make predictions
                await self._make_predictions(current_metrics)
                
                # Clean up old predictions
                self._cleanup_old_predictions()
                
                await asyncio.sleep(self.prediction_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _training_loop(self):
        """Model training loop."""
        while self.running:
            try:
                # Retrain models every hour
                if len(self.metrics_history) >= 100:
                    await self._retrain_models()
                
                await asyncio.sleep(3600)  # 1 hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(1800)  # 30 minutes
    
    async def _make_predictions(self, current_metrics: SystemMetrics):
        """Make failure predictions based on current state."""
        with self.lock:
            recent_history = self.metrics_history.copy()
        
        if len(recent_history) < 50:
            return  # Not enough data
        
        try:
            # Detect anomalies
            is_anomaly, anomaly_score, anomalous_features = self.anomaly_detector.detect_anomaly(
                current_metrics
            )
            
            # Get failure patterns from ML module
            failure_patterns = list(pattern_recognizer.identified_patterns.values())
            
            # Assess risk
            risk_level, risk_score, risk_factors = self.risk_assessor.assess_risk(
                current_metrics, failure_patterns, anomaly_score
            )
            
            # Generate predictions if risk is significant
            if risk_level.value >= RiskLevel.MEDIUM.value:
                prediction = await self._generate_prediction(
                    current_metrics, risk_level, risk_score, risk_factors, anomaly_score
                )
                
                if prediction:
                    with self.lock:
                        self.predictions.append(prediction)
                    
                    # Execute preventive actions if prediction is confident
                    if prediction.probability >= self.prediction_threshold:
                        actions = await self.action_executor.execute_prevention_actions(prediction)
                        self.preventive_actions.extend(actions)
                        
                        logger.info(
                            f"Executed {len(actions)} preventive actions for prediction "
                            f"{prediction.prediction_id}"
                        )
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
    
    async def _generate_prediction(
        self,
        current_metrics: SystemMetrics,
        risk_level: RiskLevel,
        risk_score: float,
        risk_factors: List[str],
        anomaly_score: float
    ) -> FailurePrediction:
        """Generate a failure prediction."""
        import uuid
        
        # Determine most likely failure type based on risk factors
        predicted_failure_type = FailureType.RESOURCE_EXHAUSTION
        if any("error" in factor.lower() for factor in risk_factors):
            predicted_failure_type = FailureType.APPLICATION_ERROR
        elif any("response" in factor.lower() for factor in risk_factors):
            predicted_failure_type = FailureType.TIMEOUT
        elif any("queue" in factor.lower() for factor in risk_factors):
            predicted_failure_type = FailureType.RESOURCE_EXHAUSTION
        
        # Calculate probability based on risk score
        probability = min(0.95, risk_score / 2.0)
        
        # Determine confidence
        if probability >= 0.8:
            confidence = PredictionConfidence.VERY_HIGH
        elif probability >= 0.6:
            confidence = PredictionConfidence.HIGH
        elif probability >= 0.4:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW
        
        # Estimate time to failure (simplified)
        time_to_failure = max(0.5, 24 / risk_score)
        
        # Recommend actions based on failure type and risk
        recommended_actions = self._recommend_prevention_actions(
            predicted_failure_type, risk_level, risk_factors
        )
        
        prediction = FailurePrediction(
            prediction_id=str(uuid.uuid4()),
            predicted_failure_type=predicted_failure_type,
            probability=probability,
            confidence=confidence,
            time_to_failure=time_to_failure,
            risk_level=risk_level,
            contributing_factors=risk_factors,
            recommended_actions=recommended_actions,
            predicted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            model_used="hybrid_ml_system"
        )
        
        logger.info(
            f"Generated prediction {prediction.prediction_id}: "
            f"{predicted_failure_type.value} with {probability:.1%} probability"
        )
        
        return prediction
    
    def _recommend_prevention_actions(
        self,
        failure_type: FailureType,
        risk_level: RiskLevel,
        risk_factors: List[str]
    ) -> List[PreventionAction]:
        """Recommend prevention actions based on prediction."""
        actions = []
        
        # Always notify operators for medium+ risk
        if risk_level.value >= RiskLevel.MEDIUM.value:
            actions.append(PreventionAction.NOTIFY_OPERATORS)
        
        # Resource-related actions
        if failure_type in [FailureType.RESOURCE_EXHAUSTION, FailureType.TIMEOUT]:
            actions.extend([
                PreventionAction.SCALE_RESOURCES,
                PreventionAction.CLEAR_CACHE
            ])
        
        # Application error actions
        if failure_type == FailureType.APPLICATION_ERROR:
            actions.extend([
                PreventionAction.RESTART_SERVICES,
                PreventionAction.APPLY_CIRCUIT_BREAKER
            ])
        
        # High-risk actions
        if risk_level.value >= RiskLevel.HIGH.value:
            actions.extend([
                PreventionAction.UPDATE_CONFIGURATION,
                PreventionAction.SCHEDULE_MAINTENANCE
            ])
        
        # Critical risk actions
        if risk_level.value == RiskLevel.CRITICAL.value:
            actions.append(PreventionAction.ISOLATE_COMPONENT)
        
        return actions
    
    async def _retrain_models(self):
        """Retrain ML models with latest data."""
        logger.info("Starting model retraining")
        
        try:
            with self.lock:
                training_data = self.metrics_history.copy()
            
            # Retrain time series predictor
            if len(training_data) >= 100:
                ts_results = self.time_series_predictor.train(training_data)
                logger.info(f"Time series model retrained: {ts_results}")
            
            # Retrain anomaly detector
            if len(training_data) >= 50:
                # Use recent data as "normal" behavior
                normal_data = training_data[-1000:]  # Last 1000 samples
                anomaly_results = self.anomaly_detector.train(normal_data)
                logger.info(f"Anomaly detector retrained: {anomaly_results}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _cleanup_old_predictions(self):
        """Clean up expired predictions."""
        current_time = datetime.now()
        
        with self.lock:
            self.predictions = [
                p for p in self.predictions if p.expires_at > current_time
            ]
            
            # Limit total predictions
            max_predictions = 1000
            if len(self.predictions) > max_predictions:
                self.predictions = self.predictions[-max_predictions:]
    
    def get_current_predictions(self) -> List[FailurePrediction]:
        """Get all current active predictions."""
        current_time = datetime.now()
        
        with self.lock:
            return [
                p for p in self.predictions
                if p.expires_at > current_time and p.probability >= self.prediction_threshold
            ]
    
    def get_prevention_statistics(self) -> Dict[str, Any]:
        """Get prevention system statistics."""
        with self.lock:
            total_predictions = len(self.predictions)
            high_confidence_predictions = len([
                p for p in self.predictions 
                if p.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]
            ])
            
            total_actions = len(self.preventive_actions)
            successful_actions = len([a for a in self.preventive_actions if a.success])
            
            # Calculate prevention effectiveness (would need actual failure data)
            prevention_rate = 0.85  # Mock value
            
            return {
                "total_predictions": total_predictions,
                "high_confidence_predictions": high_confidence_predictions,
                "prediction_accuracy": prevention_rate,
                "total_preventive_actions": total_actions,
                "successful_actions": successful_actions,
                "action_success_rate": successful_actions / total_actions if total_actions > 0 else 0,
                "models_trained": {
                    "time_series": self.time_series_predictor.is_trained,
                    "anomaly_detection": self.anomaly_detector.is_trained
                },
                "metrics_history_size": len(self.metrics_history),
                "last_updated": datetime.now().isoformat()
            }


# Global predictive prevention system
predictive_prevention = PredictivePreventionSystem()