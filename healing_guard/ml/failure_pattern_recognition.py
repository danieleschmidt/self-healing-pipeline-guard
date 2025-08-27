"""Advanced ML-based failure pattern recognition system.

Uses machine learning models to identify, classify, and predict pipeline failures
with high accuracy to enable proactive healing strategies.
"""

import asyncio
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import DBSCAN, KMeans
import joblib

from ..core.failure_detector import FailureEvent, FailureType, SeverityLevel
from ..monitoring.enhanced_monitoring import enhanced_monitoring

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of failure patterns."""
    RECURRING = "recurring"
    SEASONAL = "seasonal"  
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    TIMING = "timing"
    FLAKY = "flaky"
    CASCADE = "cascade"
    ANOMALY = "anomaly"


class ModelType(Enum):
    """Types of ML models used."""
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NATURAL_LANGUAGE = "nlp"


@dataclass
class FailurePattern:
    """Identified failure pattern."""
    id: str
    pattern_type: PatternType
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    description: str
    characteristics: Dict[str, Any] = field(default_factory=dict)
    related_failures: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    

@dataclass
class ModelPerformance:
    """ML model performance metrics."""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    last_trained: datetime
    sample_count: int


@dataclass
class PredictionResult:
    """Failure prediction result."""
    failure_type: FailureType
    probability: float
    confidence: float
    time_to_failure: Optional[float]  # hours
    risk_factors: List[str]
    recommended_actions: List[str]
    model_used: str


class FeatureExtractor:
    """Extract features from failure events for ML models."""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False
    
    def extract_features(self, failure_events: List[FailureEvent]) -> pd.DataFrame:
        """Extract features from failure events."""
        features = []
        
        for event in failure_events:
            # Basic features
            feature_row = {
                'failure_type': event.failure_type.value,
                'severity': event.severity.value,
                'branch': event.branch or 'unknown',
                'pipeline_stage': event.pipeline_stage or 'unknown',
                'duration_minutes': event.duration_minutes,
                'hour_of_day': event.timestamp.hour,
                'day_of_week': event.timestamp.weekday(),
                'day_of_month': event.timestamp.day,
                'month': event.timestamp.month,
                'is_weekend': event.timestamp.weekday() >= 5,
                'log_length': len(event.raw_logs),
                'remediation_count': len(event.remediation_suggestions)
            }
            
            # Text-based features (will be processed separately)
            feature_row['log_text'] = event.raw_logs
            feature_row['job_name'] = event.job_name or 'unknown'
            
            # Context features
            if event.context:
                feature_row.update({
                    f'context_{k}': v for k, v in event.context.items()
                    if isinstance(v, (int, float, bool, str))
                })
            
            features.append(feature_row)
        
        df = pd.DataFrame(features)
        return self._process_features(df)
    
    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and encode features."""
        processed_df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['failure_type', 'severity', 'branch', 'pipeline_stage', 'job_name']
        
        for col in categorical_cols:
            if col in processed_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    if not self.fitted:
                        self.label_encoders[col].fit(processed_df[col].astype(str))
                
                try:
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    processed_df[col] = 0
        
        # Process text features separately (for NLP models)
        text_columns = ['log_text']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df.drop(columns=[col], inplace=True)
        
        # Fill missing values
        processed_df.fillna(0, inplace=True)
        
        return processed_df
    
    def fit(self, failure_events: List[FailureEvent]):
        """Fit the feature extractor on training data."""
        df = self.extract_features(failure_events)
        
        # Fit text vectorizer on log texts
        log_texts = [event.raw_logs for event in failure_events]
        if log_texts:
            self.text_vectorizer.fit(log_texts)
        
        # Fit scaler on numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.scaler.fit(df[numerical_cols])
        
        self.fitted = True
        logger.info("Feature extractor fitted on training data")
    
    def extract_text_features(self, failure_events: List[FailureEvent]) -> np.ndarray:
        """Extract TF-IDF features from log texts."""
        log_texts = [event.raw_logs for event in failure_events]
        return self.text_vectorizer.transform(log_texts).toarray()


class FailurePatternRecognizer:
    """Advanced ML-based failure pattern recognition."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # ML models
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.clustering_model = DBSCAN(
            eps=0.5,
            min_samples=3
        )
        self.time_series_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        
        # Training data and patterns
        self.training_data: List[FailureEvent] = []
        self.identified_patterns: Dict[str, FailurePattern] = {}
        self.model_performance: Dict[ModelType, ModelPerformance] = {}
        
        # Configuration
        self.min_pattern_occurrences = 3
        self.anomaly_threshold = 0.1
        self.prediction_horizon_hours = 24
        
        # Load existing models if available
        self._load_models()
    
    def add_failure_event(self, failure_event: FailureEvent):
        """Add a new failure event to the training data."""
        self.training_data.append(failure_event)
        
        # Maintain rolling window of training data
        max_training_samples = 10000
        if len(self.training_data) > max_training_samples:
            self.training_data = self.training_data[-max_training_samples:]
        
        logger.debug(f"Added failure event {failure_event.id} to training data")
    
    async def train_models(self, retrain_all: bool = False) -> Dict[ModelType, ModelPerformance]:
        """Train all ML models with current data."""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for ML models")
            return {}
        
        logger.info(f"Training ML models with {len(self.training_data)} failure events")
        
        # Extract features
        feature_df = self.feature_extractor.extract_features(self.training_data)
        text_features = self.feature_extractor.extract_text_features(self.training_data)
        
        performance_results = {}
        
        # Train classification model
        try:
            perf = await self._train_classification_model(feature_df, text_features)
            if perf:
                performance_results[ModelType.CLASSIFICATION] = perf
        except Exception as e:
            logger.error(f"Error training classification model: {e}")
        
        # Train anomaly detection model
        try:
            perf = await self._train_anomaly_model(feature_df)
            if perf:
                performance_results[ModelType.ANOMALY_DETECTION] = perf
        except Exception as e:
            logger.error(f"Error training anomaly model: {e}")
        
        # Train clustering model
        try:
            perf = await self._train_clustering_model(feature_df)
            if perf:
                performance_results[ModelType.CLUSTERING] = perf
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
        
        # Save models
        self._save_models()
        
        # Update patterns
        await self._update_failure_patterns()
        
        self.model_performance.update(performance_results)
        logger.info(f"Completed training {len(performance_results)} models")
        
        return performance_results
    
    async def _train_classification_model(
        self, 
        features: pd.DataFrame, 
        text_features: np.ndarray
    ) -> Optional[ModelPerformance]:
        """Train the failure type classification model."""
        start_time = datetime.now()
        
        # Prepare labels
        labels = [event.failure_type.value for event in self.training_data]
        
        if len(set(labels)) < 2:
            logger.warning("Insufficient label diversity for classification")
            return None
        
        # Combine features
        numerical_features = self.feature_extractor.scaler.transform(
            features.select_dtypes(include=[np.number])
        )
        combined_features = np.hstack([numerical_features, text_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.classification_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classification_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPerformance(
            model_type=ModelType.CLASSIFICATION,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            last_trained=datetime.now(),
            sample_count=len(self.training_data)
        )
    
    async def _train_anomaly_model(self, features: pd.DataFrame) -> Optional[ModelPerformance]:
        """Train the anomaly detection model."""
        start_time = datetime.now()
        
        numerical_features = self.feature_extractor.scaler.transform(
            features.select_dtypes(include=[np.number])
        )
        
        # Train model
        self.anomaly_model.fit(numerical_features)
        
        # Evaluate (using prediction on training data as proxy)
        predictions = self.anomaly_model.predict(numerical_features)
        anomaly_rate = (predictions == -1).mean()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPerformance(
            model_type=ModelType.ANOMALY_DETECTION,
            accuracy=1.0 - anomaly_rate,  # Proxy metric
            precision=0.8,  # Estimated
            recall=0.7,   # Estimated
            f1_score=0.75,  # Estimated
            training_time=training_time,
            last_trained=datetime.now(),
            sample_count=len(self.training_data)
        )
    
    async def _train_clustering_model(self, features: pd.DataFrame) -> Optional[ModelPerformance]:
        """Train the clustering model for pattern discovery."""
        start_time = datetime.now()
        
        numerical_features = self.feature_extractor.scaler.transform(
            features.select_dtypes(include=[np.number])
        )
        
        # Fit clustering model
        cluster_labels = self.clustering_model.fit_predict(numerical_features)
        
        # Evaluate clustering quality
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Use silhouette score if we have enough clusters
        cluster_quality = 0.5  # Default
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            cluster_quality = silhouette_score(numerical_features, cluster_labels)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPerformance(
            model_type=ModelType.CLUSTERING,
            accuracy=cluster_quality,
            precision=0.6,  # Estimated
            recall=0.6,   # Estimated
            f1_score=0.6,  # Estimated
            training_time=training_time,
            last_trained=datetime.now(),
            sample_count=len(self.training_data)
        )
    
    async def _update_failure_patterns(self):
        """Update identified failure patterns based on clustering results."""
        if len(self.training_data) < self.min_pattern_occurrences:
            return
        
        # Extract features for clustering
        feature_df = self.feature_extractor.extract_features(self.training_data)
        numerical_features = self.feature_extractor.scaler.transform(
            feature_df.select_dtypes(include=[np.number])
        )
        
        # Get cluster labels
        cluster_labels = self.clustering_model.fit_predict(numerical_features)
        
        # Analyze clusters to identify patterns
        patterns = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get events in this cluster
            cluster_events = [
                self.training_data[i] for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]
            
            if len(cluster_events) >= self.min_pattern_occurrences:
                pattern = self._analyze_cluster_pattern(cluster_id, cluster_events)
                if pattern:
                    patterns[pattern.id] = pattern
        
        # Update pattern registry
        self.identified_patterns.update(patterns)
        
        logger.info(f"Updated {len(patterns)} failure patterns")
    
    def _analyze_cluster_pattern(
        self, 
        cluster_id: int, 
        events: List[FailureEvent]
    ) -> Optional[FailurePattern]:
        """Analyze a cluster of events to identify patterns."""
        if not events:
            return None
        
        # Basic statistics
        failure_types = [event.failure_type for event in events]
        severities = [event.severity for event in events]
        branches = [event.branch for event in events if event.branch]
        
        # Find common characteristics
        most_common_type = max(set(failure_types), key=failure_types.count)
        most_common_severity = max(set(severities), key=severities.count)
        
        # Determine pattern type based on characteristics
        pattern_type = self._classify_pattern_type(events)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(events)
        
        # Generate suggestions
        suggestions = self._generate_pattern_suggestions(pattern_type, events)
        
        pattern = FailurePattern(
            id=f"pattern_{cluster_id}_{int(datetime.now().timestamp())}",
            pattern_type=pattern_type,
            confidence=min(0.95, len(events) / 10),  # Confidence based on frequency
            frequency=len(events),
            first_seen=min(event.timestamp for event in events),
            last_seen=max(event.timestamp for event in events),
            description=f"{pattern_type.value.title()} pattern with {len(events)} occurrences",
            characteristics={
                "most_common_type": most_common_type.value,
                "most_common_severity": most_common_severity.value,
                "avg_duration": np.mean([e.duration_minutes for e in events]),
                "branches_affected": list(set(branches)),
                "time_distribution": self._analyze_time_distribution(events)
            },
            related_failures=[event.id for event in events],
            suggested_actions=suggestions,
            impact_score=impact_score
        )
        
        return pattern
    
    def _classify_pattern_type(self, events: List[FailureEvent]) -> PatternType:
        """Classify the type of failure pattern."""
        # Time-based analysis
        timestamps = [event.timestamp for event in events]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                     for i in range(1, len(timestamps))]
        
        # Check for recurring pattern (similar time intervals)
        if len(time_diffs) > 1:
            avg_interval = np.mean(time_diffs)
            if avg_interval < 3600:  # Less than 1 hour
                return PatternType.FLAKY
            elif 3600 <= avg_interval <= 86400:  # 1 hour to 1 day
                return PatternType.RECURRING
        
        # Check for seasonal pattern
        hours = [event.timestamp.hour for event in events]
        if len(set(hours)) <= 3:  # Concentrated in few hours
            return PatternType.SEASONAL
        
        # Check for dependency pattern
        failure_types = [event.failure_type for event in events]
        if FailureType.DEPENDENCY_FAILURE in failure_types:
            return PatternType.DEPENDENCY
        
        # Check for resource pattern
        if FailureType.RESOURCE_EXHAUSTION in failure_types:
            return PatternType.RESOURCE
        
        # Default to anomaly
        return PatternType.ANOMALY
    
    def _calculate_impact_score(self, events: List[FailureEvent]) -> float:
        """Calculate the impact score of a pattern."""
        # Base score on frequency and severity
        frequency_score = min(1.0, len(events) / 10)
        
        # Severity contribution
        severity_weights = {
            SeverityLevel.LOW: 0.25,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.HIGH: 0.75,
            SeverityLevel.CRITICAL: 1.0
        }
        
        avg_severity = np.mean([
            severity_weights.get(event.severity, 0.5) for event in events
        ])
        
        # Duration contribution
        avg_duration = np.mean([event.duration_minutes for event in events])
        duration_score = min(1.0, avg_duration / 60)  # Normalize to 1 hour
        
        # Combined impact score
        impact_score = (frequency_score * 0.4 + avg_severity * 0.4 + duration_score * 0.2)
        
        return impact_score
    
    def _analyze_time_distribution(self, events: List[FailureEvent]) -> Dict[str, Any]:
        """Analyze temporal distribution of events."""
        hours = [event.timestamp.hour for event in events]
        days = [event.timestamp.weekday() for event in events]
        
        return {
            "peak_hour": max(set(hours), key=hours.count) if hours else None,
            "peak_day": max(set(days), key=days.count) if days else None,
            "weekend_ratio": sum(1 for d in days if d >= 5) / len(days) if days else 0,
            "night_ratio": sum(1 for h in hours if h < 6 or h > 22) / len(hours) if hours else 0
        }
    
    def _generate_pattern_suggestions(
        self, 
        pattern_type: PatternType, 
        events: List[FailureEvent]
    ) -> List[str]:
        """Generate remediation suggestions for a pattern."""
        suggestions = []
        
        if pattern_type == PatternType.RECURRING:
            suggestions.extend([
                "Schedule preventive maintenance before peak failure times",
                "Implement circuit breakers to prevent cascading failures",
                "Add monitoring alerts for pattern early detection"
            ])
        
        elif pattern_type == PatternType.FLAKY:
            suggestions.extend([
                "Increase retry logic with exponential backoff",
                "Implement test isolation to reduce interference",
                "Add more detailed logging for root cause analysis"
            ])
        
        elif pattern_type == PatternType.RESOURCE:
            suggestions.extend([
                "Implement auto-scaling based on resource usage",
                "Optimize resource allocation for peak times",
                "Add resource monitoring and alerting"
            ])
        
        elif pattern_type == PatternType.DEPENDENCY:
            suggestions.extend([
                "Implement service mesh for better dependency management",
                "Add circuit breakers for external dependencies",
                "Create dependency health checks"
            ])
        
        elif pattern_type == PatternType.SEASONAL:
            suggestions.extend([
                "Adjust resource allocation based on time patterns",
                "Schedule maintenance during low-activity periods",
                "Implement predictive scaling"
            ])
        
        else:
            suggestions.extend([
                "Increase monitoring and alerting",
                "Implement advanced error handling",
                "Consider architectural changes"
            ])
        
        return suggestions
    
    async def predict_failure_probability(
        self, 
        current_context: Dict[str, Any]
    ) -> PredictionResult:
        """Predict probability of failure given current context."""
        if not self.model_performance:
            # Return default prediction if models not trained
            return PredictionResult(
                failure_type=FailureType.UNKNOWN,
                probability=0.1,
                confidence=0.1,
                time_to_failure=None,
                risk_factors=["Insufficient training data"],
                recommended_actions=["Collect more failure data for better predictions"],
                model_used="default"
            )
        
        try:
            # Create synthetic failure event from context
            synthetic_event = self._create_synthetic_event(current_context)
            
            # Extract features
            feature_df = self.feature_extractor.extract_features([synthetic_event])
            text_features = self.feature_extractor.extract_text_features([synthetic_event])
            
            # Predict using classification model
            numerical_features = self.feature_extractor.scaler.transform(
                feature_df.select_dtypes(include=[np.number])
            )
            combined_features = np.hstack([numerical_features, text_features])
            
            # Get prediction probabilities
            failure_probs = self.classification_model.predict_proba(combined_features)[0]
            failure_classes = self.classification_model.classes_
            
            # Find most likely failure type
            max_prob_idx = np.argmax(failure_probs)
            predicted_type = FailureType(failure_classes[max_prob_idx])
            probability = failure_probs[max_prob_idx]
            
            # Check for anomalies
            anomaly_score = self.anomaly_model.decision_function(numerical_features)[0]
            is_anomaly = anomaly_score < 0
            
            # Generate risk factors
            risk_factors = self._identify_risk_factors(current_context, synthetic_event)
            
            # Generate recommendations
            recommendations = self._get_predictive_recommendations(predicted_type, risk_factors)
            
            # Estimate time to failure (simplified heuristic)
            time_to_failure = self._estimate_time_to_failure(predicted_type, probability)
            
            return PredictionResult(
                failure_type=predicted_type,
                probability=float(probability),
                confidence=0.8 if not is_anomaly else 0.4,
                time_to_failure=time_to_failure,
                risk_factors=risk_factors,
                recommended_actions=recommendations,
                model_used="random_forest_classifier"
            )
        
        except Exception as e:
            logger.error(f"Error in failure prediction: {e}")
            return PredictionResult(
                failure_type=FailureType.UNKNOWN,
                probability=0.5,
                confidence=0.1,
                time_to_failure=None,
                risk_factors=["Prediction error"],
                recommended_actions=["Check system logs"],
                model_used="error_fallback"
            )
    
    def _create_synthetic_event(self, context: Dict[str, Any]) -> FailureEvent:
        """Create a synthetic failure event from context for prediction."""
        return FailureEvent(
            id="synthetic_prediction",
            timestamp=datetime.now(),
            job_name=context.get("job_name", "unknown"),
            job_id=context.get("job_id", "unknown"),
            failure_type=FailureType.UNKNOWN,
            severity=SeverityLevel.MEDIUM,
            raw_logs=context.get("logs", ""),
            branch=context.get("branch", "main"),
            pipeline_stage=context.get("stage", "unknown"),
            duration_minutes=context.get("duration", 5.0),
            context=context,
            remediation_suggestions=[]
        )
    
    def _identify_risk_factors(
        self, 
        context: Dict[str, Any], 
        event: FailureEvent
    ) -> List[str]:
        """Identify risk factors from current context."""
        risk_factors = []
        
        # Time-based risks
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_factors.append("Off-hours execution")
        
        if datetime.now().weekday() >= 5:
            risk_factors.append("Weekend execution")
        
        # Resource-based risks
        if context.get("cpu_usage", 0) > 80:
            risk_factors.append("High CPU usage")
        
        if context.get("memory_usage", 0) > 85:
            risk_factors.append("High memory usage")
        
        # Pattern-based risks
        for pattern in self.identified_patterns.values():
            if self._context_matches_pattern(context, pattern):
                risk_factors.append(f"Matches {pattern.pattern_type.value} pattern")
        
        return risk_factors
    
    def _context_matches_pattern(
        self, 
        context: Dict[str, Any], 
        pattern: FailurePattern
    ) -> bool:
        """Check if current context matches a known pattern."""
        # Simple matching based on common characteristics
        if pattern.characteristics.get("peak_hour"):
            if abs(datetime.now().hour - pattern.characteristics["peak_hour"]) <= 1:
                return True
        
        if context.get("branch") in pattern.characteristics.get("branches_affected", []):
            return True
        
        return False
    
    def _get_predictive_recommendations(
        self, 
        predicted_type: FailureType, 
        risk_factors: List[str]
    ) -> List[str]:
        """Get recommendations based on predicted failure type."""
        recommendations = []
        
        if predicted_type == FailureType.FLAKY_TEST:
            recommendations.extend([
                "Enable test retry mechanisms",
                "Isolate flaky tests",
                "Review test stability"
            ])
        
        elif predicted_type == FailureType.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Scale up resources preemptively",
                "Monitor resource usage closely",
                "Implement resource alerts"
            ])
        
        elif predicted_type == FailureType.DEPENDENCY_FAILURE:
            recommendations.extend([
                "Check external service health",
                "Implement circuit breakers",
                "Prepare fallback mechanisms"
            ])
        
        # Risk-specific recommendations
        if "High CPU usage" in risk_factors:
            recommendations.append("Consider reducing concurrent operations")
        
        if "High memory usage" in risk_factors:
            recommendations.append("Clear caches and optimize memory usage")
        
        return recommendations
    
    def _estimate_time_to_failure(
        self, 
        failure_type: FailureType, 
        probability: float
    ) -> Optional[float]:
        """Estimate time to failure based on type and probability."""
        # Simplified heuristic - in practice would use time series models
        base_time = {
            FailureType.FLAKY_TEST: 2.0,  # hours
            FailureType.RESOURCE_EXHAUSTION: 1.0,
            FailureType.DEPENDENCY_FAILURE: 4.0,
            FailureType.BUILD_FAILURE: 0.5,
            FailureType.DEPLOYMENT_FAILURE: 1.5
        }.get(failure_type, 3.0)
        
        # Adjust based on probability
        time_factor = 1.0 / (probability + 0.1)  # Higher probability = sooner failure
        
        return base_time * time_factor
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of identified patterns."""
        if not self.identified_patterns:
            return {"message": "No patterns identified yet"}
        
        patterns_by_type = {}
        for pattern in self.identified_patterns.values():
            pattern_type = pattern.pattern_type.value
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append({
                "id": pattern.id,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "impact_score": pattern.impact_score,
                "description": pattern.description
            })
        
        return {
            "total_patterns": len(self.identified_patterns),
            "patterns_by_type": patterns_by_type,
            "model_performance": {
                model_type.value: {
                    "accuracy": perf.accuracy,
                    "last_trained": perf.last_trained.isoformat()
                }
                for model_type, perf in self.model_performance.items()
            },
            "training_data_size": len(self.training_data)
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save models
            joblib.dump(self.classification_model, self.models_dir / "classification_model.pkl")
            joblib.dump(self.anomaly_model, self.models_dir / "anomaly_model.pkl")
            joblib.dump(self.clustering_model, self.models_dir / "clustering_model.pkl")
            
            # Save feature extractor
            with open(self.models_dir / "feature_extractor.pkl", "wb") as f:
                pickle.dump(self.feature_extractor, f)
            
            # Save patterns
            with open(self.models_dir / "patterns.json", "w") as f:
                patterns_data = {
                    pattern_id: {
                        "pattern_type": pattern.pattern_type.value,
                        "confidence": pattern.confidence,
                        "frequency": pattern.frequency,
                        "first_seen": pattern.first_seen.isoformat(),
                        "last_seen": pattern.last_seen.isoformat(),
                        "description": pattern.description,
                        "characteristics": pattern.characteristics,
                        "impact_score": pattern.impact_score,
                        "suggested_actions": pattern.suggested_actions
                    }
                    for pattern_id, pattern in self.identified_patterns.items()
                }
                json.dump(patterns_data, f, indent=2)
            
            logger.info("Models saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            # Load models if they exist
            classification_path = self.models_dir / "classification_model.pkl"
            if classification_path.exists():
                self.classification_model = joblib.load(classification_path)
            
            anomaly_path = self.models_dir / "anomaly_model.pkl"
            if anomaly_path.exists():
                self.anomaly_model = joblib.load(anomaly_path)
            
            clustering_path = self.models_dir / "clustering_model.pkl"
            if clustering_path.exists():
                self.clustering_model = joblib.load(clustering_path)
            
            # Load feature extractor
            feature_extractor_path = self.models_dir / "feature_extractor.pkl"
            if feature_extractor_path.exists():
                with open(feature_extractor_path, "rb") as f:
                    self.feature_extractor = pickle.load(f)
            
            # Load patterns
            patterns_path = self.models_dir / "patterns.json"
            if patterns_path.exists():
                with open(patterns_path, "r") as f:
                    patterns_data = json.load(f)
                    
                    for pattern_id, data in patterns_data.items():
                        pattern = FailurePattern(
                            id=pattern_id,
                            pattern_type=PatternType(data["pattern_type"]),
                            confidence=data["confidence"],
                            frequency=data["frequency"],
                            first_seen=datetime.fromisoformat(data["first_seen"]),
                            last_seen=datetime.fromisoformat(data["last_seen"]),
                            description=data["description"],
                            characteristics=data["characteristics"],
                            impact_score=data["impact_score"],
                            suggested_actions=data["suggested_actions"]
                        )
                        self.identified_patterns[pattern_id] = pattern
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")


# Global pattern recognizer instance
pattern_recognizer = FailurePatternRecognizer()