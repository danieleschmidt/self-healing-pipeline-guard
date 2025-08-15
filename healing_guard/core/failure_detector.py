"""Advanced failure detection and classification system.

Uses ML-powered pattern matching and quantum-inspired feature extraction
to identify and classify CI/CD pipeline failures in real-time.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import Counter, defaultdict, deque

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations for environments without sklearn
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using fallback implementations")

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of pipeline failures."""
    FLAKY_TEST = "flaky_test"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    NETWORK_TIMEOUT = "network_timeout"
    COMPILATION_ERROR = "compilation_error"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_ERROR = "configuration_error"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Failure severity levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class FailurePattern:
    """Represents a failure pattern for detection."""
    name: str
    type: FailureType
    severity: SeverityLevel
    regex_patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    context_indicators: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    remediation_strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "severity": self.severity.value,
            "regex_patterns": self.regex_patterns,
            "keywords": self.keywords,
            "context_indicators": self.context_indicators,
            "confidence_threshold": self.confidence_threshold,
            "remediation_strategies": self.remediation_strategies,
            "metadata": self.metadata
        }


@dataclass
class FailureEvent:
    """Represents a detected failure event."""
    id: str
    timestamp: datetime
    job_id: str
    repository: str
    branch: str
    commit_sha: str
    failure_type: FailureType
    severity: SeverityLevel
    confidence: float
    raw_logs: str
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    matched_patterns: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "job_id": self.job_id,
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "extracted_features": self.extracted_features,
            "matched_patterns": self.matched_patterns,
            "context": self.context,
            "remediation_suggestions": self.remediation_suggestions
        }


class FailureDetector:
    """Advanced failure detection and classification engine."""
    
    def __init__(self):
        self.patterns: Dict[str, FailurePattern] = {}
        self.failure_history: List[FailureEvent] = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_clusters = None
        self.feature_weights = {
            "log_content": 0.4,
            "error_frequency": 0.2,
            "timing_patterns": 0.15,
            "resource_metrics": 0.15,
            "context_similarity": 0.1
        }
        self._initialize_default_patterns()
        
    def _initialize_default_patterns(self) -> None:
        """Initialize default failure detection patterns."""
        default_patterns = [
            FailurePattern(
                name="flaky_test_timeout",
                type=FailureType.FLAKY_TEST,
                severity=SeverityLevel.MEDIUM,
                regex_patterns=[
                    r"timeout.*test.*exceeded",
                    r"test.*timed out after \d+",
                    r"assertion.*failed.*timing"
                ],
                keywords=["timeout", "flaky", "intermittent", "timing"],
                context_indicators=["test", "spec", "assertion"],
                remediation_strategies=["retry_with_isolation", "increase_timeout", "mock_external_deps"]
            ),
            FailurePattern(
                name="memory_exhaustion",
                type=FailureType.RESOURCE_EXHAUSTION,
                severity=SeverityLevel.HIGH,
                regex_patterns=[
                    r"OutOfMemoryError",
                    r"Cannot allocate memory",
                    r"OOMKilled",
                    r"Memory limit exceeded"
                ],
                keywords=["memory", "OOM", "allocation", "heap"],
                context_indicators=["java", "node", "docker", "container"],
                remediation_strategies=["increase_memory", "optimize_heap", "scale_resources"]
            ),
            FailurePattern(
                name="dependency_resolution_failure",
                type=FailureType.DEPENDENCY_FAILURE,
                severity=SeverityLevel.MEDIUM,
                regex_patterns=[
                    r"dependency.*not found",
                    r"package.*not available",
                    r"version conflict.*dependency",
                    r"npm ERR!.*dependency"
                ],
                keywords=["dependency", "package", "version", "conflict", "resolution"],
                context_indicators=["npm", "pip", "maven", "gradle", "yarn"],
                remediation_strategies=["clear_cache", "pin_versions", "update_lockfile"]
            ),
            FailurePattern(
                name="network_connectivity_failure",
                type=FailureType.NETWORK_TIMEOUT,
                severity=SeverityLevel.MEDIUM,
                regex_patterns=[
                    r"connection.*refused",
                    r"network.*timeout",
                    r"unable to connect",
                    r"DNS resolution failed"
                ],
                keywords=["network", "connection", "timeout", "DNS", "unreachable"],
                context_indicators=["curl", "wget", "http", "api", "service"],
                remediation_strategies=["retry_with_backoff", "check_network", "use_cache"]
            ),
            FailurePattern(
                name="compilation_syntax_error",
                type=FailureType.COMPILATION_ERROR,
                severity=SeverityLevel.HIGH,
                regex_patterns=[
                    r"SyntaxError",
                    r"compilation.*failed",
                    r"build.*error",
                    r"parse.*error"
                ],
                keywords=["syntax", "compilation", "build", "parse", "error"],
                context_indicators=["javac", "gcc", "tsc", "rustc", "go build"],
                remediation_strategies=["lint_check", "format_code", "update_compiler"]
            ),
            FailurePattern(
                name="security_vulnerability_scan",
                type=FailureType.SECURITY_VIOLATION,
                severity=SeverityLevel.CRITICAL,
                regex_patterns=[
                    r"security.*vulnerability.*found",
                    r"CVE-\d{4}-\d+",
                    r"high.*severity.*issue",
                    r"security.*scan.*failed"
                ],
                keywords=["security", "vulnerability", "CVE", "scan", "threat"],
                context_indicators=["snyk", "sonar", "checkmarx", "audit"],
                remediation_strategies=["update_dependencies", "apply_patch", "security_review"]
            )
        ]
        
        for pattern in default_patterns:
            self.add_pattern(pattern)
            
    def add_pattern(self, pattern: FailurePattern) -> None:
        """Add a failure detection pattern."""
        self.patterns[pattern.name] = pattern
        logger.info(f"Added failure pattern: {pattern.name} ({pattern.type.value})")
        
    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a failure detection pattern."""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            logger.info(f"Removed failure pattern: {pattern_name}")
            return True
        return False
        
    def _extract_log_features(self, logs: str) -> Dict[str, Any]:
        """Extract features from log content using quantum-inspired techniques."""
        features = {}
        
        # Basic text statistics
        features["log_length"] = len(logs)
        features["line_count"] = len(logs.split('\n'))
        features["word_count"] = len(logs.split())
        
        # Error patterns and frequencies
        error_patterns = [
            r"error",
            r"exception",
            r"fail(ed|ure)?",
            r"timeout",
            r"abort",
            r"crash"
        ]
        
        error_counts = {}
        for pattern in error_patterns:
            count = len(re.findall(pattern, logs, re.IGNORECASE))
            error_counts[pattern] = count
            
        features["error_frequencies"] = error_counts
        features["total_errors"] = sum(error_counts.values())
        
        # Timing indicators
        timing_patterns = [
            r"\d+\.?\d*\s*(ms|milliseconds?)",
            r"\d+\.?\d*\s*(s|seconds?)",
            r"\d+\.?\d*\s*(m|minutes?)"
        ]
        
        timing_matches = []
        for pattern in timing_patterns:
            matches = re.findall(pattern, logs, re.IGNORECASE)
            timing_matches.extend(matches)
            
        features["timing_indicators"] = len(timing_matches)
        
        # Resource indicators
        resource_patterns = [
            r"\d+\.?\d*\s*(MB|GB|KB)",
            r"\d+\.?\d*%\s*(cpu|memory|disk)",
            r"load.*average"
        ]
        
        resource_matches = []
        for pattern in resource_patterns:
            matches = re.findall(pattern, logs, re.IGNORECASE)
            resource_matches.extend(matches)
            
        features["resource_indicators"] = len(resource_matches)
        
        # Stack trace depth
        stack_trace_lines = [
            line for line in logs.split('\n')
            if re.match(r'\s+at\s+', line) or re.match(r'\s+File\s+"', line)
        ]
        features["stack_trace_depth"] = len(stack_trace_lines)
        
        # Unique error types
        unique_errors = set()
        for line in logs.split('\n'):
            if 'error' in line.lower() or 'exception' in line.lower():
                # Extract potential error class names
                error_matches = re.findall(r'(\w*Error|\w*Exception)', line)
                unique_errors.update(error_matches)
                
        features["unique_error_types"] = len(unique_errors)
        features["error_type_list"] = list(unique_errors)
        
        return features
    
    def _calculate_pattern_confidence(self, logs: str, pattern: FailurePattern) -> float:
        """Calculate confidence score for pattern match using quantum-inspired scoring."""
        confidence_scores = []
        
        # Regex pattern matching
        regex_score = 0.0
        for regex_pattern in pattern.regex_patterns:
            try:
                matches = re.findall(regex_pattern, logs, re.IGNORECASE)
                if matches:
                    regex_score += len(matches) * 0.3
            except re.error:
                logger.warning(f"Invalid regex pattern: {regex_pattern}")
                
        regex_score = min(1.0, regex_score)
        confidence_scores.append(regex_score)
        
        # Keyword density scoring
        keyword_score = 0.0
        log_words = logs.lower().split()
        total_words = len(log_words)
        
        if total_words > 0:
            keyword_matches = 0
            for keyword in pattern.keywords:
                keyword_count = log_words.count(keyword.lower())
                keyword_matches += keyword_count
                
            keyword_score = min(1.0, keyword_matches / (total_words * 0.01))  # Normalize by 1% of total words
            
        confidence_scores.append(keyword_score)
        
        # Context indicator scoring
        context_score = 0.0
        for indicator in pattern.context_indicators:
            if indicator.lower() in logs.lower():
                context_score += 0.2
                
        context_score = min(1.0, context_score)
        confidence_scores.append(context_score)
        
        # Weighted average confidence
        weights = [0.5, 0.3, 0.2]  # regex, keywords, context
        weighted_confidence = sum(score * weight for score, weight in zip(confidence_scores, weights))
        
        return weighted_confidence
    
    def _classify_failure_type(self, logs: str, features: Dict[str, Any]) -> Tuple[FailureType, float, List[str]]:
        """Classify failure type using enhanced ensemble methods."""
        # Try ensemble classification first if available
        if SKLEARN_AVAILABLE and len(self.failure_history) >= 50:
            try:
                return self._enhanced_ensemble_classification(logs, features)
            except Exception as e:
                logger.warning(f"Ensemble classification failed, falling back: {e}")
        
        # Traditional pattern-based classification
        return self._rule_based_classification(logs, features)
    
    def _rule_based_classification(self, logs: str, features: Dict[str, Any]) -> Tuple[FailureType, float, List[str]]:
        """Traditional rule-based pattern matching classification."""
        best_match = None
        best_confidence = 0.0
        matched_patterns = []
        
        # Pattern-based classification
        for pattern_name, pattern in self.patterns.items():
            confidence = self._calculate_pattern_confidence(logs, pattern)
            
            if confidence >= pattern.confidence_threshold:
                matched_patterns.append(pattern_name)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern
                    
        if best_match:
            return best_match.type, best_confidence, matched_patterns
            
        # Fallback to feature-based heuristics if no patterns match
        return self._heuristic_classification(features)
    
    def _heuristic_classification(self, features: Dict[str, Any]) -> Tuple[FailureType, float, List[str]]:
        """Fallback classification using feature heuristics."""
        # Simple heuristic rules
        total_errors = features.get("total_errors", 0)
        timing_indicators = features.get("timing_indicators", 0)
        resource_indicators = features.get("resource_indicators", 0)
        stack_trace_depth = features.get("stack_trace_depth", 0)
        
        if resource_indicators > 5:
            return FailureType.RESOURCE_EXHAUSTION, 0.6, ["heuristic_resource"]
        elif timing_indicators > 3:
            return FailureType.NETWORK_TIMEOUT, 0.5, ["heuristic_timing"]
        elif stack_trace_depth > 10:
            return FailureType.COMPILATION_ERROR, 0.5, ["heuristic_compilation"]
        elif total_errors > 10:
            return FailureType.UNKNOWN, 0.4, ["heuristic_high_error"]
        else:
            return FailureType.UNKNOWN, 0.3, ["heuristic_fallback"]
    
    def _get_remediation_suggestions(self, failure_type: FailureType, matched_patterns: List[str]) -> List[str]:
        """Get remediation suggestions based on failure type and matched patterns."""
        suggestions = set()
        
        # Collect suggestions from matched patterns
        for pattern_name in matched_patterns:
            if pattern_name in self.patterns:
                suggestions.update(self.patterns[pattern_name].remediation_strategies)
                
        # Add default suggestions based on failure type
        default_strategies = {
            FailureType.FLAKY_TEST: ["retry_with_isolation", "increase_timeout"],
            FailureType.RESOURCE_EXHAUSTION: ["increase_resources", "optimize_usage"],
            FailureType.DEPENDENCY_FAILURE: ["clear_cache", "update_dependencies"],
            FailureType.NETWORK_TIMEOUT: ["retry_with_backoff", "check_connectivity"],
            FailureType.COMPILATION_ERROR: ["lint_check", "syntax_review"],
            FailureType.SECURITY_VIOLATION: ["security_audit", "update_dependencies"],
            FailureType.CONFIGURATION_ERROR: ["validate_config", "check_environment"],
            FailureType.INFRASTRUCTURE_FAILURE: ["restart_services", "check_resources"]
        }
        
        if failure_type in default_strategies:
            suggestions.update(default_strategies[failure_type])
            
        return list(suggestions)
    
    async def detect_failure(
        self,
        job_id: str,
        repository: str,
        branch: str,
        commit_sha: str,
        logs: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[FailureEvent]:
        """Detect and classify a pipeline failure."""
        logger.info(f"Analyzing failure for job {job_id} in {repository}:{branch}")
        
        # Extract features from logs
        features = self._extract_log_features(logs)
        
        # Classify failure type
        failure_type, confidence, matched_patterns = self._classify_failure_type(logs, features)
        
        # Determine severity based on failure type and context
        severity = self._determine_severity(failure_type, features, context or {})
        
        # Get remediation suggestions
        remediation_suggestions = self._get_remediation_suggestions(failure_type, matched_patterns)
        
        # Create failure event
        failure_event = FailureEvent(
            id=f"{job_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            job_id=job_id,
            repository=repository,
            branch=branch,
            commit_sha=commit_sha,
            failure_type=failure_type,
            severity=severity,
            confidence=confidence,
            raw_logs=logs,
            extracted_features=features,
            matched_patterns=matched_patterns,
            context=context or {},
            remediation_suggestions=remediation_suggestions
        )
        
        # Add to history
        self.failure_history.append(failure_event)
        
        # Keep only recent history
        if len(self.failure_history) > 10000:
            self.failure_history = self.failure_history[-5000:]
            
        logger.info(
            f"Detected {failure_type.value} failure with {confidence:.2f} confidence "
            f"(severity: {severity.value})"
        )
        
        return failure_event
    
    def _determine_severity(
        self,
        failure_type: FailureType,
        features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SeverityLevel:
        """Determine failure severity based on type, features, and context."""
        # Base severity from failure type
        base_severity = {
            FailureType.SECURITY_VIOLATION: SeverityLevel.CRITICAL,
            FailureType.COMPILATION_ERROR: SeverityLevel.HIGH,
            FailureType.RESOURCE_EXHAUSTION: SeverityLevel.HIGH,
            FailureType.INFRASTRUCTURE_FAILURE: SeverityLevel.HIGH,
            FailureType.DEPENDENCY_FAILURE: SeverityLevel.MEDIUM,
            FailureType.NETWORK_TIMEOUT: SeverityLevel.MEDIUM,
            FailureType.CONFIGURATION_ERROR: SeverityLevel.MEDIUM,
            FailureType.FLAKY_TEST: SeverityLevel.LOW,
            FailureType.UNKNOWN: SeverityLevel.MEDIUM
        }.get(failure_type, SeverityLevel.MEDIUM)
        
        # Adjust based on context
        if context.get("is_main_branch", False):
            # Increase severity for main branch failures
            if base_severity.value > 1:
                base_severity = SeverityLevel(base_severity.value - 1)
                
        if context.get("is_release_candidate", False):
            # Critical severity for release candidate failures
            base_severity = SeverityLevel.CRITICAL
            
        # Adjust based on features
        total_errors = features.get("total_errors", 0)
        if total_errors > 50:
            # Many errors indicate severe issues
            if base_severity.value > 2:
                base_severity = SeverityLevel(base_severity.value - 1)
                
        return base_severity
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure detection and classification statistics."""
        if not self.failure_history:
            return {"message": "No failure history available"}
            
        total_failures = len(self.failure_history)
        
        # Failure type distribution
        type_counts = Counter(failure.failure_type.value for failure in self.failure_history)
        
        # Severity distribution
        severity_counts = Counter(failure.severity.value for failure in self.failure_history)
        
        # Average confidence
        avg_confidence = np.mean([failure.confidence for failure in self.failure_history])
        
        # Recent failures (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_failures = [
            failure for failure in self.failure_history
            if failure.timestamp > recent_cutoff
        ]
        
        # Most common patterns
        pattern_counts = Counter()
        for failure in self.failure_history:
            pattern_counts.update(failure.matched_patterns)
            
        return {
            "total_failures": total_failures,
            "failure_types": dict(type_counts),
            "severity_distribution": dict(severity_counts),
            "average_confidence": avg_confidence,
            "recent_failures_24h": len(recent_failures),
            "most_common_patterns": dict(pattern_counts.most_common(10)),
            "pattern_count": len(self.patterns),
            "feature_weights": self.feature_weights
        }
    
    def get_failure_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze failure trends over specified time period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_failures = [
            failure for failure in self.failure_history
            if failure.timestamp > cutoff_date
        ]
        
        if not recent_failures:
            return {"message": f"No failures in the last {days} days"}
            
        # Daily failure counts
        daily_counts = defaultdict(int)
        for failure in recent_failures:
            day_key = failure.timestamp.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1
            
        # Repository failure distribution
        repo_counts = Counter(failure.repository for failure in recent_failures)
        
        # Branch failure distribution
        branch_counts = Counter(failure.branch for failure in recent_failures)
        
        # Trending failure types
        type_trend = Counter(failure.failure_type.value for failure in recent_failures)
        
        return {
            "period_days": days,
            "total_failures": len(recent_failures),
            "daily_failure_counts": dict(daily_counts),
            "top_failing_repositories": dict(repo_counts.most_common(10)),
            "top_failing_branches": dict(branch_counts.most_common(10)),
            "trending_failure_types": dict(type_trend.most_common(5))
        }
    
    async def learn_from_feedback(self, failure_id: str, feedback: Dict[str, Any]) -> bool:
        """Learn from user feedback to improve detection accuracy."""
        # Find the failure event
        failure_event = None
        for failure in self.failure_history:
            if failure.id == failure_id:
                failure_event = failure
                break
                
        if not failure_event:
            logger.warning(f"Failure event {failure_id} not found for learning")
            return False
            
        # Process feedback
        correct_type = feedback.get("correct_failure_type")
        if correct_type and correct_type != failure_event.failure_type.value:
            logger.info(f"Learning: {failure_id} should be {correct_type}, not {failure_event.failure_type.value}")
            
            # Update pattern confidence thresholds based on feedback
            # This is a simplified learning mechanism
            for pattern_name in failure_event.matched_patterns:
                if pattern_name in self.patterns:
                    pattern = self.patterns[pattern_name]
                    if pattern.type.value != correct_type:
                        # Reduce confidence threshold for incorrect matches
                        pattern.confidence_threshold = min(0.9, pattern.confidence_threshold + 0.05)
                        
        return True
    
    def _enhanced_ensemble_classification(self, logs: str, features: Dict[str, Any]) -> Tuple[FailureType, float, List[str]]:
        """Advanced ensemble classification using multiple ML algorithms."""
        if not SKLEARN_AVAILABLE or len(self.failure_history) < 50:
            return self._rule_based_classification(logs, features)
        
        try:
            # Prepare training data from historical failures
            X, y = self._prepare_training_data()
            
            if len(X) < 10:  # Insufficient training data
                return self._rule_based_classification(logs, features)
            
            # Create ensemble of classifiers
            classifiers = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Train ensemble
            trained_models = {}
            ensemble_predictions = []
            
            for name, classifier in classifiers.items():
                try:
                    classifier.fit(X, y)
                    trained_models[name] = classifier
                    
                    # Predict current failure
                    current_features = self._extract_ml_features(logs, features)
                    prediction_proba = classifier.predict_proba([current_features])[0]
                    prediction = classifier.predict([current_features])[0]
                    
                    ensemble_predictions.append({
                        'model': name,
                        'prediction': prediction,
                        'confidence': max(prediction_proba),
                        'probabilities': dict(zip(classifier.classes_, prediction_proba))
                    })
                    
                except Exception as e:
                    logger.warning(f"Classifier {name} failed: {e}")
                    continue
            
            if not ensemble_predictions:
                return self._rule_based_classification(logs, features)
            
            # Ensemble voting with weighted confidence
            failure_type_votes = defaultdict(float)
            total_confidence = 0.0
            
            for pred in ensemble_predictions:
                weight = pred['confidence'] ** 2  # Square confidence for better discrimination
                failure_type_votes[pred['prediction']] += weight
                total_confidence += weight
            
            # Select most voted failure type
            best_failure_type = max(failure_type_votes.items(), key=lambda x: x[1])[0]
            ensemble_confidence = failure_type_votes[best_failure_type] / total_confidence if total_confidence > 0 else 0.5
            
            # Convert string back to enum
            try:
                failure_enum = FailureType(best_failure_type)
            except ValueError:
                failure_enum = FailureType.UNKNOWN
            
            # Get matched patterns for ensemble result
            matched_patterns = [f"ensemble_{model['model']}" for model in ensemble_predictions]
            
            logger.info(f"Ensemble classification: {failure_enum.value} with {ensemble_confidence:.3f} confidence")
            return failure_enum, ensemble_confidence, matched_patterns
            
        except Exception as e:
            logger.warning(f"Ensemble classification failed: {e}")
            return self._rule_based_classification(logs, features)
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[str]]:
        """Prepare training data from failure history."""
        X = []
        y = []
        
        for failure in self.failure_history[-1000:]:  # Use recent history
            if failure.extracted_features and failure.confidence > 0.7:  # High-confidence samples only
                features = self._extract_ml_features(failure.raw_logs, failure.extracted_features)
                X.append(features)
                y.append(failure.failure_type.value)
        
        return X, y
    
    def _extract_ml_features(self, logs: str, base_features: Dict[str, Any]) -> List[float]:
        """Extract numerical features for ML classification."""
        features = []
        
        # Text-based features
        features.append(base_features.get("log_length", 0) / 10000)  # Normalized
        features.append(base_features.get("line_count", 0) / 1000)
        features.append(base_features.get("word_count", 0) / 5000)
        features.append(base_features.get("total_errors", 0) / 100)
        features.append(base_features.get("timing_indicators", 0) / 50)
        features.append(base_features.get("resource_indicators", 0) / 20)
        features.append(base_features.get("stack_trace_depth", 0) / 100)
        features.append(len(base_features.get("unique_errors", set())) / 20)
        
        # Pattern matching features
        error_freq = base_features.get("error_frequencies", {})
        features.extend([
            error_freq.get("error", 0) / 50,
            error_freq.get("exception", 0) / 30,
            error_freq.get("fail(ed|ure)?", 0) / 30,
            error_freq.get("timeout", 0) / 10,
            error_freq.get("abort", 0) / 5,
            error_freq.get("crash", 0) / 5
        ])
        
        # Keyword presence (binary features)
        keywords = [
            "memory", "cpu", "disk", "network", "connection",
            "timeout", "security", "vulnerability", "test",
            "compile", "syntax", "dependency", "import"
        ]
        
        logs_lower = logs.lower()
        for keyword in keywords:
            features.append(1.0 if keyword in logs_lower else 0.0)
        
        # Advanced text features
        features.extend([
            len(re.findall(r'\d+', logs)) / 1000,  # Number density
            len(re.findall(r'http[s]?://', logs)) / 10,  # URL density
            len(re.findall(r'\b[A-Z][a-zA-Z]*Exception\b', logs)) / 20,  # Exception class density
            len(re.findall(r'at line \d+', logs)) / 50,  # Line number references
        ])
        
        return features
    
    def _deep_pattern_analysis(self, logs: str) -> Dict[str, Any]:
        """Perform deep pattern analysis using advanced text processing."""
        analysis = {}
        
        # Temporal pattern analysis
        timestamps = re.findall(r'\d{2}:\d{2}:\d{2}', logs)
        if len(timestamps) > 1:
            analysis['temporal_density'] = len(timestamps) / len(logs.split('\n'))
        else:
            analysis['temporal_density'] = 0.0
        
        # Error cascade detection
        error_lines = [i for i, line in enumerate(logs.split('\n')) 
                      if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed'])]
        
        if len(error_lines) > 1:
            # Calculate error clustering
            error_gaps = [error_lines[i+1] - error_lines[i] for i in range(len(error_lines)-1)]
            analysis['error_clustering'] = sum(1 for gap in error_gaps if gap <= 3) / max(len(error_gaps), 1)
        else:
            analysis['error_clustering'] = 0.0
        
        # Severity indicator analysis
        severity_indicators = {
            'critical': len(re.findall(r'\bcritical\b|\bfatal\b', logs, re.IGNORECASE)),
            'warning': len(re.findall(r'\bwarn(ing)?\b|\bcaution\b', logs, re.IGNORECASE)),
            'info': len(re.findall(r'\binfo\b|\bdebug\b', logs, re.IGNORECASE))
        }
        
        total_indicators = sum(severity_indicators.values())
        if total_indicators > 0:
            analysis['severity_distribution'] = {
                k: v / total_indicators for k, v in severity_indicators.items()
            }
        else:
            analysis['severity_distribution'] = {'critical': 0, 'warning': 0, 'info': 0}
        
        # Code context analysis
        code_indicators = {
            'file_references': len(re.findall(r'\.py|\\.java|\\.js|\\.cpp|\\.c\b', logs)),
            'function_references': len(re.findall(r'function\s+\w+|def\s+\w+', logs)),
            'line_numbers': len(re.findall(r'line\s+\d+|:\d+:', logs)),
            'variable_names': len(re.findall(r'\b[a-z_][a-zA-Z0-9_]*\b', logs))
        }
        
        analysis['code_context_strength'] = sum(code_indicators.values()) / 100  # Normalized
        
        return analysis
    
    def get_classification_confidence_report(self) -> Dict[str, Any]:
        """Generate a comprehensive confidence report for classification accuracy."""
        if not self.failure_history:
            return {"error": "No failure history available"}
        
        # Analyze recent classifications
        recent_failures = self.failure_history[-100:]  # Last 100 failures
        
        # Confidence distribution
        confidence_bins = {
            'high': sum(1 for f in recent_failures if f.confidence >= 0.8),
            'medium': sum(1 for f in recent_failures if 0.5 <= f.confidence < 0.8),
            'low': sum(1 for f in recent_failures if f.confidence < 0.5)
        }
        
        # Failure type distribution
        type_distribution = Counter(f.failure_type.value for f in recent_failures)
        
        # Average confidence by type
        avg_confidence_by_type = {}
        for failure_type in FailureType:
            type_failures = [f for f in recent_failures if f.failure_type == failure_type]
            if type_failures:
                avg_confidence_by_type[failure_type.value] = sum(f.confidence for f in type_failures) / len(type_failures)
        
        return {
            'total_classified': len(recent_failures),
            'confidence_distribution': confidence_bins,
            'confidence_percentage': {
                k: (v / len(recent_failures)) * 100 for k, v in confidence_bins.items()
            },
            'failure_type_distribution': dict(type_distribution),
            'average_confidence_by_type': avg_confidence_by_type,
            'overall_average_confidence': sum(f.confidence for f in recent_failures) / len(recent_failures),
            'ensemble_availability': SKLEARN_AVAILABLE and len(self.failure_history) >= 50
        }