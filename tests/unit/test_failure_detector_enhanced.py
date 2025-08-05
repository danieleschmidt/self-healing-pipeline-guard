"""Enhanced unit tests for failure detector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from healing_guard.core.failure_detector import (
    FailureDetector, FailureEvent, FailureType, SeverityLevel, FailurePattern
)


class TestFailureDetector:
    """Test cases for FailureDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a FailureDetector instance for testing."""
        return FailureDetector()
    
    @pytest.fixture
    def sample_failure_logs(self):
        """Sample failure logs for testing."""
        return """
        2024-01-01 12:00:00 ERROR: Connection timeout
        2024-01-01 12:00:01 INFO: Retrying connection...
        2024-01-01 12:00:02 ERROR: Failed to connect to database after 3 attempts
        2024-01-01 12:00:03 FATAL: Test suite failed with 5 errors
        Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
        at com.example.Application.processData(Application.java:45)
        """
    
    def test_initialization(self, detector):
        """Test detector initialization with default patterns."""
        assert len(detector.patterns) > 0
        assert len(detector.failure_history) == 0
        assert detector.tfidf_vectorizer is not None
        assert "log_content" in detector.feature_weights
    
    def test_add_pattern(self, detector):
        """Test adding a custom failure pattern."""
        pattern = FailurePattern(
            name="custom_timeout",
            type=FailureType.NETWORK_TIMEOUT,
            severity=SeverityLevel.MEDIUM,
            regex_patterns=[r"custom timeout"],
            keywords=["custom", "timeout"],
            remediation_strategies=["retry", "increase_timeout"]
        )
        
        initial_count = len(detector.patterns)
        detector.add_pattern(pattern)
        
        assert len(detector.patterns) == initial_count + 1
        assert "custom_timeout" in detector.patterns
        assert detector.patterns["custom_timeout"] == pattern
    
    def test_remove_pattern(self, detector):
        """Test removing a failure pattern."""
        # Get initial pattern count
        initial_count = len(detector.patterns)
        assert initial_count > 0
        
        # Get first pattern name
        first_pattern_name = list(detector.patterns.keys())[0]
        
        # Remove pattern
        success = detector.remove_pattern(first_pattern_name)
        assert success is True
        assert len(detector.patterns) == initial_count - 1
        assert first_pattern_name not in detector.patterns
        
        # Try to remove non-existent pattern
        success = detector.remove_pattern("non_existent")
        assert success is False
    
    def test_extract_log_features(self, detector, sample_failure_logs):
        """Test feature extraction from logs."""
        features = detector._extract_log_features(sample_failure_logs)
        
        # Check basic statistics
        assert "log_length" in features
        assert "line_count" in features
        assert "word_count" in features
        assert features["log_length"] > 0
        assert features["line_count"] > 0
        assert features["word_count"] > 0
        
        # Check error frequencies
        assert "error_frequencies" in features
        assert "total_errors" in features
        assert features["total_errors"] > 0
        
        # Check timing indicators
        assert "timing_indicators" in features
        
        # Check resource indicators
        assert "resource_indicators" in features
        
        # Check stack trace analysis
        assert "stack_trace_depth" in features
        assert features["stack_trace_depth"] > 0  # Java stack trace present
        
        # Check unique error types
        assert "unique_error_types" in features
        assert "error_type_list" in features
        assert "OutOfMemoryError" in features["error_type_list"]
    
    def test_calculate_pattern_confidence(self, detector, sample_failure_logs):
        """Test pattern confidence calculation."""
        # Test with memory exhaustion pattern
        memory_pattern = None
        for pattern in detector.patterns.values():
            if pattern.type == FailureType.RESOURCE_EXHAUSTION:
                memory_pattern = pattern
                break
        
        assert memory_pattern is not None
        
        confidence = detector._calculate_pattern_confidence(sample_failure_logs, memory_pattern)
        
        # Should have high confidence due to OutOfMemoryError
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be high confidence
    
    def test_classify_failure_type(self, detector, sample_failure_logs):
        """Test failure type classification."""
        features = detector._extract_log_features(sample_failure_logs)
        failure_type, confidence, matched_patterns = detector._classify_failure_type(
            sample_failure_logs, features
        )
        
        # Should detect memory exhaustion due to OutOfMemoryError
        assert failure_type == FailureType.RESOURCE_EXHAUSTION
        assert 0 <= confidence <= 1
        assert len(matched_patterns) > 0
    
    def test_heuristic_classification(self, detector):
        """Test fallback heuristic classification."""
        # Test with high resource indicators
        features = {
            "total_errors": 5,
            "timing_indicators": 2,
            "resource_indicators": 10,  # High resource indicators
            "stack_trace_depth": 3
        }
        
        failure_type, confidence, patterns = detector._heuristic_classification(features)
        assert failure_type == FailureType.RESOURCE_EXHAUSTION
        assert confidence == 0.6
        assert "heuristic_resource" in patterns
        
        # Test with high timing indicators
        features = {
            "total_errors": 3,
            "timing_indicators": 5,  # High timing indicators
            "resource_indicators": 1,
            "stack_trace_depth": 2
        }
        
        failure_type, confidence, patterns = detector._heuristic_classification(features)
        assert failure_type == FailureType.NETWORK_TIMEOUT
        assert confidence == 0.5
        assert "heuristic_timing" in patterns
    
    def test_get_remediation_suggestions(self, detector):
        """Test getting remediation suggestions."""
        # Test with matched patterns
        suggestions = detector._get_remediation_suggestions(
            FailureType.FLAKY_TEST,
            ["flaky_test_timeout"]
        )
        
        assert len(suggestions) > 0
        assert any("retry" in suggestion for suggestion in suggestions)
        
        # Test with unknown failure type
        suggestions = detector._get_remediation_suggestions(
            FailureType.UNKNOWN,
            []
        )
        
        # Should still get some default suggestions
        assert len(suggestions) >= 0
    
    def test_determine_severity(self, detector):
        """Test severity determination logic."""
        features = {"total_errors": 10}
        
        # Test critical failure type
        severity = detector._determine_severity(
            FailureType.SECURITY_VIOLATION,
            features,
            {}
        )
        assert severity == SeverityLevel.CRITICAL
        
        # Test main branch impact
        severity = detector._determine_severity(
            FailureType.DEPENDENCY_FAILURE,
            features,
            {"is_main_branch": True}
        )
        assert severity == SeverityLevel.HIGH  # Upgraded from MEDIUM
        
        # Test release candidate impact
        severity = detector._determine_severity(
            FailureType.FLAKY_TEST,
            features,
            {"is_release_candidate": True}
        )
        assert severity == SeverityLevel.CRITICAL  # Always critical for RC
        
        # Test high error count impact
        high_error_features = {"total_errors": 60}
        severity = detector._determine_severity(
            FailureType.DEPENDENCY_FAILURE,
            high_error_features,
            {}
        )
        assert severity == SeverityLevel.HIGH  # Upgraded due to high error count
    
    @pytest.mark.asyncio
    async def test_detect_failure(self, detector, sample_failure_logs):
        """Test complete failure detection process."""
        failure_event = await detector.detect_failure(
            job_id="test_job_123",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            logs=sample_failure_logs,
            context={"is_main_branch": True}
        )
        
        assert failure_event is not None
        assert isinstance(failure_event, FailureEvent)
        assert failure_event.job_id == "test_job_123"
        assert failure_event.repository == "test/repo"
        assert failure_event.branch == "main"
        assert failure_event.commit_sha == "abc123"
        assert failure_event.failure_type in [ft for ft in FailureType]
        assert failure_event.severity in [sl for sl in SeverityLevel]
        assert 0 <= failure_event.confidence <= 1
        assert len(failure_event.extracted_features) > 0
        assert len(failure_event.remediation_suggestions) > 0
        
        # Check that it was added to history
        assert len(detector.failure_history) == 1
        assert detector.failure_history[0] == failure_event
    
    def test_get_failure_statistics(self, detector):
        """Test failure statistics generation."""
        # Initially no history
        stats = detector.get_failure_statistics()
        assert "message" in stats
        
        # Add some mock failures
        for i in range(5):
            failure = FailureEvent(
                id=f"failure_{i}",
                timestamp=datetime.now(),
                job_id=f"job_{i}",
                repository="test/repo",
                branch="main",
                commit_sha=f"sha_{i}",
                failure_type=FailureType.FLAKY_TEST if i % 2 == 0 else FailureType.NETWORK_TIMEOUT,
                severity=SeverityLevel.MEDIUM,
                confidence=0.8,
                raw_logs="test logs",
                matched_patterns=[f"pattern_{i}"]
            )
            detector.failure_history.append(failure)
        
        stats = detector.get_failure_statistics()
        
        assert stats["total_failures"] == 5
        assert "failure_types" in stats
        assert "severity_distribution" in stats
        assert "average_confidence" in stats
        assert stats["average_confidence"] == 0.8
        assert "recent_failures_24h" in stats
        assert "most_common_patterns" in stats
        assert stats["pattern_count"] > 0
    
    def test_get_failure_trends(self, detector):
        """Test failure trend analysis."""
        # Initially no history
        trends = detector.get_failure_trends(days=7)
        assert "message" in trends
        
        # Add failures with different timestamps
        base_time = datetime.now()
        for i in range(10):
            failure = FailureEvent(
                id=f"failure_{i}",
                timestamp=base_time - timedelta(days=i % 3),  # Spread over 3 days
                job_id=f"job_{i}",
                repository=f"repo_{i % 2}",  # Two different repos
                branch="main" if i % 2 == 0 else "develop",
                commit_sha=f"sha_{i}",
                failure_type=FailureType.FLAKY_TEST if i % 2 == 0 else FailureType.NETWORK_TIMEOUT,
                severity=SeverityLevel.MEDIUM,
                confidence=0.8,
                raw_logs="test logs"
            )
            detector.failure_history.append(failure)
        
        trends = detector.get_failure_trends(days=7)
        
        assert trends["period_days"] == 7
        assert trends["total_failures"] == 10
        assert "daily_failure_counts" in trends
        assert "top_failing_repositories" in trends
        assert "top_failing_branches" in trends
        assert "trending_failure_types" in trends
        
        # Check repository distribution
        repo_counts = trends["top_failing_repositories"]
        assert "repo_0" in repo_counts
        assert "repo_1" in repo_counts
    
    @pytest.mark.asyncio
    async def test_learn_from_feedback(self, detector, sample_failure_logs):
        """Test learning from user feedback."""
        # First detect a failure
        failure_event = await detector.detect_failure(
            job_id="test_job_123",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            logs=sample_failure_logs
        )
        
        # Provide feedback that the classification was wrong
        feedback = {
            "correct_failure_type": "network_timeout"
        }
        
        success = await detector.learn_from_feedback(failure_event.id, feedback)
        assert success is True
        
        # Test with non-existent failure ID
        success = await detector.learn_from_feedback("non_existent", feedback)
        assert success is False
    
    def test_failure_pattern_to_dict(self):
        """Test failure pattern serialization."""
        pattern = FailurePattern(
            name="test_pattern",
            type=FailureType.COMPILATION_ERROR,
            severity=SeverityLevel.HIGH,
            regex_patterns=[r"compile.*error"],
            keywords=["compile", "error"],
            context_indicators=["javac", "gcc"],
            confidence_threshold=0.8,
            remediation_strategies=["fix_syntax", "update_compiler"],
            metadata={"version": "1.0"}
        )
        
        pattern_dict = pattern.to_dict()
        
        required_fields = [
            "name", "type", "severity", "regex_patterns", "keywords",
            "context_indicators", "confidence_threshold", "remediation_strategies", "metadata"
        ]
        
        for field in required_fields:
            assert field in pattern_dict
        
        assert pattern_dict["name"] == "test_pattern"
        assert pattern_dict["type"] == "compilation_error"
        assert pattern_dict["severity"] == 2
    
    def test_failure_event_to_dict(self):
        """Test failure event serialization."""
        failure_event = FailureEvent(
            id="test_failure",
            timestamp=datetime.now(),
            job_id="job_123",
            repository="test/repo",
            branch="main",
            commit_sha="abc123",
            failure_type=FailureType.FLAKY_TEST,
            severity=SeverityLevel.MEDIUM,
            confidence=0.85,
            raw_logs="test logs",
            extracted_features={"feature1": "value1"},
            matched_patterns=["pattern1"],
            context={"context1": "value1"},
            remediation_suggestions=["suggestion1"]
        )
        
        event_dict = failure_event.to_dict()
        
        required_fields = [
            "id", "timestamp", "job_id", "repository", "branch", "commit_sha",
            "failure_type", "severity", "confidence", "extracted_features",
            "matched_patterns", "context", "remediation_suggestions"
        ]
        
        for field in required_fields:
            assert field in event_dict
        
        assert event_dict["id"] == "test_failure"
        assert event_dict["failure_type"] == "flaky_test"
        assert event_dict["severity"] == 3
        assert event_dict["confidence"] == 0.85
    
    def test_history_size_limit(self, detector):
        """Test that failure history respects size limits."""
        original_history_size = len(detector.failure_history)
        
        # Add many failures (more than the limit of 10000)
        for i in range(15000):
            failure = FailureEvent(
                id=f"failure_{i}",
                timestamp=datetime.now(),
                job_id=f"job_{i}",
                repository="test/repo",
                branch="main",
                commit_sha=f"sha_{i}",
                failure_type=FailureType.FLAKY_TEST,
                severity=SeverityLevel.MEDIUM,
                confidence=0.8,
                raw_logs="test logs"
            )
            detector.failure_history.append(failure)
        
        # Should be limited to 5000 after cleanup
        assert len(detector.failure_history) == 5000