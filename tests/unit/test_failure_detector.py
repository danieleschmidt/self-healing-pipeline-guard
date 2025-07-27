"""
Unit tests for the failure detection module.

Tests the core logic for detecting and classifying pipeline failures
without external dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch

from healing_guard.core.failure_detector import (
    FailureDetector,
    FailureClassification,
    FailureType,
)
from healing_guard.models.pipeline import PipelineFailure


class TestFailureDetector:
    """Test suite for FailureDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a FailureDetector instance for testing."""
        return FailureDetector()

    @pytest.fixture
    def sample_logs(self):
        """Sample log data for testing."""
        return [
            "Starting test execution...",
            "Running test_calculation...",
            "AssertionError: Expected 5 but got 3",
            "Test failed with exit code 1",
            "Build finished with errors"
        ]

    @pytest.fixture
    def timeout_logs(self):
        """Sample timeout log data."""
        return [
            "Starting integration tests...",
            "Connecting to external service...",
            "Request timeout after 30 seconds",
            "Connection failed: timeout",
            "Build terminated due to timeout"
        ]

    @pytest.fixture
    def oom_logs(self):
        """Sample out-of-memory log data."""
        return [
            "Starting memory-intensive operation...",
            "Processing large dataset...",
            "java.lang.OutOfMemoryError: Java heap space",
            "Process killed: OOMKilled",
            "Build failed due to resource exhaustion"
        ]

    def test_detect_failure_basic(self, detector, sample_logs):
        """Test basic failure detection."""
        failure = PipelineFailure(
            id="test-123",
            platform="github",
            repository="test/repo",
            logs=sample_logs,
            exit_code=1
        )
        
        classification = detector.detect(failure)
        
        assert classification is not None
        assert classification.failure_type == FailureType.TEST_FAILURE
        assert classification.confidence > 0.7
        assert "AssertionError" in classification.error_patterns

    def test_detect_timeout_failure(self, detector, timeout_logs):
        """Test timeout failure detection."""
        failure = PipelineFailure(
            id="test-timeout",
            platform="github",
            repository="test/repo",
            logs=timeout_logs,
            exit_code=124  # Timeout exit code
        )
        
        classification = detector.detect(failure)
        
        assert classification.failure_type == FailureType.TIMEOUT
        assert classification.confidence > 0.8
        assert "timeout" in classification.error_patterns

    def test_detect_oom_failure(self, detector, oom_logs):
        """Test out-of-memory failure detection."""
        failure = PipelineFailure(
            id="test-oom",
            platform="github",
            repository="test/repo",
            logs=oom_logs,
            exit_code=137  # OOM kill exit code
        )
        
        classification = detector.detect(failure)
        
        assert classification.failure_type == FailureType.RESOURCE_EXHAUSTION
        assert classification.confidence > 0.9
        assert "OutOfMemoryError" in classification.error_patterns

    def test_extract_error_patterns(self, detector, sample_logs):
        """Test error pattern extraction."""
        patterns = detector._extract_error_patterns(sample_logs)
        
        assert "AssertionError" in patterns
        assert "Expected 5 but got 3" in patterns
        assert len(patterns) >= 2

    def test_calculate_confidence_high(self, detector):
        """Test confidence calculation for clear failure patterns."""
        patterns = ["OutOfMemoryError", "OOMKilled", "heap space"]
        confidence = detector._calculate_confidence(
            FailureType.RESOURCE_EXHAUSTION, 
            patterns
        )
        
        assert confidence > 0.85

    def test_calculate_confidence_low(self, detector):
        """Test confidence calculation for unclear patterns."""
        patterns = ["error", "failed"]  # Generic patterns
        confidence = detector._calculate_confidence(
            FailureType.UNKNOWN, 
            patterns
        )
        
        assert confidence < 0.6

    def test_classify_by_exit_code(self, detector):
        """Test classification based on exit codes."""
        # Test timeout exit code
        failure_type = detector._classify_by_exit_code(124)
        assert failure_type == FailureType.TIMEOUT
        
        # Test OOM exit code
        failure_type = detector._classify_by_exit_code(137)
        assert failure_type == FailureType.RESOURCE_EXHAUSTION
        
        # Test general failure
        failure_type = detector._classify_by_exit_code(1)
        assert failure_type == FailureType.UNKNOWN

    def test_classify_by_patterns_test_failure(self, detector):
        """Test pattern-based classification for test failures."""
        patterns = ["AssertionError", "Test failed", "expected"]
        failure_type = detector._classify_by_patterns(patterns)
        
        assert failure_type == FailureType.TEST_FAILURE

    def test_classify_by_patterns_dependency_failure(self, detector):
        """Test pattern-based classification for dependency failures."""
        patterns = ["Could not resolve", "dependency", "package not found"]
        failure_type = detector._classify_by_patterns(patterns)
        
        assert failure_type == FailureType.DEPENDENCY_FAILURE

    def test_classify_by_patterns_network_failure(self, detector):
        """Test pattern-based classification for network failures."""
        patterns = ["Connection refused", "network", "unable to connect"]
        failure_type = detector._classify_by_patterns(patterns)
        
        assert failure_type == FailureType.NETWORK_FAILURE

    def test_is_flaky_test_pattern(self, detector):
        """Test flaky test pattern detection."""
        # Test flaky patterns
        assert detector._is_flaky_test_pattern("Connection refused")
        assert detector._is_flaky_test_pattern("timeout")
        assert detector._is_flaky_test_pattern("race condition")
        
        # Test non-flaky patterns
        assert not detector._is_flaky_test_pattern("AssertionError")
        assert not detector._is_flaky_test_pattern("syntax error")

    def test_empty_logs(self, detector):
        """Test handling of empty logs."""
        failure = PipelineFailure(
            id="test-empty",
            platform="github",
            repository="test/repo",
            logs=[],
            exit_code=1
        )
        
        classification = detector.detect(failure)
        
        assert classification.failure_type == FailureType.UNKNOWN
        assert classification.confidence < 0.5

    def test_none_logs(self, detector):
        """Test handling of None logs."""
        failure = PipelineFailure(
            id="test-none",
            platform="github",
            repository="test/repo",
            logs=None,
            exit_code=1
        )
        
        classification = detector.detect(failure)
        
        assert classification.failure_type == FailureType.UNKNOWN
        assert classification.confidence == 0.0

    @patch('healing_guard.core.failure_detector.ML_ENABLED', True)
    def test_ml_enhanced_detection(self, detector, mock_ml_model):
        """Test ML-enhanced failure detection."""
        detector.ml_model = mock_ml_model
        
        failure = PipelineFailure(
            id="test-ml",
            platform="github",
            repository="test/repo",
            logs=["Test timeout after 30 seconds"],
            exit_code=1
        )
        
        classification = detector.detect(failure)
        
        # ML model should enhance confidence
        assert classification.confidence > 0.8
        mock_ml_model.predict_proba.assert_called_once()

    def test_historical_pattern_learning(self, detector):
        """Test learning from historical failure patterns."""
        # Simulate multiple similar failures
        for i in range(5):
            failure = PipelineFailure(
                id=f"test-{i}",
                platform="github",
                repository="test/repo",
                logs=["npm install failed", "network error"],
                exit_code=1
            )
            detector.detect(failure)
        
        # Pattern should be learned and confidence improved
        new_failure = PipelineFailure(
            id="test-new",
            platform="github",
            repository="test/repo",
            logs=["npm install failed", "network error"],
            exit_code=1
        )
        
        classification = detector.detect(new_failure)
        assert classification.confidence > 0.7

    def test_platform_specific_patterns(self, detector):
        """Test platform-specific failure pattern recognition."""
        # GitHub Actions specific pattern
        github_logs = [
            "##[error]Process completed with exit code 1",
            "Error: The process '/usr/bin/docker' failed"
        ]
        
        failure = PipelineFailure(
            id="test-github",
            platform="github",
            repository="test/repo",
            logs=github_logs,
            exit_code=1
        )
        
        classification = detector.detect(failure)
        assert classification.failure_type == FailureType.BUILD_FAILURE

    def test_multi_pattern_failure(self, detector):
        """Test detection of failures with multiple error patterns."""
        mixed_logs = [
            "Starting tests...",
            "OutOfMemoryError: Java heap space",
            "Connection timeout to database",
            "Test failed: AssertionError",
            "Build terminated"
        ]
        
        failure = PipelineFailure(
            id="test-multi",
            platform="github",
            repository="test/repo",
            logs=mixed_logs,
            exit_code=137
        )
        
        classification = detector.detect(failure)
        
        # Should prioritize based on exit code and most confident pattern
        assert classification.failure_type == FailureType.RESOURCE_EXHAUSTION
        assert len(classification.error_patterns) > 1

    def test_confidence_threshold_filtering(self, detector):
        """Test filtering based on confidence thresholds."""
        detector.confidence_threshold = 0.8
        
        # Low confidence failure
        failure = PipelineFailure(
            id="test-low-conf",
            platform="github",
            repository="test/repo",
            logs=["error", "failed"],  # Generic patterns
            exit_code=1
        )
        
        classification = detector.detect(failure)
        
        # Should return UNKNOWN for low confidence
        assert classification.failure_type == FailureType.UNKNOWN

    def test_performance_with_large_logs(self, detector):
        """Test performance with large log files."""
        import time
        
        # Generate large log data
        large_logs = ["Log line " + str(i) for i in range(10000)]
        large_logs.append("OutOfMemoryError: heap space")
        
        failure = PipelineFailure(
            id="test-large",
            platform="github",
            repository="test/repo",
            logs=large_logs,
            exit_code=137
        )
        
        start_time = time.time()
        classification = detector.detect(failure)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
        assert classification.failure_type == FailureType.RESOURCE_EXHAUSTION