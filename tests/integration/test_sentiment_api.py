"""Integration tests for sentiment analysis API endpoints."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from healing_guard.api.main import create_app
from healing_guard.core.sentiment_analyzer import SentimentResult, SentimentLabel


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_analyzer():
    """Create mock sentiment analyzer."""
    mock = AsyncMock()
    
    # Default mock response
    mock_result = SentimentResult(
        label=SentimentLabel.NEUTRAL,
        confidence=0.7,
        polarity=0.1,
        subjectivity=0.5,
        urgency_score=0.2,
        emotional_intensity=0.3,
        keywords=['test'],
        context_factors={'text_length': 12},
        processing_time_ms=15.5
    )
    
    mock.analyze_text.return_value = mock_result
    mock.analyze_pipeline_event.return_value = mock_result
    mock.analyze_batch.return_value = [mock_result]
    mock.get_sentiment_summary.return_value = {
        'total_analyses': 1,
        'average_polarity': 0.1,
        'urgent_events': 0,
        'concerning_pattern_detected': False
    }
    
    return mock


class TestSentimentAnalysisEndpoints:
    """Test sentiment analysis API endpoints."""
    
    @patch('healing_guard.api.sentiment_routes.sentiment_analyzer')
    def test_analyze_sentiment_success(self, mock_analyzer, client):
        """Test successful sentiment analysis."""
        # Mock the analyzer response
        mock_result = SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=0.85,
            polarity=0.6,
            subjectivity=0.7,
            urgency_score=0.1,
            emotional_intensity=0.2,
            keywords=['great', 'success'],
            context_factors={'text_length': 25},
            processing_time_ms=12.3
        )
        mock_analyzer.analyze_text = AsyncMock(return_value=mock_result)
        
        # Make request
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": "Great job on this success!",
            "context": {"event_type": "build_success"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["label"] == "positive"
        assert data["confidence"] == 0.85
        assert data["polarity"] == 0.6
        assert data["is_urgent"] == False
        assert data["is_negative"] == False
        assert "great" in data["keywords"]
        assert "success" in data["keywords"]
        assert data["processing_time_ms"] == 12.3
        
        # Verify analyzer was called correctly
        mock_analyzer.analyze_text.assert_called_once_with(
            "Great job on this success!",
            {"event_type": "build_success"}
        )
    
    @patch('healing_guard.api.sentiment_routes.sentiment_analyzer')
    def test_analyze_sentiment_urgent(self, mock_analyzer, client):
        """Test urgent sentiment detection."""
        mock_result = SentimentResult(
            label=SentimentLabel.URGENT,
            confidence=0.95,
            polarity=-0.4,
            subjectivity=0.8,
            urgency_score=0.9,
            emotional_intensity=0.7,
            keywords=['urgent', 'critical', 'failed'],
            context_factors={'text_length': 35, 'has_excessive_punctuation': True},
            processing_time_ms=18.7
        )
        mock_analyzer.analyze_text = AsyncMock(return_value=mock_result)
        
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": "URGENT: Critical system failure!!!"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["label"] == "urgent"
        assert data["confidence"] == 0.95
        assert data["urgency_score"] == 0.9
        assert data["is_urgent"] == True
        assert data["is_negative"] == True
        assert "urgent" in data["keywords"]
    
    def test_analyze_sentiment_validation_errors(self, client):
        """Test validation errors in sentiment analysis."""
        # Empty text
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": ""
        })
        assert response.status_code == 422
        
        # Text too long
        long_text = "x" * 10001
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": long_text
        })
        assert response.status_code == 422
        
        # Missing required field
        response = client.post("/api/v1/sentiment/analyze", json={
            "context": {"some": "data"}
        })
        assert response.status_code == 422
    
    @patch('healing_guard.api.sentiment_routes.sentiment_analyzer')
    def test_analyze_pipeline_event_success(self, mock_analyzer, client):
        """Test pipeline event sentiment analysis."""
        mock_result = SentimentResult(
            label=SentimentLabel.FRUSTRATED,
            confidence=0.8,
            polarity=-0.5,
            subjectivity=0.9,
            urgency_score=0.6,
            emotional_intensity=0.8,
            keywords=['failed', 'broken', 'frustrated'],
            context_factors={
                'event_type': 'pipeline_failure',
                'is_production': True,
                'consecutive_failures': 3
            },
            processing_time_ms=22.1
        )
        mock_analyzer.analyze_pipeline_event = AsyncMock(return_value=mock_result)
        
        response = client.post("/api/v1/sentiment/analyze/pipeline-event", json={
            "event_type": "pipeline_failure",
            "message": "Pipeline failed again, so frustrated with these broken tests",
            "metadata": {
                "environment": "production",
                "consecutive_failures": 3,
                "repository": "critical-service"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["label"] == "frustrated"
        assert data["is_frustrated"] == True
        assert data["urgency_score"] == 0.6
        assert "frustrated" in data["keywords"]
        
        # Verify analyzer was called correctly
        mock_analyzer.analyze_pipeline_event.assert_called_once_with(
            "pipeline_failure",
            "Pipeline failed again, so frustrated with these broken tests",
            {
                "environment": "production",
                "consecutive_failures": 3,
                "repository": "critical-service"
            }
        )
    
    def test_pipeline_event_validation_errors(self, client):
        """Test validation errors in pipeline event analysis."""
        # Invalid event type
        response = client.post("/api/v1/sentiment/analyze/pipeline-event", json={
            "event_type": "invalid_event",
            "message": "Some message"
        })
        assert response.status_code == 422
        
        # Valid event types should work
        valid_event_types = [
            'pipeline_failure', 'test_failure', 'build_error', 'deployment_error',
            'commit_message', 'pr_comment', 'issue_comment', 'code_review',
            'merge_request', 'release_notes', 'incident_report'
        ]
        
        for event_type in valid_event_types:
            with patch('healing_guard.api.sentiment_routes.sentiment_analyzer') as mock:
                mock_result = SentimentResult(
                    label=SentimentLabel.NEUTRAL, confidence=0.5, polarity=0.0,
                    subjectivity=0.0, urgency_score=0.0, emotional_intensity=0.0,
                    keywords=[], context_factors={}, processing_time_ms=10.0
                )
                mock.analyze_pipeline_event = AsyncMock(return_value=mock_result)
                
                response = client.post("/api/v1/sentiment/analyze/pipeline-event", json={
                    "event_type": event_type,
                    "message": "Test message"
                })
                assert response.status_code == 200, f"Failed for event_type: {event_type}"
    
    @patch('healing_guard.api.sentiment_routes.sentiment_analyzer')
    def test_batch_analysis_success(self, mock_analyzer, client):
        """Test batch sentiment analysis."""
        mock_results = [
            SentimentResult(
                label=SentimentLabel.POSITIVE, confidence=0.8, polarity=0.6,
                subjectivity=0.7, urgency_score=0.1, emotional_intensity=0.2,
                keywords=['great'], context_factors={}, processing_time_ms=10.0
            ),
            SentimentResult(
                label=SentimentLabel.NEGATIVE, confidence=0.7, polarity=-0.4,
                subjectivity=0.6, urgency_score=0.3, emotional_intensity=0.4,
                keywords=['failed'], context_factors={}, processing_time_ms=12.0
            ),
            SentimentResult(
                label=SentimentLabel.URGENT, confidence=0.9, polarity=-0.6,
                subjectivity=0.9, urgency_score=0.9, emotional_intensity=0.8,
                keywords=['urgent', 'critical'], context_factors={}, processing_time_ms=15.0
            )
        ]
        
        mock_summary = {
            'total_analyses': 3,
            'average_polarity': -0.13,
            'average_urgency': 0.43,
            'urgent_events': 1,
            'negative_events': 2,
            'concerning_pattern_detected': True,
            'overall_sentiment_trend': 'negative'
        }
        
        mock_analyzer.analyze_batch = AsyncMock(return_value=mock_results)
        mock_analyzer.get_sentiment_summary = lambda results: mock_summary
        
        response = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": [
                "Great work on this feature!",
                "Build failed with errors",
                "URGENT: Production system down!"
            ],
            "contexts": [
                {"event_type": "feature_complete"},
                {"event_type": "build_failure"},
                {"event_type": "incident", "severity": "critical"}
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 3
        assert data["results"][0]["label"] == "positive"
        assert data["results"][1]["label"] == "negative"
        assert data["results"][2]["label"] == "urgent"
        
        assert data["summary"]["total_analyses"] == 3
        assert data["summary"]["concerning_pattern_detected"] == True
        assert data["total_processing_time_ms"] > 0
    
    def test_batch_analysis_validation_errors(self, client):
        """Test validation errors in batch analysis."""
        # Empty texts list
        response = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": []
        })
        assert response.status_code == 422
        
        # Too many texts
        too_many_texts = ["text"] * 101
        response = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": too_many_texts
        })
        assert response.status_code == 422
        
        # Empty text in list
        response = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": ["valid text", "", "another valid text"]
        })
        assert response.status_code == 422
    
    @patch('healing_guard.api.sentiment_routes.sentiment_analyzer')
    def test_sentiment_health_check(self, mock_analyzer, client):
        """Test sentiment analyzer health check."""
        mock_result = SentimentResult(
            label=SentimentLabel.NEUTRAL, confidence=0.5, polarity=0.0,
            subjectivity=0.0, urgency_score=0.0, emotional_intensity=0.0,
            keywords=[], context_factors={}, processing_time_ms=8.5
        )
        mock_analyzer.analyze_text = AsyncMock(return_value=mock_result)
        
        response = client.get("/api/v1/sentiment/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["analyzer_ready"] == True
        assert data["test_analysis_time_ms"] == 8.5
        assert "timestamp" in data
    
    def test_sentiment_stats_endpoint(self, client):
        """Test sentiment statistics endpoint."""
        response = client.get("/api/v1/sentiment/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that all expected fields are present
        expected_fields = [
            'total_analyses_performed', 'average_processing_time_ms',
            'most_common_sentiment', 'urgent_events_today',
            'frustrated_events_today', 'analyzer_version',
            'supported_languages', 'uptime_hours', 'last_updated'
        ]
        
        for field in expected_fields:
            assert field in data
    
    @patch('healing_guard.api.sentiment_routes.check_rate_limit')
    def test_rate_limiting(self, mock_rate_limit, client):
        """Test rate limiting functionality."""
        # Simulate rate limit exceeded
        from healing_guard.api.exceptions import ValidationError
        mock_rate_limit.side_effect = ValidationError("Rate limit exceeded")
        
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": "Test message"
        })
        
        assert response.status_code == 400
        assert "Rate limit exceeded" in response.json()["detail"]
    
    def test_request_validation_edge_cases(self, client):
        """Test edge cases in request validation."""
        # Whitespace only text should be rejected
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": "   \n\t   "
        })
        assert response.status_code == 422
        
        # Very long context should be accepted
        large_context = {"key": "x" * 1000}
        with patch('healing_guard.api.sentiment_routes.sentiment_analyzer') as mock:
            mock_result = SentimentResult(
                label=SentimentLabel.NEUTRAL, confidence=0.5, polarity=0.0,
                subjectivity=0.0, urgency_score=0.0, emotional_intensity=0.0,
                keywords=[], context_factors={}, processing_time_ms=10.0
            )
            mock.analyze_text = AsyncMock(return_value=mock_result)
            
            response = client.post("/api/v1/sentiment/analyze", json={
                "text": "Test with large context",
                "context": large_context
            })
            assert response.status_code == 200


class TestSentimentAPIIntegration:
    """Integration tests with real analyzer (no mocking)."""
    
    def test_real_analyzer_integration(self, client):
        """Test with real sentiment analyzer (integration test)."""
        # This test uses the real analyzer to ensure end-to-end functionality
        response = client.post("/api/v1/sentiment/analyze", json={
            "text": "Build successful! All tests passed perfectly.",
            "context": {"event_type": "build_success"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect positive sentiment
        assert data["polarity"] > 0
        assert data["confidence"] > 0
        assert data["is_negative"] == False
        assert len(data["keywords"]) > 0
        assert data["processing_time_ms"] > 0
        
        # Check response structure
        required_fields = [
            'label', 'confidence', 'polarity', 'subjectivity',
            'urgency_score', 'emotional_intensity', 'keywords',
            'context_factors', 'processing_time_ms', 'is_urgent',
            'is_negative', 'is_frustrated', 'analyzed_at'
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_real_batch_analysis_integration(self, client):
        """Test real batch analysis integration."""
        response = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": [
                "Excellent deployment, everything works!",
                "Tests failed again, this is frustrating.",
                "URGENT: Production issue needs immediate fix!"
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 3
        assert data["total_processing_time_ms"] > 0
        
        # Check that different sentiments were detected
        sentiments = [result["label"] for result in data["results"]]
        assert len(set(sentiments)) > 1  # Should have different sentiments
        
        # Check summary
        summary = data["summary"]
        assert summary["total_analyses"] == 3
        assert "average_polarity" in summary
        assert "concerning_pattern_detected" in summary
    
    def test_real_pipeline_event_integration(self, client):
        """Test real pipeline event analysis integration."""
        response = client.post("/api/v1/sentiment/analyze/pipeline-event", json={
            "event_type": "pipeline_failure",
            "message": "Deployment failed due to configuration error in production environment",
            "metadata": {
                "environment": "production",
                "consecutive_failures": 2,
                "repository": "web-service"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect negative sentiment and some urgency due to production context
        assert data["is_negative"] == True
        assert data["urgency_score"] > 0.3  # Should have some urgency due to production
        assert data["context_factors"]["event_type"] == "pipeline_failure"
        assert data["context_factors"]["is_production"] == True