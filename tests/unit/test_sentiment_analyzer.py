"""Unit tests for the sentiment analyzer module."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from healing_guard.core.sentiment_analyzer import (
    PipelineSentimentAnalyzer,
    SentimentLabel,
    SentimentResult,
    sentiment_analyzer
)


class TestPipelineSentimentAnalyzer:
    """Test suite for PipelineSentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer instance for testing."""
        return PipelineSentimentAnalyzer()
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, analyzer):
        """Test analysis of empty or whitespace-only text."""
        # Test empty string
        result = await analyzer.analyze_text("")
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        assert result.polarity == 0.0
        
        # Test whitespace only
        result = await analyzer.analyze_text("   \n\t   ")
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        
        # Test None (shouldn't happen in real use but let's be safe)
        result = await analyzer.analyze_text(None)
        assert result.label == SentimentLabel.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_analyze_positive_sentiment(self, analyzer):
        """Test analysis of positive sentiment text."""
        positive_texts = [
            "Tests passed successfully! Great work on this feature.",
            "Excellent deployment, everything is working perfectly.",
            "Build completed, all checks are green and ready to merge.",
            "Amazing optimization, performance improved significantly!"
        ]
        
        for text in positive_texts:
            result = await analyzer.analyze_text(text)
            assert result.polarity > 0, f"Expected positive polarity for: {text}"
            assert result.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE, SentimentLabel.CONFIDENT]
            assert result.confidence > 0.1
            assert len(result.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_negative_sentiment(self, analyzer):
        """Test analysis of negative sentiment text."""
        negative_texts = [
            "Build failed again, this is getting frustrating.",
            "Tests are broken, nothing works anymore.",
            "Pipeline crashed with terrible error messages.",
            "This is a disaster, everything is failing."
        ]
        
        for text in negative_texts:
            result = await analyzer.analyze_text(text)
            assert result.polarity < 0, f"Expected negative polarity for: {text}"
            assert result.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE, SentimentLabel.FRUSTRATED]
            assert result.confidence > 0.1
            assert result.is_negative
    
    @pytest.mark.asyncio
    async def test_analyze_urgent_sentiment(self, analyzer):
        """Test analysis of urgent sentiment indicators."""
        urgent_texts = [
            "URGENT: Production is down, need immediate fix!",
            "Critical error in deployment, hotfix needed ASAP!",
            "Emergency: Pipeline blocker affecting all teams NOW.",
            "Production outage - immediate attention required!!!"
        ]
        
        for text in urgent_texts:
            result = await analyzer.analyze_text(text)
            assert result.urgency_score > 0.5, f"Expected high urgency for: {text}"
            assert result.is_urgent
            assert result.label == SentimentLabel.URGENT or result.urgency_score > 0.7
    
    @pytest.mark.asyncio
    async def test_analyze_frustrated_sentiment(self, analyzer):
        """Test analysis of frustrated sentiment indicators."""
        frustrated_texts = [
            "This build is stuck again, so annoying!",
            "Tests keep failing for no reason, really frustrated.",
            "Deployment broke everything, this is ridiculous!",
            "Another timeout, this system is so unreliable."
        ]
        
        for text in frustrated_texts:
            result = await analyzer.analyze_text(text)
            assert result.is_frustrated or result.label == SentimentLabel.FRUSTRATED
            assert result.emotional_intensity > 0.3
    
    @pytest.mark.asyncio
    async def test_analyze_with_context(self, analyzer):
        """Test analysis with contextual information."""
        text = "Build failed"
        
        # Context with production environment should increase urgency
        prod_context = {
            'event_type': 'pipeline_failure',
            'environment': 'production',
            'consecutive_failures': 3
        }
        result_prod = await analyzer.analyze_text(text, prod_context)
        
        # Context with development environment
        dev_context = {
            'event_type': 'pipeline_failure',
            'environment': 'development',
            'consecutive_failures': 1
        }
        result_dev = await analyzer.analyze_text(text, dev_context)
        
        # Production context should result in higher urgency
        assert result_prod.urgency_score > result_dev.urgency_score
        assert result_prod.context_factors['is_production'] == True
        assert result_dev.context_factors['is_production'] == False
    
    @pytest.mark.asyncio
    async def test_analyze_emotional_intensity(self, analyzer):
        """Test emotional intensity detection."""
        high_intensity_texts = [
            "FAILED AGAIN!!! This is soooo annoying!!!",
            "Build crashed... again... seriously???",
            "Tests are brokennnnn, fix this now!!!"
        ]
        
        low_intensity_text = "Build completed with some warnings."
        
        for text in high_intensity_texts:
            result = await analyzer.analyze_text(text)
            assert result.emotional_intensity > 0.4, f"Expected high intensity for: {text}"
            assert result.context_factors['has_excessive_punctuation'] == True or \
                   result.context_factors['has_repeated_chars'] == True
        
        result_low = await analyzer.analyze_text(low_intensity_text)
        assert result_low.emotional_intensity < 0.3
    
    @pytest.mark.asyncio
    async def test_keyword_extraction(self, analyzer):
        """Test sentiment keyword extraction."""
        text = "Build failed with error, urgent fix needed!"
        result = await analyzer.analyze_text(text)
        
        expected_keywords = {'failed', 'error', 'urgent'}
        actual_keywords = set(result.keywords)
        
        # Should extract at least some expected keywords
        assert len(expected_keywords.intersection(actual_keywords)) > 0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, analyzer):
        """Test confidence score calculation."""
        # Clear positive text should have high confidence
        clear_positive = "Excellent work! Tests passed perfectly and deployment successful."
        result_clear = await analyzer.analyze_text(clear_positive)
        assert result_clear.confidence > 0.5
        
        # Ambiguous text should have lower confidence
        ambiguous = "The build finished."
        result_ambiguous = await analyzer.analyze_text(ambiguous)
        assert result_ambiguous.confidence < 0.4
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, analyzer):
        """Test batch sentiment analysis."""
        texts = [
            "Build successful!",
            "Tests failed miserably.",
            "Deployment completed normally.",
            "URGENT: Production down!"
        ]
        
        contexts = [
            {'event_type': 'build_success'},
            {'event_type': 'test_failure'},
            {'event_type': 'deployment'},
            {'event_type': 'pipeline_failure', 'environment': 'production'}
        ]
        
        results = await analyzer.analyze_batch(texts, contexts)
        
        assert len(results) == len(texts)
        assert all(isinstance(result, SentimentResult) for result in results)
        
        # Check that results make sense
        assert results[0].polarity > 0  # "Build successful!"
        assert results[1].polarity < 0  # "Tests failed miserably."
        assert results[3].is_urgent     # "URGENT: Production down!"
    
    @pytest.mark.asyncio
    async def test_pipeline_event_analysis(self, analyzer):
        """Test pipeline-specific event analysis."""
        result = await analyzer.analyze_pipeline_event(
            event_type='pipeline_failure',
            message='Build timeout after 30 minutes, blocked deployment.',
            metadata={
                'environment': 'production',
                'consecutive_failures': 2,
                'repository': 'critical-service'
            }
        )
        
        assert result.context_factors['event_type'] == 'pipeline_failure'
        assert result.context_factors['is_production'] == True
        assert result.context_factors['consecutive_failures'] == 2
        assert result.urgency_score > 0.5  # Should be urgent due to production context
    
    @pytest.mark.asyncio
    async def test_sentiment_summary(self, analyzer):
        """Test sentiment summary generation."""
        # Create multiple results with different sentiments
        results = []
        texts_and_labels = [
            ("Great success!", SentimentLabel.POSITIVE),
            ("URGENT: Fix needed!", SentimentLabel.URGENT),
            ("Build failed badly", SentimentLabel.NEGATIVE),
            ("Tests are broken", SentimentLabel.FRUSTRATED),
            ("Deployment works fine", SentimentLabel.NEUTRAL)
        ]
        
        for text, _ in texts_and_labels:
            result = await analyzer.analyze_text(text)
            results.append(result)
        
        summary = analyzer.get_sentiment_summary(results)
        
        assert summary['total_analyses'] == len(results)
        assert 'average_polarity' in summary
        assert 'average_urgency' in summary
        assert 'sentiment_distribution' in summary
        assert 'urgent_events' in summary
        assert 'concerning_pattern_detected' in summary
        assert 'overall_sentiment_trend' in summary
        
        # Should detect concerning pattern (urgent + frustrated events)
        assert summary['concerning_pattern_detected'] == True
    
    def test_clean_text(self, analyzer):
        """Test text cleaning functionality."""
        dirty_text = "Check https://example.com and email test@example.com   with    lots   of   spaces"
        cleaned = analyzer._clean_text(dirty_text)
        
        # Should remove URLs and emails, normalize whitespace
        assert "https://example.com" not in cleaned
        assert "test@example.com" not in cleaned
        assert "   " not in cleaned  # Multiple spaces should be normalized
    
    def test_urgency_patterns(self, analyzer):
        """Test urgency pattern detection."""
        # Test excessive punctuation pattern
        assert analyzer.excessive_punct_pattern.search("Help!!!")
        assert analyzer.excessive_punct_pattern.search("Really???")
        assert not analyzer.excessive_punct_pattern.search("Normal text.")
        
        # Test caps pattern  
        assert analyzer.caps_pattern.search("URGENT HELP NEEDED")
        assert not analyzer.caps_pattern.search("normal text here")
        
        # Test time urgency pattern
        assert analyzer.time_urgency_pattern.search("Fix this today")
        assert analyzer.time_urgency_pattern.search("Need ASAP")
        assert not analyzer.time_urgency_pattern.search("scheduled for next week")
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, analyzer):
        """Test that processing time is tracked."""
        text = "Test message for processing time"
        result = await analyzer.analyze_text(text)
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, analyzer):
        """Test that analyzer can handle concurrent requests."""
        texts = [f"Test message {i}" for i in range(10)]
        
        # Start multiple analyses concurrently
        tasks = [analyzer.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(texts)
        assert all(isinstance(result, SentimentResult) for result in results)
    
    def test_sentiment_properties(self, analyzer):
        """Test SentimentResult properties."""
        # Create a mock result
        result = SentimentResult(
            label=SentimentLabel.URGENT,
            confidence=0.8,
            polarity=-0.3,
            subjectivity=0.7,
            urgency_score=0.9,
            emotional_intensity=0.6,
            keywords=['urgent', 'failed'],
            context_factors={},
            processing_time_ms=10.5
        )
        
        assert result.is_urgent == True  # urgency_score > 0.7
        assert result.is_negative == True  # polarity < -0.1
        assert result.is_frustrated == False  # not explicitly frustrated
        
        # Test frustrated result
        frustrated_result = SentimentResult(
            label=SentimentLabel.FRUSTRATED,
            confidence=0.7,
            polarity=-0.4,
            subjectivity=0.8,
            urgency_score=0.5,
            emotional_intensity=0.7,
            keywords=['frustrated', 'annoying'],
            context_factors={},
            processing_time_ms=8.2
        )
        
        assert frustrated_result.is_frustrated == True


class TestGlobalSentimentAnalyzer:
    """Test the global sentiment analyzer instance."""
    
    def test_global_instance_exists(self):
        """Test that global sentiment analyzer instance is available."""
        from healing_guard.core.sentiment_analyzer import sentiment_analyzer
        assert sentiment_analyzer is not None
        assert isinstance(sentiment_analyzer, PipelineSentimentAnalyzer)
    
    @pytest.mark.asyncio
    async def test_global_instance_functionality(self):
        """Test that global instance works correctly."""
        result = await sentiment_analyzer.analyze_text("Test message")
        assert isinstance(result, SentimentResult)
        assert result.processing_time_ms > 0


# Integration-style tests
class TestSentimentAnalyzerIntegration:
    """Integration tests for sentiment analyzer with realistic scenarios."""
    
    @pytest.fixture
    def analyzer(self):
        return PipelineSentimentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_ci_cd_failure_scenarios(self, analyzer):
        """Test realistic CI/CD failure scenarios."""
        scenarios = [
            {
                'text': 'Docker build failed: No space left on device',
                'context': {'event_type': 'build_error'},
                'expected_urgency': 'medium',
                'expected_sentiment': 'negative'
            },
            {
                'text': 'Tests flaky again, 3rd retry failed. Really annoying!',
                'context': {'event_type': 'test_failure', 'consecutive_failures': 3},
                'expected_urgency': 'high',
                'expected_sentiment': 'frustrated'
            },
            {
                'text': 'PRODUCTION DEPLOYMENT FAILED - Customer impact! Need immediate rollback!',
                'context': {'event_type': 'deployment_error', 'environment': 'production'},
                'expected_urgency': 'critical',
                'expected_sentiment': 'urgent'
            },
            {
                'text': 'All tests passing, ready to merge. Great work team!',
                'context': {'event_type': 'test_success'},
                'expected_urgency': 'low',
                'expected_sentiment': 'positive'
            }
        ]
        
        for scenario in scenarios:
            result = await analyzer.analyze_text(scenario['text'], scenario['context'])
            
            # Check urgency expectations
            if scenario['expected_urgency'] == 'critical':
                assert result.urgency_score > 0.8 or result.is_urgent
            elif scenario['expected_urgency'] == 'high':
                assert result.urgency_score > 0.6
            elif scenario['expected_urgency'] == 'medium':
                assert 0.3 < result.urgency_score <= 0.6
            else:  # low
                assert result.urgency_score <= 0.4
            
            # Check sentiment expectations
            if scenario['expected_sentiment'] == 'urgent':
                assert result.label == SentimentLabel.URGENT or result.is_urgent
            elif scenario['expected_sentiment'] == 'frustrated':
                assert result.is_frustrated or result.label == SentimentLabel.FRUSTRATED
            elif scenario['expected_sentiment'] == 'negative':
                assert result.is_negative
            elif scenario['expected_sentiment'] == 'positive':
                assert result.polarity > 0
    
    @pytest.mark.asyncio
    async def test_developer_communication_scenarios(self, analyzer):
        """Test realistic developer communication scenarios."""
        commit_messages = [
            "fix: resolve critical memory leak in authentication service",
            "feat: add awesome new dashboard with real-time metrics",
            "refactor: clean up terrible legacy code that nobody understands",
            "hotfix: URGENT production bug causing user login failures"
        ]
        
        pr_comments = [
            "LGTM! Great implementation, very clean code.",
            "This looks broken, tests are failing everywhere.",
            "Please fix the obvious bugs before merging.",
            "Awesome work! This will solve our performance issues."
        ]
        
        all_texts = commit_messages + pr_comments
        results = await analyzer.analyze_batch(all_texts)
        
        # Should have varied sentiment results
        sentiments = [result.label for result in results]
        assert len(set(sentiments)) > 2  # Should have variety in sentiments
        
        # Should detect at least one urgent/critical item
        urgent_count = sum(1 for result in results if result.is_urgent)
        assert urgent_count >= 1
    
    @pytest.mark.asyncio
    async def test_performance_with_large_text(self, analyzer):
        """Test performance with large text inputs."""
        # Create a large text (near the limit)
        large_text = "Build failed. " * 1000  # About 13KB
        
        result = await analyzer.analyze_text(large_text)
        
        # Should complete in reasonable time (< 1 second)
        assert result.processing_time_ms < 1000
        assert isinstance(result, SentimentResult)
        assert result.confidence > 0  # Should still produce meaningful results