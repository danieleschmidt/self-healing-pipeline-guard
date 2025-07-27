"""
Integration tests for webhook processing pipeline.

Tests the complete flow from webhook reception to failure analysis
and healing initiation.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestWebhookProcessing:
    """Integration tests for webhook processing."""

    @pytest.mark.asyncio
    async def test_github_webhook_processing(
        self, 
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client
    ):
        """Test complete GitHub webhook processing flow."""
        # Mock external dependencies
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Send webhook
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "accepted"
            assert "event_id" in response_data

    @pytest.mark.asyncio
    async def test_gitlab_webhook_processing(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_gitlab_client
    ):
        """Test GitLab webhook processing."""
        gitlab_webhook = {
            "object_kind": "pipeline",
            "object_attributes": {
                "id": 12345,
                "status": "failed",
                "ref": "main",
                "sha": "abc123def456",
                "web_url": "https://gitlab.com/test/repo/-/pipelines/12345"
            },
            "project": {
                "path_with_namespace": "test/repo",
                "web_url": "https://gitlab.com/test/repo"
            },
            "commit": {
                "id": "abc123def456",
                "message": "Fix bug in calculation",
                "author_email": "developer@example.com"
            }
        }
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.gitlab_client', mock_gitlab_client):
            
            response = await async_client.post(
                "/webhooks/gitlab",
                json=gitlab_webhook,
                headers={
                    "X-Gitlab-Event": "Pipeline Hook",
                    "X-Gitlab-Token": "test-token"
                }
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_jenkins_webhook_processing(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_jenkins_client
    ):
        """Test Jenkins webhook processing."""
        jenkins_webhook = {
            "name": "test-job",
            "url": "job/test-job/",
            "build": {
                "number": 42,
                "phase": "COMPLETED",
                "status": "FAILURE",
                "url": "job/test-job/42/",
                "full_url": "http://jenkins.example.com/job/test-job/42/",
                "parameters": {
                    "BRANCH": "main",
                    "COMMIT": "abc123def456"
                }
            }
        }
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.jenkins_client', mock_jenkins_client):
            
            response = await async_client.post(
                "/webhooks/jenkins",
                json=jenkins_webhook,
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_webhook_signature_validation(
        self,
        async_client: AsyncClient,
        sample_github_webhook
    ):
        """Test webhook signature validation."""
        # Test with invalid signature
        response = await async_client.post(
            "/webhooks/github",
            json=sample_github_webhook,
            headers={
                "X-GitHub-Event": "workflow_run",
                "X-Hub-Signature-256": "sha256=invalid-signature"
            }
        )
        
        assert response.status_code == 401
        assert "Invalid signature" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_webhook_duplicate_handling(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis
    ):
        """Test handling of duplicate webhooks."""
        # Mock Redis to return existing event
        mock_redis.exists.return_value = True
        
        with patch('healing_guard.services.redis_client', mock_redis):
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_event_processing_pipeline(
        self,
        mock_redis,
        sample_pipeline_failure,
        mock_ml_model
    ):
        """Test the complete event processing pipeline."""
        from healing_guard.services.event_processor import EventProcessor
        from healing_guard.core.healing_engine import HealingEngine
        
        # Setup mocks
        processor = EventProcessor()
        healing_engine = HealingEngine()
        
        with patch.object(processor, 'redis_client', mock_redis), \
             patch.object(healing_engine, 'ml_model', mock_ml_model):
            
            # Process failure event
            result = await processor.process_failure_event(sample_pipeline_failure)
            
            assert result.success is True
            assert result.healing_attempted is True
            assert result.strategy_used is not None

    @pytest.mark.asyncio
    async def test_rate_limiting(
        self,
        async_client: AsyncClient,
        sample_github_webhook
    ):
        """Test webhook rate limiting."""
        # Send multiple requests quickly
        responses = []
        for i in range(10):
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            responses.append(response)
        
        # Some requests should be rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0

    @pytest.mark.asyncio
    async def test_webhook_payload_validation(
        self,
        async_client: AsyncClient
    ):
        """Test webhook payload validation."""
        # Test with invalid payload
        invalid_payload = {"invalid": "data"}
        
        response = await async_client.post(
            "/webhooks/github",
            json=invalid_payload,
            headers={
                "X-GitHub-Event": "workflow_run",
                "X-Hub-Signature-256": "sha256=test-signature"
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_event_queue_processing(
        self,
        mock_redis,
        sample_pipeline_failure
    ):
        """Test event queue processing with Redis Streams."""
        from healing_guard.services.event_queue import EventQueue
        
        queue = EventQueue(mock_redis)
        
        # Add event to queue
        event_id = await queue.add_event("failures", sample_pipeline_failure)
        assert event_id is not None
        
        # Process events from queue
        events = await queue.get_events("failures", count=1)
        assert len(events) == 1
        assert events[0]["data"] == sample_pipeline_failure

    @pytest.mark.asyncio
    async def test_failure_escalation_workflow(
        self,
        mock_redis,
        mock_slack_client,
        mock_jira_client,
        sample_pipeline_failure
    ):
        """Test failure escalation workflow."""
        from healing_guard.services.escalation import EscalationService
        
        # Mock failed healing attempts
        sample_pipeline_failure["healing_attempts"] = 5
        sample_pipeline_failure["last_healing_failed"] = True
        
        escalation_service = EscalationService(
            slack_client=mock_slack_client,
            jira_client=mock_jira_client
        )
        
        with patch('healing_guard.services.redis_client', mock_redis):
            result = await escalation_service.escalate_failure(sample_pipeline_failure)
            
            assert result.incident_created is True
            assert result.notifications_sent is True
            mock_slack_client.chat_postMessage.assert_called()
            mock_jira_client.create_issue.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_collection_during_processing(
        self,
        mock_redis,
        mock_metrics_collector,
        sample_pipeline_failure
    ):
        """Test metrics collection during event processing."""
        from healing_guard.services.event_processor import EventProcessor
        
        processor = EventProcessor(metrics_collector=mock_metrics_collector)
        
        with patch.object(processor, 'redis_client', mock_redis):
            await processor.process_failure_event(sample_pipeline_failure)
            
            # Verify metrics were recorded
            mock_metrics_collector.record_failure.assert_called()
            mock_metrics_collector.record_healing_attempt.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_webhook_processing(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis
    ):
        """Test concurrent webhook processing."""
        import asyncio
        
        # Create multiple webhook payloads
        webhooks = []
        for i in range(5):
            webhook = sample_github_webhook.copy()
            webhook["workflow_run"]["id"] = 123456789 + i
            webhooks.append(webhook)
        
        with patch('healing_guard.services.redis_client', mock_redis):
            # Process webhooks concurrently
            tasks = [
                async_client.post(
                    "/webhooks/github",
                    json=webhook,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": "sha256=test-signature"
                    }
                )
                for webhook in webhooks
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_webhook_error_handling(
        self,
        async_client: AsyncClient,
        sample_github_webhook
    ):
        """Test webhook processing error handling."""
        # Mock a processing error
        with patch('healing_guard.services.event_processor.EventProcessor.process_failure_event', 
                   side_effect=Exception("Processing error")):
            
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            # Should handle error gracefully
            assert response.status_code == 500
            assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_platform_specific_processing(
        self,
        async_client: AsyncClient,
        mock_redis
    ):
        """Test platform-specific processing logic."""
        # Test different platforms have different processing
        platforms = ["github", "gitlab", "jenkins", "circleci"]
        
        for platform in platforms:
            with patch('healing_guard.services.redis_client', mock_redis):
                response = await async_client.get(f"/webhooks/{platform}/health")
                assert response.status_code == 200
                
                health_data = response.json()
                assert health_data["platform"] == platform
                assert "supported_events" in health_data