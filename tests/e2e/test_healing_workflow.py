"""
End-to-end tests for the complete healing workflow.

Tests the entire pipeline from failure detection through healing
execution and verification.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient


class TestHealingWorkflowE2E:
    """End-to-end tests for the healing workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_healing_workflow_success(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client,
        mock_ml_model
    ):
        """Test complete successful healing workflow."""
        # Setup successful healing scenario
        mock_github_client.trigger_workflow_run.return_value = {"id": "new-run-123"}
        mock_ml_model.predict_proba.return_value = [[0.1, 0.9]]  # High flaky confidence
        
        # Step 1: Webhook triggers failure detection
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.ml.models.flaky_test_model', mock_ml_model):
            
            webhook_response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            assert webhook_response.status_code == 200
            event_id = webhook_response.json()["event_id"]
            
            # Step 2: Wait for processing (simulate async processing)
            await asyncio.sleep(0.1)
            
            # Step 3: Check healing status
            status_response = await async_client.get(f"/healing/{event_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["status"] == "completed"
            assert status_data["healing_attempted"] is True
            assert status_data["healing_successful"] is True
            assert status_data["strategy_used"] == "flaky_test_retry"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_healing_workflow_with_escalation(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client,
        mock_slack_client,
        mock_jira_client
    ):
        """Test healing workflow that requires escalation."""
        # Setup failed healing scenario
        mock_github_client.trigger_workflow_run.side_effect = Exception("API Error")
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.integrations.slack_client', mock_slack_client), \
             patch('healing_guard.integrations.jira_client', mock_jira_client):
            
            # Trigger webhook
            webhook_response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            event_id = webhook_response.json()["event_id"]
            
            # Wait for processing and escalation
            await asyncio.sleep(0.2)
            
            # Check that escalation occurred
            status_response = await async_client.get(f"/healing/{event_id}/status")
            status_data = status_response.json()
            
            assert status_data["healing_attempted"] is True
            assert status_data["healing_successful"] is False
            assert status_data["escalated"] is True
            
            # Verify notifications were sent
            mock_slack_client.chat_postMessage.assert_called()
            mock_jira_client.create_issue.assert_called()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_strategy_healing_workflow(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_github_client
    ):
        """Test workflow with multiple healing strategies."""
        # Create failure that matches multiple strategies
        complex_webhook = {
            "action": "completed",
            "workflow_run": {
                "id": 987654321,
                "status": "completed",
                "conclusion": "failure",
                "repository": {"full_name": "test/complex-repo"},
                "head_commit": {"id": "xyz789abc"},
                "head_branch": "main"
            }
        }
        
        # Mock multiple strategy attempts
        strategy_results = [
            {"success": False, "strategy": "flaky_test_retry"},
            {"success": False, "strategy": "resource_scaling"},
            {"success": True, "strategy": "cache_invalidation"}
        ]
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.core.healing_engine.HealingEngine.execute_strategies', 
                   return_value=strategy_results):
            
            response = await async_client.post(
                "/webhooks/github",
                json=complex_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            event_id = response.json()["event_id"]
            await asyncio.sleep(0.1)
            
            # Check final status
            status_response = await async_client.get(f"/healing/{event_id}/status")
            status_data = status_response.json()
            
            assert status_data["healing_successful"] is True
            assert status_data["strategy_used"] == "cache_invalidation"
            assert len(status_data["attempted_strategies"]) == 3

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_cost_tracking_throughout_workflow(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client
    ):
        """Test cost tracking throughout the healing workflow."""
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Trigger healing
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            event_id = response.json()["event_id"]
            await asyncio.sleep(0.1)
            
            # Check cost tracking
            cost_response = await async_client.get(f"/healing/{event_id}/cost")
            assert cost_response.status_code == 200
            
            cost_data = cost_response.json()
            assert "analysis_cost" in cost_data
            assert "healing_cost" in cost_data
            assert "total_cost" in cost_data
            assert cost_data["total_cost"] >= 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_metrics_collection_e2e(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client
    ):
        """Test metrics collection throughout the workflow."""
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Process multiple failures
            for i in range(3):
                webhook = sample_github_webhook.copy()
                webhook["workflow_run"]["id"] = 123456789 + i
                
                await async_client.post(
                    "/webhooks/github",
                    json=webhook,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": "sha256=test-signature"
                    }
                )
            
            await asyncio.sleep(0.2)
            
            # Check aggregated metrics
            metrics_response = await async_client.get("/metrics/summary")
            assert metrics_response.status_code == 200
            
            metrics_data = metrics_response.json()
            assert metrics_data["total_failures"] >= 3
            assert "healing_success_rate" in metrics_data
            assert "avg_resolution_time" in metrics_data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_historical_learning_workflow(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_github_client,
        mock_ml_model
    ):
        """Test that the system learns from historical patterns."""
        # Create multiple similar failures
        base_webhook = {
            "action": "completed",
            "workflow_run": {
                "status": "completed",
                "conclusion": "failure",
                "repository": {"full_name": "test/learning-repo"},
                "head_branch": "main"
            }
        }
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.ml.models.flaky_test_model', mock_ml_model):
            
            # Process several similar failures
            for i in range(5):
                webhook = base_webhook.copy()
                webhook["workflow_run"]["id"] = 555000000 + i
                
                await async_client.post(
                    "/webhooks/github",
                    json=webhook,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": "sha256=test-signature"
                    }
                )
            
            await asyncio.sleep(0.3)
            
            # Check that patterns were learned
            patterns_response = await async_client.get(
                "/patterns/repository/test/learning-repo"
            )
            assert patterns_response.status_code == 200
            
            patterns_data = patterns_response.json()
            assert len(patterns_data["learned_patterns"]) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_real_time_dashboard_updates(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client
    ):
        """Test real-time dashboard updates during workflow."""
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Trigger healing
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            event_id = response.json()["event_id"]
            
            # Check dashboard updates at different stages
            stages = ["processing", "analyzing", "healing", "completed"]
            
            for stage in stages:
                await asyncio.sleep(0.05)
                
                dashboard_response = await async_client.get("/dashboard/live")
                assert dashboard_response.status_code == 200
                
                dashboard_data = dashboard_response.json()
                assert "active_healings" in dashboard_data
                assert "recent_activities" in dashboard_data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_api_rate_limiting_during_load(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis,
        mock_github_client
    ):
        """Test API rate limiting during high load."""
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Send many requests quickly
            tasks = []
            for i in range(20):
                webhook = sample_github_webhook.copy()
                webhook["workflow_run"]["id"] = 777000000 + i
                
                task = async_client.post(
                    "/webhooks/github",
                    json=webhook,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": "sha256=test-signature"
                    }
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some should be rate limited
            successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
            rate_limited = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 429]
            
            assert len(successful) > 0
            assert len(rate_limited) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_cross_platform_workflow(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_github_client,
        mock_gitlab_client
    ):
        """Test workflow across multiple CI/CD platforms."""
        # GitHub failure
        github_webhook = sample_github_webhook
        
        # GitLab failure
        gitlab_webhook = {
            "object_kind": "pipeline",
            "object_attributes": {
                "id": 98765,
                "status": "failed",
                "ref": "main",
                "sha": "def456ghi789"
            },
            "project": {"path_with_namespace": "test/cross-platform"}
        }
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.integrations.gitlab_client', mock_gitlab_client):
            
            # Process both platforms
            github_response = await async_client.post(
                "/webhooks/github",
                json=github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            gitlab_response = await async_client.post(
                "/webhooks/gitlab",
                json=gitlab_webhook,
                headers={
                    "X-Gitlab-Event": "Pipeline Hook",
                    "X-Gitlab-Token": "test-token"
                }
            )
            
            assert github_response.status_code == 200
            assert gitlab_response.status_code == 200
            
            await asyncio.sleep(0.2)
            
            # Check cross-platform analytics
            analytics_response = await async_client.get("/analytics/cross-platform")
            assert analytics_response.status_code == 200
            
            analytics_data = analytics_response.json()
            assert "platform_statistics" in analytics_data
            assert len(analytics_data["platform_statistics"]) >= 2

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_running_healing_workflow(
        self,
        async_client: AsyncClient,
        mock_redis,
        mock_github_client
    ):
        """Test long-running healing workflow with timeouts."""
        # Mock slow healing process
        async def slow_healing(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow healing
            return {"success": True, "strategy": "slow_strategy"}
        
        with patch('healing_guard.services.redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client), \
             patch('healing_guard.core.healing_engine.HealingEngine.execute_healing', 
                   side_effect=slow_healing):
            
            # Trigger slow healing
            response = await async_client.post(
                "/webhooks/github",
                json=sample_github_webhook,
                headers={
                    "X-GitHub-Event": "workflow_run",
                    "X-Hub-Signature-256": "sha256=test-signature"
                }
            )
            
            event_id = response.json()["event_id"]
            
            # Check status during processing
            status_response = await async_client.get(f"/healing/{event_id}/status")
            status_data = status_response.json()
            assert status_data["status"] in ["processing", "healing"]
            
            # Wait for completion
            await asyncio.sleep(3)
            
            # Check final status
            final_response = await async_client.get(f"/healing/{event_id}/status")
            final_data = final_response.json()
            assert final_data["status"] == "completed"