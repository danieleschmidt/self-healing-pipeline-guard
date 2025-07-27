"""
Performance tests for the Self-Healing Pipeline Guard.

Tests system performance under various load conditions and validates
response times, throughput, and resource usage.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, AsyncMock

import pytest
from httpx import AsyncClient


class TestLoadPerformance:
    """Performance tests for high-load scenarios."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_webhook_throughput(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis
    ):
        """Test webhook processing throughput."""
        num_requests = 100
        start_time = time.time()
        
        with patch('healing_guard.services.redis_client', mock_redis):
            # Create tasks for concurrent requests
            tasks = []
            for i in range(num_requests):
                webhook = sample_github_webhook.copy()
                webhook["workflow_run"]["id"] = 1000000 + i
                
                task = async_client.post(
                    "/webhooks/github",
                    json=webhook,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": "sha256=test-signature"
                    }
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_responses = [
            r for r in responses 
            if hasattr(r, 'status_code') and r.status_code == 200
        ]
        
        throughput = len(successful_responses) / duration
        
        # Performance assertions
        assert len(successful_responses) >= num_requests * 0.95  # 95% success rate
        assert throughput >= 50  # At least 50 requests/second
        assert duration < 5.0  # Complete within 5 seconds
        
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Success rate: {len(successful_responses)/num_requests*100:.1f}%")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_healing_engine_performance(
        self,
        mock_redis,
        sample_pipeline_failure,
        mock_ml_model
    ):
        """Test healing engine performance with multiple concurrent failures."""
        from healing_guard.core.healing_engine import HealingEngine
        
        engine = HealingEngine()
        num_failures = 50
        
        # Create multiple failure scenarios
        failures = []
        for i in range(num_failures):
            failure = sample_pipeline_failure.copy()
            failure["id"] = f"perf-test-{i}"
            failures.append(failure)
        
        start_time = time.time()
        
        with patch.object(engine, 'redis_client', mock_redis), \
             patch.object(engine, 'ml_model', mock_ml_model):
            
            # Process failures concurrently
            tasks = [engine.heal_failure(failure) for failure in failures]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze performance
        successful_healings = [r for r in results if not isinstance(r, Exception)]
        healing_rate = len(successful_healings) / duration
        
        assert len(successful_healings) >= num_failures * 0.9  # 90% success rate
        assert healing_rate >= 20  # At least 20 healings/second
        assert duration < 5.0  # Complete within 5 seconds
        
        print(f"Healing rate: {healing_rate:.2f} healings/second")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis
    ):
        """Test memory usage during high load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        num_requests = 200
        
        with patch('healing_guard.services.redis_client', mock_redis):
            # Send many requests
            for batch in range(10):
                tasks = []
                for i in range(20):
                    webhook = sample_github_webhook.copy()
                    webhook["workflow_run"]["id"] = 2000000 + (batch * 20) + i
                    
                    task = async_client.post(
                        "/webhooks/github",
                        json=webhook,
                        headers={
                            "X-GitHub-Event": "workflow_run",
                            "X-Hub-Signature-256": "sha256=test-signature"
                        }
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check memory after each batch
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_increase < 100  # Less than 100MB increase
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Memory increase: {total_increase:.2f} MB")
        assert total_increase < 150  # Total increase should be reasonable

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_database_query_performance(
        self,
        db_session,
        sample_pipeline_failure
    ):
        """Test database query performance under load."""
        from healing_guard.database.repositories import FailureRepository
        
        repo = FailureRepository(db_session)
        num_records = 1000
        
        # Insert test data
        start_time = time.time()
        
        for i in range(num_records):
            failure = sample_pipeline_failure.copy()
            failure["id"] = f"db-perf-{i}"
            await repo.create_failure(failure)
        
        insert_time = time.time() - start_time
        
        # Query performance tests
        start_time = time.time()
        
        # Test various query patterns
        await repo.get_failures_by_repository("test-org/test-repo")
        await repo.get_recent_failures(limit=100)
        await repo.get_failure_statistics()
        
        query_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 10.0  # Bulk insert within 10 seconds
        assert query_time < 2.0   # Queries within 2 seconds
        
        print(f"Insert rate: {num_records/insert_time:.2f} records/second")
        print(f"Query time: {query_time:.3f} seconds")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ml_model_inference_performance(
        self,
        mock_ml_model,
        sample_pipeline_failure
    ):
        """Test ML model inference performance."""
        from healing_guard.ml.inference import ModelInference
        
        inference = ModelInference(mock_ml_model)
        num_inferences = 500
        
        # Prepare test data
        test_cases = []
        for i in range(num_inferences):
            failure = sample_pipeline_failure.copy()
            failure["logs"] = [f"Test log {i}", "Error occurred", "Build failed"]
            test_cases.append(failure)
        
        start_time = time.time()
        
        # Run batch inference
        results = await inference.batch_predict(test_cases)
        
        end_time = time.time()
        duration = end_time - start_time
        
        inference_rate = num_inferences / duration
        
        assert len(results) == num_inferences
        assert inference_rate >= 100  # At least 100 inferences/second
        assert duration < 10.0  # Complete within 10 seconds
        
        print(f"ML inference rate: {inference_rate:.2f} inferences/second")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_redis_performance(
        self,
        mock_redis
    ):
        """Test Redis operations performance."""
        num_operations = 1000
        
        # Test SET operations
        start_time = time.time()
        
        tasks = []
        for i in range(num_operations):
            task = mock_redis.set(f"perf_key_{i}", f"value_{i}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        set_time = time.time() - start_time
        
        # Test GET operations
        start_time = time.time()
        
        tasks = []
        for i in range(num_operations):
            task = mock_redis.get(f"perf_key_{i}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        get_time = time.time() - start_time
        
        set_rate = num_operations / set_time
        get_rate = num_operations / get_time
        
        # Redis should handle high operation rates
        assert set_rate >= 1000  # At least 1000 sets/second
        assert get_rate >= 1000  # At least 1000 gets/second
        
        print(f"Redis SET rate: {set_rate:.2f} ops/second")
        print(f"Redis GET rate: {get_rate:.2f} ops/second")

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_performance(
        self,
        async_client: AsyncClient,
        sample_github_webhook,
        mock_redis
    ):
        """Test performance under sustained load."""
        duration_seconds = 30  # 30 second test
        requests_per_second = 10
        
        total_requests = 0
        successful_requests = 0
        start_time = time.time()
        
        with patch('healing_guard.services.redis_client', mock_redis):
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                
                # Send batch of requests
                tasks = []
                for i in range(requests_per_second):
                    webhook = sample_github_webhook.copy()
                    webhook["workflow_run"]["id"] = 3000000 + total_requests + i
                    
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
                
                # Count successful responses
                successful_batch = sum(
                    1 for r in responses 
                    if hasattr(r, 'status_code') and r.status_code == 200
                )
                
                total_requests += len(tasks)
                successful_requests += successful_batch
                
                # Wait for next second
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        success_rate = successful_requests / total_requests
        avg_throughput = successful_requests / actual_duration
        
        print(f"Sustained load test results:")
        print(f"Duration: {actual_duration:.2f} seconds")
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {successful_requests}")
        print(f"Success rate: {success_rate*100:.2f}%")
        print(f"Average throughput: {avg_throughput:.2f} requests/second")
        
        # Performance criteria for sustained load
        assert success_rate >= 0.95  # 95% success rate
        assert avg_throughput >= 8   # At least 8 requests/second sustained

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_healing_performance(
        self,
        mock_redis,
        mock_github_client,
        sample_pipeline_failure
    ):
        """Test performance of concurrent healing operations."""
        from healing_guard.core.healing_engine import HealingEngine
        
        engine = HealingEngine()
        num_concurrent_healings = 25
        
        # Create different failure scenarios
        failures = []
        for i in range(num_concurrent_healings):
            failure = sample_pipeline_failure.copy()
            failure["id"] = f"concurrent-{i}"
            failure["failure_type"] = ["test_failure", "timeout", "oom"][i % 3]
            failures.append(failure)
        
        start_time = time.time()
        
        with patch.object(engine, 'redis_client', mock_redis), \
             patch('healing_guard.integrations.github_client', mock_github_client):
            
            # Execute healing operations concurrently
            tasks = [engine.heal_failure(failure) for failure in failures]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful_healings = [r for r in results if not isinstance(r, Exception)]
        
        assert len(successful_healings) >= num_concurrent_healings * 0.9
        assert duration < 3.0  # Should complete within 3 seconds
        
        healing_rate = len(successful_healings) / duration
        print(f"Concurrent healing rate: {healing_rate:.2f} healings/second")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_response_times(
        self,
        async_client: AsyncClient,
        mock_redis
    ):
        """Test API endpoint response times."""
        endpoints = [
            ("/health", "GET"),
            ("/metrics/summary", "GET"),
            ("/patterns/recent", "GET"),
            ("/healing/status/recent", "GET")
        ]
        
        with patch('healing_guard.services.redis_client', mock_redis):
            for endpoint, method in endpoints:
                times = []
                
                # Test each endpoint multiple times
                for _ in range(10):
                    start_time = time.time()
                    
                    if method == "GET":
                        response = await async_client.get(endpoint)
                    
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # milliseconds
                    times.append(response_time)
                
                avg_time = sum(times) / len(times)
                max_time = max(times)
                
                print(f"{endpoint}: avg={avg_time:.2f}ms, max={max_time:.2f}ms")
                
                # Response time assertions
                assert avg_time < 100  # Average under 100ms
                assert max_time < 500  # Max under 500ms