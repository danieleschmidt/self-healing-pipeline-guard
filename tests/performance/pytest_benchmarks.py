"""
Pytest benchmark tests for Self-Healing Pipeline Guard.
Tests performance of critical functions and algorithms.
"""

import asyncio
import json
import random
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures.pipeline_data import (
    create_sample_pipeline_failure,
    create_github_webhook_payload,
    create_test_failure_scenarios
)


class TestFailureDetectionPerformance:
    """Benchmark tests for failure detection algorithms."""
    
    @pytest.mark.benchmark(group="failure_detection")
    def test_parse_github_webhook_performance(self, benchmark: BenchmarkFixture):
        """Benchmark GitHub webhook parsing performance."""
        webhook_payload = create_github_webhook_payload()
        
        def parse_webhook():
            # Simulate webhook parsing logic
            return {
                "platform": "github",
                "repository": webhook_payload["workflow_run"]["repository"]["full_name"],
                "status": webhook_payload["workflow_run"]["conclusion"],
                "commit_sha": webhook_payload["workflow_run"]["head_commit"]["id"],
                "branch": webhook_payload["workflow_run"]["head_branch"]
            }
        
        result = benchmark(parse_webhook)
        assert result["platform"] == "github"
    
    @pytest.mark.benchmark(group="failure_detection")
    def test_failure_classification_performance(self, benchmark: BenchmarkFixture):
        """Benchmark failure classification algorithm."""
        failure_scenarios = create_test_failure_scenarios()
        
        def classify_failures():
            classifications = []
            for failure in failure_scenarios:
                # Simulate ML classification logic
                error_message = failure["error_message"].lower()
                
                if "timeout" in error_message:
                    classification = "network_timeout"
                elif "memory" in error_message:
                    classification = "resource_exhaustion"
                elif "flaky" in error_message or "random" in error_message:
                    classification = "flaky_test"
                elif "dependency" in error_message:
                    classification = "dependency_conflict"
                else:
                    classification = "unknown"
                
                classifications.append({
                    "failure_id": failure["id"],
                    "classification": classification,
                    "confidence": random.uniform(0.7, 0.95)
                })
            
            return classifications
        
        result = benchmark(classify_failures)
        assert len(result) == len(failure_scenarios)
    
    @pytest.mark.benchmark(group="failure_detection")
    def test_log_analysis_performance(self, benchmark: BenchmarkFixture):
        """Benchmark log analysis performance."""
        # Create large log set for performance testing
        logs = []
        for i in range(1000):
            logs.extend([
                f"[{time.time():.3f}] INFO: Starting process {i}",
                f"[{time.time():.3f}] DEBUG: Processing item {i}",
                f"[{time.time():.3f}] ERROR: Failed to process item {i}",
                f"[{time.time():.3f}] INFO: Retrying process {i}"
            ])
        
        def analyze_logs():
            error_patterns = []
            error_keywords = ["error", "failed", "exception", "timeout", "killed"]
            
            for log_line in logs:
                log_lower = log_line.lower()
                for keyword in error_keywords:
                    if keyword in log_lower:
                        error_patterns.append({
                            "line": log_line,
                            "keyword": keyword,
                            "timestamp": time.time()
                        })
                        break
            
            return error_patterns
        
        result = benchmark(analyze_logs)
        assert len(result) > 0


class TestHealingStrategyPerformance:
    """Benchmark tests for healing strategy algorithms."""
    
    @pytest.mark.benchmark(group="healing_strategies")
    def test_strategy_selection_performance(self, benchmark: BenchmarkFixture):
        """Benchmark healing strategy selection algorithm."""
        failure_data = create_sample_pipeline_failure()
        
        strategies = {
            "flaky_test_retry": {"confidence_threshold": 0.8, "cost": 1},
            "resource_scaling": {"confidence_threshold": 0.7, "cost": 5},
            "cache_invalidation": {"confidence_threshold": 0.6, "cost": 2},
            "environment_reset": {"confidence_threshold": 0.9, "cost": 10},
            "dependency_update": {"confidence_threshold": 0.75, "cost": 3}
        }
        
        def select_strategy():
            failure_type = failure_data["failure_type"]
            
            # Simulate strategy scoring
            strategy_scores = {}
            for strategy_name, config in strategies.items():
                base_score = random.uniform(0.3, 0.9)
                
                # Type-specific bonuses
                if failure_type == "flaky_test" and "retry" in strategy_name:
                    base_score += 0.2
                elif failure_type == "resource_exhaustion" and "scaling" in strategy_name:
                    base_score += 0.3
                elif failure_type == "dependency_conflict" and "dependency" in strategy_name:
                    base_score += 0.25
                
                # Cost penalty
                cost_penalty = config["cost"] * 0.01
                final_score = max(0, base_score - cost_penalty)
                
                if final_score >= config["confidence_threshold"]:
                    strategy_scores[strategy_name] = final_score
            
            # Select best strategy
            if strategy_scores:
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                return {
                    "strategy": best_strategy,
                    "confidence": strategy_scores[best_strategy],
                    "alternatives": strategy_scores
                }
            
            return {"strategy": None, "confidence": 0, "alternatives": {}}
        
        result = benchmark(select_strategy)
        assert "strategy" in result
    
    @pytest.mark.benchmark(group="healing_strategies")
    def test_retry_backoff_calculation_performance(self, benchmark: BenchmarkFixture):
        """Benchmark retry backoff calculation."""
        def calculate_backoff():
            backoff_times = []
            base_delay = 1.0
            max_delay = 300.0
            backoff_factor = 2.0
            max_retries = 10
            
            for attempt in range(max_retries):
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.1, 0.3) * delay
                final_delay = delay + jitter
                backoff_times.append(final_delay)
            
            return backoff_times
        
        result = benchmark(calculate_backoff)
        assert len(result) == 10
        assert all(t > 0 for t in result)


class TestAsyncPerformance:
    """Benchmark tests for async operations."""
    
    @pytest.mark.benchmark(group="async_operations")
    @pytest.mark.asyncio
    async def test_concurrent_webhook_processing_performance(self, benchmark: BenchmarkFixture):
        """Benchmark concurrent webhook processing."""
        async def process_webhooks():
            webhooks = [create_github_webhook_payload() for _ in range(50)]
            
            async def process_single_webhook(webhook):
                # Simulate async processing
                await asyncio.sleep(0.001)  # Simulate I/O
                return {
                    "webhook_id": webhook["workflow_run"]["id"],
                    "processed": True,
                    "repository": webhook["workflow_run"]["repository"]["full_name"]
                }
            
            # Process webhooks concurrently
            tasks = [process_single_webhook(webhook) for webhook in webhooks]
            results = await asyncio.gather(*tasks)
            return results
        
        result = await benchmark.pedantic(process_webhooks, iterations=10, rounds=3)
        assert len(result) == 50
    
    @pytest.mark.benchmark(group="async_operations")
    @pytest.mark.asyncio
    async def test_database_bulk_operations_performance(self, benchmark: BenchmarkFixture):
        """Benchmark bulk database operations."""
        async def bulk_database_operations():
            # Simulate bulk insert operations
            records = []
            for i in range(100):
                record = {
                    "id": f"record_{i}",
                    "failure_type": random.choice(["test_failure", "build_failure", "deployment_failure"]),
                    "timestamp": time.time(),
                    "metadata": {"attempt": i, "batch": True}
                }
                records.append(record)
            
            # Simulate async database bulk insert
            await asyncio.sleep(0.01)  # Simulate database I/O
            return {"inserted": len(records), "records": records}
        
        result = await benchmark.pedantic(bulk_database_operations, iterations=5, rounds=2)
        assert result["inserted"] == 100


class TestMLModelPerformance:
    """Benchmark tests for ML model operations."""
    
    @pytest.mark.benchmark(group="ml_models")
    def test_feature_extraction_performance(self, benchmark: BenchmarkFixture):
        """Benchmark feature extraction from failure data."""
        failure_data = create_sample_pipeline_failure()
        
        def extract_features():
            features = {}
            
            # Text features from error message
            error_message = failure_data["error_message"].lower()
            features["error_length"] = len(error_message)
            features["has_timeout"] = 1 if "timeout" in error_message else 0
            features["has_memory"] = 1 if "memory" in error_message else 0
            features["has_network"] = 1 if "network" in error_message else 0
            
            # Numeric features from metadata
            metadata = failure_data["metadata"]
            features["duration"] = metadata.get("duration", 0)
            features["exit_code"] = metadata.get("exit_code", 0)
            features["retry_count"] = metadata.get("retry_count", 0)
            
            # Log-based features
            logs = failure_data["logs"]
            features["log_count"] = len(logs)
            features["error_log_count"] = sum(1 for log in logs if "error" in log.lower())
            features["warning_log_count"] = sum(1 for log in logs if "warning" in log.lower())
            
            # Platform features
            features["platform_github"] = 1 if failure_data["platform"] == "github" else 0
            features["platform_gitlab"] = 1 if failure_data["platform"] == "gitlab" else 0
            features["platform_jenkins"] = 1 if failure_data["platform"] == "jenkins" else 0
            
            # Branch features
            features["is_main_branch"] = 1 if failure_data["branch"] in ["main", "master"] else 0
            
            return features
        
        result = benchmark(extract_features)
        assert len(result) > 0
        assert "error_length" in result
    
    @pytest.mark.benchmark(group="ml_models")
    def test_prediction_performance(self, benchmark: BenchmarkFixture):
        """Benchmark ML model prediction."""
        # Create mock model with realistic prediction logic
        def mock_predict(features):
            # Simulate feature preprocessing
            feature_vector = [
                features.get("error_length", 0) / 100,  # Normalize
                features.get("duration", 0) / 3600,  # Normalize to hours
                features.get("exit_code", 0) / 255,  # Normalize
                features.get("has_timeout", 0),
                features.get("has_memory", 0),
                features.get("log_count", 0) / 100  # Normalize
            ]
            
            # Simulate model computation (simplified linear model)
            weights = [0.2, 0.3, 0.1, 0.4, 0.5, 0.15]
            score = sum(f * w for f, w in zip(feature_vector, weights))
            probability = 1 / (1 + pow(2.718, -score))  # Sigmoid
            
            return {
                "prediction": "flaky" if probability > 0.5 else "not_flaky",
                "probability": probability,
                "confidence": abs(probability - 0.5) * 2
            }
        
        features = {
            "error_length": 150,
            "duration": 120,
            "exit_code": 1,
            "has_timeout": 0,
            "has_memory": 1,
            "log_count": 25
        }
        
        result = benchmark(mock_predict, features)
        assert "prediction" in result
        assert "probability" in result
        assert 0 <= result["probability"] <= 1


class TestCachePerformance:
    """Benchmark tests for caching operations."""
    
    @pytest.mark.benchmark(group="caching")
    def test_cache_key_generation_performance(self, benchmark: BenchmarkFixture):
        """Benchmark cache key generation."""
        def generate_cache_keys():
            keys = []
            base_data = {
                "repository": "test-org/test-repo",
                "platform": "github",
                "failure_type": "test_failure"
            }
            
            for i in range(1000):
                # Simulate cache key generation for different scenarios
                key_data = {**base_data, "attempt": i, "timestamp": time.time()}
                key_string = json.dumps(key_data, sort_keys=True)
                
                # Simple hash function (in real implementation, would use proper hashing)
                hash_value = hash(key_string) % (10**12)
                cache_key = f"healing_cache:{hash_value}"
                keys.append(cache_key)
            
            return keys
        
        result = benchmark(generate_cache_keys)
        assert len(result) == 1000
        assert len(set(result)) == len(result)  # All keys should be unique
    
    @pytest.mark.benchmark(group="caching")
    def test_cache_serialization_performance(self, benchmark: BenchmarkFixture):
        """Benchmark cache data serialization."""
        cache_data = {
            "failure_data": create_sample_pipeline_failure(),
            "healing_results": [
                {"strategy": "retry", "success": True, "duration": 45.2},
                {"strategy": "scaling", "success": False, "duration": 120.1}
            ],
            "metadata": {
                "cached_at": time.time(),
                "ttl": 3600,
                "version": "1.0"
            }
        }
        
        def serialize_cache_data():
            # Simulate serialization for cache storage
            serialized = json.dumps(cache_data, default=str)
            # Simulate compression (simplified)
            compressed_size = len(serialized) * 0.7  # Assume 30% compression
            
            return {
                "serialized": serialized,
                "original_size": len(json.dumps(cache_data)),
                "compressed_size": int(compressed_size),
                "compression_ratio": compressed_size / len(serialized)
            }
        
        result = benchmark(serialize_cache_data)
        assert "serialized" in result
        assert result["compression_ratio"] < 1.0


# Custom benchmark markers
pytest.mark.performance = pytest.mark.benchmark
pytest.mark.slow_benchmark = pytest.mark.benchmark