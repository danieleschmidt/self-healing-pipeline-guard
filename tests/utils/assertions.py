"""
Custom assertion helpers for Self-Healing Pipeline Guard tests.
Provides domain-specific assertion functions for better test readability.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import Response


def assert_pipeline_failure_valid(failure_data: Dict[str, Any]) -> None:
    """Assert that pipeline failure data has all required fields."""
    required_fields = [
        "id", "platform", "repository", "pipeline_id", "job_id",
        "commit_sha", "branch", "failure_type", "error_message", "logs", "metadata"
    ]
    
    for field in required_fields:
        assert field in failure_data, f"Missing required field: {field}"
    
    # Validate field types
    assert isinstance(failure_data["id"], str), "ID must be string"
    assert isinstance(failure_data["logs"], list), "Logs must be list"
    assert isinstance(failure_data["metadata"], dict), "Metadata must be dict"
    
    # Validate metadata fields
    metadata = failure_data["metadata"]
    assert "duration" in metadata, "Metadata missing duration"
    assert "exit_code" in metadata, "Metadata missing exit_code"
    assert "timestamp" in metadata, "Metadata missing timestamp"


def assert_healing_strategy_result(
    result: Dict[str, Any],
    expected_success: bool = True,
    expected_strategy: Optional[str] = None,
    min_confidence: float = 0.0
) -> None:
    """Assert healing strategy result matches expectations."""
    required_fields = ["strategy_name", "success", "duration", "attempts", "confidence"]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    assert result["success"] == expected_success, f"Expected success={expected_success}"
    assert result["confidence"] >= min_confidence, f"Confidence too low: {result['confidence']}"
    assert result["duration"] >= 0, "Duration must be non-negative"
    assert result["attempts"] >= 1, "Attempts must be at least 1"
    
    if expected_strategy:
        assert result["strategy_name"] == expected_strategy, f"Expected strategy: {expected_strategy}"


def assert_api_response(
    response: Union[Response, TestClient],
    expected_status: int = 200,
    expected_keys: Optional[List[str]] = None,
    expected_content_type: str = "application/json"
) -> Dict[str, Any]:
    """Assert API response matches expectations and return parsed data."""
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.text}"
    )
    
    if expected_content_type == "application/json":
        try:
            data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {response.text}")
        
        if expected_keys:
            for key in expected_keys:
                assert key in data, f"Missing expected key '{key}' in response"
        
        return data
    
    return {"content": response.text}


def assert_webhook_payload_valid(
    payload: Dict[str, Any],
    platform: str = "github"
) -> None:
    """Assert webhook payload is valid for the specified platform."""
    if platform == "github":
        assert "action" in payload, "GitHub webhook missing 'action'"
        assert "workflow_run" in payload, "GitHub webhook missing 'workflow_run'"
        
        workflow_run = payload["workflow_run"]
        required_fields = ["id", "status", "conclusion", "repository", "head_commit"]
        for field in required_fields:
            assert field in workflow_run, f"Missing field '{field}' in workflow_run"
    
    elif platform == "gitlab":
        assert "object_kind" in payload, "GitLab webhook missing 'object_kind'"
        assert payload["object_kind"] == "pipeline", "Expected pipeline webhook"
        assert "object_attributes" in payload, "GitLab webhook missing 'object_attributes'"
        assert "project" in payload, "GitLab webhook missing 'project'"
    
    elif platform == "jenkins":
        assert "number" in payload, "Jenkins webhook missing 'number'"
        assert "result" in payload, "Jenkins webhook missing 'result'"
        assert "url" in payload, "Jenkins webhook missing 'url'"
    
    else:
        pytest.fail(f"Unsupported platform: {platform}")


def assert_notification_sent(
    mock_client: MagicMock,
    method_name: str,
    expected_calls: int = 1,
    check_content: Optional[str] = None
) -> None:
    """Assert that a notification was sent using the mock client."""
    method = getattr(mock_client, method_name)
    assert method.call_count == expected_calls, (
        f"Expected {expected_calls} calls to {method_name}, got {method.call_count}"
    )
    
    if check_content and expected_calls > 0:
        call_args = method.call_args
        if call_args:
            # Check if content appears in any of the call arguments
            found_content = False
            for arg in call_args[0] + tuple(call_args[1].values()):
                if isinstance(arg, str) and check_content in arg:
                    found_content = True
                    break
            assert found_content, f"Content '{check_content}' not found in notification"


def assert_metrics_recorded(
    mock_metrics: MagicMock,
    metric_name: str,
    expected_calls: int = 1,
    expected_value: Optional[Union[int, float]] = None
) -> None:
    """Assert that metrics were recorded correctly."""
    method = getattr(mock_metrics, f"record_{metric_name}")
    assert method.call_count == expected_calls, (
        f"Expected {expected_calls} calls to record_{metric_name}, got {method.call_count}"
    )
    
    if expected_value is not None and expected_calls > 0:
        call_args = method.call_args[0]
        assert call_args[0] == expected_value, (
            f"Expected metric value {expected_value}, got {call_args[0]}"
        )


def assert_database_state(
    session,
    model_class,
    expected_count: Optional[int] = None,
    filter_conditions: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """Assert database state matches expectations."""
    query = session.query(model_class)
    
    if filter_conditions:
        for field, value in filter_conditions.items():
            query = query.filter(getattr(model_class, field) == value)
    
    results = query.all()
    
    if expected_count is not None:
        assert len(results) == expected_count, (
            f"Expected {expected_count} records, found {len(results)}"
        )
    
    return results


def assert_log_contains(
    caplog,
    level: str,
    message: str,
    count: Optional[int] = None
) -> None:
    """Assert that log contains specific message at given level."""
    level_records = [record for record in caplog.records if record.levelname == level.upper()]
    matching_records = [record for record in level_records if message in record.message]
    
    if count is not None:
        assert len(matching_records) == count, (
            f"Expected {count} log messages containing '{message}', found {len(matching_records)}"
        )
    else:
        assert len(matching_records) > 0, (
            f"No log messages found containing '{message}' at level {level}"
        )


def assert_timestamp_recent(
    timestamp: Union[str, datetime],
    max_age_seconds: int = 60
) -> None:
    """Assert that timestamp is recent (within max_age_seconds)."""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    now = datetime.now(timezone.utc)
    age = (now - timestamp).total_seconds()
    
    assert age <= max_age_seconds, (
        f"Timestamp is too old: {age} seconds > {max_age_seconds} seconds"
    )


def assert_healing_configuration_valid(config: Dict[str, Any]) -> None:
    """Assert healing configuration is valid."""
    assert "strategies" in config, "Missing 'strategies' in configuration"
    assert "escalation" in config, "Missing 'escalation' in configuration"
    
    strategies = config["strategies"]
    assert isinstance(strategies, dict), "Strategies must be a dictionary"
    
    for strategy_name, strategy_config in strategies.items():
        assert "enabled" in strategy_config, f"Strategy {strategy_name} missing 'enabled'"
        assert isinstance(strategy_config["enabled"], bool), f"Strategy {strategy_name} 'enabled' must be boolean"
        
        if strategy_config["enabled"]:
            assert "confidence_threshold" in strategy_config, (
                f"Enabled strategy {strategy_name} missing 'confidence_threshold'"
            )
    
    escalation = config["escalation"]
    required_escalation_fields = ["max_auto_attempts", "escalation_delay", "create_incidents"]
    for field in required_escalation_fields:
        assert field in escalation, f"Escalation missing '{field}'"


def assert_ml_prediction_valid(
    prediction: Dict[str, Any],
    expected_classes: Optional[List[str]] = None
) -> None:
    """Assert ML prediction result is valid."""
    required_fields = ["prediction", "confidence", "model_version"]
    for field in required_fields:
        assert field in prediction, f"Missing field '{field}' in prediction"
    
    assert 0 <= prediction["confidence"] <= 1, "Confidence must be between 0 and 1"
    
    if expected_classes:
        assert prediction["prediction"] in expected_classes, (
            f"Prediction '{prediction['prediction']}' not in expected classes {expected_classes}"
        )


def assert_security_compliance(test_data: Dict[str, Any]) -> None:
    """Assert that test data doesn't contain real secrets or sensitive information."""
    # Convert data to string for checking
    data_str = json.dumps(test_data, default=str).lower()
    
    # Patterns that might indicate real secrets
    forbidden_patterns = [
        "ghp_",  # GitHub personal access tokens
        "gho_",  # GitHub OAuth tokens
        "ghu_",  # GitHub user tokens
        "ghs_",  # GitHub server tokens
        "glpat-",  # GitLab personal access tokens
        "xoxb-",  # Slack bot tokens
        "xoxp-",  # Slack user tokens
        "sk_live_",  # Stripe live keys
        "pk_live_",  # Stripe live public keys
    ]
    
    for pattern in forbidden_patterns:
        assert pattern not in data_str, f"Test data contains potential real secret: {pattern}"
    
    # Check for suspicious patterns
    import re
    
    # Base64-like patterns (40+ chars)
    base64_pattern = r'[A-Za-z0-9+/]{40,}={0,2}'
    if re.search(base64_pattern, data_str):
        # Allow known test patterns
        test_patterns = ["test", "fake", "mock", "sample", "example"]
        if not any(pattern in data_str for pattern in test_patterns):
            pytest.fail("Test data contains suspicious base64-like string")
    
    # Hex patterns (32+ chars)
    hex_pattern = r'[a-fA-F0-9]{32,}'
    if re.search(hex_pattern, data_str):
        # Allow known test patterns
        if not any(pattern in data_str for pattern in ["abc123", "test", "fake"]):
            pytest.fail("Test data contains suspicious hex string")