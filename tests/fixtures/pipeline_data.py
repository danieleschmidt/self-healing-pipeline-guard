"""
Test fixtures for pipeline-related data.
Contains sample pipeline failures, webhook payloads, and related test data.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any


def create_sample_pipeline_failure(
    platform: str = "github",
    failure_type: str = "test_failure",
    repository: str = "test-org/test-repo",
    **kwargs
) -> Dict[str, Any]:
    """Create a sample pipeline failure for testing."""
    base_failure = {
        "id": f"test-failure-{hash(str(kwargs))}"[:20],
        "platform": platform,
        "repository": repository,
        "pipeline_id": f"pipeline-{hash(repository)}"[:15],
        "job_id": f"job-{hash(failure_type)}"[:15],
        "commit_sha": "abc123def456789012345678901234567890abcd",
        "branch": "main",
        "failure_type": failure_type,
        "error_message": "Test failure occurred",
        "logs": [
            "Starting pipeline execution...",
            "Running tests...",
            "ERROR: Test failed",
            "Pipeline execution failed"
        ],
        "metadata": {
            "duration": 120,
            "exit_code": 1,
            "author": "developer@example.com",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "production",
            "retry_count": 0
        }
    }
    
    # Override with any provided kwargs
    base_failure.update(kwargs)
    return base_failure


def create_github_webhook_payload(
    action: str = "completed",
    conclusion: str = "failure",
    repository: str = "test-org/test-repo",
    **kwargs
) -> Dict[str, Any]:
    """Create a GitHub workflow webhook payload for testing."""
    base_payload = {
        "action": action,
        "workflow_run": {
            "id": 123456789,
            "name": "CI",
            "status": "completed",
            "conclusion": conclusion,
            "html_url": f"https://github.com/{repository}/actions/runs/123456789",
            "repository": {
                "full_name": repository,
                "html_url": f"https://github.com/{repository}",
                "private": False,
                "default_branch": "main"
            },
            "head_commit": {
                "id": "abc123def456789012345678901234567890abcd",
                "message": "Fix calculation bug",
                "author": {
                    "email": "developer@example.com",
                    "name": "Developer"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "head_branch": "main",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "jobs_url": f"https://api.github.com/repos/{repository}/actions/runs/123456789/jobs",
            "logs_url": f"https://api.github.com/repos/{repository}/actions/runs/123456789/logs"
        }
    }
    
    # Override with any provided kwargs
    if kwargs:
        base_payload.update(kwargs)
    
    return base_payload


def create_gitlab_webhook_payload(
    status: str = "failed",
    project_name: str = "test-group/test-repo",
    **kwargs
) -> Dict[str, Any]:
    """Create a GitLab pipeline webhook payload for testing."""
    base_payload = {
        "object_kind": "pipeline",
        "object_attributes": {
            "id": 987654321,
            "status": status,
            "stage": "test",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration": 120,
            "ref": "main",
            "sha": "abc123def456789012345678901234567890abcd",
            "web_url": f"https://gitlab.com/{project_name}/-/pipelines/987654321"
        },
        "project": {
            "id": 12345,
            "name": project_name.split("/")[-1],
            "path_with_namespace": project_name,
            "web_url": f"https://gitlab.com/{project_name}"
        },
        "commit": {
            "id": "abc123def456789012345678901234567890abcd",
            "message": "Fix calculation bug",
            "author": {
                "name": "Developer",
                "email": "developer@example.com"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "builds": [
            {
                "id": 111111,
                "stage": "test",
                "name": "unit-tests",
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat()
            }
        ]
    }
    
    # Override with any provided kwargs
    if kwargs:
        base_payload.update(kwargs)
    
    return base_payload


def create_jenkins_build_data(
    result: str = "FAILURE",
    job_name: str = "test-job",
    **kwargs
) -> Dict[str, Any]:
    """Create Jenkins build data for testing."""
    base_data = {
        "number": 42,
        "result": result,
        "url": f"http://jenkins.example.com/job/{job_name}/42/",
        "fullDisplayName": f"{job_name} #42",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "duration": 120000,  # milliseconds
        "changeSets": [
            {
                "items": [
                    {
                        "commitId": "abc123def456789012345678901234567890abcd",
                        "msg": "Fix calculation bug",
                        "author": {
                            "fullName": "Developer",
                            "absoluteUrl": "http://jenkins.example.com/user/developer"
                        }
                    }
                ]
            }
        ],
        "actions": [
            {
                "_class": "hudson.model.CauseAction",
                "causes": [
                    {
                        "_class": "hudson.triggers.SCMTrigger$SCMTriggerCause",
                        "shortDescription": "Started by an SCM change"
                    }
                ]
            }
        ]
    }
    
    # Override with any provided kwargs
    if kwargs:
        base_data.update(kwargs)
    
    return base_data


def create_test_failure_scenarios() -> List[Dict[str, Any]]:
    """Create a list of various test failure scenarios for comprehensive testing."""
    scenarios = [
        # Flaky test
        create_sample_pipeline_failure(
            failure_type="flaky_test",
            error_message="Test passed locally but failed in CI",
            logs=[
                "Running test_random_behavior...",
                "AssertionError: Random value was 7, expected < 5",
                "This test has a 30% failure rate"
            ],
            metadata={
                "duration": 45,
                "exit_code": 1,
                "author": "developer@example.com",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "flaky_score": 0.85,
                "historical_failures": 15
            }
        ),
        
        # Resource exhaustion
        create_sample_pipeline_failure(
            failure_type="resource_exhaustion",
            error_message="Out of memory: Cannot allocate memory",
            logs=[
                "Starting memory-intensive operation...",
                "Allocating large arrays...",
                "fatal error: runtime: out of memory",
                "Process killed"
            ],
            metadata={
                "duration": 180,
                "exit_code": 137,  # SIGKILL
                "memory_used": "2048MB",
                "memory_limit": "2048MB"
            }
        ),
        
        # Network timeout
        create_sample_pipeline_failure(
            failure_type="network_timeout",
            error_message="Connection timeout after 30 seconds",
            logs=[
                "Downloading dependencies...",
                "Connecting to registry.npmjs.org...",
                "Error: ETIMEDOUT",
                "Request timed out"
            ],
            metadata={
                "duration": 35,
                "exit_code": 1,
                "timeout_duration": 30,
                "target_host": "registry.npmjs.org"
            }
        ),
        
        # Dependency conflict
        create_sample_pipeline_failure(
            failure_type="dependency_conflict",
            error_message="Package version conflict detected",
            logs=[
                "Resolving dependencies...",
                "Found: react@17.0.0",
                "Peer dependency warning: react-dom@18.0.0 requires react@18.0.0",
                "Error: Could not resolve dependency tree"
            ],
            metadata={
                "duration": 60,
                "exit_code": 1,
                "conflicting_packages": ["react", "react-dom"],
                "package_manager": "npm"
            }
        ),
        
        # Race condition
        create_sample_pipeline_failure(
            failure_type="race_condition",
            error_message="Database deadlock detected",
            logs=[
                "Running parallel tests...",
                "Test 1: Updating user record...",
                "Test 2: Updating same user record...",
                "ERROR: Deadlock detected",
                "Transaction rolled back"
            ],
            metadata={
                "duration": 90,
                "exit_code": 1,
                "parallel_processes": 4,
                "database_error_code": "40001"
            }
        )
    ]
    
    return scenarios


def create_healing_strategy_results() -> List[Dict[str, Any]]:
    """Create sample healing strategy results for testing."""
    return [
        {
            "strategy_name": "flaky_test_retry",
            "success": True,
            "duration": 45.2,
            "attempts": 2,
            "confidence": 0.87,
            "metadata": {
                "retry_intervals": [1, 2],
                "final_exit_code": 0,
                "logs_analyzed": True
            }
        },
        {
            "strategy_name": "resource_scaling",
            "success": True,
            "duration": 120.5,
            "attempts": 1,
            "confidence": 0.92,
            "metadata": {
                "memory_increased": "1GB",
                "cpu_increased": "0.5",
                "cost_impact": 0.15
            }
        },
        {
            "strategy_name": "cache_invalidation",
            "success": False,
            "duration": 30.1,
            "attempts": 1,
            "confidence": 0.45,
            "metadata": {
                "caches_cleared": ["npm", "pip"],
                "cache_sizes": {"npm": "150MB", "pip": "75MB"},
                "error": "Cache clearing succeeded but build still failed"
            }
        }
    ]