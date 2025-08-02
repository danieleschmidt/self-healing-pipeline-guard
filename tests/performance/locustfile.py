"""
Locust performance testing configuration for Self-Healing Pipeline Guard.
Defines load testing scenarios for API endpoints and webhook processing.
"""

import json
import random
from typing import Dict, Any

from locust import HttpUser, TaskSet, task, between, events
from locust.exception import StopUser


class WebhookProcessingTasks(TaskSet):
    """Task set for testing webhook processing performance."""
    
    def on_start(self):
        """Initialize test data when user starts."""
        self.repositories = [
            "test-org/repo-1",
            "test-org/repo-2", 
            "test-org/repo-3",
            "test-org/repo-4",
            "test-org/repo-5"
        ]
        
        self.failure_types = [
            "test_failure",
            "build_failure", 
            "deployment_failure",
            "flaky_test",
            "resource_exhaustion"
        ]
    
    def create_github_webhook_payload(self) -> Dict[str, Any]:
        """Create a realistic GitHub webhook payload."""
        return {
            "action": "completed",
            "workflow_run": {
                "id": random.randint(100000, 999999),
                "name": "CI",
                "status": "completed",
                "conclusion": random.choice(["failure", "success"]),
                "html_url": f"https://github.com/{random.choice(self.repositories)}/actions/runs/{random.randint(100000, 999999)}",
                "repository": {
                    "full_name": random.choice(self.repositories),
                    "html_url": f"https://github.com/{random.choice(self.repositories)}"
                },
                "head_commit": {
                    "id": ''.join(random.choices('abcdef0123456789', k=40)),
                    "message": f"Fix {random.choice(self.failure_types)} issue",
                    "author": {
                        "email": "developer@example.com",
                        "name": "Test Developer"
                    }
                },
                "head_branch": random.choice(["main", "develop", "feature/test"]),
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:32:00Z"
            }
        }
    
    @task(3)
    def process_github_webhook(self):
        """Test GitHub webhook processing endpoint."""
        payload = self.create_github_webhook_payload()
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "workflow_run",
            "X-GitHub-Delivery": f"12345678-1234-1234-1234-{random.randint(100000000000, 999999999999)}"
        }
        
        with self.client.post(
            "/webhooks/github",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 422:
                # Validation error is acceptable for load testing
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def get_healing_status(self):
        """Test healing status endpoint."""
        healing_id = f"healing-{random.randint(1000, 9999)}"
        
        with self.client.get(f"/api/healing/{healing_id}/status", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def get_metrics_summary(self):
        """Test metrics summary endpoint."""
        with self.client.get("/api/metrics/summary", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class APIUserTasks(TaskSet):
    """Task set for testing API user interactions."""
    
    def on_start(self):
        """Authenticate user when starting."""
        # Simulate authentication
        login_data = {
            "username": f"test_user_{random.randint(1, 100)}",
            "password": "test_password"
        }
        
        response = self.client.post("/auth/login", json=login_data)
        if response.status_code == 200:
            self.token = response.json().get("access_token")
        else:
            # For load testing, continue without authentication
            self.token = "test-token"
    
    def get_auth_headers(self):
        """Get authentication headers."""
        return {"Authorization": f"Bearer {self.token}"}
    
    @task(4)
    def list_failures(self):
        """Test listing pipeline failures."""
        params = {
            "limit": random.randint(10, 50),
            "offset": random.randint(0, 100),
            "platform": random.choice(["github", "gitlab", "jenkins", None])
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        with self.client.get(
            "/api/failures",
            params=params,
            headers=self.get_auth_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(3)
    def get_failure_details(self):
        """Test getting failure details."""
        failure_id = f"failure-{random.randint(1000, 9999)}"
        
        with self.client.get(
            f"/api/failures/{failure_id}",
            headers=self.get_auth_headers(),
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def list_healing_attempts(self):
        """Test listing healing attempts."""
        params = {
            "limit": random.randint(10, 30),
            "status": random.choice(["pending", "in_progress", "completed", "failed", None])
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        with self.client.get(
            "/api/healing",
            params=params,
            headers=self.get_auth_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def get_dashboard_data(self):
        """Test dashboard data endpoint."""
        with self.client.get(
            "/api/dashboard",
            headers=self.get_auth_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class WebhookUser(HttpUser):
    """User that primarily processes webhooks."""
    tasks = [WebhookProcessingTasks]
    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between tasks
    weight = 3  # 3x more likely to be selected than APIUser


class APIUser(HttpUser):
    """User that primarily interacts with the API."""
    tasks = [APIUserTasks]
    wait_time = between(1.0, 3.0)  # Wait 1-3 seconds between tasks
    weight = 1


class HeavyLoadUser(HttpUser):
    """User that simulates heavy load scenarios."""
    wait_time = between(0.1, 0.5)  # Very short wait times
    weight = 1
    
    @task
    def heavy_webhook_load(self):
        """Send rapid webhook requests."""
        payload = {
            "action": "completed",
            "workflow_run": {
                "id": random.randint(100000, 999999),
                "status": "completed",
                "conclusion": "failure",
                "repository": {"full_name": "load-test/repo"}
            }
        }
        
        with self.client.post(
            "/webhooks/github",
            json=payload,
            headers={"Content-Type": "application/json", "X-GitHub-Event": "workflow_run"},
            catch_response=True
        ) as response:
            if response.status_code in [200, 422, 429]:  # Accept rate limiting
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("🚀 Starting Self-Healing Pipeline Guard load test")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("🏁 Load test completed")
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Called for each request."""
    # Log slow requests
    if response_time > 1000:  # Requests slower than 1 second
        print(f"⚠️  Slow request: {request_type} {name} took {response_time:.2f}ms")
    
    # Log errors
    if exception:
        print(f"❌ Request failed: {request_type} {name} - {exception}")


# Custom task for stress testing specific endpoints
class StressTestTasks(TaskSet):
    """Stress test specific endpoints with high load."""
    
    @task
    def stress_webhook_endpoint(self):
        """Stress test the webhook endpoint."""
        payload = {"action": "completed", "workflow_run": {"id": 1, "status": "completed"}}
        
        # Send multiple rapid requests
        for _ in range(5):
            self.client.post("/webhooks/github", json=payload, headers={"X-GitHub-Event": "workflow_run"})


class StressTestUser(HttpUser):
    """User for stress testing."""
    tasks = [StressTestTasks]
    wait_time = between(0.01, 0.1)  # Very rapid requests
    weight = 0  # Don't include by default


# Configuration for different test scenarios
class ScenarioConfig:
    """Configuration for different testing scenarios."""
    
    @staticmethod
    def light_load():
        """Configuration for light load testing."""
        return {
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m"
        }
    
    @staticmethod
    def normal_load():
        """Configuration for normal load testing."""
        return {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "10m"
        }
    
    @staticmethod
    def heavy_load():
        """Configuration for heavy load testing."""
        return {
            "users": 200,
            "spawn_rate": 10,
            "run_time": "15m"
        }
    
    @staticmethod
    def stress_test():
        """Configuration for stress testing."""
        return {
            "users": 500,
            "spawn_rate": 20,
            "run_time": "20m"
        }