"""
Test helper utilities for Self-Healing Pipeline Guard.
Provides utility functions to make testing easier and more readable.
"""

import asyncio
import json
import random
import string
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_commit_sha() -> str:
    """Generate a realistic-looking Git commit SHA."""
    return ''.join(random.choices(string.hexdigits.lower(), k=40))


def generate_test_id(prefix: str = "test") -> str:
    """Generate a unique test ID."""
    timestamp = int(time.time() * 1000)
    random_suffix = generate_random_string(8)
    return f"{prefix}-{timestamp}-{random_suffix}"


@contextmanager
def temporary_json_file(data: Dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary JSON file with the given data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def temporary_yaml_file(data: Dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary YAML file with the given data."""
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(data, f, default_flow_style=False)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def mock_environment_variables(**env_vars: str) -> Generator[None, None, None]:
    """Mock environment variables for the duration of the context."""
    with patch.dict('os.environ', env_vars, clear=False):
        yield


@asynccontextmanager
async def mock_async_dependency(
    dependency_path: str,
    mock_return_value: Any = None
) -> AsyncGenerator[AsyncMock, None]:
    """Mock an async dependency for testing."""
    mock = AsyncMock(return_value=mock_return_value)
    with patch(dependency_path, mock):
        yield mock


@contextmanager
def mock_sync_dependency(
    dependency_path: str,
    mock_return_value: Any = None
) -> Generator[MagicMock, None, None]:
    """Mock a sync dependency for testing."""
    mock = MagicMock(return_value=mock_return_value)
    with patch(dependency_path, mock):
        yield mock


def create_mock_response(
    status_code: int = 200,
    json_data: Optional[Dict[str, Any]] = None,
    text_data: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> MagicMock:
    """Create a mock HTTP response object."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = headers or {}
    
    if json_data is not None:
        mock_response.json.return_value = json_data
        mock_response.text = json.dumps(json_data)
    elif text_data is not None:
        mock_response.text = text_data
        mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)
    else:
        mock_response.text = ""
        mock_response.json.return_value = {}
    
    return mock_response


def create_mock_async_response(
    status_code: int = 200,
    json_data: Optional[Dict[str, Any]] = None,
    text_data: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> AsyncMock:
    """Create a mock async HTTP response object."""
    mock_response = AsyncMock()
    mock_response.status_code = status_code
    mock_response.headers = headers or {}
    
    if json_data is not None:
        mock_response.json.return_value = json_data
        mock_response.text = json.dumps(json_data)
    elif text_data is not None:
        mock_response.text = text_data
        mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)
    else:
        mock_response.text = ""
        mock_response.json.return_value = {}
    
    return mock_response


class MockTimer:
    """Mock timer for testing time-dependent code."""
    
    def __init__(self, initial_time: float = 0.0):
        self.current_time = initial_time
    
    def time(self) -> float:
        """Return current mock time."""
        return self.current_time
    
    def sleep(self, duration: float) -> None:
        """Advance mock time by duration."""
        self.current_time += duration
    
    def advance(self, duration: float) -> None:
        """Advance mock time by duration (alias for sleep)."""
        self.sleep(duration)


@contextmanager
def mock_time(initial_time: float = 0.0) -> Generator[MockTimer, None, None]:
    """Mock time.time() and time.sleep() for testing."""
    timer = MockTimer(initial_time)
    
    with patch('time.time', timer.time), \
         patch('time.sleep', timer.sleep):
        yield timer


class AsyncContextManager:
    """Helper for creating async context managers in tests."""
    
    def __init__(self, value: Any):
        self.value = value
    
    async def __aenter__(self):
        return self.value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_async_context_manager(value: Any) -> AsyncContextManager:
    """Create an async context manager that yields the given value."""
    return AsyncContextManager(value)


def wait_for_condition(
    condition_func: callable,
    timeout: float = 5.0,
    check_interval: float = 0.1
) -> None:
    """Wait for a condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return
        time.sleep(check_interval)
    
    raise TimeoutError(f"Condition did not become true within {timeout} seconds")


async def wait_for_async_condition(
    condition_func: callable,
    timeout: float = 5.0,
    check_interval: float = 0.1
) -> None:
    """Wait for an async condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func():
            return
        await asyncio.sleep(check_interval)
    
    raise TimeoutError(f"Condition did not become true within {timeout} seconds")


def parametrize_with_cases(test_cases: List[Dict[str, Any]]):
    """Decorator to parametrize test with named test cases."""
    def decorator(func):
        case_ids = [case.get('id', f'case_{i}') for i, case in enumerate(test_cases)]
        case_values = [tuple(case.values()) for case in test_cases]
        case_keys = list(test_cases[0].keys()) if test_cases else []
        
        return pytest.mark.parametrize(
            ','.join(case_keys),
            case_values,
            ids=case_ids
        )(func)
    
    return decorator


def skip_if_external_deps_unavailable():
    """Skip test if external dependencies are not available."""
    def decorator(func):
        # Check if we're in CI or if external dependencies should be skipped
        skip_external = (
            pytest.config.getoption("--skip-external", default=False) or
            os.environ.get("SKIP_EXTERNAL_DEPS", "false").lower() == "true"
        )
        
        return pytest.mark.skipif(
            skip_external,
            reason="External dependencies not available"
        )(func)
    
    return decorator


class TestDataBuilder:
    """Builder pattern for creating test data."""
    
    def __init__(self):
        self.data = {}
    
    def with_field(self, key: str, value: Any) -> 'TestDataBuilder':
        """Add a field to the test data."""
        self.data[key] = value
        return self
    
    def with_nested_field(self, path: str, value: Any) -> 'TestDataBuilder':
        """Add a nested field using dot notation (e.g., 'metadata.duration')."""
        keys = path.split('.')
        current = self.data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return self
    
    def with_random_id(self, prefix: str = "test") -> 'TestDataBuilder':
        """Add a random ID field."""
        return self.with_field("id", generate_test_id(prefix))
    
    def with_timestamp(self, field_name: str = "timestamp") -> 'TestDataBuilder':
        """Add a current timestamp field."""
        from datetime import datetime, timezone
        return self.with_field(field_name, datetime.now(timezone.utc).isoformat())
    
    def build(self) -> Dict[str, Any]:
        """Build and return the test data."""
        return self.data.copy()


def create_test_data() -> TestDataBuilder:
    """Create a new test data builder."""
    return TestDataBuilder()


class MockDatabase:
    """Mock database for testing database operations."""
    
    def __init__(self):
        self.data = {}
        self.transaction_active = False
    
    def insert(self, table: str, record: Dict[str, Any]) -> str:
        """Insert a record and return its ID."""
        if table not in self.data:
            self.data[table] = []
        
        record_id = record.get('id', generate_test_id())
        record_with_id = {**record, 'id': record_id}
        self.data[table].append(record_with_id)
        return record_id
    
    def find(self, table: str, conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find records matching conditions."""
        if table not in self.data:
            return []
        
        records = self.data[table]
        
        if conditions:
            filtered_records = []
            for record in records:
                match = True
                for key, value in conditions.items():
                    if record.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            return filtered_records
        
        return records.copy()
    
    def update(self, table: str, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a record by ID."""
        if table not in self.data:
            return False
        
        for record in self.data[table]:
            if record.get('id') == record_id:
                record.update(updates)
                return True
        
        return False
    
    def delete(self, table: str, record_id: str) -> bool:
        """Delete a record by ID."""
        if table not in self.data:
            return False
        
        original_length = len(self.data[table])
        self.data[table] = [r for r in self.data[table] if r.get('id') != record_id]
        return len(self.data[table]) < original_length
    
    def begin_transaction(self):
        """Begin a transaction."""
        self.transaction_active = True
    
    def commit_transaction(self):
        """Commit the current transaction."""
        self.transaction_active = False
    
    def rollback_transaction(self):
        """Rollback the current transaction."""
        self.transaction_active = False
        # In a real implementation, this would restore previous state
    
    def clear(self):
        """Clear all data."""
        self.data.clear()


@contextmanager
def mock_database() -> Generator[MockDatabase, None, None]:
    """Create a mock database for testing."""
    db = MockDatabase()
    yield db
    db.clear()