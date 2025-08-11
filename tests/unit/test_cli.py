"""Tests for the CLI module."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch

from healing_guard.cli import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_logs(tmp_path):
    """Create sample log file for testing."""
    log_file = tmp_path / "test.log"
    log_content = """
    ERROR: Test failed with timeout
    Connection refused: Unable to connect to database
    java.lang.OutOfMemoryError: Java heap space
    Build failed with exit code 1
    """
    log_file.write_text(log_content.strip())
    return log_file


class TestCLI:
    """Test suite for CLI commands."""
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Healing Guard v" in result.stdout
    
    def test_help_command(self, runner):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Self-Healing Pipeline Guard" in result.stdout
        assert "analyze" in result.stdout
        assert "heal" in result.stdout
        assert "status" in result.stdout
        assert "server" in result.stdout
        assert "config" in result.stdout
    
    def test_analyze_command_file_not_found(self, runner):
        """Test analysis with non-existent log file."""
        result = runner.invoke(app, ["analyze", "nonexistent.log"])
        
        assert result.exit_code == 1
        assert "Log file nonexistent.log not found" in result.stdout
    
    def test_status_command(self, runner):
        """Test status command."""
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "System Status" in result.stdout
        assert "Failure Detector" in result.stdout
        assert "Healing Engine" in result.stdout
        assert "Quantum Planner" in result.stdout
    
    def test_config_generate_command(self, runner):
        """Test config generate command."""
        result = runner.invoke(app, ["config", "--generate"])
        
        assert result.exit_code == 0
        assert "Example Configuration" in result.stdout
        assert "healing_guard" in result.stdout