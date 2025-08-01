#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Executor
Repository: self-healing-pipeline-guard
Maturity Level: Advanced (90%)

This module executes the next best value item autonomously with safety controls.
"""

import json
import subprocess
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

from value_discovery import ValueDiscoveryEngine, ValueItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of autonomous task execution."""
    success: bool
    item_id: str
    title: str
    execution_time: float
    changes_made: List[str]
    error_message: Optional[str] = None
    rollback_performed: bool = False


class AutonomousExecutor:
    """Autonomous task execution with safety controls."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.discovery_engine = ValueDiscoveryEngine(repo_path)
        self.execution_log = self.repo_path / ".terragon" / "execution_log.json"
        
    def execute_next_best_item(self) -> Optional[ExecutionResult]:
        """Execute the next highest-value item autonomously."""
        logger.info("Starting autonomous execution cycle...")
        
        # Get next best item
        item = self.discovery_engine.get_next_best_item()
        if not item:
            logger.info("No qualifying items found for execution")
            return None
            
        logger.info(f"Executing item: {item.title} (Score: {item.composite_score:.1f})")
        
        # Pre-execution safety checks
        if not self._safety_checks(item):
            logger.warning(f"Safety checks failed for item {item.id}")
            return ExecutionResult(
                success=False,
                item_id=item.id,
                title=item.title,
                execution_time=0,
                changes_made=[],
                error_message="Safety checks failed"
            )
        
        # Create execution branch
        branch_name = f"auto-value/{item.id}-{item.category}"
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], 
                         cwd=self.repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return ExecutionResult(
                success=False, item_id=item.id, title=item.title,
                execution_time=0, changes_made=[],
                error_message=f"Branch creation failed: {e}"
            )
        
        # Execute based on item type
        start_time = datetime.datetime.now()
        result = self._execute_item(item)
        execution_time = (datetime.datetime.now() - start_time).total_seconds() / 60.0
        
        result.execution_time = execution_time
        
        # Post-execution validation
        if result.success:
            validation_result = self._validate_execution(item)
            if not validation_result:
                logger.warning("Post-execution validation failed, rolling back...")
                self._rollback_changes()
                result.success = False
                result.rollback_performed = True
                result.error_message = "Post-execution validation failed"
        
        # Log execution result
        self._log_execution(result)
        
        if result.success:
            logger.info(f"Successfully executed {item.title} in {execution_time:.1f} minutes")
        else:
            logger.error(f"Execution failed for {item.title}: {result.error_message}")
            
        return result
    
    def _safety_checks(self, item: ValueItem) -> bool:
        """Perform pre-execution safety checks."""
        # Check if working directory is clean
        try:
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  cwd=self.repo_path, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                logger.warning("Working directory not clean")
                return False
        except subprocess.CalledProcessError:
            return False
            
        # Check risk threshold
        max_risk = self.discovery_engine.config['execution']['maxConcurrentTasks']
        if item.risk_level > 0.8:  # Very high risk items need manual approval
            logger.warning(f"Risk level {item.risk_level} exceeds safe automation threshold")
            return False
            
        # Check if required tools are available
        if item.category == 'security' and not self._check_security_tools():
            logger.warning("Security tools not available for security tasks")
            return False
            
        return True
    
    def _execute_item(self, item: ValueItem) -> ExecutionResult:
        """Execute specific item based on its category and requirements."""
        try:
            if item.id == "deps-001":  # Poetry lock file
                return self._execute_poetry_lock(item)
            elif item.id == "git-001":  # Technical debt cleanup
                return self._execute_tech_debt_cleanup(item)
            elif item.id == "cicd-001":  # GitHub Actions setup
                return self._execute_github_actions(item)
            elif item.id == "quality-002":  # Pre-commit setup
                return self._execute_precommit_setup(item)
            else:
                return ExecutionResult(
                    success=False, item_id=item.id, title=item.title,
                    execution_time=0, changes_made=[],
                    error_message="Unsupported item type"
                )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False, item_id=item.id, title=item.title,
                execution_time=0, changes_made=[],
                error_message=str(e)
            )
    
    def _execute_poetry_lock(self, item: ValueItem) -> ExecutionResult:
        """Execute poetry lock file creation."""
        try:
            # Check if poetry.lock already exists
            if (self.repo_path / "poetry.lock").exists():
                return ExecutionResult(
                    success=True, item_id=item.id, title=item.title,
                    execution_time=0, changes_made=[],
                    error_message="poetry.lock already exists"
                )
            
            # Run poetry lock
            result = subprocess.run(["poetry", "lock"], 
                                  cwd=self.repo_path, capture_output=True, text=True, check=True)
            
            return ExecutionResult(
                success=True, item_id=item.id, title=item.title,
                execution_time=0, changes_made=["poetry.lock"]
            )
        except subprocess.CalledProcessError as e:
            return ExecutionResult(
                success=False, item_id=item.id, title=item.title,
                execution_time=0, changes_made=[],
                error_message=f"Poetry lock failed: {e}"
            )
    
    def _execute_tech_debt_cleanup(self, item: ValueItem) -> ExecutionResult:
        """Execute technical debt cleanup tasks."""
        changes_made = []
        
        try:
            # Search for TODO/FIXME in codebase
            result = subprocess.run([
                "grep", "-r", "-n", "TODO\\|FIXME\\|HACK", ".", 
                "--include=*.py", "--include=*.md"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            debt_markers = result.stdout.strip().split('\n') if result.stdout else []
            
            # For now, just document the debt markers
            if debt_markers:
                debt_report = f"# Technical Debt Report\\n\\nGenerated: {datetime.datetime.now().isoformat()}\\n\\n"
                debt_report += "## Found Debt Markers\\n\\n"
                for marker in debt_markers[:10]:  # Limit to first 10
                    debt_report += f"- {marker}\\n"
                
                debt_file = self.repo_path / "TECHNICAL_DEBT.md"
                debt_file.write_text(debt_report)
                changes_made.append("TECHNICAL_DEBT.md")
            
            return ExecutionResult(
                success=True, item_id=item.id, title=item.title,
                execution_time=0, changes_made=changes_made
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False, item_id=item.id, title=item.title,
                execution_time=0, changes_made=[],
                error_message=str(e)
            )
    
    def _execute_github_actions(self, item: ValueItem) -> ExecutionResult:
        """Execute GitHub Actions workflow creation."""
        workflows_dir = self.repo_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        changes_made = []
        
        # Basic CI workflow
        ci_workflow = '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install
      
    - name: Run tests
      run: poetry run pytest
      
    - name: Run linting
      run: |
        poetry run ruff check .
        poetry run mypy .
'''
        
        ci_file = workflows_dir / "ci.yml"
        ci_file.write_text(ci_workflow)
        changes_made.append(".github/workflows/ci.yml")
        
        return ExecutionResult(
            success=True, item_id=item.id, title=item.title,
            execution_time=0, changes_made=changes_made
        )
    
    def _execute_precommit_setup(self, item: ValueItem) -> ExecutionResult:
        """Execute pre-commit hooks setup."""
        precommit_config = '''repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.7
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
'''
        
        config_file = self.repo_path / ".pre-commit-config.yaml"
        config_file.write_text(precommit_config)
        
        return ExecutionResult(
            success=True, item_id=item.id, title=item.title,
            execution_time=0, changes_made=[".pre-commit-config.yaml"]
        )
    
    def _validate_execution(self, item: ValueItem) -> bool:
        """Validate that execution was successful."""
        try:
            # Check if any files were actually changed
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  cwd=self.repo_path, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                logger.warning("No changes detected after execution")
                return False
                
            # Try to run basic tests if they exist
            if (self.repo_path / "tests").exists():
                try:
                    subprocess.run(["python3", "-m", "pytest", "--tb=short"], 
                                 cwd=self.repo_path, capture_output=True, check=True, timeout=120)
                except subprocess.CalledProcessError:
                    logger.warning("Tests failed after execution")
                    return False
                except subprocess.TimeoutExpired:
                    logger.warning("Tests timed out")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _rollback_changes(self):
        """Rollback changes if validation fails."""
        try:
            subprocess.run(["git", "reset", "--hard", "HEAD"], 
                         cwd=self.repo_path, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "main"], 
                         cwd=self.repo_path, check=True, capture_output=True)
            logger.info("Successfully rolled back changes")
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
    
    def _check_security_tools(self) -> bool:
        """Check if security scanning tools are available."""
        tools = ["bandit", "safety"]
        for tool in tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        return True
    
    def _log_execution(self, result: ExecutionResult):
        """Log execution result for learning and metrics."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "item_id": result.item_id,
            "title": result.title,
            "success": result.success,
            "execution_time_minutes": result.execution_time,
            "changes_made": result.changes_made,
            "error_message": result.error_message,
            "rollback_performed": result.rollback_performed
        }
        
        # Load existing log
        existing_log = []
        if self.execution_log.exists():
            with open(self.execution_log, 'r') as f:
                existing_log = json.load(f)
        
        # Append new entry
        existing_log.append(log_entry)
        
        # Keep only last 100 entries
        if len(existing_log) > 100:
            existing_log = existing_log[-100:]
        
        # Save updated log
        self.execution_log.parent.mkdir(exist_ok=True)
        with open(self.execution_log, 'w') as f:
            json.dump(existing_log, f, indent=2)


if __name__ == "__main__":
    executor = AutonomousExecutor()
    result = executor.execute_next_best_item()
    
    if result:
        if result.success:
            print(f"✅ Successfully executed: {result.title}")
            print(f"⏱️  Execution time: {result.execution_time:.1f} minutes")
            print(f"📝 Changes made: {', '.join(result.changes_made)}")
        else:
            print(f"❌ Execution failed: {result.title}")
            print(f"🚨 Error: {result.error_message}")
    else:
        print("🔍 No qualifying items found for execution")