#!/usr/bin/env python3
"""
Configuration validation script for Self-Healing Pipeline Guard.
Validates configuration files for correctness and security.
"""

import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

try:
    import toml
except ImportError:
    try:
        import tomllib as toml
    except ImportError:
        print("Warning: TOML library not available. Skipping TOML validation.")
        toml = None


class ConfigValidator:
    """Validates configuration files for the healing guard application."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.project_root = Path(__file__).parent.parent

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(f"ERROR: {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"WARNING: {message}")

    def validate_env_example(self) -> None:
        """Validate .env.example file."""
        env_file = self.project_root / ".env.example"
        
        if not env_file.exists():
            self.add_error(".env.example file not found")
            return

        required_vars = [
            "APP_NAME", "APP_VERSION", "APP_ENV", "DEBUG", "LOG_LEVEL",
            "HOST", "PORT", "DATABASE_URL", "REDIS_URL", "SECRET_KEY",
            "JWT_SECRET_KEY", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"
        ]

        try:
            content = env_file.read_text()
            for var in required_vars:
                if f"{var}=" not in content:
                    self.add_error(f"Required environment variable {var} not found in .env.example")

            # Check for potential security issues
            if "password" in content.lower() and "your-password" not in content.lower():
                self.add_warning("Potential real password found in .env.example")
                
            if "secret" in content.lower() and "your-" not in content.lower():
                self.add_warning("Potential real secret found in .env.example")

        except Exception as e:
            self.add_error(f"Failed to read .env.example: {e}")

    def validate_pyproject_toml(self) -> None:
        """Validate pyproject.toml file."""
        toml_file = self.project_root / "pyproject.toml"
        
        if not toml_file.exists():
            self.add_error("pyproject.toml file not found")
            return

        if toml is None:
            self.add_warning("TOML library not available, skipping pyproject.toml validation")
            return

        try:
            with open(toml_file, 'rb') as f:
                data = toml.load(f)

            # Validate poetry configuration
            if "tool" not in data or "poetry" not in data["tool"]:
                self.add_error("Poetry configuration not found in pyproject.toml")
                return

            poetry_config = data["tool"]["poetry"]
            required_fields = ["name", "version", "description", "authors"]
            
            for field in required_fields:
                if field not in poetry_config:
                    self.add_error(f"Required field '{field}' not found in poetry configuration")

            # Check dependencies
            if "dependencies" not in poetry_config:
                self.add_error("Dependencies section not found in poetry configuration")
            else:
                deps = poetry_config["dependencies"]
                required_deps = ["fastapi", "uvicorn", "pydantic", "sqlalchemy"]
                
                for dep in required_deps:
                    if dep not in deps:
                        self.add_warning(f"Recommended dependency '{dep}' not found")

            # Validate test configuration
            if "pytest" in data.get("tool", {}):
                pytest_config = data["tool"]["pytest"]["ini_options"]
                if "testpaths" not in pytest_config:
                    self.add_warning("testpaths not configured in pytest configuration")

        except Exception as e:
            self.add_error(f"Failed to parse pyproject.toml: {e}")

    def validate_docker_configs(self) -> None:
        """Validate Docker configuration files."""
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "docker-compose.prod.yml"
        ]

        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    
                    # Basic Docker security checks
                    if docker_file.startswith("Dockerfile"):
                        if "USER root" in content and "USER " not in content.replace("USER root", ""):
                            self.add_warning(f"{docker_file}: Running as root user without switching")
                        
                        if "COPY . ." in content:
                            self.add_warning(f"{docker_file}: Copying entire context, consider using .dockerignore")
                            
                    elif docker_file.startswith("docker-compose"):
                        # Validate YAML syntax
                        try:
                            yaml.safe_load(content)
                        except yaml.YAMLError as e:
                            self.add_error(f"{docker_file}: Invalid YAML syntax - {e}")
                            
                except Exception as e:
                    self.add_error(f"Failed to validate {docker_file}: {e}")

    def validate_config_directory(self) -> None:
        """Validate configuration files in config/ directory."""
        config_dir = self.project_root / "config"
        
        if not config_dir.exists():
            self.add_warning("config/ directory not found")
            return

        yaml_files = list(config_dir.glob("**/*.yml")) + list(config_dir.glob("**/*.yaml"))
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.add_error(f"{yaml_file.relative_to(self.project_root)}: Invalid YAML - {e}")
            except Exception as e:
                self.add_error(f"Failed to read {yaml_file.relative_to(self.project_root)}: {e}")

    def validate_github_workflows(self) -> None:
        """Validate GitHub workflow files."""
        workflows_dir = self.project_root / ".github" / "workflows"
        
        if not workflows_dir.exists():
            self.add_warning("GitHub workflows directory not found")
            return

        for workflow_file in workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                    
                if not isinstance(workflow_data, dict):
                    self.add_error(f"{workflow_file.name}: Invalid workflow structure")
                    continue
                    
                if "on" not in workflow_data:
                    self.add_error(f"{workflow_file.name}: Missing 'on' trigger")
                    
                if "jobs" not in workflow_data:
                    self.add_error(f"{workflow_file.name}: Missing 'jobs' section")

            except yaml.YAMLError as e:
                self.add_error(f"{workflow_file.name}: Invalid YAML - {e}")
            except Exception as e:
                self.add_error(f"Failed to validate {workflow_file.name}: {e}")

    def validate_package_json(self) -> None:
        """Validate package.json if it exists."""
        package_file = self.project_root / "package.json"
        
        if not package_file.exists():
            return  # package.json is optional for Python projects

        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)
                
            required_fields = ["name", "version"]
            for field in required_fields:
                if field not in package_data:
                    self.add_error(f"Required field '{field}' not found in package.json")

            # Check scripts
            if "scripts" in package_data:
                recommended_scripts = ["test", "lint", "build"]
                for script in recommended_scripts:
                    if script not in package_data["scripts"]:
                        self.add_warning(f"Recommended script '{script}' not found in package.json")

        except json.JSONDecodeError as e:
            self.add_error(f"Invalid JSON in package.json: {e}")
        except Exception as e:
            self.add_error(f"Failed to validate package.json: {e}")

    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating configuration files...")
        
        self.validate_env_example()
        self.validate_pyproject_toml()
        self.validate_docker_configs()
        self.validate_config_directory()
        self.validate_github_workflows()
        self.validate_package_json()

        # Report results
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"  {error}")
            print(f"\n💥 Validation failed with {len(self.errors)} error(s)")
            return False
        else:
            warning_count = len(self.warnings)
            if warning_count > 0:
                print(f"\n✅ Validation passed with {warning_count} warning(s)")
            else:
                print("\n✅ All configuration files are valid!")
            return True


def main():
    """Main entry point."""
    validator = ConfigValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()