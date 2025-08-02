#!/usr/bin/env python3
"""
Security policy validation script for Self-Healing Pipeline Guard.
Checks code for security policy compliance and best practices.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Set, Dict, Optional


class SecurityPolicyChecker:
    """Checks Python code for security policy compliance."""

    def __init__(self):
        self.violations: List[str] = []
        self.warnings: List[str] = []
        self.project_root = Path(__file__).parent.parent

        # Security patterns to check
        self.forbidden_patterns = {
            r'eval\s*\(': 'Use of eval() is forbidden',
            r'exec\s*\(': 'Use of exec() is forbidden',
            r'__import__\s*\(': 'Dynamic imports should be avoided',
            r'input\s*\(': 'Use of input() in production code should be avoided',
            r'os\.system\s*\(': 'Use of os.system() is forbidden, use subprocess instead',
            r'shell=True': 'Use of shell=True in subprocess is dangerous',
            r'pickle\.load': 'Use of pickle.load() can be dangerous with untrusted data',
            r'yaml\.load\s*\(': 'Use yaml.safe_load() instead of yaml.load()',
            r'\.format\s*\([^)]*\%': 'String formatting with % operator can be dangerous',
            r'random\.random\s*\(\)': 'Use secrets module for cryptographic randomness',
            r'md5\s*\(': 'MD5 is cryptographically broken, use SHA-256 or better',
            r'sha1\s*\(': 'SHA-1 is deprecated, use SHA-256 or better'
        }

        # Dangerous imports
        self.dangerous_imports = {
            'pickle': 'Pickle can execute arbitrary code when deserializing',
            'marshal': 'Marshal can execute arbitrary code when deserializing',
            'dill': 'Dill can execute arbitrary code when deserializing',
            'subprocess': 'Subprocess usage should be carefully reviewed for shell injection'
        }

        # Required security imports for certain operations
        self.security_imports = {
            'secrets': 'for cryptographic randomness',
            'hashlib': 'for secure hashing',
            'hmac': 'for message authentication',
            'cryptography': 'for encryption operations'
        }

        # Patterns that should use environment variables
        self.env_var_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']'
        ]

    def add_violation(self, message: str) -> None:
        """Add a security violation."""
        self.violations.append(f"SECURITY VIOLATION: {message}")

    def add_warning(self, message: str) -> None:
        """Add a security warning."""
        self.warnings.append(f"SECURITY WARNING: {message}")

    def check_file_content(self, file_path: Path) -> None:
        """Check a Python file for security issues."""
        try:
            content = file_path.read_text(encoding='utf-8')
            relative_path = file_path.relative_to(self.project_root)

            # Check for forbidden patterns
            for pattern, message in self.forbidden_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.add_violation(f"{relative_path}:{line_num} - {message}")

            # Check for hardcoded secrets
            for pattern in self.env_var_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.add_violation(
                        f"{relative_path}:{line_num} - Hardcoded credential detected, use environment variables"
                    )

            # Check for SQL injection vulnerabilities
            sql_patterns = [
                r'f["\'].*SELECT.*{.*}.*["\']',
                r'["\'].*SELECT.*["\']\s*\+',
                r'["\'].*INSERT.*["\']\s*\+',
                r'["\'].*UPDATE.*["\']\s*\+',
                r'["\'].*DELETE.*["\']\s*\+'
            ]

            for pattern in sql_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.add_violation(
                        f"{relative_path}:{line_num} - Potential SQL injection vulnerability"
                    )

            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self.check_ast(tree, relative_path)
            except SyntaxError:
                self.add_warning(f"{relative_path} - Could not parse AST (syntax error)")

        except Exception as e:
            self.add_warning(f"Failed to check {file_path}: {e}")

    def check_ast(self, tree: ast.AST, file_path: Path) -> None:
        """Check AST for security issues."""
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_imports:
                        self.add_warning(
                            f"{file_path}:{node.lineno} - {self.dangerous_imports[alias.name]}"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module in self.dangerous_imports:
                    self.add_warning(
                        f"{file_path}:{node.lineno} - {self.dangerous_imports[node.module]}"
                    )

            # Check for unsafe function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        self.add_violation(
                            f"{file_path}:{node.lineno} - Use of {node.func.id}() is forbidden"
                        )

                elif isinstance(node.func, ast.Attribute):
                    # Check for os.system calls
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'os' and 
                        node.func.attr == 'system'):
                        self.add_violation(
                            f"{file_path}:{node.lineno} - Use of os.system() is forbidden"
                        )

            # Check for bare except clauses
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.add_warning(
                        f"{file_path}:{node.lineno} - Bare except clause can hide security issues"
                    )

            # Check for assert statements (can be disabled with -O)
            elif isinstance(node, ast.Assert):
                self.add_warning(
                    f"{file_path}:{node.lineno} - Assert statements are disabled with -O flag"
                )

    def check_environment_file(self) -> None:
        """Check .env.example for security issues."""
        env_file = self.project_root / ".env.example"
        
        if not env_file.exists():
            self.add_warning(".env.example file not found")
            return

        try:
            content = env_file.read_text()
            
            # Check for real secrets in example file
            suspicious_patterns = [
                r'=.*[A-Za-z0-9+/]{40,}',  # Base64-like strings
                r'=.*[a-fA-F0-9]{32,}',    # Hex strings
                r'=.*sk_[a-zA-Z0-9]+',     # Stripe-like keys
                r'=.*pk_[a-zA-Z0-9]+',     # Public keys
                r'=ghp_[a-zA-Z0-9]+',      # GitHub tokens
            ]

            for pattern in suspicious_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.add_warning(
                        f".env.example:{line_num} - Potential real secret in example file"
                    )

        except Exception as e:
            self.add_warning(f"Failed to check .env.example: {e}")

    def check_docker_security(self) -> None:
        """Check Docker files for security issues."""
        dockerfile_paths = [
            self.project_root / "Dockerfile",
            self.project_root / "Dockerfile.dev",
            self.project_root / "Dockerfile.prod"
        ]

        for dockerfile in dockerfile_paths:
            if not dockerfile.exists():
                continue

            try:
                content = dockerfile.read_text()
                relative_path = dockerfile.relative_to(self.project_root)

                # Check for security issues
                if "USER root" in content and not re.search(r'USER\s+(?!root)', content):
                    self.add_violation(f"{relative_path} - Running as root without switching user")

                if re.search(r'ADD\s+http', content):
                    self.add_warning(f"{relative_path} - Using ADD with URL, consider COPY")

                if "--privileged" in content:
                    self.add_violation(f"{relative_path} - Using privileged mode")

                if re.search(r'COPY\s+\.\s+\.', content):
                    self.add_warning(f"{relative_path} - Copying entire context, use .dockerignore")

            except Exception as e:
                self.add_warning(f"Failed to check {dockerfile}: {e}")

    def check_python_files(self) -> None:
        """Check all Python files in the project."""
        python_files = []
        
        # Find all Python files
        for pattern in ["**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))

        # Filter out excluded directories
        excluded_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules'}
        
        for py_file in python_files:
            # Skip files in excluded directories
            if any(excluded_dir in py_file.parts for excluded_dir in excluded_dirs):
                continue
                
            self.check_file_content(py_file)

    def run_security_check(self) -> bool:
        """Run all security checks."""
        print("🔒 Checking security policies...")
        
        self.check_python_files()
        self.check_environment_file()
        self.check_docker_security()

        # Report results
        if self.warnings:
            print("\n⚠️  Security Warnings:")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.violations:
            print("\n❌ Security Violations:")
            for violation in self.violations:
                print(f"  {violation}")
            print(f"\n💥 Security check failed with {len(self.violations)} violation(s)")
            return False
        else:
            warning_count = len(self.warnings)
            if warning_count > 0:
                print(f"\n✅ Security check passed with {warning_count} warning(s)")
            else:
                print("\n✅ No security policy violations found!")
            return True


def main():
    """Main entry point."""
    checker = SecurityPolicyChecker()
    success = checker.run_security_check()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()