#!/usr/bin/env python3
"""Quality gates runner - validates code quality, security, and performance."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success = False
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.execution_time = 0.0
    
    async def run(self) -> bool:
        """Run the quality gate check."""
        start_time = time.time()
        try:
            logger.info(f"Running quality gate: {self.name}")
            self.success = await self._execute()
        except Exception as e:
            self.success = False
            self.errors.append(f"Quality gate execution failed: {str(e)}")
            logger.error(f"Quality gate {self.name} failed: {e}")
        finally:
            self.execution_time = time.time() - start_time
        
        status = "PASS" if self.success else "FAIL"
        logger.info(f"Quality gate {self.name}: {status} ({self.execution_time:.2f}s)")
        
        return self.success
    
    async def _execute(self) -> bool:
        """Execute the quality gate check. Override in subclasses."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "execution_time": self.execution_time
        }


class CodeQualityGate(QualityGate):
    """Code quality and style checks."""
    
    def __init__(self):
        super().__init__("Code Quality", "Check code style, complexity, and quality metrics")
    
    async def _execute(self) -> bool:
        """Execute code quality checks."""
        success = True
        
        # Check Python syntax
        if not await self._check_python_syntax():
            success = False
        
        # Check code structure
        if not await self._check_code_structure():
            success = False
        
        # Check for TODO/FIXME comments
        await self._check_code_comments()
        
        return success
    
    async def _check_python_syntax(self) -> bool:
        """Check Python syntax in all Python files."""
        python_files = list(Path(".").rglob("*.py"))
        syntax_errors = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {py_file}: {e}")
                syntax_errors += 1
            except Exception as e:
                self.warnings.append(f"Could not check syntax in {py_file}: {e}")
        
        self.metrics["python_files_checked"] = len(python_files)
        self.metrics["syntax_errors"] = syntax_errors
        
        return syntax_errors == 0
    
    async def _check_code_structure(self) -> bool:
        """Check code structure and organization."""
        issues = 0
        
        # Check for required directories
        required_dirs = ["healing_guard", "tests", "config"]
        for required_dir in required_dirs:
            if not Path(required_dir).exists():
                self.errors.append(f"Required directory missing: {required_dir}")
                issues += 1
        
        # Check for required files
        required_files = ["README.md", "pyproject.toml", "Dockerfile"]
        for required_file in required_files:
            if not Path(required_file).exists():
                self.warnings.append(f"Recommended file missing: {required_file}")
        
        # Check module structure
        healing_guard_path = Path("healing_guard")
        if healing_guard_path.exists():
            required_modules = ["core", "api", "monitoring", "security"]
            for module in required_modules:
                module_path = healing_guard_path / module
                if not module_path.exists():
                    self.errors.append(f"Required module missing: healing_guard.{module}")
                    issues += 1
                elif not (module_path / "__init__.py").exists():
                    self.warnings.append(f"Module init file missing: healing_guard.{module}.__init__.py")
        
        self.metrics["structure_issues"] = issues
        return issues == 0
    
    async def _check_code_comments(self):
        """Check for TODO/FIXME comments that need attention."""
        python_files = list(Path(".").rglob("*.py"))
        todo_count = 0
        fixme_count = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count TODO/FIXME comments
                import re
                todos = re.findall(r'#.*TODO', content, re.IGNORECASE)
                fixmes = re.findall(r'#.*FIXME', content, re.IGNORECASE)
                
                todo_count += len(todos)
                fixme_count += len(fixmes)
                
            except Exception:
                continue
        
        self.metrics["todo_comments"] = todo_count
        self.metrics["fixme_comments"] = fixme_count
        
        if fixme_count > 0:
            self.warnings.append(f"Found {fixme_count} FIXME comments that should be addressed")


class SecurityGate(QualityGate):
    """Security vulnerability checks."""
    
    def __init__(self):
        super().__init__("Security", "Check for security vulnerabilities and issues")
    
    async def _execute(self) -> bool:
        """Execute security checks."""
        success = True
        
        # Check for hardcoded secrets
        if not await self._check_hardcoded_secrets():
            success = False
        
        # Check for insecure patterns
        if not await self._check_insecure_patterns():
            success = False
        
        # Check file permissions
        await self._check_file_permissions()
        
        return success
    
    async def _check_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets in code."""
        python_files = list(Path(".").rglob("*.py"))
        secrets_found = 0
        
        # Common secret patterns
        secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*[=:]\s*[\'"][^\'"]{3,}[\'"]', "password"),
            (r'(?i)(api_key|apikey)\s*[=:]\s*[\'"][^\'"]{10,}[\'"]', "api_key"),
            (r'(?i)(secret|token)\s*[=:]\s*[\'"][^\'"]{10,}[\'"]', "secret"),
            (r'(?i)(access_key|access_secret)\s*[=:]\s*[\'"][^\'"]{10,}[\'"]', "access_key")
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                import re
                for pattern, secret_type in secret_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Filter out obvious test/example values
                        filtered_matches = [
                            match for match in matches
                            if not any(test_val in str(match).lower() 
                                     for test_val in ['test', 'example', 'dummy', 'fake', 'sample'])
                        ]
                        
                        if filtered_matches:
                            self.errors.append(f"Potential hardcoded {secret_type} in {py_file}")
                            secrets_found += len(filtered_matches)
                            
            except Exception:
                continue
        
        self.metrics["potential_secrets"] = secrets_found
        return secrets_found == 0
    
    async def _check_insecure_patterns(self) -> bool:
        """Check for insecure code patterns."""
        python_files = list(Path(".").rglob("*.py"))
        insecure_patterns_found = 0
        
        # Insecure patterns to check for
        insecure_patterns = [
            (r'(?i)eval\s*\(', "Use of eval() function"),
            (r'(?i)exec\s*\(', "Use of exec() function"),
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', "Shell injection via subprocess"),
            (r'(?i)md5\s*\(', "Use of weak MD5 hash"),
            (r'(?i)debug\s*=\s*True', "Debug mode enabled"),
            (r'(?i)ssl_verify\s*=\s*False', "SSL verification disabled")
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                import re
                for pattern, description in insecure_patterns:
                    if re.search(pattern, content):
                        self.warnings.append(f"{description} found in {py_file}")
                        insecure_patterns_found += 1
                        
            except Exception:
                continue
        
        self.metrics["insecure_patterns"] = insecure_patterns_found
        
        # Don't fail the gate for patterns, just warn
        return True
    
    async def _check_file_permissions(self):
        """Check file permissions for security issues."""
        sensitive_files = [
            "config/secrets.yml",
            "config/production.yml", 
            ".env",
            "docker-compose.prod.yml"
        ]
        
        permission_issues = 0
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                # Check if file is world-readable
                stat_info = path.stat()
                if stat_info.st_mode & 0o044:  # World or group readable
                    self.warnings.append(f"File {file_path} may have overly permissive permissions")
                    permission_issues += 1
        
        self.metrics["permission_issues"] = permission_issues


class PerformanceGate(QualityGate):
    """Performance and scalability checks."""
    
    def __init__(self):
        super().__init__("Performance", "Check performance characteristics and scalability")
    
    async def _execute(self) -> bool:
        """Execute performance checks."""
        success = True
        
        # Check for performance anti-patterns
        if not await self._check_performance_patterns():
            success = False
        
        # Analyze code complexity
        await self._analyze_code_complexity()
        
        # Check resource usage patterns
        await self._check_resource_patterns()
        
        return success
    
    async def _check_performance_patterns(self) -> bool:
        """Check for performance anti-patterns."""
        python_files = list(Path(".").rglob("*.py"))
        performance_issues = 0
        
        # Performance anti-patterns
        anti_patterns = [
            (r'time\.sleep\(\d+\)', "Long sleep() calls can impact performance"),
            (r'for.*in.*range\(.*\d{4,}', "Large range loops may impact performance"),
            (r'(?i)n\+1.*query', "Potential N+1 query problem"),
            (r'\.append\(.*\)\s*\n.*for.*in', "List append in loop - consider list comprehension")
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                import re
                for pattern, description in anti_patterns:
                    if re.search(pattern, content):
                        self.warnings.append(f"{description} in {py_file}")
                        performance_issues += 1
                        
            except Exception:
                continue
        
        self.metrics["performance_warnings"] = performance_issues
        return True  # Don't fail for performance warnings
    
    async def _analyze_code_complexity(self):
        """Analyze code complexity metrics."""
        python_files = list(Path(".").rglob("*.py"))
        total_lines = 0
        total_functions = 0
        complex_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                # Simple function counting
                import re
                functions = re.findall(r'^\s*def\s+\w+', ''.join(lines), re.MULTILINE)
                total_functions += len(functions)
                
                # Check for very long functions (> 50 lines)
                current_function_lines = 0
                in_function = False
                
                for line in lines:
                    if re.match(r'^\s*def\s+\w+', line):
                        if in_function and current_function_lines > 50:
                            complex_functions += 1
                        in_function = True
                        current_function_lines = 0
                    elif re.match(r'^\s*class\s+\w+', line):
                        if in_function and current_function_lines > 50:
                            complex_functions += 1
                        in_function = False
                        current_function_lines = 0
                    elif in_function:
                        current_function_lines += 1
                        
            except Exception:
                continue
        
        self.metrics["total_lines"] = total_lines
        self.metrics["total_functions"] = total_functions
        self.metrics["complex_functions"] = complex_functions
        
        if complex_functions > 0:
            self.warnings.append(f"Found {complex_functions} functions with >50 lines - consider refactoring")
    
    async def _check_resource_patterns(self):
        """Check for resource usage patterns."""
        python_files = list(Path(".").rglob("*.py"))
        resource_issues = 0
        
        # Resource usage patterns to check
        resource_patterns = [
            (r'open\([^)]*\)(?!\s*\.close\(\))', "File opened without explicit close"),
            (r'(?i)while\s+True:', "Infinite loop detected - ensure proper exit conditions"),
            (r'(?i)recursion.*limit', "Recursion limit modification")
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                import re
                for pattern, description in resource_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Filter out with context manager usage
                        if "file opened" in description.lower():
                            # Check if using context manager
                            if "with open" in content:
                                continue
                        
                        self.warnings.append(f"{description} in {py_file}")
                        resource_issues += 1
                        
            except Exception:
                continue
        
        self.metrics["resource_warnings"] = resource_issues


class TestGate(QualityGate):
    """Test coverage and quality checks."""
    
    def __init__(self):
        super().__init__("Tests", "Check test coverage and test quality")
    
    async def _execute(self) -> bool:
        """Execute test checks."""
        success = True
        
        # Check test structure
        if not await self._check_test_structure():
            success = False
        
        # Analyze test coverage
        await self._analyze_test_coverage()
        
        return success
    
    async def _check_test_structure(self) -> bool:
        """Check test directory structure and organization."""
        issues = 0
        
        # Check if tests directory exists
        tests_dir = Path("tests")
        if not tests_dir.exists():
            self.errors.append("Tests directory not found")
            return False
        
        # Check for test subdirectories
        expected_test_dirs = ["unit", "integration", "e2e"]
        for test_dir in expected_test_dirs:
            test_path = tests_dir / test_dir
            if not test_path.exists():
                self.warnings.append(f"Recommended test directory not found: tests/{test_dir}")
        
        # Count test files
        test_files = list(tests_dir.rglob("test_*.py"))
        if len(test_files) == 0:
            self.errors.append("No test files found")
            issues += 1
        
        self.metrics["test_files"] = len(test_files)
        
        # Check if conftest.py exists (pytest configuration)
        if (tests_dir / "conftest.py").exists():
            self.metrics["has_pytest_config"] = True
        else:
            self.warnings.append("No pytest configuration (conftest.py) found")
        
        return issues == 0
    
    async def _analyze_test_coverage(self):
        """Analyze test coverage by comparing source and test files."""
        source_files = list(Path("healing_guard").rglob("*.py"))
        test_files = list(Path("tests").rglob("test_*.py"))
        
        # Simple coverage estimation based on file names
        source_modules = set()
        for source_file in source_files:
            if source_file.name != "__init__.py":
                module_name = source_file.stem
                source_modules.add(module_name)
        
        tested_modules = set()
        for test_file in test_files:
            # Extract module name from test file (e.g., test_quantum_planner.py -> quantum_planner)
            if test_file.name.startswith("test_"):
                module_name = test_file.stem[5:]  # Remove "test_" prefix
                tested_modules.add(module_name)
        
        coverage_ratio = len(tested_modules) / len(source_modules) if source_modules else 0
        
        self.metrics["source_modules"] = len(source_modules)
        self.metrics["tested_modules"] = len(tested_modules)
        self.metrics["estimated_coverage"] = coverage_ratio * 100
        
        if coverage_ratio < 0.8:  # Less than 80% coverage
            self.warnings.append(f"Low test coverage: {coverage_ratio*100:.1f}% (target: 80%)")


class QualityGateRunner:
    """Runner for all quality gates."""
    
    def __init__(self):
        self.gates: List[QualityGate] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def register_gates(self):
        """Register all quality gates."""
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            PerformanceGate(),
            TestGate()
        ]
    
    async def run_all_gates(self) -> bool:
        """Run all quality gates."""
        logger.info("Starting quality gates execution...")
        start_time = time.time()
        
        all_passed = True
        
        for gate in self.gates:
            success = await gate.run()
            self.results[gate.name] = gate.to_dict()
            
            if not success:
                all_passed = False
        
        total_time = time.time() - start_time
        
        # Generate summary report
        self._generate_report(all_passed, total_time)
        
        return all_passed
    
    def _generate_report(self, all_passed: bool, total_time: float):
        """Generate quality gates report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PASS" if all_passed else "FAIL",
            "total_execution_time": total_time,
            "gates": self.results,
            "summary": {
                "total_gates": len(self.gates),
                "passed_gates": sum(1 for gate in self.results.values() if gate["success"]),
                "failed_gates": sum(1 for gate in self.results.values() if not gate["success"]),
                "total_errors": sum(len(gate["errors"]) for gate in self.results.values()),
                "total_warnings": sum(len(gate["warnings"]) for gate in self.results.values())
            }
        }
        
        # Save report to file
        report_file = Path("quality_gates_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("QUALITY GATES REPORT")
        print("="*80)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Gates Passed: {report['summary']['passed_gates']}/{report['summary']['total_gates']}")
        print(f"Total Errors: {report['summary']['total_errors']}")
        print(f"Total Warnings: {report['summary']['total_warnings']}")
        print()
        
        # Print individual gate results
        for gate_name, gate_result in self.results.items():
            status = "PASS" if gate_result["success"] else "FAIL"
            errors = len(gate_result["errors"])
            warnings = len(gate_result["warnings"])
            time_taken = gate_result["execution_time"]
            
            print(f"  {gate_name}: {status} ({time_taken:.2f}s)")
            if errors > 0:
                print(f"    Errors: {errors}")
                for error in gate_result["errors"][:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if errors > 3:
                    print(f"      ... and {errors - 3} more")
            
            if warnings > 0:
                print(f"    Warnings: {warnings}")
                for warning in gate_result["warnings"][:2]:  # Show first 2 warnings
                    print(f"      - {warning}")
                if warnings > 2:
                    print(f"      ... and {warnings - 2} more")
            print()
        
        print(f"Detailed report saved to: {report_file}")
        print("="*80)


async def main():
    """Main entry point for quality gates runner."""
    runner = QualityGateRunner()
    runner.register_gates()
    
    success = await runner.run_all_gates()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())