"""Security scanning and vulnerability detection."""

import asyncio
import hashlib
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import tempfile
import os

import aiofiles
import aiohttp
from packaging import version

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CODE_INJECTION = "code_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    INSECURE_DEPENDENCIES = "insecure_dependencies"
    SECRETS_EXPOSURE = "secrets_exposure"
    INSECURE_CONFIG = "insecure_config"
    WEAK_CRYPTO = "weak_crypto"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FLAW = "authorization_flaw"
    DATA_EXPOSURE = "data_exposure"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    vulnerability_type: VulnerabilityType
    affected_files: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    cve_ids: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    fixed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "vulnerability_type": self.vulnerability_type.value,
            "affected_files": self.affected_files,
            "line_numbers": self.line_numbers,
            "cve_ids": self.cve_ids,
            "cvss_score": self.cvss_score,
            "remediation": self.remediation,
            "references": self.references,
            "discovered_at": self.discovered_at.isoformat(),
            "fixed": self.fixed
        }


@dataclass
class ScanResult:
    """Results of a security scan."""
    scan_id: str
    scan_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    total_files_scanned: int = 0
    scan_duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def critical_count(self) -> int:
        """Count of critical vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.CRITICAL)
    
    @property
    def high_count(self) -> int:
        """Count of high severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.HIGH)
    
    @property
    def medium_count(self) -> int:
        """Count of medium severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.MEDIUM)
    
    @property
    def low_count(self) -> int:
        """Count of low severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.LOW)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary."""
        return {
            "scan_id": self.scan_id,
            "scan_type": self.scan_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "total_files_scanned": self.total_files_scanned,
            "scan_duration": self.scan_duration,
            "success": self.success,
            "error_message": self.error_message,
            "summary": {
                "total_vulnerabilities": len(self.vulnerabilities),
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count
            }
        }


class SecurityScanner:
    """Comprehensive security scanner for code and dependencies."""
    
    def __init__(self):
        self.scan_history: List[ScanResult] = []
        self.patterns = self._load_security_patterns()
        self.dependency_cache: Dict[str, Dict[str, Any]] = {}
        
    def _load_security_patterns(self) -> Dict[VulnerabilityType, List[Dict[str, Any]]]:
        """Load security vulnerability patterns."""
        return {
            VulnerabilityType.SQL_INJECTION: [
                {
                    "pattern": r"(?i)(SELECT|INSERT|UPDATE|DELETE).*\+.*\+",
                    "description": "Potential SQL injection via string concatenation",
                    "severity": SeverityLevel.HIGH
                },
                {
                    "pattern": r"(?i)cursor\.execute\([^)]*%[^)]*\)",
                    "description": "Potential SQL injection via string formatting",
                    "severity": SeverityLevel.HIGH
                }
            ],
            VulnerabilityType.CODE_INJECTION: [
                {
                    "pattern": r"(?i)(eval|exec|compile)\s*\(",
                    "description": "Code execution via eval/exec",
                    "severity": SeverityLevel.CRITICAL
                },
                {
                    "pattern": r"(?i)subprocess\.(call|run|Popen).*shell\s*=\s*True",
                    "description": "Shell injection vulnerability",
                    "severity": SeverityLevel.HIGH
                }
            ],
            VulnerabilityType.SECRETS_EXPOSURE: [
                {
                    "pattern": r"(?i)(password|passwd|pwd|secret|key|token)\s*[=:]\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded secrets in source code",
                    "severity": SeverityLevel.CRITICAL
                },
                {
                    "pattern": r"(?i)(api_key|access_key|secret_key)\s*[=:]\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded API keys",
                    "severity": SeverityLevel.CRITICAL
                },
                {
                    "pattern": r"-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----",
                    "description": "Private key in source code",
                    "severity": SeverityLevel.CRITICAL
                }
            ],
            VulnerabilityType.WEAK_CRYPTO: [
                {
                    "pattern": r"(?i)(md5|sha1)\s*\(",
                    "description": "Weak cryptographic hash function",
                    "severity": SeverityLevel.MEDIUM
                },
                {
                    "pattern": r"(?i)random\.random\(\)",
                    "description": "Non-cryptographically secure random number generation",
                    "severity": SeverityLevel.LOW
                }
            ],
            VulnerabilityType.INSECURE_CONFIG: [
                {
                    "pattern": r"(?i)(DEBUG|debug)\s*=\s*(True|true|1)",
                    "description": "Debug mode enabled",
                    "severity": SeverityLevel.MEDIUM
                },
                {
                    "pattern": r"(?i)(ssl_verify|verify)\s*=\s*(False|false|0)",
                    "description": "SSL verification disabled",
                    "severity": SeverityLevel.HIGH
                }
            ]
        }
    
    async def scan_code(
        self, 
        scan_path: str, 
        file_patterns: Optional[List[str]] = None
    ) -> ScanResult:
        """Scan source code for security vulnerabilities."""
        scan_id = f"code_scan_{datetime.now().timestamp()}"
        result = ScanResult(
            scan_id=scan_id,
            scan_type="code_scan",
            started_at=datetime.now()
        )
        
        try:
            path = Path(scan_path)
            if not path.exists():
                raise ValueError(f"Scan path does not exist: {scan_path}")
            
            # Default file patterns for code scanning
            if file_patterns is None:
                file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rb", "*.php"]
            
            # Collect files to scan
            files_to_scan = []
            for pattern in file_patterns:
                files_to_scan.extend(path.rglob(pattern))
            
            result.total_files_scanned = len(files_to_scan)
            
            # Scan each file
            for file_path in files_to_scan:
                vulnerabilities = await self._scan_file(file_path)
                result.vulnerabilities.extend(vulnerabilities)
            
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = True
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = False
            result.error_message = str(e)
            logger.error(f"Code scan failed: {e}")
        
        self.scan_history.append(result)
        return result
    
    async def _scan_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a single file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                lines = content.split('\n')
                
                # Apply pattern matching
                for vuln_type, patterns in self.patterns.items():
                    for pattern_config in patterns:
                        pattern = pattern_config["pattern"]
                        description = pattern_config["description"]
                        severity = pattern_config["severity"]
                        
                        # Search for pattern in each line
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line):
                                vuln_id = hashlib.md5(
                                    f"{file_path}:{line_num}:{pattern}".encode()
                                ).hexdigest()[:12]
                                
                                vulnerability = SecurityVulnerability(
                                    id=vuln_id,
                                    title=f"{vuln_type.value.replace('_', ' ').title()} Vulnerability",
                                    description=description,
                                    severity=severity,
                                    vulnerability_type=vuln_type,
                                    affected_files=[str(file_path)],
                                    line_numbers=[line_num],
                                    remediation=self._get_remediation_advice(vuln_type)
                                )
                                
                                vulnerabilities.append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Failed to scan file {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_remediation_advice(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice for vulnerability type."""
        remediation_map = {
            VulnerabilityType.SQL_INJECTION: (
                "Use parameterized queries or ORM methods. "
                "Never concatenate user input directly into SQL statements."
            ),
            VulnerabilityType.CODE_INJECTION: (
                "Avoid using eval() or exec(). "
                "Validate and sanitize all user inputs. Use subprocess with explicit arguments."
            ),
            VulnerabilityType.SECRETS_EXPOSURE: (
                "Move secrets to environment variables or secure configuration files. "
                "Use a secrets management system for production environments."
            ),
            VulnerabilityType.WEAK_CRYPTO: (
                "Use strong cryptographic algorithms like SHA-256 or SHA-3. "
                "Use secrets.SystemRandom() for cryptographically secure random numbers."
            ),
            VulnerabilityType.INSECURE_CONFIG: (
                "Disable debug mode in production. "
                "Always verify SSL certificates in production environments."
            )
        }
        
        return remediation_map.get(
            vuln_type,
            "Review the code and apply security best practices for this vulnerability type."
        )
    
    async def scan_dependencies(self, requirements_file: str = "requirements.txt") -> ScanResult:
        """Scan dependencies for known vulnerabilities."""
        scan_id = f"dep_scan_{datetime.now().timestamp()}"
        result = ScanResult(
            scan_id=scan_id,
            scan_type="dependency_scan",
            started_at=datetime.now()
        )
        
        try:
            # Read requirements file
            requirements_path = Path(requirements_file)
            if not requirements_path.exists():
                raise ValueError(f"Requirements file not found: {requirements_file}")
            
            async with aiofiles.open(requirements_path, 'r') as f:
                requirements_content = await f.read()
            
            # Parse dependencies
            dependencies = self._parse_requirements(requirements_content)
            result.total_files_scanned = len(dependencies)
            
            # Check each dependency
            for dep_name, dep_version in dependencies.items():
                vulnerabilities = await self._check_dependency_vulnerabilities(dep_name, dep_version)
                result.vulnerabilities.extend(vulnerabilities)
            
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = True
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = False
            result.error_message = str(e)
            logger.error(f"Dependency scan failed: {e}")
        
        self.scan_history.append(result)
        return result
    
    def _parse_requirements(self, content: str) -> Dict[str, str]:
        """Parse requirements.txt content."""
        dependencies = {}
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle various requirement formats
                if '==' in line:
                    name, version_spec = line.split('==', 1)
                    dependencies[name.strip()] = version_spec.strip()
                elif '>=' in line:
                    name, version_spec = line.split('>=', 1)
                    dependencies[name.strip()] = version_spec.strip()
                elif '>' in line:
                    name, version_spec = line.split('>', 1)
                    dependencies[name.strip()] = version_spec.strip()
                else:
                    # No version specified
                    dependencies[line.strip()] = "latest"
        
        return dependencies
    
    async def _check_dependency_vulnerabilities(
        self, 
        package_name: str, 
        package_version: str
    ) -> List[SecurityVulnerability]:
        """Check a specific dependency for vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Use PyPI vulnerability database or OSV database
            cache_key = f"{package_name}:{package_version}"
            
            if cache_key in self.dependency_cache:
                vuln_data = self.dependency_cache[cache_key]
            else:
                vuln_data = await self._fetch_vulnerability_data(package_name, package_version)
                self.dependency_cache[cache_key] = vuln_data
            
            # Process vulnerability data
            for vuln in vuln_data.get("vulnerabilities", []):
                vuln_id = vuln.get("id", f"vuln_{package_name}_{len(vulnerabilities)}")
                
                vulnerability = SecurityVulnerability(
                    id=vuln_id,
                    title=f"Vulnerable dependency: {package_name}",
                    description=vuln.get("summary", "Known vulnerability in dependency"),
                    severity=self._map_severity(vuln.get("severity", "medium")),
                    vulnerability_type=VulnerabilityType.INSECURE_DEPENDENCIES,
                    affected_files=[f"requirements.txt ({package_name}=={package_version})"],
                    cve_ids=vuln.get("cve_ids", []),
                    cvss_score=vuln.get("cvss_score"),
                    remediation=f"Update {package_name} to version {vuln.get('fixed_version', 'latest')} or later",
                    references=vuln.get("references", [])
                )
                
                vulnerabilities.append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Failed to check vulnerabilities for {package_name}: {e}")
        
        return vulnerabilities
    
    async def _fetch_vulnerability_data(self, package_name: str, package_version: str) -> Dict[str, Any]:
        """Fetch vulnerability data from external sources."""
        # This is a simplified implementation
        # In production, you would integrate with:
        # - OSV (Open Source Vulnerabilities) database
        # - PyUp.io Safety database
        # - GitHub Advisory Database
        # - Snyk database
        
        try:
            # Example: OSV API call
            url = f"https://api.osv.dev/v1/query"
            payload = {
                "package": {
                    "name": package_name,
                    "ecosystem": "PyPI"
                },
                "version": package_version
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"vulnerabilities": data.get("vulns", [])}
        
        except Exception as e:
            logger.debug(f"Failed to fetch vulnerability data for {package_name}: {e}")
        
        return {"vulnerabilities": []}
    
    def _map_severity(self, severity_str: str) -> SeverityLevel:
        """Map external severity string to internal severity level."""
        severity_map = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "moderate": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "info": SeverityLevel.INFO
        }
        
        return severity_map.get(severity_str.lower(), SeverityLevel.MEDIUM)
    
    def get_scan_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get security scan summary for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scans = [
            scan for scan in self.scan_history
            if scan.started_at > cutoff_date
        ]
        
        if not recent_scans:
            return {"message": f"No scans in the last {days} days"}
        
        total_vulnerabilities = sum(len(scan.vulnerabilities) for scan in recent_scans)
        critical_vulnerabilities = sum(scan.critical_count for scan in recent_scans)
        high_vulnerabilities = sum(scan.high_count for scan in recent_scans)
        
        return {
            "period_days": days,
            "total_scans": len(recent_scans),
            "total_vulnerabilities": total_vulnerabilities,
            "critical_vulnerabilities": critical_vulnerabilities,
            "high_vulnerabilities": high_vulnerabilities,
            "scan_types": {
                scan_type: len([s for s in recent_scans if s.scan_type == scan_type])
                for scan_type in set(scan.scan_type for scan in recent_scans)
            },
            "avg_scan_duration": (
                sum(scan.scan_duration or 0 for scan in recent_scans) / len(recent_scans)
            )
        }


class VulnerabilityScanner:
    """Specialized vulnerability scanner with advanced detection capabilities."""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.custom_rules: List[Dict[str, Any]] = []
        
    def add_custom_rule(
        self,
        name: str,
        pattern: str,
        vulnerability_type: VulnerabilityType,
        severity: SeverityLevel,
        description: str,
        remediation: str
    ):
        """Add a custom vulnerability detection rule."""
        rule = {
            "name": name,
            "pattern": pattern,
            "vulnerability_type": vulnerability_type,
            "severity": severity,
            "description": description,
            "remediation": remediation
        }
        
        self.custom_rules.append(rule)
        logger.info(f"Added custom vulnerability rule: {name}")
    
    async def comprehensive_scan(self, project_path: str) -> Dict[str, ScanResult]:
        """Perform comprehensive security scanning."""
        results = {}
        
        # Code vulnerability scan
        code_result = await self.security_scanner.scan_code(project_path)
        results["code_scan"] = code_result
        
        # Dependency scan
        requirements_files = ["requirements.txt", "pyproject.toml", "Pipfile"]
        for req_file in requirements_files:
            req_path = Path(project_path) / req_file
            if req_path.exists():
                dep_result = await self.security_scanner.scan_dependencies(str(req_path))
                results[f"dependency_scan_{req_file}"] = dep_result
                break
        
        # Container scan (if Dockerfile exists)
        dockerfile_path = Path(project_path) / "Dockerfile"
        if dockerfile_path.exists():
            container_result = await self._scan_dockerfile(str(dockerfile_path))
            results["container_scan"] = container_result
        
        return results
    
    async def _scan_dockerfile(self, dockerfile_path: str) -> ScanResult:
        """Scan Dockerfile for security issues."""
        scan_id = f"container_scan_{datetime.now().timestamp()}"
        result = ScanResult(
            scan_id=scan_id,
            scan_type="container_scan",
            started_at=datetime.now()
        )
        
        try:
            async with aiofiles.open(dockerfile_path, 'r') as f:
                content = await f.read()
            
            lines = content.split('\n')
            result.total_files_scanned = 1
            
            # Container security patterns
            container_patterns = [
                {
                    "pattern": r"(?i)^FROM.*:latest",
                    "description": "Using 'latest' tag in base image",
                    "severity": SeverityLevel.MEDIUM,
                    "remediation": "Pin to specific version tags for reproducibility"
                },
                {
                    "pattern": r"(?i)^USER\s+root",
                    "description": "Running container as root user",
                    "severity": SeverityLevel.HIGH,
                    "remediation": "Create and use a non-root user"
                },
                {
                    "pattern": r"(?i)ADD\s+http",
                    "description": "Using ADD with HTTP URL",
                    "severity": SeverityLevel.MEDIUM,
                    "remediation": "Use RUN with curl/wget instead of ADD for HTTP resources"
                }
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern_config in container_patterns:
                    if re.search(pattern_config["pattern"], line):
                        vuln_id = hashlib.md5(
                            f"{dockerfile_path}:{line_num}:{pattern_config['pattern']}".encode()
                        ).hexdigest()[:12]
                        
                        vulnerability = SecurityVulnerability(
                            id=vuln_id,
                            title="Container Security Issue",
                            description=pattern_config["description"],
                            severity=pattern_config["severity"],
                            vulnerability_type=VulnerabilityType.INSECURE_CONFIG,
                            affected_files=[dockerfile_path],
                            line_numbers=[line_num],
                            remediation=pattern_config["remediation"]
                        )
                        
                        result.vulnerabilities.append(vulnerability)
            
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = True
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.scan_duration = (result.completed_at - result.started_at).total_seconds()
            result.success = False
            result.error_message = str(e)
            logger.error(f"Container scan failed: {e}")
        
        return result
    
    async def generate_security_report(self, scan_results: Dict[str, ScanResult]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "scan_summary": {},
            "vulnerability_breakdown": {},
            "recommendations": [],
            "risk_score": 0
        }
        
        all_vulnerabilities = []
        for scan_result in scan_results.values():
            all_vulnerabilities.extend(scan_result.vulnerabilities)
        
        # Summary statistics
        report["scan_summary"] = {
            "total_scans": len(scan_results),
            "total_vulnerabilities": len(all_vulnerabilities),
            "critical": sum(1 for v in all_vulnerabilities if v.severity == SeverityLevel.CRITICAL),
            "high": sum(1 for v in all_vulnerabilities if v.severity == SeverityLevel.HIGH),
            "medium": sum(1 for v in all_vulnerabilities if v.severity == SeverityLevel.MEDIUM),
            "low": sum(1 for v in all_vulnerabilities if v.severity == SeverityLevel.LOW)
        }
        
        # Vulnerability breakdown by type
        vuln_types = {}
        for vuln in all_vulnerabilities:
            vuln_type = vuln.vulnerability_type.value
            if vuln_type not in vuln_types:
                vuln_types[vuln_type] = {"count": 0, "severities": {}}
            
            vuln_types[vuln_type]["count"] += 1
            severity = vuln.severity.value
            vuln_types[vuln_type]["severities"][severity] = (
                vuln_types[vuln_type]["severities"].get(severity, 0) + 1
            )
        
        report["vulnerability_breakdown"] = vuln_types
        
        # Generate recommendations
        recommendations = []
        if report["scan_summary"]["critical"] > 0:
            recommendations.append({
                "priority": "URGENT",
                "action": "Fix critical vulnerabilities immediately",
                "details": f"Found {report['scan_summary']['critical']} critical vulnerabilities that require immediate attention"
            })
        
        if report["scan_summary"]["high"] > 5:
            recommendations.append({
                "priority": "HIGH",
                "action": "Address high-severity vulnerabilities",
                "details": f"Found {report['scan_summary']['high']} high-severity vulnerabilities"
            })
        
        # Calculate risk score (0-100)
        risk_score = min(100, (
            report["scan_summary"]["critical"] * 25 +
            report["scan_summary"]["high"] * 10 +
            report["scan_summary"]["medium"] * 3 +
            report["scan_summary"]["low"] * 1
        ))
        
        report["risk_score"] = risk_score
        report["recommendations"] = recommendations
        
        return report


# Global scanner instances
security_scanner = SecurityScanner()
vulnerability_scanner = VulnerabilityScanner()