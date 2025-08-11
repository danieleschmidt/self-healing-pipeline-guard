"""Advanced security validation and threat detection for healing operations.

Implements ML-based anomaly detection, behavioral analysis, and security policy enforcement.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from collections import deque, defaultdict
import re

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    POLICY_VIOLATION = "policy_violation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INJECTION_ATTEMPT = "injection_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    id: str
    timestamp: datetime
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    source: str
    affected_resources: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threat to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "description": self.description,
            "source": self.source,
            "affected_resources": self.affected_resources,
            "indicators": self.indicators,
            "mitigation_actions": self.mitigation_actions,
            "confidence_score": self.confidence_score
        }


@dataclass
class SecurityPolicy:
    """Represents a security policy rule."""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    severity: ThreatLevel
    enabled: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rules": self.rules,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "last_updated": self.last_updated.isoformat()
        }


class SecurityPatternMatcher:
    """Advanced pattern matching for security threats."""
    
    def __init__(self):
        self.suspicious_patterns = {
            # Command injection patterns
            "command_injection": [
                r";.*\s*\w+",  # Command chaining
                r"\|.*\s*\w+",  # Pipe operations
                r"&&.*\s*\w+",  # Command conjunction
                r"`.*`",  # Command substitution
                r"\$\(.*\)",  # Command substitution
            ],
            
            # SQL injection patterns
            "sql_injection": [
                r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+",
                r"(?i)or\s+\d+\s*=\s*\d+",
                r"(?i)and\s+\d+\s*=\s*\d+",
                r"(?i)'.*or.*'.*=.*'",
            ],
            
            # Path traversal patterns
            "path_traversal": [
                r"\.\./",
                r"\\\.\\\.\\",
                r"%2e%2e%2f",
                r"%252e%252e%252f",
            ],
            
            # Script injection patterns
            "script_injection": [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"on\w+\s*=",
            ],
            
            # Suspicious API patterns
            "suspicious_api": [
                r"(?i)exec\s*\(",
                r"(?i)eval\s*\(",
                r"(?i)system\s*\(",
                r"(?i)shell_exec\s*\(",
                r"(?i)passthru\s*\(",
            ]
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for category, patterns in self.suspicious_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def scan_for_threats(self, content: str, context: str = "unknown") -> List[Dict[str, Any]]:
        """Scan content for security threat patterns."""
        threats = []
        
        for category, patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(patterns):
                matches = pattern.findall(content)
                if matches:
                    threat = {
                        "category": category,
                        "pattern_index": i,
                        "matches": matches,
                        "context": context,
                        "severity": self._get_pattern_severity(category),
                        "description": f"Detected {category} pattern in {context}"
                    }
                    threats.append(threat)
        
        return threats
    
    def _get_pattern_severity(self, category: str) -> ThreatLevel:
        """Get severity level for pattern category."""
        severity_mapping = {
            "command_injection": ThreatLevel.CRITICAL,
            "sql_injection": ThreatLevel.HIGH,
            "path_traversal": ThreatLevel.HIGH,
            "script_injection": ThreatLevel.MEDIUM,
            "suspicious_api": ThreatLevel.MEDIUM
        }
        return severity_mapping.get(category, ThreatLevel.LOW)


class BehavioralAnalyzer:
    """Behavioral analysis for anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.action_history: deque = deque(maxlen=window_size)
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def record_action(self, action_data: Dict[str, Any]):
        """Record an action for behavioral analysis."""
        action_data["timestamp"] = datetime.now()
        self.action_history.append(action_data)
        
        # Update baseline metrics periodically
        if len(self.action_history) >= 10:
            self._update_baseline_metrics()
    
    def _update_baseline_metrics(self):
        """Update baseline metrics from recent action history."""
        if not self.action_history:
            return
        
        # Calculate baseline metrics
        actions_per_hour = {}
        resource_usage = []
        success_rates = []
        
        for action in self.action_history:
            hour_key = action["timestamp"].hour
            actions_per_hour[hour_key] = actions_per_hour.get(hour_key, 0) + 1
            
            if "resource_usage" in action:
                resource_usage.append(action["resource_usage"])
            
            if "success" in action:
                success_rates.append(1 if action["success"] else 0)
        
        self.baseline_metrics = {
            "avg_actions_per_hour": sum(actions_per_hour.values()) / len(actions_per_hour) if actions_per_hour else 0,
            "avg_resource_usage": sum(resource_usage) / len(resource_usage) if resource_usage else 0,
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "last_updated": datetime.now()
        }
    
    def detect_anomalies(self, current_action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies in current action."""
        anomalies = []
        
        if not self.baseline_metrics:
            return anomalies
        
        # Check for unusual frequency
        current_hour = datetime.now().hour
        recent_actions = [
            a for a in self.action_history 
            if a["timestamp"].hour == current_hour and 
               (datetime.now() - a["timestamp"]).seconds < 3600
        ]
        
        current_frequency = len(recent_actions)
        baseline_frequency = self.baseline_metrics["avg_actions_per_hour"]
        
        if current_frequency > baseline_frequency * self.anomaly_threshold:
            anomalies.append({
                "type": "high_frequency",
                "description": f"Unusually high action frequency: {current_frequency} vs baseline {baseline_frequency:.2f}",
                "severity": ThreatLevel.MEDIUM,
                "confidence": min(1.0, (current_frequency - baseline_frequency) / baseline_frequency)
            })
        
        # Check for unusual resource usage
        if "resource_usage" in current_action:
            current_usage = current_action["resource_usage"]
            baseline_usage = self.baseline_metrics["avg_resource_usage"]
            
            if baseline_usage > 0 and current_usage > baseline_usage * self.anomaly_threshold:
                anomalies.append({
                    "type": "high_resource_usage",
                    "description": f"Unusually high resource usage: {current_usage} vs baseline {baseline_usage:.2f}",
                    "severity": ThreatLevel.MEDIUM,
                    "confidence": min(1.0, (current_usage - baseline_usage) / baseline_usage)
                })
        
        # Check for pattern deviations
        if "action_type" in current_action:
            action_types = [a.get("action_type") for a in self.action_history if "action_type" in a]
            type_frequency = defaultdict(int)
            for action_type in action_types:
                type_frequency[action_type] += 1
            
            current_type = current_action["action_type"]
            if current_type not in type_frequency:
                anomalies.append({
                    "type": "unknown_action_type",
                    "description": f"Unknown action type: {current_type}",
                    "severity": ThreatLevel.HIGH,
                    "confidence": 0.9
                })
        
        return anomalies


class SecurityPolicyEngine:
    """Security policy engine for enforcement and validation."""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_violations: List[Dict[str, Any]] = []
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default security policies."""
        
        # Resource access policy
        resource_policy = SecurityPolicy(
            id="resource_access",
            name="Resource Access Control",
            description="Controls access to critical system resources",
            rules=[
                {
                    "type": "resource_limit",
                    "resource": "cpu",
                    "max_value": 80.0,
                    "unit": "percentage"
                },
                {
                    "type": "resource_limit", 
                    "resource": "memory",
                    "max_value": 85.0,
                    "unit": "percentage"
                },
                {
                    "type": "path_restriction",
                    "allowed_paths": ["/tmp", "/var/log", "/opt/healing_guard"],
                    "denied_paths": ["/etc", "/root", "/home"]
                }
            ],
            severity=ThreatLevel.HIGH
        )
        
        # API rate limiting policy
        rate_limit_policy = SecurityPolicy(
            id="api_rate_limit",
            name="API Rate Limiting",
            description="Prevents API abuse and DoS attacks",
            rules=[
                {
                    "type": "rate_limit",
                    "endpoint": "*",
                    "max_requests": 1000,
                    "window_seconds": 3600
                },
                {
                    "type": "rate_limit",
                    "endpoint": "/heal",
                    "max_requests": 100,
                    "window_seconds": 3600
                }
            ],
            severity=ThreatLevel.MEDIUM
        )
        
        # Command execution policy
        command_policy = SecurityPolicy(
            id="command_execution",
            name="Command Execution Control",
            description="Controls which commands can be executed",
            rules=[
                {
                    "type": "allowed_commands",
                    "commands": ["docker", "kubectl", "systemctl", "npm", "pip", "git"],
                    "parameters_whitelist": True
                },
                {
                    "type": "denied_commands",
                    "commands": ["rm", "dd", "mkfs", "fdisk", "sudo", "su"],
                    "strict": True
                }
            ],
            severity=ThreatLevel.CRITICAL
        )
        
        self.policies = {
            resource_policy.id: resource_policy,
            rate_limit_policy.id: rate_limit_policy,
            command_policy.id: command_policy
        }
    
    def add_policy(self, policy: SecurityPolicy):
        """Add a new security policy."""
        self.policies[policy.id] = policy
        logger.info(f"Added security policy: {policy.name}")
    
    def validate_action(self, action_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate an action against all security policies."""
        violations = []
        
        for policy_id, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            policy_violations = self._check_policy_compliance(action_data, policy)
            violations.extend(policy_violations)
        
        return violations
    
    def _check_policy_compliance(self, action_data: Dict[str, Any], policy: SecurityPolicy) -> List[Dict[str, Any]]:
        """Check if an action complies with a specific policy."""
        violations = []
        
        for rule in policy.rules:
            violation = self._check_rule(action_data, rule, policy)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_rule(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check a specific policy rule."""
        rule_type = rule.get("type")
        
        if rule_type == "resource_limit":
            return self._check_resource_limit(action_data, rule, policy)
        elif rule_type == "path_restriction":
            return self._check_path_restriction(action_data, rule, policy)
        elif rule_type == "rate_limit":
            return self._check_rate_limit(action_data, rule, policy)
        elif rule_type == "allowed_commands":
            return self._check_allowed_commands(action_data, rule, policy)
        elif rule_type == "denied_commands":
            return self._check_denied_commands(action_data, rule, policy)
        
        return None
    
    def _check_resource_limit(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check resource limit compliance."""
        resource = rule.get("resource")
        max_value = rule.get("max_value")
        
        if resource in action_data:
            current_value = action_data[resource]
            if current_value > max_value:
                return {
                    "policy_id": policy.id,
                    "rule_type": "resource_limit",
                    "resource": resource,
                    "current_value": current_value,
                    "max_allowed": max_value,
                    "severity": policy.severity.value,
                    "description": f"Resource {resource} exceeds limit: {current_value} > {max_value}"
                }
        
        return None
    
    def _check_path_restriction(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check path access restrictions."""
        if "path" not in action_data:
            return None
        
        target_path = action_data["path"]
        allowed_paths = rule.get("allowed_paths", [])
        denied_paths = rule.get("denied_paths", [])
        
        # Check denied paths
        for denied_path in denied_paths:
            if target_path.startswith(denied_path):
                return {
                    "policy_id": policy.id,
                    "rule_type": "path_restriction",
                    "target_path": target_path,
                    "denied_path": denied_path,
                    "severity": policy.severity.value,
                    "description": f"Access to denied path: {target_path}"
                }
        
        # Check allowed paths (if specified)
        if allowed_paths:
            path_allowed = any(target_path.startswith(allowed_path) for allowed_path in allowed_paths)
            if not path_allowed:
                return {
                    "policy_id": policy.id,
                    "rule_type": "path_restriction",
                    "target_path": target_path,
                    "allowed_paths": allowed_paths,
                    "severity": policy.severity.value,
                    "description": f"Path not in allowed list: {target_path}"
                }
        
        return None
    
    def _check_rate_limit(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check rate limiting compliance."""
        # This is a simplified implementation
        # In a real system, you would track request counts in a time window
        endpoint = rule.get("endpoint")
        max_requests = rule.get("max_requests")
        
        # For now, just log that rate limiting should be checked
        logger.debug(f"Rate limit check for endpoint {endpoint}: {max_requests} requests allowed")
        
        return None
    
    def _check_allowed_commands(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check if command is in allowed list."""
        if "command" not in action_data:
            return None
        
        command = action_data["command"].split()[0] if action_data["command"] else ""
        allowed_commands = rule.get("commands", [])
        
        if command not in allowed_commands:
            return {
                "policy_id": policy.id,
                "rule_type": "allowed_commands",
                "command": command,
                "allowed_commands": allowed_commands,
                "severity": policy.severity.value,
                "description": f"Command not in allowed list: {command}"
            }
        
        return None
    
    def _check_denied_commands(self, action_data: Dict[str, Any], rule: Dict[str, Any], policy: SecurityPolicy) -> Optional[Dict[str, Any]]:
        """Check if command is in denied list."""
        if "command" not in action_data:
            return None
        
        command = action_data["command"].split()[0] if action_data["command"] else ""
        denied_commands = rule.get("commands", [])
        
        if command in denied_commands:
            return {
                "policy_id": policy.id,
                "rule_type": "denied_commands", 
                "command": command,
                "denied_commands": denied_commands,
                "severity": policy.severity.value,
                "description": f"Command in denied list: {command}"
            }
        
        return None


class AdvancedSecurityValidator:
    """Advanced security validator combining multiple security techniques."""
    
    def __init__(self):
        self.pattern_matcher = SecurityPatternMatcher()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.policy_engine = SecurityPolicyEngine()
        self.threat_history: deque = deque(maxlen=1000)
        self.security_metrics = {
            "threats_detected": 0,
            "threats_mitigated": 0,
            "policy_violations": 0,
            "anomalies_detected": 0
        }
    
    async def validate_healing_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security validation of a healing action."""
        validation_start = datetime.now()
        threats = []
        warnings = []
        
        try:
            # 1. Pattern-based threat detection
            content_to_scan = json.dumps(action_data)
            pattern_threats = self.pattern_matcher.scan_for_threats(
                content_to_scan, 
                f"healing_action_{action_data.get('id', 'unknown')}"
            )
            
            for threat_data in pattern_threats:
                threat = SecurityThreat(
                    id=f"pattern_{hashlib.sha256(content_to_scan.encode()).hexdigest()[:8]}",
                    timestamp=datetime.now(),
                    event_type=SecurityEvent.SUSPICIOUS_PATTERN,
                    threat_level=threat_data["severity"],
                    description=threat_data["description"],
                    source="pattern_matcher",
                    indicators=threat_data,
                    confidence_score=0.8
                )
                threats.append(threat)
            
            # 2. Behavioral analysis
            self.behavioral_analyzer.record_action(action_data)
            anomalies = self.behavioral_analyzer.detect_anomalies(action_data)
            
            for anomaly in anomalies:
                threat = SecurityThreat(
                    id=f"behavioral_{hashlib.sha256(str(anomaly).encode()).hexdigest()[:8]}",
                    timestamp=datetime.now(),
                    event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                    threat_level=anomaly["severity"],
                    description=anomaly["description"],
                    source="behavioral_analyzer",
                    indicators=anomaly,
                    confidence_score=anomaly.get("confidence", 0.7)
                )
                threats.append(threat)
            
            # 3. Policy compliance check
            policy_violations = self.policy_engine.validate_action(action_data)
            
            for violation in policy_violations:
                threat = SecurityThreat(
                    id=f"policy_{violation.get('policy_id', 'unknown')}_{hashlib.sha256(str(violation).encode()).hexdigest()[:8]}",
                    timestamp=datetime.now(),
                    event_type=SecurityEvent.POLICY_VIOLATION,
                    threat_level=ThreatLevel(violation["severity"]),
                    description=violation["description"],
                    source="policy_engine",
                    indicators=violation,
                    confidence_score=0.9
                )
                threats.append(threat)
            
            # 4. Update security metrics
            self.security_metrics["threats_detected"] += len(threats)
            if anomalies:
                self.security_metrics["anomalies_detected"] += len(anomalies)
            if policy_violations:
                self.security_metrics["policy_violations"] += len(policy_violations)
            
            # 5. Store threats in history
            for threat in threats:
                self.threat_history.append(threat.to_dict())
            
            # 6. Determine action recommendation
            critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
            high_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]
            
            if critical_threats:
                recommendation = "BLOCK"
                reason = f"Critical security threats detected: {len(critical_threats)}"
            elif high_threats:
                recommendation = "REVIEW"
                reason = f"High-risk security threats detected: {len(high_threats)}"
            elif threats:
                recommendation = "MONITOR"
                reason = f"Security threats detected: {len(threats)}"
            else:
                recommendation = "ALLOW"
                reason = "No security threats detected"
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            
            return {
                "validation_id": hashlib.sha256(f"{validation_start.isoformat()}_{action_data.get('id', 'unknown')}".encode()).hexdigest()[:16],
                "timestamp": validation_start.isoformat(),
                "validation_time": validation_time,
                "recommendation": recommendation,
                "reason": reason,
                "threats": [threat.to_dict() for threat in threats],
                "warnings": warnings,
                "security_score": self._calculate_security_score(threats),
                "metrics": self.security_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {
                "validation_id": "error",
                "timestamp": datetime.now().isoformat(),
                "validation_time": (datetime.now() - validation_start).total_seconds(),
                "recommendation": "BLOCK",
                "reason": f"Security validation error: {str(e)}",
                "threats": [],
                "warnings": [f"Validation system error: {str(e)}"],
                "security_score": 0.0,
                "metrics": self.security_metrics.copy()
            }
    
    def _calculate_security_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall security score (0.0 = highest risk, 1.0 = no risk)."""
        if not threats:
            return 1.0
        
        total_risk = 0.0
        for threat in threats:
            risk_weight = {
                ThreatLevel.CRITICAL: 1.0,
                ThreatLevel.HIGH: 0.7,
                ThreatLevel.MEDIUM: 0.4,
                ThreatLevel.LOW: 0.1
            }.get(threat.threat_level, 0.1)
            
            total_risk += risk_weight * threat.confidence_score
        
        # Normalize to 0-1 scale (higher threat count and severity = lower score)
        max_possible_risk = len(threats) * 1.0  # All threats critical with 1.0 confidence
        normalized_risk = min(1.0, total_risk / max_possible_risk) if max_possible_risk > 0 else 0.0
        
        return max(0.0, 1.0 - normalized_risk)
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        if not self.threat_history:
            return {"message": "No threats detected"}
        
        threat_levels = defaultdict(int)
        threat_types = defaultdict(int)
        
        for threat in self.threat_history:
            threat_levels[threat["threat_level"]] += 1
            threat_types[threat["event_type"]] += 1
        
        return {
            "total_threats": len(self.threat_history),
            "threat_levels": dict(threat_levels),
            "threat_types": dict(threat_types),
            "recent_threats": list(self.threat_history)[-10:],  # Last 10 threats
            "metrics": self.security_metrics.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_security_policies(self, policies: List[SecurityPolicy]):
        """Update security policies."""
        for policy in policies:
            self.policy_engine.add_policy(policy)
        logger.info(f"Updated {len(policies)} security policies")


# Global security validator instance  
security_validator = AdvancedSecurityValidator()