"""Advanced security features for the self-healing pipeline system.

Implements multi-layered security controls including threat detection,
access control, audit logging, and security policy enforcement.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict, deque
import re

import jwt
from cryptography.fernet import Fernet
import bcrypt

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(Enum):
    """Access control levels."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3
    SUPER_ADMIN = 4


@dataclass
class SecurityEvent:
    """Represents a security event."""
    id: str
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "resource": self.resource,
            "details": self.details,
            "mitigated": self.mitigated
        }


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enabled: bool = True
    enforcement_level: str = "warn"  # warn, block, audit
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy against context."""
        violations = []
        
        for rule in self.rules:
            if self._evaluate_rule(rule, context):
                violations.append(rule)
        
        return {
            "policy": self.name,
            "violations": violations,
            "passed": len(violations) == 0,
            "enforcement_level": self.enforcement_level
        }
    
    def _evaluate_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single rule."""
        rule_type = rule.get("type")
        
        if rule_type == "rate_limit":
            return self._evaluate_rate_limit(rule, context)
        elif rule_type == "ip_whitelist":
            return self._evaluate_ip_whitelist(rule, context)
        elif rule_type == "resource_access":
            return self._evaluate_resource_access(rule, context)
        elif rule_type == "time_window":
            return self._evaluate_time_window(rule, context)
        
        return False
    
    def _evaluate_rate_limit(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate rate limiting rule."""
        max_requests = rule.get("max_requests", 100)
        time_window = rule.get("time_window", 3600)  # seconds
        
        current_count = context.get("request_count", 0)
        return current_count > max_requests
    
    def _evaluate_ip_whitelist(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate IP whitelist rule."""
        allowed_ips = rule.get("allowed_ips", [])
        source_ip = context.get("source_ip")
        
        return source_ip not in allowed_ips if source_ip else False
    
    def _evaluate_resource_access(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate resource access rule."""
        required_level = AccessLevel(rule.get("required_level", 0))
        user_level = AccessLevel(context.get("user_access_level", 0))
        
        return user_level.value < required_level.value
    
    def _evaluate_time_window(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate time window rule."""
        allowed_hours = rule.get("allowed_hours", [])
        current_hour = datetime.now().hour
        
        return current_hour not in allowed_hours if allowed_hours else False


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.suspicious_patterns = {
            "sql_injection": [
                r"'.*(?:;|--|\|\|)",
                r"(?:union|select|insert|update|delete|drop).*(?:from|into|set|table)",
                r"(?:exec|execute|sp_|xp_)"
            ],
            "xss_attempt": [
                r"<script[^>]*>.*</script>",
                r"javascript:",
                r"on(?:load|error|click|mouseover)=",
                r"<(?:iframe|object|embed|applet)"
            ],
            "path_traversal": [
                r"\.\.\/",
                r"\.\.\\\\",
                r"(?:\/|\\\\)(?:etc|windows|system32|boot)",
                r"(?:file|ftp|http|https):\/\/"
            ],
            "command_injection": [
                r"(?:;|&|\|).*(?:rm|del|format|shutdown|reboot)",
                r"`.*`",
                r"\$\(.*\)",
                r"(?:bash|sh|cmd|powershell).*(?:-c|\/c)"
            ],
            "anomalous_behavior": [
                r"(?:password|passwd|secret|key|token).*=.*[^a-zA-Z0-9]",
                r"(?:admin|root|administrator).*(?:login|auth)",
                r"(?:drop|delete|truncate).*(?:table|database|schema)"
            ]
        }
        
        self.threat_scores = defaultdict(float)
        self.request_history: deque = deque(maxlen=10000)
        
    async def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request for security threats."""
        threat_analysis = {
            "threats_detected": [],
            "threat_level": ThreatLevel.LOW,
            "confidence": 0.0,
            "recommended_action": "allow"
        }
        
        # Analyze request content
        content = str(request_data.get("content", ""))
        headers = request_data.get("headers", {})
        params = request_data.get("params", {})
        
        # Combined analysis text
        analysis_text = f"{content} {json.dumps(headers)} {json.dumps(params)}".lower()
        
        total_score = 0.0
        detected_threats = []
        
        for threat_type, patterns in self.suspicious_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, analysis_text, re.IGNORECASE):
                    matches.append(pattern)
                    score += 1.0
            
            if matches:
                detected_threats.append({
                    "type": threat_type,
                    "score": score,
                    "matches": matches
                })
                total_score += score
        
        # Calculate threat level and confidence
        if total_score >= 5.0:
            threat_analysis["threat_level"] = ThreatLevel.CRITICAL
            threat_analysis["recommended_action"] = "block"
        elif total_score >= 3.0:
            threat_analysis["threat_level"] = ThreatLevel.HIGH
            threat_analysis["recommended_action"] = "quarantine"
        elif total_score >= 1.5:
            threat_analysis["threat_level"] = ThreatLevel.MEDIUM
            threat_analysis["recommended_action"] = "monitor"
        
        threat_analysis["threats_detected"] = detected_threats
        threat_analysis["confidence"] = min(1.0, total_score / 10.0)
        
        # Update request history
        self.request_history.append({
            "timestamp": datetime.now(),
            "source_ip": request_data.get("source_ip"),
            "threat_level": threat_analysis["threat_level"],
            "score": total_score
        })
        
        return threat_analysis
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        if not self.request_history:
            return {"message": "No request history available"}
        
        threat_counts = defaultdict(int)
        source_ip_counts = defaultdict(int)
        
        for request in self.request_history:
            threat_counts[request["threat_level"].value] += 1
            if request["source_ip"]:
                source_ip_counts[request["source_ip"]] += 1
        
        return {
            "total_requests_analyzed": len(self.request_history),
            "threat_level_distribution": dict(threat_counts),
            "top_source_ips": dict(list(source_ip_counts.most_common(10))),
            "detection_patterns": len(sum(self.suspicious_patterns.values(), [])),
            "average_threat_score": sum(r["score"] for r in self.request_history) / len(self.request_history)
        }


class AccessController:
    """Role-based access control system."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.access_logs: deque = deque(maxlen=5000)
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        self.roles = {
            "viewer": {
                "access_level": AccessLevel.READ,
                "permissions": ["read_healing_status", "read_metrics"],
                "description": "Read-only access to system status"
            },
            "operator": {
                "access_level": AccessLevel.WRITE,
                "permissions": ["read_healing_status", "read_metrics", "trigger_healing", "modify_config"],
                "description": "Operational access for healing management"
            },
            "admin": {
                "access_level": AccessLevel.ADMIN,
                "permissions": ["*"],
                "description": "Administrative access to all features"
            },
            "super_admin": {
                "access_level": AccessLevel.SUPER_ADMIN,
                "permissions": ["*", "manage_users", "manage_security"],
                "description": "Full system access including user management"
            }
        }
    
    def create_user(self, user_id: str, password: str, role: str, email: str = None) -> bool:
        """Create a new user account."""
        if user_id in self.users:
            return False
        
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        self.users[user_id] = {
            "user_id": user_id,
            "password_hash": password_hash,
            "role": role,
            "email": email,
            "created_at": datetime.now(),
            "last_login": None,
            "active": True,
            "failed_login_attempts": 0,
            "locked_until": None
        }
        
        logger.info(f"Created user account: {user_id} with role: {role}")
        return True
    
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        user = self.users.get(user_id)
        if not user or not user["active"]:
            self._log_access_attempt(user_id, None, "failed", "user_not_found_or_inactive")
            return None
        
        # Check if user is locked
        if user["locked_until"] and datetime.now() < user["locked_until"]:
            self._log_access_attempt(user_id, None, "failed", "account_locked")
            return None
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), user["password_hash"]):
            # Successful login
            user["last_login"] = datetime.now()
            user["failed_login_attempts"] = 0
            user["locked_until"] = None
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            self.sessions[session_token] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "role": user["role"],
                "access_level": self.roles[user["role"]]["access_level"]
            }
            
            self._log_access_attempt(user_id, session_token, "success", "login_successful")
            return session_token
        else:
            # Failed login
            user["failed_login_attempts"] += 1
            if user["failed_login_attempts"] >= 5:
                # Lock account for 30 minutes
                user["locked_until"] = datetime.now() + timedelta(minutes=30)
                self._log_access_attempt(user_id, None, "failed", "account_locked_due_to_failures")
            else:
                self._log_access_attempt(user_id, None, "failed", "invalid_password")
            
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return session info."""
        session = self.sessions.get(session_token)
        if not session:
            return None
        
        # Check session expiry (24 hours)
        if datetime.now() - session["created_at"] > timedelta(hours=24):
            del self.sessions[session_token]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now()
        return session
    
    def check_permission(self, session_token: str, permission: str, resource: str = None) -> bool:
        """Check if user has required permission."""
        session = self.validate_session(session_token)
        if not session:
            return False
        
        user_role = session["role"]
        role_permissions = self.roles[user_role]["permissions"]
        
        # Super permissions
        if "*" in role_permissions:
            return True
        
        # Exact permission match
        if permission in role_permissions:
            return True
        
        # Resource-specific permissions
        if resource and f"{permission}:{resource}" in role_permissions:
            return True
        
        self._log_access_attempt(session["user_id"], session_token, "denied", f"insufficient_permission_{permission}")
        return False
    
    def _log_access_attempt(self, user_id: str, session_token: Optional[str], result: str, details: str):
        """Log access attempt."""
        self.access_logs.append({
            "timestamp": datetime.now(),
            "user_id": user_id,
            "session_token": session_token,
            "result": result,
            "details": details
        })
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics."""
        if not self.access_logs:
            return {"message": "No access logs available"}
        
        total_attempts = len(self.access_logs)
        successful_logins = sum(1 for log in self.access_logs if log["result"] == "success")
        failed_attempts = total_attempts - successful_logins
        
        user_activity = defaultdict(int)
        for log in self.access_logs:
            user_activity[log["user_id"]] += 1
        
        return {
            "total_users": len(self.users),
            "active_sessions": len(self.sessions),
            "total_access_attempts": total_attempts,
            "successful_logins": successful_logins,
            "failed_attempts": failed_attempts,
            "success_rate": successful_logins / total_attempts if total_attempts > 0 else 0,
            "most_active_users": dict(list(user_activity.most_common(10))),
            "available_roles": list(self.roles.keys())
        }


class SecurityManager:
    """Main security management orchestrator."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.access_controller = AccessController()
        self.policies: Dict[str, SecurityPolicy] = {}
        self.security_events: deque = deque(maxlen=10000)
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize default policies
        self._initialize_default_policies()
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        self.policies = {
            "api_rate_limiting": SecurityPolicy(
                name="api_rate_limiting",
                description="Limit API requests per IP",
                rules=[{
                    "type": "rate_limit",
                    "max_requests": 100,
                    "time_window": 3600
                }],
                enforcement_level="block"
            ),
            "admin_access_time": SecurityPolicy(
                name="admin_access_time",
                description="Restrict admin access to business hours",
                rules=[{
                    "type": "time_window",
                    "allowed_hours": list(range(8, 18))  # 8 AM to 6 PM
                }],
                enforcement_level="warn"
            ),
            "critical_resource_access": SecurityPolicy(
                name="critical_resource_access",
                description="Require admin access for critical resources",
                rules=[{
                    "type": "resource_access",
                    "required_level": AccessLevel.ADMIN.value
                }],
                enforcement_level="block"
            )
        }
    
    async def process_security_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through security pipeline."""
        source_ip = request_data.get("source_ip", "unknown")
        user_id = request_data.get("user_id")
        resource = request_data.get("resource")
        action = request_data.get("action", "access")
        
        security_result = {
            "allowed": True,
            "threat_analysis": {},
            "policy_violations": [],
            "rate_limit_status": {},
            "access_granted": True,
            "security_events": []
        }
        
        # 1. Threat Detection
        threat_analysis = await self.threat_detector.analyze_request(request_data)
        security_result["threat_analysis"] = threat_analysis
        
        if threat_analysis["recommended_action"] == "block":
            security_result["allowed"] = False
            await self._create_security_event(
                "threat_blocked",
                ThreatLevel.HIGH,
                source_ip,
                user_id,
                resource,
                {"threat_analysis": threat_analysis}
            )
        
        # 2. Rate Limiting
        rate_limit_result = self._check_rate_limits(source_ip, user_id)
        security_result["rate_limit_status"] = rate_limit_result
        
        if not rate_limit_result["allowed"]:
            security_result["allowed"] = False
            await self._create_security_event(
                "rate_limit_exceeded",
                ThreatLevel.MEDIUM,
                source_ip,
                user_id,
                resource,
                {"rate_limit": rate_limit_result}
            )
        
        # 3. Access Control
        session_token = request_data.get("session_token")
        if session_token:
            permission = request_data.get("permission", "read")
            access_granted = self.access_controller.check_permission(session_token, permission, resource)
            security_result["access_granted"] = access_granted
            
            if not access_granted:
                security_result["allowed"] = False
                await self._create_security_event(
                    "access_denied",
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    resource,
                    {"permission": permission}
                )
        
        # 4. Policy Evaluation
        policy_context = {
            "source_ip": source_ip,
            "user_id": user_id,
            "resource": resource,
            "request_count": len([r for r in self.rate_limits[source_ip] 
                                if r > time.time() - 3600]),
            "user_access_level": request_data.get("user_access_level", 0)
        }
        
        policy_violations = []
        for policy_name, policy in self.policies.items():
            if policy.enabled:
                result = policy.evaluate(policy_context)
                if not result["passed"]:
                    policy_violations.append(result)
                    
                    if result["enforcement_level"] == "block":
                        security_result["allowed"] = False
                        await self._create_security_event(
                            "policy_violation",
                            ThreatLevel.HIGH,
                            source_ip,
                            user_id,
                            resource,
                            {"policy": policy_name, "violations": result["violations"]}
                        )
        
        security_result["policy_violations"] = policy_violations
        
        return security_result
    
    def _check_rate_limits(self, source_ip: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Check rate limits for source IP and user."""
        current_time = time.time()
        
        # Clean old entries
        ip_requests = self.rate_limits[source_ip]
        while ip_requests and ip_requests[0] < current_time - 3600:  # 1 hour window
            ip_requests.popleft()
        
        # Check IP rate limit
        ip_count = len(ip_requests)
        ip_limit = 100  # 100 requests per hour per IP
        
        if ip_count >= ip_limit:
            return {
                "allowed": False,
                "reason": "ip_rate_limit_exceeded",
                "ip_count": ip_count,
                "ip_limit": ip_limit,
                "reset_time": current_time + 3600
            }
        
        # Record this request
        ip_requests.append(current_time)
        
        return {
            "allowed": True,
            "ip_count": ip_count + 1,
            "ip_limit": ip_limit,
            "remaining": ip_limit - ip_count - 1
        }
    
    async def _create_security_event(
        self, 
        event_type: str, 
        threat_level: ThreatLevel, 
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None, 
        resource: Optional[str] = None, 
        details: Dict[str, Any] = None
    ):
        """Create and log security event."""
        event = SecurityEvent(
            id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            details=details or {}
        )
        
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} ({threat_level.value}) from {source_ip or 'unknown'}")
        
        # Auto-mitigation for critical events
        if threat_level == ThreatLevel.CRITICAL:
            await self._auto_mitigate_threat(event)
    
    async def _auto_mitigate_threat(self, event: SecurityEvent):
        """Automatically mitigate critical threats."""
        if event.event_type == "threat_blocked" and event.source_ip:
            # Temporarily block IP for critical threats
            await self._temporary_ip_block(event.source_ip, duration_minutes=60)
            event.mitigated = True
            logger.info(f"Auto-mitigated threat: blocked IP {event.source_ip} for 60 minutes")
    
    async def _temporary_ip_block(self, ip_address: str, duration_minutes: int = 60):
        """Temporarily block an IP address."""
        # In a real implementation, this would integrate with firewall/load balancer
        block_until = time.time() + (duration_minutes * 60)
        
        # Store in memory for now - in production would use Redis or similar
        if not hasattr(self, '_blocked_ips'):
            self._blocked_ips = {}
        
        self._blocked_ips[ip_address] = block_until
        logger.info(f"Temporarily blocked IP {ip_address} until {datetime.fromtimestamp(block_until)}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        
        threat_level_counts = defaultdict(int)
        event_type_counts = defaultdict(int)
        
        for event in recent_events:
            threat_level_counts[event.threat_level.value] += 1
            event_type_counts[event.event_type] += 1
        
        return {
            "security_overview": {
                "total_security_events": len(self.security_events),
                "recent_events_24h": len([e for e in recent_events if 
                                         (datetime.now() - e.timestamp).days < 1]),
                "threat_level_distribution": dict(threat_level_counts),
                "event_type_distribution": dict(event_type_counts)
            },
            "threat_detection": self.threat_detector.get_threat_statistics(),
            "access_control": self.access_controller.get_access_statistics(),
            "active_policies": len(self.policies),
            "mitigation_status": {
                "auto_mitigated_events": len([e for e in recent_events if e.mitigated]),
                "blocked_ips": len(getattr(self, '_blocked_ips', {}))
            }
        }


# Global security manager instance
security_manager = SecurityManager()