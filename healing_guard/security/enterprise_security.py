"""Enterprise-grade security features for healing guard system.

Implements advanced security controls, threat detection, and compliance
features for production enterprise environments.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

from ..core.healing_engine import HealingEngine
from ..monitoring.enhanced_monitoring import enhanced_monitoring, Alert, AlertSeverity

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of security-audited actions."""
    LOGIN = "login"
    LOGOUT = "logout"
    HEALING_EXECUTE = "healing_execute"
    CONFIG_CHANGE = "config_change"
    DATA_ACCESS = "data_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SYSTEM_COMMAND = "system_command"
    API_ACCESS = "api_access"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: Set[str]
    security_level: SecurityLevel
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authentication_method: str = "password"
    mfa_verified: bool = False
    expires_at: Optional[datetime] = None


@dataclass
class AuditEvent:
    """Security audit event."""
    id: str
    timestamp: datetime
    user_id: str
    session_id: Optional[str]
    action_type: ActionType
    resource: str
    success: bool
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action_type": self.action_type.value,
            "resource": self.resource,
            "success": self.success,
            "ip_address": self.ip_address,
            "details": self.details,
            "risk_score": self.risk_score
        }


@dataclass
class ThreatEvent:
    """Security threat event."""
    id: str
    timestamp: datetime
    threat_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    indicators: List[str] = field(default_factory=list)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


class EncryptionManager:
    """Manages encryption/decryption operations."""
    
    def __init__(self, key_file: str = "security/master.key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(exist_ok=True)
        
        # Initialize encryption key
        self.fernet_key = self._load_or_create_key()
        self.fernet = Fernet(self.fernet_key)
        
        # Initialize RSA keys for asymmetric encryption
        self.private_key, self.public_key = self._load_or_create_rsa_keys()
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one."""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            logger.info("Generated new encryption key")
            return key
    
    def _load_or_create_rsa_keys(self) -> Tuple[Any, Any]:
        """Load or create RSA key pair."""
        private_key_file = self.key_file.parent / "private_key.pem"
        public_key_file = self.key_file.parent / "public_key.pem"
        
        if private_key_file.exists() and public_key_file.exists():
            # Load existing keys
            with open(private_key_file, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            with open(public_key_file, "rb") as f:
                public_key = serialization.load_pem_public_key(f.read())
            
            return private_key, public_key
        else:
            # Generate new keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Save keys
            with open(private_key_file, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(public_key_file, "wb") as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            logger.info("Generated new RSA key pair")
            return private_key, public_key
    
    def encrypt_symmetric(self, data: str) -> bytes:
        """Encrypt data using symmetric encryption."""
        return self.fernet.encrypt(data.encode('utf-8'))
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> str:
        """Decrypt data using symmetric encryption."""
        return self.fernet.decrypt(encrypted_data).decode('utf-8')
    
    def encrypt_asymmetric(self, data: str) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return self.public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> str:
        """Decrypt data using asymmetric encryption."""
        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        ).decode('utf-8')
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class AuthenticationManager:
    """Handles user authentication and session management."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.jwt_secret = secrets.token_urlsafe(32)
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=8)
        self.jwt_expiration = timedelta(hours=24)
        
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate user and create security context."""
        # Check for account lockout
        if self._is_account_locked(username):
            logger.warning(f"Login attempt for locked account: {username}")
            return None
        
        # Simulate user lookup (in production, would query database)
        user_data = self._get_user_data(username)
        if not user_data:
            self._record_failed_attempt(username)
            return None
        
        # Verify password
        if not self.encryption_manager.verify_password(password, user_data["password_hash"]):
            self._record_failed_attempt(username)
            return None
        
        # Clear failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        # Create security context
        session_id = str(uuid.uuid4())
        context = SecurityContext(
            user_id=user_data["user_id"],
            session_id=session_id,
            permissions=set(user_data.get("permissions", [])),
            security_level=SecurityLevel(user_data.get("security_level", SecurityLevel.INTERNAL.value)),
            ip_address=ip_address,
            user_agent=user_agent,
            authentication_method="password",
            mfa_verified=False,  # Would require additional MFA step
            expires_at=datetime.now() + self.session_timeout
        )
        
        self.active_sessions[session_id] = context
        
        logger.info(f"User {username} authenticated successfully")
        return context
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[username]
            if datetime.now() - attempt < self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now())
        
        # Clean up old attempts
        cutoff = datetime.now() - self.lockout_duration
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]
    
    def _get_user_data(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data (mock implementation)."""
        # Mock user database - in production would query actual user store
        mock_users = {
            "admin": {
                "user_id": "admin",
                "username": "admin",
                "password_hash": self.encryption_manager.hash_password("admin123"),
                "permissions": ["healing:execute", "config:write", "audit:read", "system:admin"],
                "security_level": SecurityLevel.RESTRICTED.value
            },
            "operator": {
                "user_id": "operator",
                "username": "operator", 
                "password_hash": self.encryption_manager.hash_password("operator123"),
                "permissions": ["healing:execute", "audit:read"],
                "security_level": SecurityLevel.CONFIDENTIAL.value
            },
            "viewer": {
                "user_id": "viewer",
                "username": "viewer",
                "password_hash": self.encryption_manager.hash_password("viewer123"),
                "permissions": ["audit:read"],
                "security_level": SecurityLevel.INTERNAL.value
            }
        }
        
        return mock_users.get(username)
    
    def generate_jwt_token(self, context: SecurityContext) -> str:
        """Generate JWT token for API authentication."""
        payload = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "permissions": list(context.permissions),
            "security_level": context.security_level.value,
            "iat": int(time.time()),
            "exp": int(time.time()) + int(self.jwt_expiration.total_seconds())
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if session still exists
            session_id = payload.get("session_id")
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                
                # Check if session has expired
                if context.expires_at and datetime.now() > context.expires_at:
                    del self.active_sessions[session_id]
                    return None
                
                return context
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, context in self.active_sessions.items()
            if context.expires_at and current_time > context.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class AuthorizationManager:
    """Handles authorization and permission checking."""
    
    def __init__(self):
        # Permission hierarchy
        self.permission_hierarchy = {
            "system:admin": ["healing:execute", "config:write", "audit:read", "audit:write"],
            "healing:execute": ["healing:read"],
            "config:write": ["config:read"],
            "audit:write": ["audit:read"]
        }
        
        # Resource-based permissions
        self.resource_permissions = {
            "healing_operations": ["healing:execute", "healing:read"],
            "system_config": ["config:write", "config:read"],
            "audit_logs": ["audit:read", "audit:write"],
            "user_management": ["system:admin"]
        }
    
    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if user has specific permission."""
        # Direct permission check
        if permission in context.permissions:
            return True
        
        # Check hierarchical permissions
        for user_perm in context.permissions:
            if user_perm in self.permission_hierarchy:
                if permission in self.permission_hierarchy[user_perm]:
                    return True
        
        return False
    
    def check_resource_access(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Check if user can access specific resource."""
        required_permissions = self.resource_permissions.get(resource, [])
        
        for perm in required_permissions:
            if self.check_permission(context, perm):
                return True
        
        return False
    
    def get_user_permissions(self, context: SecurityContext) -> Set[str]:
        """Get all effective permissions for user."""
        effective_permissions = set(context.permissions)
        
        # Add hierarchical permissions
        for user_perm in context.permissions:
            if user_perm in self.permission_hierarchy:
                effective_permissions.update(self.permission_hierarchy[user_perm])
        
        return effective_permissions


class SecurityAuditor:
    """Handles security auditing and compliance logging."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.audit_events: List[AuditEvent] = []
        self.threat_events: List[ThreatEvent] = []
        
        # Compliance settings
        self.audit_retention_days = 2555  # 7 years for compliance
        self.encrypt_audit_logs = True
        self.tamper_protection = True
        
        # Risk scoring weights
        self.risk_weights = {
            "failed_login": 2.0,
            "privilege_escalation": 8.0,
            "config_change": 4.0,
            "data_access": 3.0,
            "off_hours_access": 1.5,
            "unusual_ip": 3.0,
            "multiple_failures": 2.5
        }
    
    def log_audit_event(
        self,
        user_id: str,
        action_type: ActionType,
        resource: str,
        success: bool,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log a security audit event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            resource=resource,
            success=success,
            ip_address=ip_address,
            details=details or {},
            risk_score=self._calculate_risk_score(action_type, success, details or {})
        )
        
        self.audit_events.append(event)
        
        # Trigger alerts for high-risk events
        if event.risk_score > 7.0:
            asyncio.create_task(self._generate_security_alert(event))
        
        # Clean up old events
        self._cleanup_old_events()
        
        logger.info(f"Audit event logged: {action_type.value} by {user_id}")
        return event
    
    def _calculate_risk_score(
        self,
        action_type: ActionType,
        success: bool,
        details: Dict[str, Any]
    ) -> float:
        """Calculate risk score for audit event."""
        base_score = 1.0
        
        # Action type risk
        action_risks = {
            ActionType.LOGIN: 1.0,
            ActionType.HEALING_EXECUTE: 3.0,
            ActionType.CONFIG_CHANGE: 5.0,
            ActionType.PRIVILEGE_ESCALATION: 9.0,
            ActionType.SYSTEM_COMMAND: 7.0,
            ActionType.DATA_ACCESS: 2.0
        }
        base_score *= action_risks.get(action_type, 1.0)
        
        # Failure increases risk
        if not success:
            base_score *= self.risk_weights["failed_login"]
        
        # Time-based risk (off hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            base_score *= self.risk_weights["off_hours_access"]
        
        # Context-based risks
        if details.get("unusual_ip"):
            base_score *= self.risk_weights["unusual_ip"]
        
        if details.get("multiple_failures"):
            base_score *= self.risk_weights["multiple_failures"]
        
        return min(10.0, base_score)  # Cap at 10
    
    async def _generate_security_alert(self, event: AuditEvent):
        """Generate security alert for high-risk events."""
        alert = Alert(
            id=f"security_{event.id}",
            message=f"High-risk security event: {event.action_type.value} by {event.user_id}",
            severity=AlertSeverity.HIGH if event.risk_score < 9 else AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            component="security",
            metadata={
                "audit_event_id": event.id,
                "risk_score": event.risk_score,
                "user_id": event.user_id,
                "action": event.action_type.value
            }
        )
        
        # Send to monitoring system
        enhanced_monitoring.active_alerts[alert.id] = alert
        
        logger.warning(f"Security alert generated: {alert.message}")
    
    def log_threat_event(
        self,
        threat_type: str,
        threat_level: ThreatLevel,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        indicators: Optional[List[str]] = None
    ) -> ThreatEvent:
        """Log a security threat event."""
        event = ThreatEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            threat_type=threat_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            indicators=indicators or []
        )
        
        self.threat_events.append(event)
        
        # Generate alert for threats
        asyncio.create_task(self._generate_threat_alert(event))
        
        logger.warning(f"Threat event logged: {threat_type} - {description}")
        return event
    
    async def _generate_threat_alert(self, event: ThreatEvent):
        """Generate alert for threat events."""
        severity_mapping = {
            ThreatLevel.LOW: AlertSeverity.MEDIUM,
            ThreatLevel.MEDIUM: AlertSeverity.HIGH,
            ThreatLevel.HIGH: AlertSeverity.HIGH,
            ThreatLevel.CRITICAL: AlertSeverity.CRITICAL
        }
        
        alert = Alert(
            id=f"threat_{event.id}",
            message=f"Security threat detected: {event.threat_type} - {event.description}",
            severity=severity_mapping[event.threat_level],
            timestamp=datetime.now(),
            component="security_threats",
            metadata={
                "threat_event_id": event.id,
                "threat_type": event.threat_type,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip
            }
        )
        
        enhanced_monitoring.active_alerts[alert.id] = alert
    
    def _cleanup_old_events(self):
        """Clean up old audit events based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.audit_retention_days)
        
        # Keep only recent events in memory (full events would be persisted to database)
        memory_cutoff = datetime.now() - timedelta(days=7)
        self.audit_events = [
            event for event in self.audit_events
            if event.timestamp > memory_cutoff
        ]
        
        self.threat_events = [
            event for event in self.threat_events
            if event.timestamp > memory_cutoff
        ]
    
    def get_audit_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action_type: Optional[ActionType] = None
    ) -> Dict[str, Any]:
        """Generate audit report with filtering."""
        # Filter events
        filtered_events = self.audit_events.copy()
        
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        if action_type:
            filtered_events = [e for e in filtered_events if e.action_type == action_type]
        
        # Generate statistics
        total_events = len(filtered_events)
        successful_events = len([e for e in filtered_events if e.success])
        failed_events = total_events - successful_events
        
        # Risk analysis
        high_risk_events = [e for e in filtered_events if e.risk_score > 7.0]
        avg_risk_score = sum(e.risk_score for e in filtered_events) / total_events if total_events > 0 else 0
        
        # Action type breakdown
        action_counts = {}
        for event in filtered_events:
            action_type_str = event.action_type.value
            action_counts[action_type_str] = action_counts.get(action_type_str, 0) + 1
        
        # User activity
        user_counts = {}
        for event in filtered_events:
            user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1
        
        return {
            "report_generated": datetime.now().isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "summary": {
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "success_rate": successful_events / total_events if total_events > 0 else 0,
                "high_risk_events": len(high_risk_events),
                "average_risk_score": avg_risk_score
            },
            "breakdown": {
                "by_action_type": action_counts,
                "by_user": user_counts
            },
            "high_risk_events": [e.to_dict() for e in high_risk_events[:10]],  # Top 10
            "threat_events": len(self.threat_events)
        }


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self, auditor: SecurityAuditor):
        self.auditor = auditor
        self.detection_rules: List[Dict[str, Any]] = []
        self.behavioral_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Initialize detection rules
        self._initialize_detection_rules()
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    def _initialize_detection_rules(self):
        """Initialize threat detection rules."""
        self.detection_rules = [
            {
                "id": "brute_force_login",
                "name": "Brute Force Login Attempt",
                "description": "Multiple failed login attempts from same IP",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "severity": ThreatLevel.HIGH
            },
            {
                "id": "privilege_escalation_attempt",
                "name": "Privilege Escalation Attempt",
                "description": "User attempting actions beyond their permissions",
                "threshold": 3,
                "time_window": 600,  # 10 minutes
                "severity": ThreatLevel.CRITICAL
            },
            {
                "id": "unusual_access_time",
                "name": "Unusual Access Time",
                "description": "Access during unusual hours for this user",
                "threshold": 1,
                "time_window": 0,
                "severity": ThreatLevel.MEDIUM
            },
            {
                "id": "rapid_config_changes",
                "name": "Rapid Configuration Changes",
                "description": "Multiple configuration changes in short time",
                "threshold": 10,
                "time_window": 900,  # 15 minutes
                "severity": ThreatLevel.HIGH
            },
            {
                "id": "anomalous_healing_activity",
                "name": "Anomalous Healing Activity",
                "description": "Unusual pattern of healing executions",
                "threshold": 20,
                "time_window": 1800,  # 30 minutes
                "severity": ThreatLevel.MEDIUM
            }
        ]
    
    async def start_monitoring(self):
        """Start threat monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Threat detection monitoring started")
    
    async def stop_monitoring(self):
        """Stop threat monitoring."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Threat detection monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main threat monitoring loop."""
        while self.running:
            try:
                await self._check_threat_rules()
                await self._update_behavioral_baselines()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in threat monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_threat_rules(self):
        """Check all threat detection rules."""
        current_time = datetime.now()
        
        for rule in self.detection_rules:
            try:
                await self._check_individual_rule(rule, current_time)
            except Exception as e:
                logger.error(f"Error checking rule {rule['id']}: {e}")
    
    async def _check_individual_rule(self, rule: Dict[str, Any], current_time: datetime):
        """Check individual threat detection rule."""
        rule_id = rule["id"]
        time_window = rule["time_window"]
        threshold = rule["threshold"]
        
        if time_window > 0:
            cutoff_time = current_time - timedelta(seconds=time_window)
            relevant_events = [
                event for event in self.auditor.audit_events
                if event.timestamp >= cutoff_time
            ]
        else:
            relevant_events = self.auditor.audit_events
        
        # Apply rule-specific logic
        if rule_id == "brute_force_login":
            await self._check_brute_force_login(rule, relevant_events)
        elif rule_id == "privilege_escalation_attempt":
            await self._check_privilege_escalation(rule, relevant_events)
        elif rule_id == "unusual_access_time":
            await self._check_unusual_access_time(rule, relevant_events)
        elif rule_id == "rapid_config_changes":
            await self._check_rapid_config_changes(rule, relevant_events)
        elif rule_id == "anomalous_healing_activity":
            await self._check_anomalous_healing_activity(rule, relevant_events)
    
    async def _check_brute_force_login(self, rule: Dict[str, Any], events: List[AuditEvent]):
        """Check for brute force login attempts."""
        # Group failed login attempts by IP
        failed_logins_by_ip = {}
        
        for event in events:
            if (event.action_type == ActionType.LOGIN and 
                not event.success and 
                event.ip_address):
                
                ip = event.ip_address
                if ip not in failed_logins_by_ip:
                    failed_logins_by_ip[ip] = []
                failed_logins_by_ip[ip].append(event)
        
        # Check threshold
        for ip, login_events in failed_logins_by_ip.items():
            if len(login_events) >= rule["threshold"]:
                await self._generate_threat_event(
                    rule["id"],
                    rule["severity"],
                    f"Brute force login detected from IP {ip}",
                    source_ip=ip,
                    indicators=[f"{len(login_events)} failed attempts in {rule['time_window']}s"]
                )
    
    async def _check_privilege_escalation(self, rule: Dict[str, Any], events: List[AuditEvent]):
        """Check for privilege escalation attempts."""
        # Look for failed attempts to access restricted resources
        escalation_attempts = [
            event for event in events
            if not event.success and event.action_type in [
                ActionType.HEALING_EXECUTE,
                ActionType.CONFIG_CHANGE,
                ActionType.SYSTEM_COMMAND
            ]
        ]
        
        # Group by user
        attempts_by_user = {}
        for event in escalation_attempts:
            user = event.user_id
            if user not in attempts_by_user:
                attempts_by_user[user] = []
            attempts_by_user[user].append(event)
        
        # Check threshold
        for user, user_events in attempts_by_user.items():
            if len(user_events) >= rule["threshold"]:
                await self._generate_threat_event(
                    rule["id"],
                    rule["severity"],
                    f"Privilege escalation attempts by user {user}",
                    user_id=user,
                    indicators=[f"{len(user_events)} failed privilege escalation attempts"]
                )
    
    async def _check_unusual_access_time(self, rule: Dict[str, Any], events: List[AuditEvent]):
        """Check for access during unusual hours."""
        # Define business hours (8 AM to 6 PM)
        business_start = 8
        business_end = 18
        
        for event in events:
            hour = event.timestamp.hour
            if hour < business_start or hour >= business_end:
                # Check if this is unusual for this user
                user_baseline = self.behavioral_baselines.get(event.user_id, {})
                typical_hours = user_baseline.get("typical_access_hours", set())
                
                if typical_hours and hour not in typical_hours:
                    await self._generate_threat_event(
                        rule["id"],
                        rule["severity"],
                        f"Unusual access time for user {event.user_id}",
                        user_id=event.user_id,
                        indicators=[f"Access at {hour}:00, typical hours: {sorted(typical_hours)}"]
                    )
    
    async def _check_rapid_config_changes(self, rule: Dict[str, Any], events: List[AuditEvent]):
        """Check for rapid configuration changes."""
        config_changes = [
            event for event in events
            if event.action_type == ActionType.CONFIG_CHANGE
        ]
        
        if len(config_changes) >= rule["threshold"]:
            await self._generate_threat_event(
                rule["id"],
                rule["severity"],
                f"Rapid configuration changes detected",
                indicators=[f"{len(config_changes)} config changes in {rule['time_window']}s"]
            )
    
    async def _check_anomalous_healing_activity(self, rule: Dict[str, Any], events: List[AuditEvent]):
        """Check for anomalous healing activity."""
        healing_events = [
            event for event in events
            if event.action_type == ActionType.HEALING_EXECUTE
        ]
        
        if len(healing_events) >= rule["threshold"]:
            # Additional analysis for anomalous patterns
            users_involved = set(event.user_id for event in healing_events)
            resources_affected = set(event.resource for event in healing_events)
            
            await self._generate_threat_event(
                rule["id"],
                rule["severity"],
                f"Anomalous healing activity detected",
                indicators=[
                    f"{len(healing_events)} healing operations",
                    f"{len(users_involved)} users involved",
                    f"{len(resources_affected)} resources affected"
                ]
            )
    
    async def _generate_threat_event(
        self,
        threat_type: str,
        threat_level: ThreatLevel,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        indicators: Optional[List[str]] = None
    ):
        """Generate and log a threat event."""
        self.auditor.log_threat_event(
            threat_type=threat_type,
            threat_level=threat_level,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            indicators=indicators
        )
    
    async def _update_behavioral_baselines(self):
        """Update behavioral baselines for users."""
        # Analyze user behavior patterns
        user_patterns = {}
        
        # Look at last 30 days of activity
        cutoff_time = datetime.now() - timedelta(days=30)
        recent_events = [
            event for event in self.auditor.audit_events
            if event.timestamp >= cutoff_time
        ]
        
        # Group by user
        for event in recent_events:
            user_id = event.user_id
            if user_id not in user_patterns:
                user_patterns[user_id] = {
                    "access_hours": [],
                    "common_actions": [],
                    "common_resources": [],
                    "common_ips": []
                }
            
            user_patterns[user_id]["access_hours"].append(event.timestamp.hour)
            user_patterns[user_id]["common_actions"].append(event.action_type.value)
            user_patterns[user_id]["common_resources"].append(event.resource)
            if event.ip_address:
                user_patterns[user_id]["common_ips"].append(event.ip_address)
        
        # Build baselines
        for user_id, patterns in user_patterns.items():
            baseline = {
                "typical_access_hours": set(patterns["access_hours"]),
                "common_actions": set(patterns["common_actions"]),
                "common_resources": set(patterns["common_resources"]),
                "common_ips": set(patterns["common_ips"]),
                "last_updated": datetime.now()
            }
            
            self.behavioral_baselines[user_id] = baseline


class EnterpriseSecurityManager:
    """Main enterprise security manager."""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager(self.encryption_manager)
        self.authz_manager = AuthorizationManager()
        self.auditor = SecurityAuditor(self.encryption_manager)
        self.threat_detector = ThreatDetector(self.auditor)
        
        # Security policies
        self.security_policies = {
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True,
                "max_age_days": 90
            },
            "session_policy": {
                "timeout_minutes": 480,  # 8 hours
                "max_concurrent_sessions": 3,
                "require_mfa_for_admin": True
            },
            "audit_policy": {
                "log_all_actions": True,
                "retention_days": 2555,  # 7 years
                "encrypt_logs": True,
                "tamper_protection": True
            }
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start enterprise security services."""
        if self.running:
            return
        
        self.running = True
        
        # Start threat detection
        await self.threat_detector.start_monitoring()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Enterprise security manager started")
    
    async def stop(self):
        """Stop enterprise security services."""
        self.running = False
        
        # Stop threat detection
        await self.threat_detector.stop_monitoring()
        
        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("Enterprise security manager stopped")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                # Clean up expired sessions
                self.auth_manager.cleanup_expired_sessions()
                
                # Clean up old audit events
                self.auditor._cleanup_old_events()
                
                # Wait for next cleanup
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security cleanup loop: {e}")
                await asyncio.sleep(300)  # 5 minutes
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate user and create security context."""
        context = self.auth_manager.authenticate_user(username, password, ip_address, user_agent)
        
        # Log authentication attempt
        self.auditor.log_audit_event(
            user_id=username,
            action_type=ActionType.LOGIN,
            resource="authentication",
            success=context is not None,
            session_id=context.session_id if context else None,
            ip_address=ip_address,
            details={"user_agent": user_agent}
        )
        
        return context
    
    def authorize_action(
        self,
        context: SecurityContext,
        action: str,
        resource: str,
        log_attempt: bool = True
    ) -> bool:
        """Authorize user action."""
        authorized = self.authz_manager.check_permission(context, action)
        
        if log_attempt:
            self.auditor.log_audit_event(
                user_id=context.user_id,
                action_type=self._map_action_to_audit_type(action),
                resource=resource,
                success=authorized,
                session_id=context.session_id,
                ip_address=context.ip_address
            )
        
        return authorized
    
    def _map_action_to_audit_type(self, action: str) -> ActionType:
        """Map action string to audit action type."""
        mapping = {
            "healing:execute": ActionType.HEALING_EXECUTE,
            "config:write": ActionType.CONFIG_CHANGE,
            "config:read": ActionType.DATA_ACCESS,
            "audit:read": ActionType.DATA_ACCESS,
            "audit:write": ActionType.DATA_ACCESS,
            "system:admin": ActionType.PRIVILEGE_ESCALATION
        }
        return mapping.get(action, ActionType.API_ACCESS)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            "active_sessions": len(self.auth_manager.active_sessions),
            "failed_login_attempts": sum(
                len(attempts) for attempts in self.auth_manager.failed_attempts.values()
            ),
            "audit_events_24h": len([
                e for e in self.auditor.audit_events
                if e.timestamp >= datetime.now() - timedelta(days=1)
            ]),
            "threat_events_24h": len([
                e for e in self.auditor.threat_events
                if e.timestamp >= datetime.now() - timedelta(days=1)
            ]),
            "high_risk_events": len([
                e for e in self.auditor.audit_events
                if e.risk_score > 7.0 and e.timestamp >= datetime.now() - timedelta(days=1)
            ]),
            "threat_monitoring_active": self.threat_detector.running,
            "security_policies": self.security_policies
        }


# Global security manager instance
security_manager = EnterpriseSecurityManager()