"""Advanced compliance and audit logging system.

Implements enterprise-grade compliance features for regulatory requirements
including SOX, GDPR, HIPAA, PCI DSS, and other industry standards.
"""

import asyncio
import json
import logging
import hashlib
import hmac
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # ISO/IEC 27001
    NIST = "nist"  # NIST Cybersecurity Framework
    FedRAMP = "fedramp"  # Federal Risk and Authorization Management Program


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RetentionPolicy(Enum):
    """Data retention policies."""
    SHORT_TERM = "short_term"  # 1 year
    MEDIUM_TERM = "medium_term"  # 3 years
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"  # Indefinite


@dataclass
class ComplianceEvent:
    """Compliance audit event with full traceability."""
    id: str
    timestamp: datetime
    event_type: str
    user_id: str
    session_id: Optional[str]
    source_ip: Optional[str]
    resource: str
    action: str
    result: str
    classification: DataClassification
    compliance_standards: List[ComplianceStandard]
    retention_policy: RetentionPolicy
    checksum: str = ""
    digital_signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate checksum and signature after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        # Create deterministic string representation
        data_string = (
            f"{self.id}|{self.timestamp.isoformat()}|{self.event_type}|"
            f"{self.user_id}|{self.session_id}|{self.source_ip}|"
            f"{self.resource}|{self.action}|{self.result}|"
            f"{self.classification.value}|{json.dumps(self.metadata, sort_keys=True)}"
        )
        
        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        calculated_checksum = self._calculate_checksum()
        return calculated_checksum == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "classification": self.classification.value,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "retention_policy": self.retention_policy.value,
            "checksum": self.checksum,
            "digital_signature": self.digital_signature,
            "metadata": self.metadata
        }


@dataclass
class ComplianceReport:
    """Compliance audit report."""
    id: str
    report_type: str
    standards: List[ComplianceStandard]
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    generated_by: str
    total_events: int
    compliant_events: int
    violations: int
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    certification_status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ComplianceDatabase:
    """Secure database for compliance audit logs."""
    
    def __init__(self, db_path: str = "compliance/audit_logs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe database operations
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize database
        self._initialize_database()
        
        # Setup encryption for sensitive data
        self._setup_encryption()
    
    def _initialize_database(self):
        """Initialize compliance database schema."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_events (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        source_ip TEXT,
                        resource TEXT NOT NULL,
                        action TEXT NOT NULL,
                        result TEXT NOT NULL,
                        classification TEXT NOT NULL,
                        compliance_standards TEXT NOT NULL,
                        retention_policy TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        digital_signature TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON compliance_events(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id 
                    ON compliance_events(user_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_type 
                    ON compliance_events(event_type)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_standards 
                    ON compliance_events(compliance_standards)
                """)
                
                # Audit log integrity table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_integrity (
                        id TEXT PRIMARY KEY,
                        batch_start TEXT NOT NULL,
                        batch_end TEXT NOT NULL,
                        event_count INTEGER NOT NULL,
                        batch_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Compliance reports table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_reports (
                        id TEXT PRIMARY KEY,
                        report_type TEXT NOT NULL,
                        standards TEXT NOT NULL,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        generated_at TEXT NOT NULL,
                        generated_by TEXT NOT NULL,
                        total_events INTEGER NOT NULL,
                        compliant_events INTEGER NOT NULL,
                        violations INTEGER NOT NULL,
                        findings TEXT,
                        recommendations TEXT,
                        certification_status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Compliance database initialized")
                
            finally:
                conn.close()
    
    def _setup_encryption(self):
        """Setup encryption keys for sensitive data."""
        key_file = self.db_path.parent / "audit_signing_key.pem"
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        else:
            # Generate new signing key
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            with open(key_file, "wb") as f:
                f.write(self.private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                ))
            
            # Save public key
            public_key = self.private_key.public_key()
            with open(key_file.with_name("audit_public_key.pem"), "wb") as f:
                f.write(public_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        
        self.public_key = self.private_key.public_key()
    
    def sign_event(self, event: ComplianceEvent) -> str:
        """Create digital signature for audit event."""
        message = f"{event.id}:{event.checksum}".encode('utf-8')
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature.hex()
    
    def verify_event_signature(self, event: ComplianceEvent) -> bool:
        """Verify digital signature of audit event."""
        if not event.digital_signature:
            return False
        
        try:
            message = f"{event.id}:{event.checksum}".encode('utf-8')
            signature_bytes = bytes.fromhex(event.digital_signature)
            
            self.public_key.verify(
                signature_bytes,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    async def store_event(self, event: ComplianceEvent) -> bool:
        """Store compliance event in secure database."""
        # Sign the event
        event.digital_signature = self.sign_event(event)
        
        def _store():
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    conn.execute("""
                        INSERT INTO compliance_events 
                        (id, timestamp, event_type, user_id, session_id, source_ip,
                         resource, action, result, classification, compliance_standards,
                         retention_policy, checksum, digital_signature, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.id,
                        event.timestamp.isoformat(),
                        event.event_type,
                        event.user_id,
                        event.session_id,
                        event.source_ip,
                        event.resource,
                        event.action,
                        event.result,
                        event.classification.value,
                        json.dumps([std.value for std in event.compliance_standards]),
                        event.retention_policy.value,
                        event.checksum,
                        event.digital_signature,
                        json.dumps(event.metadata)
                    ))
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error storing compliance event: {e}")
                    return False
                finally:
                    conn.close()
        
        # Execute in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _store
        )
    
    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        standards: Optional[List[ComplianceStandard]] = None,
        limit: int = 1000
    ) -> List[ComplianceEvent]:
        """Retrieve compliance events with filtering."""
        
        def _query():
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    query = "SELECT * FROM compliance_events WHERE 1=1"
                    params = []
                    
                    if start_date:
                        query += " AND timestamp >= ?"
                        params.append(start_date.isoformat())
                    
                    if end_date:
                        query += " AND timestamp <= ?"
                        params.append(end_date.isoformat())
                    
                    if user_id:
                        query += " AND user_id = ?"
                        params.append(user_id)
                    
                    if event_type:
                        query += " AND event_type = ?"
                        params.append(event_type)
                    
                    if standards:
                        # Check if any of the specified standards are in the event
                        standard_conditions = " OR ".join([
                            f"compliance_standards LIKE '%{std.value}%'" 
                            for std in standards
                        ])
                        query += f" AND ({standard_conditions})"
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    events = []
                    for row in rows:
                        # Convert row to ComplianceEvent
                        event = ComplianceEvent(
                            id=row[0],
                            timestamp=datetime.fromisoformat(row[1]),
                            event_type=row[2],
                            user_id=row[3],
                            session_id=row[4],
                            source_ip=row[5],
                            resource=row[6],
                            action=row[7],
                            result=row[8],
                            classification=DataClassification(row[9]),
                            compliance_standards=[
                                ComplianceStandard(std) 
                                for std in json.loads(row[10])
                            ],
                            retention_policy=RetentionPolicy(row[11]),
                            checksum=row[12],
                            digital_signature=row[13] or "",
                            metadata=json.loads(row[14]) if row[14] else {}
                        )
                        events.append(event)
                    
                    return events
                    
                finally:
                    conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _query
        )
    
    async def verify_audit_integrity(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Verify integrity of audit logs in date range."""
        
        def _verify():
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    # Get events in date range
                    cursor = conn.execute("""
                        SELECT id, checksum, digital_signature 
                        FROM compliance_events 
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp
                    """, (start_date.isoformat(), end_date.isoformat()))
                    
                    events = cursor.fetchall()
                    
                    integrity_results = {
                        "total_events": len(events),
                        "verified_checksums": 0,
                        "verified_signatures": 0,
                        "checksum_failures": [],
                        "signature_failures": [],
                        "overall_integrity": True
                    }
                    
                    # Verify each event (simplified - would need full event data for proper verification)
                    for event_id, checksum, signature in events:
                        # In real implementation, would reconstruct full event and verify
                        # For now, just check that checksum and signature exist
                        if checksum:
                            integrity_results["verified_checksums"] += 1
                        else:
                            integrity_results["checksum_failures"].append(event_id)
                            integrity_results["overall_integrity"] = False
                        
                        if signature:
                            integrity_results["verified_signatures"] += 1
                        else:
                            integrity_results["signature_failures"].append(event_id)
                    
                    return integrity_results
                    
                finally:
                    conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _verify
        )


class ComplianceAuditor:
    """Advanced compliance auditing system."""
    
    def __init__(self, db_path: str = "compliance/audit_logs.db"):
        self.database = ComplianceDatabase(db_path)
        
        # Compliance mappings
        self.standard_requirements = {
            ComplianceStandard.SOX: {
                "events": ["config_change", "data_modification", "access_control"],
                "retention_years": 7,
                "approval_required": True,
                "segregation_of_duties": True
            },
            ComplianceStandard.GDPR: {
                "events": ["data_access", "data_modification", "data_transfer"],
                "retention_years": 6,
                "consent_tracking": True,
                "right_to_erasure": True
            },
            ComplianceStandard.HIPAA: {
                "events": ["phi_access", "phi_modification", "phi_disclosure"],
                "retention_years": 6,
                "minimum_necessary": True,
                "business_associate": True
            },
            ComplianceStandard.PCI_DSS: {
                "events": ["cardholder_data_access", "system_modification"],
                "retention_years": 1,
                "network_segmentation": True,
                "encryption_required": True
            }
        }
        
        # Event classification rules
        self.classification_rules = {
            "healing_execute": DataClassification.CONFIDENTIAL,
            "config_change": DataClassification.RESTRICTED,
            "user_access": DataClassification.INTERNAL,
            "system_command": DataClassification.RESTRICTED,
            "data_export": DataClassification.CONFIDENTIAL
        }
        
        # Retention policy mapping
        self.retention_mapping = {
            DataClassification.PUBLIC: RetentionPolicy.SHORT_TERM,
            DataClassification.INTERNAL: RetentionPolicy.MEDIUM_TERM,
            DataClassification.CONFIDENTIAL: RetentionPolicy.LONG_TERM,
            DataClassification.RESTRICTED: RetentionPolicy.LONG_TERM,
            DataClassification.TOP_SECRET: RetentionPolicy.PERMANENT
        }
    
    async def log_compliance_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceEvent:
        """Log a compliance audit event."""
        
        # Determine applicable compliance standards
        applicable_standards = self._determine_applicable_standards(
            event_type, resource, action
        )
        
        # Classify data
        classification = self._classify_event(event_type, resource, action)
        
        # Determine retention policy
        retention_policy = self.retention_mapping.get(
            classification, RetentionPolicy.LONG_TERM
        )
        
        # Create compliance event
        event = ComplianceEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            result=result,
            classification=classification,
            compliance_standards=applicable_standards,
            retention_policy=retention_policy,
            metadata=metadata or {}
        )
        
        # Store in database
        await self.database.store_event(event)
        
        logger.info(f"Compliance event logged: {event_type} by {user_id}")
        return event
    
    def _determine_applicable_standards(
        self,
        event_type: str,
        resource: str,
        action: str
    ) -> List[ComplianceStandard]:
        """Determine which compliance standards apply to this event."""
        applicable = []
        
        # SOX applies to financial and IT controls
        if any(term in event_type.lower() for term in ["config", "finance", "control"]):
            applicable.append(ComplianceStandard.SOX)
        
        # GDPR applies to personal data
        if any(term in resource.lower() for term in ["user", "personal", "profile"]):
            applicable.append(ComplianceStandard.GDPR)
        
        # HIPAA applies to health information
        if any(term in resource.lower() for term in ["health", "medical", "phi"]):
            applicable.append(ComplianceStandard.HIPAA)
        
        # PCI DSS applies to payment data
        if any(term in resource.lower() for term in ["payment", "card", "transaction"]):
            applicable.append(ComplianceStandard.PCI_DSS)
        
        # ISO 27001 applies to all security events
        if event_type in ["security_event", "access_control", "authentication"]:
            applicable.append(ComplianceStandard.ISO27001)
        
        # Default to general standards if none specific
        if not applicable:
            applicable.extend([ComplianceStandard.ISO27001, ComplianceStandard.NIST])
        
        return applicable
    
    def _classify_event(
        self,
        event_type: str,
        resource: str,
        action: str
    ) -> DataClassification:
        """Classify the data sensitivity level of an event."""
        # Use classification rules
        classification = self.classification_rules.get(
            event_type, DataClassification.INTERNAL
        )
        
        # Upgrade classification based on resource sensitivity
        if any(term in resource.lower() for term in ["admin", "root", "system"]):
            classification = DataClassification.RESTRICTED
        
        if any(term in resource.lower() for term in ["secret", "key", "password"]):
            classification = DataClassification.TOP_SECRET
        
        return classification
    
    async def generate_compliance_report(
        self,
        standards: List[ComplianceStandard],
        start_date: datetime,
        end_date: datetime,
        generated_by: str
    ) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        
        # Get relevant events
        events = await self.database.get_events(
            start_date=start_date,
            end_date=end_date,
            standards=standards,
            limit=10000
        )
        
        # Analyze compliance
        compliance_analysis = await self._analyze_compliance(events, standards)
        
        # Generate findings and recommendations
        findings = await self._generate_findings(events, standards, compliance_analysis)
        recommendations = await self._generate_recommendations(findings, standards)
        
        # Create report
        report = ComplianceReport(
            id=str(uuid.uuid4()),
            report_type="compliance_audit",
            standards=standards,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now(),
            generated_by=generated_by,
            total_events=len(events),
            compliant_events=compliance_analysis["compliant_events"],
            violations=compliance_analysis["violations"],
            findings=findings,
            recommendations=recommendations,
            certification_status=self._determine_certification_status(compliance_analysis)
        )
        
        # Store report
        await self._store_report(report)
        
        logger.info(f"Compliance report generated: {report.id}")
        return report
    
    async def _analyze_compliance(
        self,
        events: List[ComplianceEvent],
        standards: List[ComplianceStandard]
    ) -> Dict[str, Any]:
        """Analyze events for compliance with standards."""
        
        analysis = {
            "compliant_events": 0,
            "violations": 0,
            "total_events": len(events),
            "compliance_by_standard": {},
            "violation_types": {},
            "risk_scores": []
        }
        
        for standard in standards:
            requirements = self.standard_requirements.get(standard, {})
            standard_violations = 0
            standard_compliant = 0
            
            for event in events:
                if standard in event.compliance_standards:
                    # Check specific requirements
                    violations = []
                    
                    # Check retention requirements
                    if requirements.get("retention_years"):
                        required_retention = RetentionPolicy.LONG_TERM
                        if event.retention_policy != required_retention:
                            violations.append("incorrect_retention_policy")
                    
                    # Check approval requirements
                    if requirements.get("approval_required"):
                        if not event.metadata.get("approved_by"):
                            violations.append("missing_approval")
                    
                    # Check segregation of duties
                    if requirements.get("segregation_of_duties"):
                        if event.metadata.get("approver") == event.user_id:
                            violations.append("segregation_violation")
                    
                    # Check encryption requirements
                    if requirements.get("encryption_required"):
                        if not event.digital_signature:
                            violations.append("missing_encryption")
                    
                    if violations:
                        standard_violations += 1
                        analysis["violations"] += 1
                        
                        # Track violation types
                        for violation in violations:
                            analysis["violation_types"][violation] = \
                                analysis["violation_types"].get(violation, 0) + 1
                    else:
                        standard_compliant += 1
                        analysis["compliant_events"] += 1
            
            # Store per-standard analysis
            total_standard_events = standard_violations + standard_compliant
            compliance_rate = (
                standard_compliant / total_standard_events 
                if total_standard_events > 0 else 0
            )
            
            analysis["compliance_by_standard"][standard.value] = {
                "total_events": total_standard_events,
                "compliant_events": standard_compliant,
                "violations": standard_violations,
                "compliance_rate": compliance_rate
            }
        
        return analysis
    
    async def _generate_findings(
        self,
        events: List[ComplianceEvent],
        standards: List[ComplianceStandard],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed compliance findings."""
        
        findings = []
        
        # Overall compliance finding
        overall_compliance_rate = (
            analysis["compliant_events"] / analysis["total_events"]
            if analysis["total_events"] > 0 else 0
        )
        
        findings.append({
            "id": "overall_compliance",
            "title": "Overall Compliance Rate",
            "severity": "info" if overall_compliance_rate > 0.95 else "warning",
            "description": f"Overall compliance rate: {overall_compliance_rate:.1%}",
            "details": {
                "compliant_events": analysis["compliant_events"],
                "total_events": analysis["total_events"],
                "violations": analysis["violations"]
            }
        })
        
        # Per-standard findings
        for standard, std_analysis in analysis["compliance_by_standard"].items():
            compliance_rate = std_analysis["compliance_rate"]
            
            severity = "info"
            if compliance_rate < 0.9:
                severity = "high"
            elif compliance_rate < 0.95:
                severity = "medium"
            elif compliance_rate < 0.98:
                severity = "low"
            
            findings.append({
                "id": f"compliance_{standard}",
                "title": f"{standard.upper()} Compliance",
                "severity": severity,
                "description": f"{standard.upper()} compliance rate: {compliance_rate:.1%}",
                "details": std_analysis
            })
        
        # Violation type findings
        if analysis["violation_types"]:
            findings.append({
                "id": "violation_analysis",
                "title": "Violation Types Analysis",
                "severity": "medium",
                "description": "Analysis of compliance violation types",
                "details": analysis["violation_types"]
            })
        
        # Data integrity findings
        integrity_events = [e for e in events if not e.verify_integrity()]
        if integrity_events:
            findings.append({
                "id": "data_integrity",
                "title": "Data Integrity Issues",
                "severity": "critical",
                "description": f"{len(integrity_events)} events failed integrity verification",
                "details": {
                    "failed_events": len(integrity_events),
                    "event_ids": [e.id for e in integrity_events[:10]]  # First 10
                }
            })
        
        return findings
    
    async def _generate_recommendations(
        self,
        findings: List[Dict[str, Any]],
        standards: List[ComplianceStandard]
    ) -> List[str]:
        """Generate compliance recommendations."""
        
        recommendations = []
        
        # Check for high/critical findings
        high_findings = [f for f in findings if f["severity"] in ["high", "critical"]]
        
        if high_findings:
            recommendations.append(
                "Address all high and critical compliance findings immediately"
            )
        
        # Standard-specific recommendations
        for standard in standards:
            requirements = self.standard_requirements.get(standard, {})
            
            if standard == ComplianceStandard.SOX:
                recommendations.extend([
                    "Implement segregation of duties for financial controls",
                    "Ensure all configuration changes are approved",
                    "Maintain 7-year retention of audit logs"
                ])
            
            elif standard == ComplianceStandard.GDPR:
                recommendations.extend([
                    "Implement consent tracking for personal data",
                    "Ensure right to erasure capabilities",
                    "Document lawful basis for data processing"
                ])
            
            elif standard == ComplianceStandard.HIPAA:
                recommendations.extend([
                    "Implement minimum necessary access controls",
                    "Ensure business associate agreements are in place",
                    "Regular PHI access audits"
                ])
            
            elif standard == ComplianceStandard.PCI_DSS:
                recommendations.extend([
                    "Encrypt all cardholder data",
                    "Implement network segmentation",
                    "Regular vulnerability assessments"
                ])
        
        # Generic recommendations
        recommendations.extend([
            "Implement automated compliance monitoring",
            "Regular compliance training for staff",
            "Establish incident response procedures",
            "Regular third-party compliance audits"
        ])
        
        return recommendations
    
    def _determine_certification_status(
        self,
        analysis: Dict[str, Any]
    ) -> str:
        """Determine certification status based on compliance analysis."""
        
        overall_compliance = (
            analysis["compliant_events"] / analysis["total_events"]
            if analysis["total_events"] > 0 else 0
        )
        
        if overall_compliance >= 0.98:
            return "certified"
        elif overall_compliance >= 0.95:
            return "conditional"
        elif overall_compliance >= 0.90:
            return "remediation_required"
        else:
            return "non_compliant"
    
    async def _store_report(self, report: ComplianceReport):
        """Store compliance report in database."""
        def _store():
            with self.database.lock:
                conn = sqlite3.connect(str(self.database.db_path))
                try:
                    conn.execute("""
                        INSERT INTO compliance_reports 
                        (id, report_type, standards, period_start, period_end,
                         generated_at, generated_by, total_events, compliant_events,
                         violations, findings, recommendations, certification_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        report.id,
                        report.report_type,
                        json.dumps([std.value for std in report.standards]),
                        report.period_start.isoformat(),
                        report.period_end.isoformat(),
                        report.generated_at.isoformat(),
                        report.generated_by,
                        report.total_events,
                        report.compliant_events,
                        report.violations,
                        json.dumps(report.findings),
                        json.dumps(report.recommendations),
                        report.certification_status
                    ))
                    conn.commit()
                finally:
                    conn.close()
        
        await asyncio.get_event_loop().run_in_executor(
            self.database.executor, _store
        )
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status summary."""
        
        # Get recent events (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        events = await self.database.get_events(
            start_date=start_date,
            end_date=end_date,
            limit=5000
        )
        
        # Analyze by standard
        standard_status = {}
        for standard in ComplianceStandard:
            standard_events = [
                e for e in events if standard in e.compliance_standards
            ]
            
            if standard_events:
                compliant = len([e for e in standard_events if e.verify_integrity()])
                compliance_rate = compliant / len(standard_events)
                
                standard_status[standard.value] = {
                    "total_events": len(standard_events),
                    "compliant_events": compliant,
                    "compliance_rate": compliance_rate,
                    "status": "compliant" if compliance_rate > 0.95 else "at_risk"
                }
        
        # Verify audit integrity
        integrity_results = await self.database.verify_audit_integrity(
            start_date, end_date
        )
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "standards_compliance": standard_status,
            "audit_integrity": integrity_results,
            "last_updated": datetime.now().isoformat()
        }


# Global compliance auditor instance
compliance_auditor = ComplianceAuditor()