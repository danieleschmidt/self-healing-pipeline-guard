"""GDPR compliance implementation for European Union deployment."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """GDPR data categories."""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    PSEUDONYMIZED_DATA = "pseudonymized_data"
    ANONYMIZED_DATA = "anonymized_data"
    TECHNICAL_DATA = "technical_data"
    METADATA = "metadata"


class ProcessingPurpose(Enum):
    """Legal bases for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRight(Enum):
    """GDPR data subject rights."""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure (right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    DATA_PORTABILITY = "data_portability"  # Right to data portability
    OBJECT = "object"  # Right to object
    AUTOMATED_DECISION_MAKING = "automated_decision_making"  # Rights related to automated decision making


@dataclass
class PersonalDataRecord:
    """Record of personal data processing."""
    record_id: str
    data_subject_id: str
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    data_fields: List[str]
    collected_at: datetime
    retention_period: Optional[timedelta] = None
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    anonymized: bool = False
    deleted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the data has exceeded its retention period."""
        if not self.retention_period:
            return False
        return datetime.now() > self.collected_at + self.retention_period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "data_subject_id": self.data_subject_id,
            "data_category": self.data_category.value,
            "processing_purpose": self.processing_purpose.value,
            "data_fields": self.data_fields,
            "collected_at": self.collected_at.isoformat(),
            "retention_period": self.retention_period.total_seconds() if self.retention_period else None,
            "consent_given": self.consent_given,
            "consent_timestamp": self.consent_timestamp.isoformat() if self.consent_timestamp else None,
            "anonymized": self.anonymized,
            "deleted": self.deleted,
            "metadata": self.metadata
        }


@dataclass
class DataSubjectRequest:
    """GDPR data subject request."""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRight
    requested_at: datetime
    description: str
    status: str = "pending"
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "data_subject_id": self.data_subject_id,
            "request_type": self.request_type.value,
            "requested_at": self.requested_at.isoformat(),
            "description": self.description,
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_data": self.response_data
        }


class DataProcessor:
    """Data processing handler with GDPR compliance."""
    
    def __init__(self, organization_name: str, dpo_contact: str):
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact  # Data Protection Officer contact
        self.personal_data_records: Dict[str, PersonalDataRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
    def record_data_collection(
        self,
        data_subject_id: str,
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        data_fields: List[str],
        retention_period: Optional[timedelta] = None,
        consent_given: bool = False
    ) -> str:
        """Record collection of personal data."""
        record_id = str(uuid.uuid4())
        
        record = PersonalDataRecord(
            record_id=record_id,
            data_subject_id=data_subject_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            data_fields=data_fields,
            collected_at=datetime.now(),
            retention_period=retention_period,
            consent_given=consent_given,
            consent_timestamp=datetime.now() if consent_given else None
        )
        
        self.personal_data_records[record_id] = record
        
        logger.info(f"Recorded data collection: {record_id} for subject {data_subject_id}")
        return record_id
    
    def record_consent(
        self,
        data_subject_id: str,
        processing_purposes: List[ProcessingPurpose],
        consent_given: bool,
        consent_text: str,
        consent_method: str = "explicit"
    ):
        """Record consent for data processing."""
        consent_record = {
            "data_subject_id": data_subject_id,
            "processing_purposes": [p.value for p in processing_purposes],
            "consent_given": consent_given,
            "consent_text": consent_text,
            "consent_method": consent_method,
            "timestamp": datetime.now().isoformat(),
            "withdrawn": False,
            "withdrawal_timestamp": None
        }
        
        consent_id = f"{data_subject_id}_{datetime.now().timestamp()}"
        self.consent_records[consent_id] = consent_record
        
        # Update related data records
        for record in self.personal_data_records.values():
            if (record.data_subject_id == data_subject_id and 
                record.processing_purpose in processing_purposes):
                record.consent_given = consent_given
                record.consent_timestamp = datetime.now() if consent_given else None
        
        logger.info(f"Recorded consent: {consent_id} - granted: {consent_given}")
    
    def withdraw_consent(self, data_subject_id: str, processing_purposes: List[ProcessingPurpose]):
        """Withdraw consent for specific processing purposes."""
        # Update consent records
        for consent_record in self.consent_records.values():
            if (consent_record["data_subject_id"] == data_subject_id and
                any(p.value in consent_record["processing_purposes"] for p in processing_purposes)):
                consent_record["withdrawn"] = True
                consent_record["withdrawal_timestamp"] = datetime.now().isoformat()
        
        # Update data records
        for record in self.personal_data_records.values():
            if (record.data_subject_id == data_subject_id and 
                record.processing_purpose in processing_purposes):
                record.consent_given = False
                record.consent_timestamp = None
        
        logger.info(f"Consent withdrawn for subject {data_subject_id}")
    
    def anonymize_data(self, record_id: str):
        """Anonymize personal data record."""
        if record_id in self.personal_data_records:
            record = self.personal_data_records[record_id]
            
            # Hash the data subject ID to anonymize
            anonymous_id = hashlib.sha256(
                f"{record.data_subject_id}_{record.collected_at}".encode()
            ).hexdigest()[:16]
            
            record.data_subject_id = f"anon_{anonymous_id}"
            record.anonymized = True
            record.consent_given = False  # No longer needed for anonymized data
            
            logger.info(f"Anonymized data record: {record_id}")
    
    def delete_data(self, data_subject_id: str, processing_purpose: Optional[ProcessingPurpose] = None):
        """Delete personal data (right to erasure)."""
        deleted_records = []
        
        for record_id, record in self.personal_data_records.items():
            if record.data_subject_id == data_subject_id:
                if processing_purpose is None or record.processing_purpose == processing_purpose:
                    record.deleted = True
                    record.data_fields = []  # Clear actual data
                    deleted_records.append(record_id)
        
        logger.info(f"Deleted {len(deleted_records)} records for subject {data_subject_id}")
        return deleted_records
    
    def get_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get all data for a data subject (right to access)."""
        subject_records = []
        
        for record in self.personal_data_records.values():
            if record.data_subject_id == data_subject_id and not record.deleted:
                subject_records.append(record.to_dict())
        
        # Get consent records
        subject_consents = []
        for consent_record in self.consent_records.values():
            if consent_record["data_subject_id"] == data_subject_id:
                subject_consents.append(consent_record)
        
        return {
            "data_subject_id": data_subject_id,
            "personal_data_records": subject_records,
            "consent_records": subject_consents,
            "generated_at": datetime.now().isoformat()
        }
    
    def export_subject_data(self, data_subject_id: str, format: str = "json") -> bytes:
        """Export data subject's data in portable format (right to data portability)."""
        data = self.get_subject_data(data_subject_id)
        
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        elif format == "csv":
            # Simple CSV export (in production, use proper CSV library)
            import io
            output = io.StringIO()
            
            # Write headers
            output.write("Record ID,Data Category,Processing Purpose,Collected At,Consent Given\n")
            
            # Write data
            for record in data["personal_data_records"]:
                output.write(f"{record['record_id']},{record['data_category']},"
                           f"{record['processing_purpose']},{record['collected_at']},"
                           f"{record['consent_given']}\n")
            
            return output.getvalue().encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_expired_data(self) -> int:
        """Clean up data that has exceeded retention periods."""
        expired_count = 0
        current_time = datetime.now()
        
        for record in self.personal_data_records.values():
            if (record.retention_period and 
                not record.deleted and 
                current_time > record.collected_at + record.retention_period):
                
                # Automatically delete expired data
                record.deleted = True
                record.data_fields = []
                expired_count += 1
                
                logger.info(f"Auto-deleted expired data record: {record.record_id}")
        
        return expired_count


class GDPRCompliance:
    """Main GDPR compliance manager."""
    
    def __init__(self, organization_name: str, dpo_contact: str):
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact
        self.data_processor = DataProcessor(organization_name, dpo_contact)
        self.compliance_checks: List[Dict[str, Any]] = []
        
        # Default retention periods for different data types
        self.default_retention_periods = {
            DataCategory.PERSONAL_DATA: timedelta(days=365 * 2),  # 2 years
            DataCategory.SENSITIVE_DATA: timedelta(days=365),     # 1 year
            DataCategory.TECHNICAL_DATA: timedelta(days=365 * 3), # 3 years
            DataCategory.METADATA: timedelta(days=30),            # 30 days
        }
    
    async def handle_data_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle a data subject request under GDPR."""
        logger.info(f"Processing GDPR request: {request.request_type.value} for {request.data_subject_id}")
        
        try:
            if request.request_type == DataSubjectRight.ACCESS:
                # Right to access
                data = self.data_processor.get_subject_data(request.data_subject_id)
                request.response_data = data
                request.status = "completed"
                
            elif request.request_type == DataSubjectRight.ERASURE:
                # Right to be forgotten
                deleted_records = self.data_processor.delete_data(request.data_subject_id)
                request.response_data = {"deleted_records": deleted_records}
                request.status = "completed"
                
            elif request.request_type == DataSubjectRight.DATA_PORTABILITY:
                # Right to data portability
                exported_data = self.data_processor.export_subject_data(
                    request.data_subject_id, 
                    format="json"
                )
                # In production, you'd store this file and provide a download link
                request.response_data = {"export_size": len(exported_data)}
                request.status = "completed"
                
            elif request.request_type == DataSubjectRight.RESTRICT_PROCESSING:
                # Right to restrict processing - mark records as restricted
                for record in self.data_processor.personal_data_records.values():
                    if record.data_subject_id == request.data_subject_id:
                        record.metadata["processing_restricted"] = True
                        record.metadata["restriction_reason"] = "data_subject_request"
                
                request.status = "completed"
                
            elif request.request_type == DataSubjectRight.RECTIFICATION:
                # Right to rectification - would need specific correction instructions
                request.status = "manual_review_required"
                request.response_data = {
                    "message": "Manual review required for data rectification"
                }
                
            else:
                request.status = "not_implemented"
                request.response_data = {
                    "message": f"Request type {request.request_type.value} not yet implemented"
                }
            
            request.completed_at = datetime.now()
            self.data_processor.data_subject_requests[request.request_id] = request
            
            logger.info(f"GDPR request {request.request_id} completed with status: {request.status}")
            
            return {
                "request_id": request.request_id,
                "status": request.status,
                "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                "response_data": request.response_data
            }
            
        except Exception as e:
            request.status = "failed"
            request.response_data = {"error": str(e)}
            request.completed_at = datetime.now()
            
            logger.error(f"GDPR request {request.request_id} failed: {e}")
            
            return {
                "request_id": request.request_id,
                "status": "failed",
                "error": str(e)
            }
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """Run comprehensive GDPR compliance check."""
        check_results = {
            "timestamp": datetime.now().isoformat(),
            "organization": self.organization_name,
            "dpo_contact": self.dpo_contact,
            "checks": []
        }
        
        # Check 1: Data retention compliance
        expired_data_count = 0
        total_records = len(self.data_processor.personal_data_records)
        
        for record in self.data_processor.personal_data_records.values():
            if record.is_expired and not record.deleted:
                expired_data_count += 1
        
        check_results["checks"].append({
            "name": "Data Retention Compliance",
            "status": "PASS" if expired_data_count == 0 else "FAIL",
            "details": f"{expired_data_count} expired records found out of {total_records} total",
            "action_required": expired_data_count > 0
        })
        
        # Check 2: Consent management
        records_without_valid_consent = 0
        consent_required_records = 0
        
        for record in self.data_processor.personal_data_records.values():
            if record.processing_purpose == ProcessingPurpose.CONSENT and not record.deleted:
                consent_required_records += 1
                if not record.consent_given:
                    records_without_valid_consent += 1
        
        check_results["checks"].append({
            "name": "Consent Management",
            "status": "PASS" if records_without_valid_consent == 0 else "FAIL", 
            "details": f"{records_without_valid_consent} records without valid consent out of {consent_required_records} requiring consent",
            "action_required": records_without_valid_consent > 0
        })
        
        # Check 3: Data subject request response times
        pending_requests = sum(
            1 for req in self.data_processor.data_subject_requests.values()
            if req.status == "pending"
        )
        
        overdue_requests = 0
        for req in self.data_processor.data_subject_requests.values():
            if req.status == "pending":
                # GDPR requires response within 30 days
                if datetime.now() > req.requested_at + timedelta(days=30):
                    overdue_requests += 1
        
        check_results["checks"].append({
            "name": "Data Subject Request Response Times",
            "status": "PASS" if overdue_requests == 0 else "FAIL",
            "details": f"{overdue_requests} overdue requests out of {pending_requests} pending",
            "action_required": overdue_requests > 0
        })
        
        # Check 4: Data minimization
        sensitive_data_records = sum(
            1 for record in self.data_processor.personal_data_records.values()
            if record.data_category == DataCategory.SENSITIVE_DATA and not record.deleted
        )
        
        check_results["checks"].append({
            "name": "Data Minimization",
            "status": "REVIEW" if sensitive_data_records > 0 else "PASS",
            "details": f"{sensitive_data_records} sensitive data records - review for necessity",
            "action_required": False
        })
        
        # Overall compliance status
        failed_checks = sum(1 for check in check_results["checks"] if check["status"] == "FAIL")
        check_results["overall_status"] = "COMPLIANT" if failed_checks == 0 else "NON_COMPLIANT"
        check_results["failed_checks"] = failed_checks
        
        self.compliance_checks.append(check_results)
        
        logger.info(f"GDPR compliance check completed: {check_results['overall_status']}")
        
        return check_results
    
    def generate_privacy_policy(self) -> str:
        """Generate a basic privacy policy template."""
        policy_template = f"""
PRIVACY POLICY - {self.organization_name}

1. DATA CONTROLLER
{self.organization_name}
Data Protection Officer: {self.dpo_contact}

2. DATA WE COLLECT
We collect the following categories of personal data:
- Technical data for system operation and monitoring
- Usage data for service improvement
- Contact information for user accounts

3. LEGAL BASIS FOR PROCESSING
We process your data based on:
- Your consent (Article 6(1)(a) GDPR)
- Performance of a contract (Article 6(1)(b) GDPR)
- Legitimate interests (Article 6(1)(f) GDPR)

4. DATA RETENTION
We retain personal data for the following periods:
- Personal data: 2 years
- Technical data: 3 years
- Sensitive data: 1 year

5. YOUR RIGHTS
Under GDPR, you have the right to:
- Access your personal data
- Rectify inaccurate data
- Erase your data
- Restrict processing
- Data portability
- Object to processing

6. CONTACT
To exercise your rights or for privacy questions, contact: {self.dpo_contact}

Last updated: {datetime.now().strftime('%Y-%m-%d')}
"""
        return policy_template.strip()
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report."""
        latest_check = self.compliance_checks[-1] if self.compliance_checks else self.run_compliance_check()
        
        return {
            "organization": self.organization_name,
            "report_generated": datetime.now().isoformat(),
            "compliance_status": latest_check["overall_status"],
            "total_data_records": len(self.data_processor.personal_data_records),
            "total_consent_records": len(self.data_processor.consent_records),
            "pending_requests": len([
                req for req in self.data_processor.data_subject_requests.values()
                if req.status == "pending"
            ]),
            "latest_compliance_check": latest_check,
            "privacy_policy": self.generate_privacy_policy()
        }


# Global GDPR compliance instance
_gdpr_compliance: Optional[GDPRCompliance] = None


def get_gdpr_compliance(organization_name: str = "Healing Guard", dpo_contact: str = "dpo@healingguard.com") -> GDPRCompliance:
    """Get global GDPR compliance instance."""
    global _gdpr_compliance
    
    if _gdpr_compliance is None:
        _gdpr_compliance = GDPRCompliance(organization_name, dpo_contact)
    
    return _gdpr_compliance