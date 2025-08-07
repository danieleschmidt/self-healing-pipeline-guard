"""PDPA compliance implementation for Personal Data Protection Act (Singapore/Thailand).

This module provides comprehensive PDPA compliance for sentiment analysis data processing,
covering both Singapore's PDPA and Thailand's PDPA requirements.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class PDPAJurisdiction(Enum):
    """PDPA jurisdictions supported."""
    SINGAPORE = "singapore"
    THAILAND = "thailand"
    MALAYSIA = "malaysia"  # Future support


class PDPAConsent(Enum):
    """Types of consent under PDPA."""
    EXPRESS_CONSENT = "express_consent"  # Clear, specific consent
    IMPLIED_CONSENT = "implied_consent"  # Reasonable person would consent
    DEEMED_CONSENT = "deemed_consent"    # Prescribed circumstances
    OPT_OUT_CONSENT = "opt_out_consent"  # Notification with opt-out option


class PDPAPurpose(Enum):
    """Purposes for personal data processing under PDPA."""
    SERVICE_PROVISION = "service_provision"
    CUSTOMER_SUPPORT = "customer_support"
    SYSTEM_ADMINISTRATION = "system_administration"
    SECURITY_MONITORING = "security_monitoring"
    BUSINESS_IMPROVEMENT = "business_improvement"
    LEGAL_COMPLIANCE = "legal_compliance"
    RESEARCH_DEVELOPMENT = "research_development"
    MARKETING = "marketing"


class PDPADataType(Enum):
    """Types of personal data under PDPA."""
    PERSONAL_DATA = "personal_data"  # Data that can identify an individual
    SENSITIVE_DATA = "sensitive_data"  # Health, religious beliefs, etc.
    BUSINESS_CONTACT_DATA = "business_contact_data"  # Business contact information


@dataclass
class PDPAConsentRecord:
    """Record of consent under PDPA."""
    consent_id: str
    individual_id: str  # Hashed individual identifier
    jurisdiction: PDPAJurisdiction
    consent_type: PDPAConsent
    purposes: List[PDPAPurpose]
    data_types: List[PDPADataType]
    consent_given: bool
    consent_timestamp: datetime
    consent_method: str
    withdrawal_timestamp: Optional[datetime] = None
    consent_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert consent record to dictionary."""
        return {
            "consent_id": self.consent_id,
            "individual_id_hash": self.individual_id,
            "jurisdiction": self.jurisdiction.value,
            "consent_type": self.consent_type.value,
            "purposes": [purpose.value for purpose in self.purposes],
            "data_types": [data_type.value for data_type in self.data_types],
            "consent_given": self.consent_given,
            "consent_timestamp": self.consent_timestamp.isoformat(),
            "consent_method": self.consent_method,
            "withdrawal_timestamp": self.withdrawal_timestamp.isoformat() if self.withdrawal_timestamp else None,
            "consent_duration_days": self.consent_duration.days if self.consent_duration else None
        }


@dataclass
class PDPAProcessingRecord:
    """Record of personal data processing under PDPA."""
    processing_id: str
    individual_id: Optional[str]  # Hashed individual identifier
    jurisdiction: PDPAJurisdiction
    data_types: List[PDPADataType]
    purposes: List[PDPAPurpose]
    processing_timestamp: datetime
    consent_reference: Optional[str]
    retention_period_days: int
    cross_border_transfer: bool = False
    third_party_disclosure: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert processing record to dictionary."""
        return {
            "processing_id": self.processing_id,
            "individual_id_hash": self.individual_id,
            "jurisdiction": self.jurisdiction.value,
            "data_types": [data_type.value for data_type in self.data_types],
            "purposes": [purpose.value for purpose in self.purposes],
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "consent_reference": self.consent_reference,
            "retention_period_days": self.retention_period_days,
            "cross_border_transfer": self.cross_border_transfer,
            "third_party_disclosure": self.third_party_disclosure
        }


@dataclass
class PDPAAccessRequest:
    """PDPA data access request from individual."""
    request_id: str
    individual_id: str
    jurisdiction: PDPAJurisdiction
    requested_at: datetime
    request_type: str  # access, correction, deletion, portability
    description: str
    status: str = "pending"
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert access request to dictionary."""
        return {
            "request_id": self.request_id,
            "individual_id_hash": hashlib.sha256(self.individual_id.encode()).hexdigest(),
            "jurisdiction": self.jurisdiction.value,
            "requested_at": self.requested_at.isoformat(),
            "request_type": self.request_type,
            "description": self.description,
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_data": self.response_data
        }


class PDPADataProcessor:
    """Handles PDPA-compliant data processing for sentiment analysis."""
    
    def __init__(self, default_jurisdiction: PDPAJurisdiction = PDPAJurisdiction.SINGAPORE):
        self.default_jurisdiction = default_jurisdiction
        self.processing_records: List[PDPAProcessingRecord] = []
        self.consent_records: Dict[str, PDPAConsentRecord] = {}
        self.access_requests: Dict[str, PDPAAccessRequest] = {}
        
        # Jurisdiction-specific retention periods
        self.retention_periods = {
            PDPAJurisdiction.SINGAPORE: {
                PDPADataType.PERSONAL_DATA: 365,  # 1 year
                PDPADataType.SENSITIVE_DATA: 180,  # 6 months
                PDPADataType.BUSINESS_CONTACT_DATA: 730  # 2 years
            },
            PDPAJurisdiction.THAILAND: {
                PDPADataType.PERSONAL_DATA: 365,  # 1 year
                PDPADataType.SENSITIVE_DATA: 90,   # 3 months
                PDPADataType.BUSINESS_CONTACT_DATA: 1095  # 3 years
            }
        }
    
    def hash_individual_id(self, individual_id: str) -> str:
        """Create privacy-preserving hash of individual identifier."""
        salt = "pdpa_healing_guard_salt_2025"  # In production, use environment variable
        combined = f"{salt}{individual_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def record_consent(
        self,
        individual_id: str,
        consent_type: PDPAConsent,
        purposes: List[PDPAPurpose],
        data_types: List[PDPADataType],
        consent_method: str,
        jurisdiction: Optional[PDPAJurisdiction] = None,
        consent_duration: Optional[timedelta] = None
    ) -> str:
        """Record consent for personal data processing under PDPA."""
        
        consent_id = f"pdpa_consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        hashed_individual = self.hash_individual_id(individual_id)
        jurisdiction = jurisdiction or self.default_jurisdiction
        
        consent_record = PDPAConsentRecord(
            consent_id=consent_id,
            individual_id=hashed_individual,
            jurisdiction=jurisdiction,
            consent_type=consent_type,
            purposes=purposes,
            data_types=data_types,
            consent_given=True,
            consent_timestamp=datetime.now(),
            consent_method=consent_method,
            consent_duration=consent_duration
        )
        
        self.consent_records[consent_id] = consent_record
        
        logger.info(f"PDPA: Recorded consent {consent_id} for {jurisdiction.value}")
        
        return consent_id
    
    def withdraw_consent(self, individual_id: str, consent_id: Optional[str] = None) -> List[str]:
        """Withdraw consent for data processing."""
        hashed_individual = self.hash_individual_id(individual_id)
        withdrawn_consents = []
        
        for consent_record in self.consent_records.values():
            should_withdraw = (
                consent_record.individual_id == hashed_individual and
                consent_record.consent_given and
                (consent_id is None or consent_record.consent_id == consent_id)
            )
            
            if should_withdraw:
                consent_record.consent_given = False
                consent_record.withdrawal_timestamp = datetime.now()
                withdrawn_consents.append(consent_record.consent_id)
        
        logger.info(f"PDPA: Withdrew {len(withdrawn_consents)} consents for individual")
        
        return withdrawn_consents
    
    def has_valid_consent(
        self,
        individual_id: str,
        purpose: PDPAPurpose,
        data_type: PDPADataType,
        jurisdiction: Optional[PDPAJurisdiction] = None
    ) -> bool:
        """Check if valid consent exists for data processing."""
        
        hashed_individual = self.hash_individual_id(individual_id)
        jurisdiction = jurisdiction or self.default_jurisdiction
        
        for consent_record in self.consent_records.values():
            if (consent_record.individual_id == hashed_individual and
                consent_record.jurisdiction == jurisdiction and
                consent_record.consent_given and
                purpose in consent_record.purposes and
                data_type in consent_record.data_types):
                
                # Check if consent has expired
                if consent_record.consent_duration:
                    expiry_time = consent_record.consent_timestamp + consent_record.consent_duration
                    if datetime.now() > expiry_time:
                        continue
                
                return True
        
        return False
    
    def record_processing_activity(
        self,
        individual_id: Optional[str],
        data_types: List[PDPADataType],
        purposes: List[PDPAPurpose],
        consent_reference: Optional[str] = None,
        jurisdiction: Optional[PDPAJurisdiction] = None,
        cross_border_transfer: bool = False,
        third_party_disclosure: bool = False
    ) -> str:
        """Record personal data processing activity."""
        
        processing_id = f"pdpa_proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_records)}"
        hashed_individual = None
        if individual_id:
            hashed_individual = self.hash_individual_id(individual_id)
        
        jurisdiction = jurisdiction or self.default_jurisdiction
        
        # Determine retention period based on data types and jurisdiction
        retention_periods = self.retention_periods.get(jurisdiction, {})
        max_retention = max(
            retention_periods.get(data_type, 365) 
            for data_type in data_types
        )
        
        record = PDPAProcessingRecord(
            processing_id=processing_id,
            individual_id=hashed_individual,
            jurisdiction=jurisdiction,
            data_types=data_types,
            purposes=purposes,
            processing_timestamp=datetime.now(),
            consent_reference=consent_reference,
            retention_period_days=max_retention,
            cross_border_transfer=cross_border_transfer,
            third_party_disclosure=third_party_disclosure
        )
        
        self.processing_records.append(record)
        
        logger.info(f"PDPA: Recorded processing activity {processing_id} in {jurisdiction.value}")
        
        return processing_id
    
    def can_process_data(
        self,
        individual_id: str,
        data_types: List[PDPADataType],
        purposes: List[PDPAPurpose],
        jurisdiction: Optional[PDPAJurisdiction] = None
    ) -> Dict[str, Any]:
        """Determine if data can be processed under PDPA."""
        
        jurisdiction = jurisdiction or self.default_jurisdiction
        processing_allowed = True
        consent_status = {}
        
        # Check consent for each combination of data type and purpose
        for data_type in data_types:
            for purpose in purposes:
                consent_key = f"{data_type.value}_{purpose.value}"
                
                # Check for sensitive data requiring express consent
                if data_type == PDPADataType.SENSITIVE_DATA:
                    has_consent = False
                    for consent_record in self.consent_records.values():
                        if (consent_record.individual_id == self.hash_individual_id(individual_id) and
                            consent_record.consent_type == PDPAConsent.EXPRESS_CONSENT and
                            data_type in consent_record.data_types and
                            purpose in consent_record.purposes):
                            has_consent = True
                            break
                    
                    consent_status[consent_key] = has_consent
                    if not has_consent:
                        processing_allowed = False
                else:
                    # Regular personal data - check for any valid consent
                    has_consent = self.has_valid_consent(individual_id, purpose, data_type, jurisdiction)
                    consent_status[consent_key] = has_consent
                    
                    # Some purposes may not require explicit consent under PDPA
                    if not has_consent and purpose in [
                        PDPAPurpose.LEGAL_COMPLIANCE,
                        PDPAPurpose.SECURITY_MONITORING
                    ]:
                        consent_status[consent_key] = True  # Legal basis exists
                    elif not has_consent:
                        processing_allowed = False
        
        return {
            "processing_allowed": processing_allowed,
            "jurisdiction": jurisdiction.value,
            "consent_status": consent_status,
            "legal_basis": [purpose.value for purpose in purposes if purpose in [
                PDPAPurpose.LEGAL_COMPLIANCE,
                PDPAPurpose.SECURITY_MONITORING
            ]]
        }
    
    def get_individual_data(self, individual_id: str) -> Dict[str, Any]:
        """Get all personal data for an individual (access right)."""
        hashed_individual = self.hash_individual_id(individual_id)
        
        individual_records = [
            record for record in self.processing_records
            if record.individual_id == hashed_individual
        ]
        
        individual_consents = [
            consent for consent in self.consent_records.values()
            if consent.individual_id == hashed_individual
        ]
        
        # Aggregate data types and purposes
        all_data_types = set()
        all_purposes = set()
        for record in individual_records:
            all_data_types.update(record.data_types)
            all_purposes.update(record.purposes)
        
        return {
            "individual_id": "HASHED_FOR_PRIVACY",
            "processing_records": [record.to_dict() for record in individual_records],
            "consent_records": [consent.to_dict() for consent in individual_consents],
            "data_types_processed": [dt.value for dt in all_data_types],
            "processing_purposes": [purpose.value for purpose in all_purposes],
            "cross_border_transfers": any(record.cross_border_transfer for record in individual_records),
            "third_party_disclosures": any(record.third_party_disclosure for record in individual_records)
        }
    
    def delete_individual_data(
        self,
        individual_id: str,
        data_types_to_delete: Optional[List[PDPADataType]] = None
    ) -> Dict[str, Any]:
        """Delete individual's personal data."""
        hashed_individual = self.hash_individual_id(individual_id)
        
        deleted_records = []
        remaining_records = []
        
        for record in self.processing_records:
            if record.individual_id == hashed_individual:
                should_delete = True
                
                # Check if specific data types were requested
                if data_types_to_delete:
                    should_delete = any(
                        data_type in record.data_types 
                        for data_type in data_types_to_delete
                    )
                
                # Check for legal obligations to retain data
                if PDPAPurpose.LEGAL_COMPLIANCE in record.purposes:
                    should_delete = False
                    logger.info(f"Retaining record {record.processing_id} for legal compliance")
                
                if should_delete:
                    deleted_records.append(record.processing_id)
                else:
                    remaining_records.append(record)
            else:
                remaining_records.append(record)
        
        # Update processing records
        self.processing_records = remaining_records
        
        # Also withdraw all consents for the individual
        withdrawn_consents = self.withdraw_consent(individual_id)
        
        logger.info(f"PDPA: Deleted {len(deleted_records)} records and withdrew {len(withdrawn_consents)} consents")
        
        return {
            "deleted_records": deleted_records,
            "withdrawn_consents": withdrawn_consents,
            "data_types_deleted": [dt.value for dt in (data_types_to_delete or [])],
            "remaining_records": len([
                r for r in remaining_records 
                if r.individual_id == hashed_individual
            ])
        }
    
    def cleanup_expired_records(self) -> int:
        """Clean up records that have exceeded PDPA retention periods."""
        current_time = datetime.now()
        deleted_count = 0
        records_to_keep = []
        
        for record in self.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            expiry_date = record.processing_timestamp + retention_period
            
            if current_time > expiry_date:
                deleted_count += 1
                logger.info(f"PDPA: Auto-deleted expired record {record.processing_id}")
            else:
                records_to_keep.append(record)
        
        self.processing_records = records_to_keep
        
        return deleted_count


class PDPACompliance:
    """Main PDPA compliance manager for sentiment analysis."""
    
    def __init__(self, jurisdiction: PDPAJurisdiction = PDPAJurisdiction.SINGAPORE):
        self.jurisdiction = jurisdiction
        self.data_processor = PDPADataProcessor(jurisdiction)
        self.compliance_checks: List[Dict[str, Any]] = []
    
    async def handle_access_request(self, request: PDPAAccessRequest) -> Dict[str, Any]:
        """Handle PDPA individual access request."""
        logger.info(f"Processing PDPA {request.request_type} request for {request.individual_id}")
        
        try:
            if request.request_type == "access":
                # Provide individual's personal data
                individual_data = self.data_processor.get_individual_data(request.individual_id)
                request.response_data = individual_data
                request.status = "completed"
                
            elif request.request_type == "deletion":
                # Delete individual's personal data
                deletion_result = self.data_processor.delete_individual_data(request.individual_id)
                request.response_data = deletion_result
                request.status = "completed"
                
            elif request.request_type == "correction":
                # Data correction would require specific implementation
                request.status = "manual_review_required"
                request.response_data = {
                    "message": "Data correction requires manual review",
                    "contact": "privacy@healing-guard.com"
                }
                
            elif request.request_type == "portability":
                # Data portability
                individual_data = self.data_processor.get_individual_data(request.individual_id)
                # In production, would generate portable format (JSON, CSV, etc.)
                request.response_data = {
                    "export_format": "json",
                    "data": individual_data,
                    "generated_at": datetime.now().isoformat()
                }
                request.status = "completed"
            
            request.completed_at = datetime.now()
            self.data_processor.access_requests[request.request_id] = request
            
            logger.info(f"PDPA request {request.request_id} completed successfully")
            
            return {
                "request_id": request.request_id,
                "status": request.status,
                "jurisdiction": request.jurisdiction.value,
                "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                "response_data": request.response_data
            }
            
        except Exception as e:
            request.status = "failed"
            request.response_data = {"error": str(e)}
            request.completed_at = datetime.now()
            
            logger.error(f"PDPA request {request.request_id} failed: {e}")
            
            return {
                "request_id": request.request_id,
                "status": "failed",
                "error": str(e)
            }
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """Run comprehensive PDPA compliance check."""
        check_results = {
            "timestamp": datetime.now().isoformat(),
            "jurisdiction": self.jurisdiction.value,
            "compliance_framework": f"PDPA ({self.jurisdiction.value.title()})",
            "checks": []
        }
        
        # Check 1: Consent management
        total_consents = len(self.data_processor.consent_records)
        active_consents = sum(
            1 for consent in self.data_processor.consent_records.values()
            if consent.consent_given
        )
        
        check_results["checks"].append({
            "name": "PDPA Consent Management",
            "status": "PASS" if active_consents > 0 else "WARNING",
            "details": f"{active_consents} active consents out of {total_consents} total",
            "regulation": f"PDPA {self.jurisdiction.value.title()} Section 13",
            "action_required": False
        })
        
        # Check 2: Data retention compliance
        expired_records = 0
        total_records = len(self.data_processor.processing_records)
        
        for record in self.data_processor.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            if datetime.now() > record.processing_timestamp + retention_period:
                expired_records += 1
        
        check_results["checks"].append({
            "name": "PDPA Data Retention Compliance",
            "status": "PASS" if expired_records == 0 else "FAIL",
            "details": f"{expired_records} expired records found out of {total_records} total",
            "regulation": f"PDPA {self.jurisdiction.value.title()} Section 25",
            "action_required": expired_records > 0
        })
        
        # Check 3: Access request response times
        pending_requests = len([
            req for req in self.data_processor.access_requests.values()
            if req.status == "pending"
        ])
        
        overdue_requests = 0
        response_timeframe = 30  # 30 days for Singapore PDPA
        if self.jurisdiction == PDPAJurisdiction.THAILAND:
            response_timeframe = 30  # Also 30 days for Thailand PDPA
        
        for req in self.data_processor.access_requests.values():
            if req.status == "pending":
                if datetime.now() > req.requested_at + timedelta(days=response_timeframe):
                    overdue_requests += 1
        
        check_results["checks"].append({
            "name": "Access Request Response Times",
            "status": "PASS" if overdue_requests == 0 else "FAIL",
            "details": f"{overdue_requests} overdue requests out of {pending_requests} pending",
            "regulation": f"PDPA {self.jurisdiction.value.title()} Section 21",
            "action_required": overdue_requests > 0
        })
        
        # Check 4: Cross-border transfer compliance
        cross_border_records = sum(
            1 for record in self.data_processor.processing_records
            if record.cross_border_transfer
        )
        
        check_results["checks"].append({
            "name": "Cross-Border Transfer Compliance",
            "status": "REVIEW" if cross_border_records > 0 else "PASS",
            "details": f"{cross_border_records} records involve cross-border transfers",
            "regulation": f"PDPA {self.jurisdiction.value.title()} Section 26",
            "action_required": cross_border_records > 0
        })
        
        # Overall compliance status
        failed_checks = sum(1 for check in check_results["checks"] if check["status"] == "FAIL")
        check_results["overall_status"] = "COMPLIANT" if failed_checks == 0 else "NON_COMPLIANT"
        check_results["failed_checks"] = failed_checks
        
        self.compliance_checks.append(check_results)
        
        logger.info(f"PDPA compliance check completed: {check_results['overall_status']}")
        
        return check_results
    
    def generate_privacy_policy_section(self) -> str:
        """Generate PDPA-specific section for privacy policy."""
        
        policy_section = f"""
PERSONAL DATA PROTECTION ACT ({self.jurisdiction.value.upper()}) PRIVACY RIGHTS

This section applies to individuals in {self.jurisdiction.value.title()} and describes your rights under the Personal Data Protection Act.

PERSONAL DATA WE COLLECT
We collect and process personal data for the following purposes:
• Service provision and customer support
• System administration and security monitoring
• Business improvement and research & development
• Legal compliance requirements

CONSENT AND LEGAL BASIS
We process your personal data based on:
• Your consent for specific purposes
• Legal obligations and compliance requirements
• Legitimate interests for system security and improvement

YOUR PDPA RIGHTS
You have the right to:

1. Access: Request access to your personal data
2. Correction: Request correction of inaccurate personal data
3. Deletion: Request deletion of your personal data
4. Portability: Request your data in a portable format
5. Withdraw Consent: Withdraw previously given consent

HOW TO EXERCISE YOUR RIGHTS
• Email: privacy@healing-guard.com
• Subject Line: "PDPA Request - [Request Type]"
• Include: Full name, contact details, specific request

DATA RETENTION
We retain personal data for the following periods:
• Personal data: {self.data_processor.retention_periods[self.jurisdiction][PDPADataType.PERSONAL_DATA]} days
• Sensitive data: {self.data_processor.retention_periods[self.jurisdiction][PDPADataType.SENSITIVE_DATA]} days
• Business contact data: {self.data_processor.retention_periods[self.jurisdiction][PDPADataType.BUSINESS_CONTACT_DATA]} days

RESPONSE TIMEFRAMES
We will respond to your requests within 30 days of verification.

DATA PROTECTION OFFICER
For privacy-related inquiries, contact our Data Protection Officer:
Email: dpo@healing-guard.com

Last Updated: {datetime.now().strftime('%B %d, %Y')}
"""
        return policy_section.strip()


class SentimentAnalysisPDPAWrapper:
    """PDPA-compliant wrapper for sentiment analysis operations."""
    
    def __init__(self, pdpa_processor: PDPADataProcessor):
        self.pdpa_processor = pdpa_processor
    
    async def analyze_with_pdpa_compliance(
        self,
        text: str,
        individual_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        jurisdiction: Optional[PDPAJurisdiction] = None
    ) -> Dict[str, Any]:
        """Perform sentiment analysis with PDPA compliance checks."""
        
        jurisdiction = jurisdiction or self.pdpa_processor.default_jurisdiction
        
        # Determine data types and purposes
        data_types = [PDPADataType.PERSONAL_DATA]
        if context and any(key in context for key in ['health', 'religion', 'politics']):
            data_types.append(PDPADataType.SENSITIVE_DATA)
        
        purposes = [
            PDPAPurpose.SERVICE_PROVISION,
            PDPAPurpose.BUSINESS_IMPROVEMENT,
            PDPAPurpose.SYSTEM_ADMINISTRATION
        ]
        
        # Check if processing is allowed
        if individual_id:
            processing_check = self.pdpa_processor.can_process_data(
                individual_id=individual_id,
                data_types=data_types,
                purposes=purposes,
                jurisdiction=jurisdiction
            )
            
            if not processing_check["processing_allowed"]:
                return {
                    "error": "PDPA_PROCESSING_NOT_ALLOWED",
                    "message": "Data processing not allowed under current consent status",
                    "jurisdiction": jurisdiction.value,
                    "consent_status": processing_check["consent_status"],
                    "individual_rights": {
                        "request_access": "Contact privacy@healing-guard.com",
                        "withdraw_consent": "Contact dpo@healing-guard.com",
                        "request_deletion": "Subject: PDPA Request - Deletion"
                    }
                }
        
        # Record processing activity
        processing_id = self.pdpa_processor.record_processing_activity(
            individual_id=individual_id,
            data_types=data_types,
            purposes=purposes,
            jurisdiction=jurisdiction,
            cross_border_transfer=False,  # Sentiment analysis processed locally
            third_party_disclosure=False
        )
        
        # In real implementation, would call actual sentiment analyzer here
        response = {
            "sentiment_analysis": {
                "processing_id": processing_id,
                "jurisdiction": jurisdiction.value,
                "data_types_processed": [dt.value for dt in data_types],
                "processing_purposes": [purpose.value for purpose in purposes],
                "retention_period_days": max(
                    self.pdpa_processor.retention_periods[jurisdiction].get(dt, 365)
                    for dt in data_types
                )
            },
            "pdpa_notice": {
                "jurisdiction": jurisdiction.value,
                "your_rights": [
                    "Access your personal data",
                    "Correct inaccurate personal data",
                    "Delete your personal data",
                    "Data portability",
                    "Withdraw consent"
                ],
                "contact_for_rights": "privacy@healing-guard.com",
                "data_protection_officer": "dpo@healing-guard.com",
                "privacy_policy": "/privacy-policy"
            }
        }
        
        return response


# Global PDPA compliance instances
pdpa_processor_singapore = PDPADataProcessor(PDPAJurisdiction.SINGAPORE)
pdpa_processor_thailand = PDPADataProcessor(PDPAJurisdiction.THAILAND)

pdpa_compliance_singapore = PDPACompliance(PDPAJurisdiction.SINGAPORE)
pdpa_compliance_thailand = PDPACompliance(PDPAJurisdiction.THAILAND)

pdpa_sentiment_wrapper_singapore = SentimentAnalysisPDPAWrapper(pdpa_processor_singapore)
pdpa_sentiment_wrapper_thailand = SentimentAnalysisPDPAWrapper(pdpa_processor_thailand)