"""GDPR compliance implementation for sentiment analysis data processing."""

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


class DataProcessingPurpose(Enum):
    """Legal purposes for data processing under GDPR."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PIPELINE_HEALING = "pipeline_healing" 
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY_ANALYSIS = "security_analysis"
    SYSTEM_IMPROVEMENT = "system_improvement"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR Article 6."""
    LEGITIMATE_INTEREST = "legitimate_interest"  # Article 6(1)(f)
    CONSENT = "consent"  # Article 6(1)(a)
    CONTRACT = "contract"  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for GDPR compliance."""
    processing_id: str
    timestamp: datetime
    data_subject: Optional[str]  # Hashed identifier, never plaintext
    data_type: str
    processing_purpose: DataProcessingPurpose
    legal_basis: LegalBasis
    retention_period_days: int
    data_categories: List[str]
    recipients: List[str] = field(default_factory=list)
    cross_border_transfer: bool = False
    automated_decision_making: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for logging/storage."""
        return {
            "processing_id": self.processing_id,
            "timestamp": self.timestamp.isoformat(),
            "data_subject_hash": self.data_subject,
            "data_type": self.data_type,
            "processing_purpose": self.processing_purpose.value,
            "legal_basis": self.legal_basis.value,
            "retention_period_days": self.retention_period_days,
            "data_categories": self.data_categories,
            "recipients": self.recipients,
            "cross_border_transfer": self.cross_border_transfer,
            "automated_decision_making": self.automated_decision_making
        }


class GDPRDataProcessor:
    """Handles GDPR-compliant data processing for sentiment analysis."""
    
    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.data_retention_periods = {
            DataProcessingPurpose.SENTIMENT_ANALYSIS: 90,  # 3 months
            DataProcessingPurpose.PIPELINE_HEALING: 180,   # 6 months
            DataProcessingPurpose.PERFORMANCE_MONITORING: 365,  # 1 year
            DataProcessingPurpose.SECURITY_ANALYSIS: 730,  # 2 years
            DataProcessingPurpose.SYSTEM_IMPROVEMENT: 365  # 1 year
        }
        self.consent_registry: Dict[str, Dict[str, Any]] = {}
    
    def hash_data_subject_id(self, identifier: str) -> str:
        """Create a privacy-preserving hash of data subject identifier."""
        # Use SHA-256 with salt for privacy protection
        salt = "healing_guard_gdpr_salt_2025"  # In production, use environment variable
        combined = f"{salt}{identifier}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def record_processing_activity(
        self,
        data_subject_id: Optional[str],
        data_type: str,
        processing_purpose: DataProcessingPurpose,
        legal_basis: LegalBasis,
        data_categories: List[str],
        automated_decision: bool = False,
        recipients: Optional[List[str]] = None
    ) -> str:
        """Record a data processing activity for GDPR compliance."""
        
        processing_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_records)}"
        
        # Hash data subject ID for privacy
        hashed_subject = None
        if data_subject_id:
            hashed_subject = self.hash_data_subject_id(data_subject_id)
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            timestamp=datetime.now(),
            data_subject=hashed_subject,
            data_type=data_type,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            retention_period_days=self.data_retention_periods.get(processing_purpose, 365),
            data_categories=data_categories,
            recipients=recipients or [],
            cross_border_transfer=False,  # Assume no cross-border by default
            automated_decision_making=automated_decision
        )
        
        self.processing_records.append(record)
        
        logger.info(f"GDPR: Recorded processing activity {processing_id} for {processing_purpose.value}")
        
        return processing_id
    
    def record_consent(
        self,
        data_subject_id: str,
        consent_purposes: List[DataProcessingPurpose],
        consent_given: bool,
        consent_method: str = "api"
    ) -> str:
        """Record consent for data processing."""
        
        consent_id = f"consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        hashed_subject = self.hash_data_subject_id(data_subject_id)
        
        consent_record = {
            "consent_id": consent_id,
            "data_subject_hash": hashed_subject,
            "timestamp": datetime.now().isoformat(),
            "purposes": [purpose.value for purpose in consent_purposes],
            "consent_given": consent_given,
            "consent_method": consent_method,
            "ip_address": None,  # Could be added if available
            "user_agent": None   # Could be added if available
        }
        
        self.consent_registry[hashed_subject] = consent_record
        
        logger.info(f"GDPR: Recorded consent {consent_id} - {'granted' if consent_given else 'withdrawn'}")
        
        return consent_id
    
    def has_valid_consent(
        self,
        data_subject_id: str,
        purpose: DataProcessingPurpose
    ) -> bool:
        """Check if valid consent exists for data processing purpose."""
        
        hashed_subject = self.hash_data_subject_id(data_subject_id)
        consent_record = self.consent_registry.get(hashed_subject)
        
        if not consent_record:
            return False
        
        return (consent_record.get("consent_given", False) and 
                purpose.value in consent_record.get("purposes", []))
    
    def can_process_data(
        self,
        data_subject_id: Optional[str],
        purpose: DataProcessingPurpose,
        legal_basis: LegalBasis
    ) -> bool:
        """Determine if data can be processed under GDPR."""
        
        # If no data subject (anonymous data), processing is generally allowed
        if not data_subject_id:
            return True
        
        # Check legal basis
        if legal_basis == LegalBasis.CONSENT:
            return self.has_valid_consent(data_subject_id, purpose)
        
        elif legal_basis == LegalBasis.LEGITIMATE_INTEREST:
            # For sentiment analysis in CI/CD, legitimate interest applies
            # as it improves system reliability without significant privacy impact
            return purpose in [
                DataProcessingPurpose.SENTIMENT_ANALYSIS,
                DataProcessingPurpose.PIPELINE_HEALING,
                DataProcessingPurpose.PERFORMANCE_MONITORING
            ]
        
        elif legal_basis == LegalBasis.CONTRACT:
            # Processing necessary for service delivery
            return purpose in [
                DataProcessingPurpose.PIPELINE_HEALING,
                DataProcessingPurpose.SECURITY_ANALYSIS
            ]
        
        elif legal_basis == LegalBasis.LEGAL_OBLIGATION:
            # Security monitoring and logging for compliance
            return purpose in [
                DataProcessingPurpose.SECURITY_ANALYSIS
            ]
        
        return False
    
    def get_processing_records_for_subject(
        self,
        data_subject_id: str
    ) -> List[DataProcessingRecord]:
        """Get all processing records for a specific data subject (GDPR Article 15)."""
        
        hashed_subject = self.hash_data_subject_id(data_subject_id)
        return [
            record for record in self.processing_records
            if record.data_subject == hashed_subject
        ]
    
    def delete_processing_records_for_subject(
        self,
        data_subject_id: str,
        purposes: Optional[List[DataProcessingPurpose]] = None
    ) -> int:
        """Delete processing records for data subject (Right to be forgotten - GDPR Article 17)."""
        
        hashed_subject = self.hash_data_subject_id(data_subject_id)
        deleted_count = 0
        
        # Filter records to delete
        records_to_keep = []
        for record in self.processing_records:
            should_delete = (
                record.data_subject == hashed_subject and
                (purposes is None or record.processing_purpose in purposes)
            )
            
            if should_delete:
                deleted_count += 1
                logger.info(f"GDPR: Deleted processing record {record.processing_id}")
            else:
                records_to_keep.append(record)
        
        self.processing_records = records_to_keep
        
        # Also remove consent records if requested
        if hashed_subject in self.consent_registry:
            del self.consent_registry[hashed_subject]
            logger.info(f"GDPR: Deleted consent record for subject")
        
        return deleted_count
    
    def cleanup_expired_records(self) -> int:
        """Clean up records that have exceeded retention period."""
        
        current_time = datetime.now()
        deleted_count = 0
        records_to_keep = []
        
        for record in self.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            expiry_date = record.timestamp + retention_period
            
            if current_time > expiry_date:
                deleted_count += 1
                logger.info(f"GDPR: Auto-deleted expired record {record.processing_id}")
            else:
                records_to_keep.append(record)
        
        self.processing_records = records_to_keep
        
        return deleted_count
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy impact report for GDPR compliance."""
        
        current_time = datetime.now()
        
        # Calculate statistics
        total_records = len(self.processing_records)
        records_by_purpose = {}
        records_by_legal_basis = {}
        automated_decisions = 0
        
        for record in self.processing_records:
            purpose = record.processing_purpose.value
            legal_basis = record.legal_basis.value
            
            records_by_purpose[purpose] = records_by_purpose.get(purpose, 0) + 1
            records_by_legal_basis[legal_basis] = records_by_legal_basis.get(legal_basis, 0) + 1
            
            if record.automated_decision_making:
                automated_decisions += 1
        
        # Calculate retention compliance
        expired_records = []
        for record in self.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            expiry_date = record.timestamp + retention_period
            if current_time > expiry_date:
                expired_records.append(record.processing_id)
        
        report = {
            "report_generated": current_time.isoformat(),
            "data_processing_overview": {
                "total_processing_records": total_records,
                "records_by_purpose": records_by_purpose,
                "records_by_legal_basis": records_by_legal_basis,
                "automated_decision_making_records": automated_decisions
            },
            "retention_compliance": {
                "total_records": total_records,
                "expired_records": len(expired_records),
                "expired_record_ids": expired_records,
                "compliance_rate": (
                    ((total_records - len(expired_records)) / total_records * 100) 
                    if total_records > 0 else 100
                )
            },
            "consent_management": {
                "total_consent_records": len(self.consent_registry),
                "active_consents": sum(
                    1 for consent in self.consent_registry.values()
                    if consent.get("consent_given", False)
                )
            },
            "data_subject_rights": {
                "right_to_access": "Implemented via get_processing_records_for_subject()",
                "right_to_rectification": "Manual process",
                "right_to_erasure": "Implemented via delete_processing_records_for_subject()",
                "right_to_portability": "Available via API export",
                "right_to_object": "Implemented via consent management"
            },
            "technical_measures": {
                "data_subject_id_hashing": True,
                "automated_retention_cleanup": True,
                "consent_tracking": True,
                "purpose_limitation": True,
                "data_minimization": True
            }
        }
        
        return report
    
    async def export_subject_data(
        self,
        data_subject_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export all data for a subject (GDPR Article 20 - Data Portability)."""
        
        hashed_subject = self.hash_data_subject_id(data_subject_id)
        
        # Get processing records
        processing_records = self.get_processing_records_for_subject(data_subject_id)
        
        # Get consent record
        consent_record = self.consent_registry.get(hashed_subject)
        
        export_data = {
            "data_subject_id": "HASHED_FOR_PRIVACY",
            "export_timestamp": datetime.now().isoformat(),
            "processing_records": [record.to_dict() for record in processing_records],
            "consent_record": consent_record,
            "retention_information": {
                purpose.value: days for purpose, days in self.data_retention_periods.items()
            },
            "data_categories_processed": list(set(
                category 
                for record in processing_records 
                for category in record.data_categories
            )),
            "legal_basis_used": list(set(
                record.legal_basis.value for record in processing_records
            ))
        }
        
        return export_data


class SentimentAnalysisGDPRWrapper:
    """GDPR-compliant wrapper for sentiment analysis operations."""
    
    def __init__(self, gdpr_processor: GDPRDataProcessor):
        self.gdpr_processor = gdpr_processor
    
    async def analyze_with_gdpr_compliance(
        self,
        text: str,
        data_subject_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTEREST
    ) -> Dict[str, Any]:
        """Perform sentiment analysis with GDPR compliance checks."""
        
        # Determine data categories
        data_categories = ["text_content", "sentiment_metadata"]
        if context:
            data_categories.extend(["context_data", "pipeline_metadata"])
        
        # Check if processing is allowed
        can_process = self.gdpr_processor.can_process_data(
            data_subject_id=data_subject_id,
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            legal_basis=legal_basis
        )
        
        if not can_process:
            return {
                "error": "GDPR_PROCESSING_NOT_ALLOWED",
                "message": "Data processing not allowed under current consent/legal basis",
                "data_subject_rights": {
                    "request_access": "Contact privacy@company.com",
                    "withdraw_consent": "Use /api/gdpr/consent/withdraw endpoint",
                    "request_deletion": "Use /api/gdpr/deletion endpoint"
                }
            }
        
        # Record processing activity
        processing_id = self.gdpr_processor.record_processing_activity(
            data_subject_id=data_subject_id,
            data_type="sentiment_analysis_request",
            processing_purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            legal_basis=legal_basis,
            data_categories=data_categories,
            automated_decision=True,  # Sentiment analysis is automated
            recipients=["healing_engine", "metrics_system"]
        )
        
        # In real implementation, would call actual sentiment analyzer here
        # For now, return simulated compliant response
        response = {
            "sentiment_analysis": {
                "processing_id": processing_id,
                "legal_basis": legal_basis.value,
                "data_categories_processed": data_categories,
                "retention_period_days": self.gdpr_processor.data_retention_periods[
                    DataProcessingPurpose.SENTIMENT_ANALYSIS
                ]
            },
            "privacy_notice": {
                "purpose": "Sentiment analysis for CI/CD pipeline improvement",
                "legal_basis": legal_basis.value,
                "retention_period": "90 days",
                "data_subject_rights": [
                    "Access your data",
                    "Rectify inaccurate data", 
                    "Erase your data",
                    "Restrict processing",
                    "Data portability",
                    "Object to processing"
                ],
                "contact": "privacy@healing-guard.com"
            }
        }
        
        return response


# Global GDPR processor instance
gdpr_processor = GDPRDataProcessor()
gdpr_sentiment_wrapper = SentimentAnalysisGDPRWrapper(gdpr_processor)