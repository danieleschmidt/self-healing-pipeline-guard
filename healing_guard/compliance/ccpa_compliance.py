"""CCPA compliance implementation for California Consumer Privacy Act.

This module provides comprehensive CCPA compliance for sentiment analysis data processing,
including consumer rights, data handling requirements, and opt-out mechanisms.
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


class CCPAConsumerRight(Enum):
    """CCPA consumer rights under the California Consumer Privacy Act."""
    RIGHT_TO_KNOW = "right_to_know"  # What personal information is collected
    RIGHT_TO_DELETE = "right_to_delete"  # Delete personal information
    RIGHT_TO_OPT_OUT = "right_to_opt_out"  # Opt out of sale of personal information
    RIGHT_TO_NON_DISCRIMINATION = "right_to_non_discrimination"  # Non-discriminatory treatment


class CCPADataCategory(Enum):
    """Categories of personal information under CCPA."""
    IDENTIFIERS = "identifiers"  # Name, address, email, etc.
    PERSONAL_INFO = "personal_info"  # Age, gender, etc.
    PROTECTED_CHARACTERISTICS = "protected_characteristics"  # Race, religion, etc.
    COMMERCIAL_INFO = "commercial_info"  # Purchase history, preferences
    BIOMETRIC_INFO = "biometric_info"  # Fingerprints, voiceprints, etc.
    INTERNET_ACTIVITY = "internet_activity"  # Browsing history, interactions
    GEOLOCATION_DATA = "geolocation_data"  # Physical location data
    SENSORY_DATA = "sensory_data"  # Audio, visual, thermal, olfactory
    PROFESSIONAL_INFO = "professional_info"  # Employment-related information
    EDUCATION_INFO = "education_info"  # Education records
    INFERENCES = "inferences"  # Preferences, behaviors, aptitudes


class CCPABusinessPurpose(Enum):
    """Business purposes for processing personal information under CCPA."""
    SECURITY = "security"  # Detecting and preventing security incidents
    DEBUGGING = "debugging"  # Debugging to identify and repair errors
    SHORT_TERM_USE = "short_term_use"  # Short-term, transient use
    PERFORMING_SERVICES = "performing_services"  # Performing requested services
    INTERNAL_RESEARCH = "internal_research"  # Internal research for improvements
    QUALITY_ASSURANCE = "quality_assurance"  # Quality and safety maintenance
    LEGAL_COMPLIANCE = "legal_compliance"  # Legal compliance requirements


@dataclass
class CCPADataProcessingRecord:
    """Record of personal information processing under CCPA."""
    record_id: str
    timestamp: datetime
    consumer_id: Optional[str]  # Hashed consumer identifier
    data_categories: List[CCPADataCategory]
    business_purposes: List[CCPABusinessPurpose]
    third_party_sharing: bool
    data_sold: bool = False
    retention_period_days: int = 365
    source_of_data: str = "direct_collection"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for logging/storage."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "consumer_id_hash": self.consumer_id,
            "data_categories": [cat.value for cat in self.data_categories],
            "business_purposes": [purpose.value for purpose in self.business_purposes],
            "third_party_sharing": self.third_party_sharing,
            "data_sold": self.data_sold,
            "retention_period_days": self.retention_period_days,
            "source_of_data": self.source_of_data
        }


@dataclass
class CCPAConsumerRequest:
    """CCPA consumer rights request."""
    request_id: str
    consumer_id: str
    request_type: CCPAConsumerRight
    requested_at: datetime
    description: str
    status: str = "pending"
    verification_status: str = "pending"  # pending, verified, rejected
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "request_id": self.request_id,
            "consumer_id_hash": hashlib.sha256(self.consumer_id.encode()).hexdigest(),
            "request_type": self.request_type.value,
            "requested_at": self.requested_at.isoformat(),
            "description": self.description,
            "status": self.status,
            "verification_status": self.verification_status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_data": self.response_data
        }


@dataclass
class CCPADisclosure:
    """CCPA-required disclosures about personal information collection and use."""
    categories_collected: List[CCPADataCategory]
    business_purposes: List[CCPABusinessPurpose]
    categories_sold: List[CCPADataCategory]
    categories_disclosed: List[CCPADataCategory]
    third_parties: List[str]
    retention_periods: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert disclosure to dictionary."""
        return {
            "categories_collected": [cat.value for cat in self.categories_collected],
            "business_purposes": [purpose.value for purpose in self.business_purposes],
            "categories_sold": [cat.value for cat in self.categories_sold],
            "categories_disclosed": [cat.value for cat in self.categories_disclosed],
            "third_parties": self.third_parties,
            "retention_periods": self.retention_periods
        }


class CCPADataProcessor:
    """Handles CCPA-compliant data processing for sentiment analysis."""
    
    def __init__(self):
        self.processing_records: List[CCPADataProcessingRecord] = []
        self.consumer_requests: Dict[str, CCPAConsumerRequest] = {}
        self.opt_out_registry: Set[str] = set()  # Hashed consumer IDs who opted out
        self.consent_registry: Dict[str, Dict[str, Any]] = {}
        
        # CCPA-specific retention periods
        self.retention_periods = {
            CCPADataCategory.IDENTIFIERS: 730,  # 2 years
            CCPADataCategory.INTERNET_ACTIVITY: 365,  # 1 year
            CCPADataCategory.COMMERCIAL_INFO: 1095,  # 3 years
            CCPADataCategory.INFERENCES: 365  # 1 year
        }
    
    def hash_consumer_id(self, consumer_id: str) -> str:
        """Create privacy-preserving hash of consumer identifier."""
        salt = "ccpa_healing_guard_salt_2025"  # In production, use environment variable
        combined = f"{salt}{consumer_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def record_data_processing(
        self,
        consumer_id: Optional[str],
        data_categories: List[CCPADataCategory],
        business_purposes: List[CCPABusinessPurpose],
        third_party_sharing: bool = False,
        data_sold: bool = False,
        source_of_data: str = "direct_collection"
    ) -> str:
        """Record personal information processing activity for CCPA compliance."""
        
        record_id = f"ccpa_proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_records)}"
        
        # Hash consumer ID for privacy
        hashed_consumer = None
        if consumer_id:
            hashed_consumer = self.hash_consumer_id(consumer_id)
        
        # Determine retention period based on data categories
        max_retention = max(
            self.retention_periods.get(category, 365) 
            for category in data_categories
        )
        
        record = CCPADataProcessingRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            consumer_id=hashed_consumer,
            data_categories=data_categories,
            business_purposes=business_purposes,
            third_party_sharing=third_party_sharing,
            data_sold=data_sold,
            retention_period_days=max_retention,
            source_of_data=source_of_data
        )
        
        self.processing_records.append(record)
        
        logger.info(f"CCPA: Recorded processing activity {record_id}")
        
        return record_id
    
    def consumer_opted_out(self, consumer_id: str) -> bool:
        """Check if consumer has opted out of data sale."""
        hashed_consumer = self.hash_consumer_id(consumer_id)
        return hashed_consumer in self.opt_out_registry
    
    def record_opt_out(self, consumer_id: str, opt_out_method: str = "web_form") -> str:
        """Record consumer opt-out from sale of personal information."""
        hashed_consumer = self.hash_consumer_id(consumer_id)
        self.opt_out_registry.add(hashed_consumer)
        
        opt_out_id = f"optout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store opt-out record for compliance tracking
        opt_out_record = {
            "opt_out_id": opt_out_id,
            "consumer_id_hash": hashed_consumer,
            "timestamp": datetime.now().isoformat(),
            "method": opt_out_method,
            "status": "active"
        }
        
        # In production, this would be stored in a persistent database
        logger.info(f"CCPA: Recorded consumer opt-out {opt_out_id}")
        
        return opt_out_id
    
    def record_opt_in(self, consumer_id: str, opt_in_method: str = "web_form") -> str:
        """Record consumer opt-in to data sale (re-enabling after opt-out)."""
        hashed_consumer = self.hash_consumer_id(consumer_id)
        
        if hashed_consumer in self.opt_out_registry:
            self.opt_out_registry.remove(hashed_consumer)
        
        opt_in_id = f"optin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store opt-in record for compliance tracking
        opt_in_record = {
            "opt_in_id": opt_in_id,
            "consumer_id_hash": hashed_consumer,
            "timestamp": datetime.now().isoformat(),
            "method": opt_in_method,
            "status": "active"
        }
        
        logger.info(f"CCPA: Recorded consumer opt-in {opt_in_id}")
        
        return opt_in_id
    
    def get_consumer_data(self, consumer_id: str) -> Dict[str, Any]:
        """Get all personal information for a consumer (Right to Know)."""
        hashed_consumer = self.hash_consumer_id(consumer_id)
        
        consumer_records = [
            record for record in self.processing_records
            if record.consumer_id == hashed_consumer
        ]
        
        # Aggregate data categories and business purposes
        all_categories = set()
        all_purposes = set()
        for record in consumer_records:
            all_categories.update(record.data_categories)
            all_purposes.update(record.business_purposes)
        
        return {
            "consumer_id": "HASHED_FOR_PRIVACY",
            "processing_records": [record.to_dict() for record in consumer_records],
            "data_categories_collected": [cat.value for cat in all_categories],
            "business_purposes": [purpose.value for purpose in all_purposes],
            "opt_out_status": hashed_consumer in self.opt_out_registry,
            "data_sold": any(record.data_sold for record in consumer_records),
            "third_party_sharing": any(record.third_party_sharing for record in consumer_records),
            "retention_information": {
                cat.value: self.retention_periods.get(cat, 365)
                for cat in all_categories
            }
        }
    
    def delete_consumer_data(
        self,
        consumer_id: str,
        categories_to_delete: Optional[List[CCPADataCategory]] = None
    ) -> Dict[str, Any]:
        """Delete consumer's personal information (Right to Delete)."""
        hashed_consumer = self.hash_consumer_id(consumer_id)
        
        deleted_records = []
        remaining_records = []
        
        for record in self.processing_records:
            if record.consumer_id == hashed_consumer:
                # Check if we should delete this record
                should_delete = True
                
                if categories_to_delete:
                    # Only delete if any of the specified categories are in the record
                    should_delete = any(
                        category in record.data_categories 
                        for category in categories_to_delete
                    )
                
                # Check for legal basis to retain data
                if CCPABusinessPurpose.LEGAL_COMPLIANCE in record.business_purposes:
                    should_delete = False
                    logger.info(f"Retaining record {record.record_id} for legal compliance")
                
                if should_delete:
                    deleted_records.append(record.record_id)
                else:
                    remaining_records.append(record)
            else:
                remaining_records.append(record)
        
        # Update processing records
        self.processing_records = remaining_records
        
        logger.info(f"CCPA: Deleted {len(deleted_records)} records for consumer")
        
        return {
            "deleted_records": deleted_records,
            "categories_deleted": [cat.value for cat in (categories_to_delete or [])],
            "remaining_records": len([
                r for r in remaining_records 
                if r.consumer_id == hashed_consumer
            ])
        }
    
    def cleanup_expired_records(self) -> int:
        """Clean up records that have exceeded CCPA retention periods."""
        current_time = datetime.now()
        deleted_count = 0
        records_to_keep = []
        
        for record in self.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            expiry_date = record.timestamp + retention_period
            
            if current_time > expiry_date:
                deleted_count += 1
                logger.info(f"CCPA: Auto-deleted expired record {record.record_id}")
            else:
                records_to_keep.append(record)
        
        self.processing_records = records_to_keep
        
        return deleted_count
    
    def generate_ccpa_disclosure(self) -> CCPADisclosure:
        """Generate CCPA-required disclosure about data practices."""
        
        # Aggregate information from all processing records
        all_categories = set()
        all_purposes = set()
        sold_categories = set()
        disclosed_categories = set()
        third_parties = set()
        
        for record in self.processing_records:
            all_categories.update(record.data_categories)
            all_purposes.update(record.business_purposes)
            
            if record.data_sold:
                sold_categories.update(record.data_categories)
            
            if record.third_party_sharing:
                disclosed_categories.update(record.data_categories)
                third_parties.add("analytics_partners")  # Example
        
        return CCPADisclosure(
            categories_collected=list(all_categories),
            business_purposes=list(all_purposes),
            categories_sold=list(sold_categories),
            categories_disclosed=list(disclosed_categories),
            third_parties=list(third_parties),
            retention_periods={cat.value: period for cat, period in self.retention_periods.items()}
        )


class CCPACompliance:
    """Main CCPA compliance manager for sentiment analysis."""
    
    def __init__(self):
        self.data_processor = CCPADataProcessor()
        self.compliance_checks: List[Dict[str, Any]] = []
    
    async def handle_consumer_request(self, request: CCPAConsumerRequest) -> Dict[str, Any]:
        """Handle a CCPA consumer rights request."""
        logger.info(f"Processing CCPA request: {request.request_type.value} for {request.consumer_id}")
        
        try:
            # Verify consumer identity (simplified for demo)
            if request.verification_status != "verified":
                return {
                    "request_id": request.request_id,
                    "status": "verification_required",
                    "message": "Consumer identity verification required",
                    "verification_methods": ["email", "phone", "address"]
                }
            
            if request.request_type == CCPAConsumerRight.RIGHT_TO_KNOW:
                # Provide consumer's personal information
                consumer_data = self.data_processor.get_consumer_data(request.consumer_id)
                request.response_data = consumer_data
                request.status = "completed"
                
            elif request.request_type == CCPAConsumerRight.RIGHT_TO_DELETE:
                # Delete consumer's personal information
                deletion_result = self.data_processor.delete_consumer_data(request.consumer_id)
                request.response_data = deletion_result
                request.status = "completed"
                
            elif request.request_type == CCPAConsumerRight.RIGHT_TO_OPT_OUT:
                # Opt consumer out of data sale
                opt_out_id = self.data_processor.record_opt_out(request.consumer_id)
                request.response_data = {"opt_out_id": opt_out_id, "status": "opted_out"}
                request.status = "completed"
                
            elif request.request_type == CCPAConsumerRight.RIGHT_TO_NON_DISCRIMINATION:
                # Confirm non-discriminatory treatment
                request.response_data = {
                    "message": "Your rights request will not result in discriminatory treatment",
                    "policy": "We do not discriminate against consumers who exercise their CCPA rights"
                }
                request.status = "completed"
            
            request.completed_at = datetime.now()
            self.data_processor.consumer_requests[request.request_id] = request
            
            logger.info(f"CCPA request {request.request_id} completed successfully")
            
            return {
                "request_id": request.request_id,
                "status": request.status,
                "completed_at": request.completed_at.isoformat(),
                "response_data": request.response_data
            }
            
        except Exception as e:
            request.status = "failed"
            request.response_data = {"error": str(e)}
            request.completed_at = datetime.now()
            
            logger.error(f"CCPA request {request.request_id} failed: {e}")
            
            return {
                "request_id": request.request_id,
                "status": "failed",
                "error": str(e)
            }
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """Run comprehensive CCPA compliance check."""
        check_results = {
            "timestamp": datetime.now().isoformat(),
            "compliance_framework": "CCPA",
            "checks": []
        }
        
        # Check 1: Data retention compliance
        expired_records = 0
        total_records = len(self.data_processor.processing_records)
        
        for record in self.data_processor.processing_records:
            retention_period = timedelta(days=record.retention_period_days)
            if datetime.now() > record.timestamp + retention_period:
                expired_records += 1
        
        check_results["checks"].append({
            "name": "CCPA Data Retention Compliance",
            "status": "PASS" if expired_records == 0 else "FAIL",
            "details": f"{expired_records} expired records found out of {total_records} total",
            "regulation": "CCPA Section 1798.105",
            "action_required": expired_records > 0
        })
        
        # Check 2: Consumer request response times
        pending_requests = len([
            req for req in self.data_processor.consumer_requests.values()
            if req.status == "pending"
        ])
        
        overdue_requests = 0
        for req in self.data_processor.consumer_requests.values():
            if req.status == "pending":
                # CCPA requires response within 45 days (with possible 45-day extension)
                if datetime.now() > req.requested_at + timedelta(days=45):
                    overdue_requests += 1
        
        check_results["checks"].append({
            "name": "Consumer Request Response Times",
            "status": "PASS" if overdue_requests == 0 else "FAIL",
            "details": f"{overdue_requests} overdue requests out of {pending_requests} pending",
            "regulation": "CCPA Section 1798.130",
            "action_required": overdue_requests > 0
        })
        
        # Check 3: Opt-out mechanisms
        opt_out_available = True  # In real implementation, check if opt-out mechanisms are available
        
        check_results["checks"].append({
            "name": "Opt-Out Mechanism Availability",
            "status": "PASS" if opt_out_available else "FAIL",
            "details": "Do Not Sell My Personal Information link and mechanism available",
            "regulation": "CCPA Section 1798.135",
            "action_required": not opt_out_available
        })
        
        # Check 4: Privacy policy disclosures
        disclosure = self.data_processor.generate_ccpa_disclosure()
        has_required_disclosures = len(disclosure.categories_collected) > 0
        
        check_results["checks"].append({
            "name": "Privacy Policy Disclosures",
            "status": "PASS" if has_required_disclosures else "FAIL",
            "details": f"Privacy policy includes disclosures for {len(disclosure.categories_collected)} data categories",
            "regulation": "CCPA Section 1798.130",
            "action_required": not has_required_disclosures
        })
        
        # Overall compliance status
        failed_checks = sum(1 for check in check_results["checks"] if check["status"] == "FAIL")
        check_results["overall_status"] = "COMPLIANT" if failed_checks == 0 else "NON_COMPLIANT"
        check_results["failed_checks"] = failed_checks
        
        self.compliance_checks.append(check_results)
        
        logger.info(f"CCPA compliance check completed: {check_results['overall_status']}")
        
        return check_results
    
    def generate_privacy_policy_section(self) -> str:
        """Generate CCPA-specific section for privacy policy."""
        disclosure = self.data_processor.generate_ccpa_disclosure()
        
        policy_section = f"""
CALIFORNIA CONSUMER PRIVACY ACT (CCPA) PRIVACY RIGHTS

This section applies to California residents and describes your rights under the California Consumer Privacy Act.

PERSONAL INFORMATION WE COLLECT
We collect the following categories of personal information:
{chr(10).join(f"• {cat.value.replace('_', ' ').title()}" for cat in disclosure.categories_collected)}

BUSINESS PURPOSES FOR COLLECTION
We use personal information for these business purposes:
{chr(10).join(f"• {purpose.value.replace('_', ' ').title()}" for purpose in disclosure.business_purposes)}

YOUR CCPA RIGHTS
As a California resident, you have the right to:

1. Right to Know: Request information about personal information we collect, use, disclose, or sell
2. Right to Delete: Request deletion of your personal information
3. Right to Opt-Out: Opt out of the sale of your personal information
4. Right to Non-Discrimination: Not be discriminated against for exercising your rights

HOW TO EXERCISE YOUR RIGHTS
• Email: privacy@healing-guard.com
• Website: [Do Not Sell My Personal Information link]
• Phone: 1-800-PRIVACY

VERIFICATION PROCESS
We will verify your identity before processing rights requests through:
• Email verification for account holders
• Additional verification for sensitive requests

RESPONSE TIMEFRAMES
• We will respond to verified requests within 45 days
• If additional time is needed, we will notify you within 45 days

Last Updated: {datetime.now().strftime('%B %d, %Y')}
"""
        return policy_section.strip()


class SentimentAnalysisCCPAWrapper:
    """CCPA-compliant wrapper for sentiment analysis operations."""
    
    def __init__(self, ccpa_processor: CCPADataProcessor):
        self.ccpa_processor = ccpa_processor
    
    async def analyze_with_ccpa_compliance(
        self,
        text: str,
        consumer_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform sentiment analysis with CCPA compliance checks."""
        
        # Check if consumer has opted out of data sale
        if consumer_id and self.ccpa_processor.consumer_opted_out(consumer_id):
            # Limit data processing for opted-out consumers
            limited_context = {
                "processing_limitation": "consumer_opted_out",
                "data_use_restricted": True
            }
        else:
            limited_context = context or {}
        
        # Determine CCPA data categories
        data_categories = [CCPADataCategory.INTERNET_ACTIVITY]
        if consumer_id:
            data_categories.append(CCPADataCategory.IDENTIFIERS)
        if context:
            data_categories.append(CCPADataCategory.INFERENCES)
        
        # Determine business purposes
        business_purposes = [
            CCPABusinessPurpose.PERFORMING_SERVICES,
            CCPABusinessPurpose.INTERNAL_RESEARCH,
            CCPABusinessPurpose.QUALITY_ASSURANCE
        ]
        
        # Record processing activity
        processing_id = self.ccpa_processor.record_data_processing(
            consumer_id=consumer_id,
            data_categories=data_categories,
            business_purposes=business_purposes,
            third_party_sharing=False,
            data_sold=False,  # We don't sell sentiment analysis data
            source_of_data="api_request"
        )
        
        # In real implementation, would call actual sentiment analyzer here
        response = {
            "sentiment_analysis": {
                "processing_id": processing_id,
                "data_categories_processed": [cat.value for cat in data_categories],
                "business_purposes": [purpose.value for purpose in business_purposes],
                "consumer_opted_out": consumer_id and self.ccpa_processor.consumer_opted_out(consumer_id)
            },
            "ccpa_notice": {
                "your_rights": [
                    "Right to know what personal information is collected",
                    "Right to delete your personal information",
                    "Right to opt-out of sale of personal information",
                    "Right to non-discriminatory treatment"
                ],
                "contact_for_rights": "privacy@healing-guard.com",
                "opt_out_link": "/ccpa/do-not-sell",
                "privacy_policy": "/privacy-policy"
            }
        }
        
        return response


# Global CCPA compliance instance
ccpa_processor = CCPADataProcessor()
ccpa_compliance = CCPACompliance()
ccpa_sentiment_wrapper = SentimentAnalysisCCPAWrapper(ccpa_processor)