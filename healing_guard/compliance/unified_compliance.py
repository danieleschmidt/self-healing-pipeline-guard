"""Unified compliance manager integrating GDPR, CCPA, and PDPA frameworks.

This module provides a single interface for managing compliance across multiple
privacy regulations, automatically determining applicable frameworks based on
user location and preferences.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from .gdpr_compliance import (
    GDPRDataProcessor, 
    SentimentAnalysisGDPRWrapper,
    DataProcessingPurpose as GDPRPurpose,
    LegalBasis
)
from .ccpa_compliance import (
    CCPADataProcessor,
    SentimentAnalysisCCPAWrapper, 
    CCPADataCategory,
    CCPABusinessPurpose
)
from .pdpa_compliance import (
    PDPADataProcessor,
    SentimentAnalysisPDPAWrapper,
    PDPAJurisdiction,
    PDPAPurpose,
    PDPADataType
)

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California, USA
    PDPA_SINGAPORE = "pdpa_singapore"  # Singapore
    PDPA_THAILAND = "pdpa_thailand"  # Thailand


class UserLocation(Enum):
    """User location regions for compliance determination."""
    EU = "eu"
    CALIFORNIA = "california"
    SINGAPORE = "singapore"
    THAILAND = "thailand"
    OTHER = "other"


@dataclass
class ComplianceContext:
    """Context for determining applicable compliance frameworks."""
    user_location: Optional[UserLocation] = None
    ip_country: Optional[str] = None
    user_preference: Optional[ComplianceFramework] = None
    data_processing_location: str = "us"
    cross_border_transfer: bool = False
    
    def get_applicable_frameworks(self) -> List[ComplianceFramework]:
        """Determine which compliance frameworks apply."""
        frameworks = []
        
        # User preference takes precedence
        if self.user_preference:
            frameworks.append(self.user_preference)
        
        # Location-based determination
        if self.user_location == UserLocation.EU or self.ip_country in [
            "DE", "FR", "IT", "ES", "NL", "BE", "AT", "SE", "DK", "FI", "NO", "IE", "PT", "GR", "PL", "CZ", "HU", "SK", "SI", "HR", "BG", "RO", "LT", "LV", "EE", "CY", "MT", "LU"
        ]:
            if ComplianceFramework.GDPR not in frameworks:
                frameworks.append(ComplianceFramework.GDPR)
        
        if self.user_location == UserLocation.CALIFORNIA or self.ip_country == "US-CA":
            if ComplianceFramework.CCPA not in frameworks:
                frameworks.append(ComplianceFramework.CCPA)
        
        if self.user_location == UserLocation.SINGAPORE or self.ip_country == "SG":
            if ComplianceFramework.PDPA_SINGAPORE not in frameworks:
                frameworks.append(ComplianceFramework.PDPA_SINGAPORE)
        
        if self.user_location == UserLocation.THAILAND or self.ip_country == "TH":
            if ComplianceFramework.PDPA_THAILAND not in frameworks:
                frameworks.append(ComplianceFramework.PDPA_THAILAND)
        
        # Default to most restrictive if no specific framework applies
        if not frameworks:
            frameworks.append(ComplianceFramework.GDPR)  # GDPR as default (most comprehensive)
        
        return frameworks


@dataclass
class UnifiedComplianceResult:
    """Result from unified compliance processing."""
    processing_allowed: bool
    applicable_frameworks: List[ComplianceFramework]
    processing_records: Dict[str, str]  # Framework -> processing ID
    compliance_notices: Dict[str, Dict[str, Any]]  # Framework -> notice
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UnifiedComplianceManager:
    """Unified manager for all privacy compliance frameworks."""
    
    def __init__(self):
        # Initialize all compliance processors
        self.gdpr_processor = GDPRDataProcessor()
        self.ccpa_processor = CCPADataProcessor()
        self.pdpa_processor_singapore = PDPADataProcessor(PDPAJurisdiction.SINGAPORE)
        self.pdpa_processor_thailand = PDPADataProcessor(PDPAJurisdiction.THAILAND)
        
        # Initialize wrapper classes
        self.gdpr_wrapper = SentimentAnalysisGDPRWrapper(self.gdpr_processor)
        self.ccpa_wrapper = SentimentAnalysisCCPAWrapper(self.ccpa_processor)
        self.pdpa_wrapper_singapore = SentimentAnalysisPDPAWrapper(self.pdpa_processor_singapore)
        self.pdpa_wrapper_thailand = SentimentAnalysisPDPAWrapper(self.pdpa_processor_thailand)
        
        # Framework mapping
        self.framework_processors = {
            ComplianceFramework.GDPR: self.gdpr_processor,
            ComplianceFramework.CCPA: self.ccpa_processor,
            ComplianceFramework.PDPA_SINGAPORE: self.pdpa_processor_singapore,
            ComplianceFramework.PDPA_THAILAND: self.pdpa_processor_thailand
        }
        
        self.framework_wrappers = {
            ComplianceFramework.GDPR: self.gdpr_wrapper,
            ComplianceFramework.CCPA: self.ccpa_wrapper,
            ComplianceFramework.PDPA_SINGAPORE: self.pdpa_wrapper_singapore,
            ComplianceFramework.PDPA_THAILAND: self.pdpa_wrapper_thailand
        }
    
    async def analyze_sentiment_with_compliance(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        compliance_context: Optional[ComplianceContext] = None
    ) -> UnifiedComplianceResult:
        """Analyze sentiment with unified compliance checking across all applicable frameworks."""
        
        # Determine compliance context
        if compliance_context is None:
            compliance_context = ComplianceContext()
        
        applicable_frameworks = compliance_context.get_applicable_frameworks()
        
        result = UnifiedComplianceResult(
            processing_allowed=True,
            applicable_frameworks=applicable_frameworks,
            processing_records={},
            compliance_notices={}
        )
        
        # Process through each applicable framework
        for framework in applicable_frameworks:
            try:
                framework_result = await self._process_framework_compliance(
                    framework, text, user_id, context, compliance_context
                )
                
                # Check if processing is allowed under this framework
                if not framework_result.get("processing_allowed", True):
                    result.processing_allowed = False
                    if "error" in framework_result:
                        result.error_messages.append(
                            f"{framework.value.upper()}: {framework_result['error']}"
                        )
                
                # Record processing ID if successful
                if "sentiment_analysis" in framework_result:
                    processing_id = framework_result["sentiment_analysis"].get("processing_id")
                    if processing_id:
                        result.processing_records[framework.value] = processing_id
                
                # Store compliance notices
                if "gdpr_notice" in framework_result:
                    result.compliance_notices[framework.value] = framework_result["gdpr_notice"]
                elif "ccpa_notice" in framework_result:
                    result.compliance_notices[framework.value] = framework_result["ccpa_notice"]
                elif "pdpa_notice" in framework_result:
                    result.compliance_notices[framework.value] = framework_result["pdpa_notice"]
                
            except Exception as e:
                logger.error(f"Error processing {framework.value} compliance: {e}")
                result.warnings.append(f"{framework.value.upper()}: Processing error - {str(e)}")
        
        # If processing is not allowed under any framework, halt processing
        if not result.processing_allowed:
            logger.warning(f"Sentiment analysis blocked due to compliance restrictions: {result.error_messages}")
        
        return result
    
    async def _process_framework_compliance(
        self,
        framework: ComplianceFramework,
        text: str,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
        compliance_context: ComplianceContext
    ) -> Dict[str, Any]:
        """Process compliance for a specific framework."""
        
        wrapper = self.framework_wrappers[framework]
        
        if framework == ComplianceFramework.GDPR:
            # Map to GDPR legal basis
            legal_basis = LegalBasis.LEGITIMATE_INTEREST
            if compliance_context.user_preference == ComplianceFramework.GDPR:
                legal_basis = LegalBasis.CONSENT
            
            return await wrapper.analyze_with_gdpr_compliance(
                text=text,
                data_subject_id=user_id,
                context=context,
                legal_basis=legal_basis
            )
        
        elif framework == ComplianceFramework.CCPA:
            return await wrapper.analyze_with_ccpa_compliance(
                text=text,
                consumer_id=user_id,
                context=context
            )
        
        elif framework in [ComplianceFramework.PDPA_SINGAPORE, ComplianceFramework.PDPA_THAILAND]:
            jurisdiction = PDPAJurisdiction.SINGAPORE if framework == ComplianceFramework.PDPA_SINGAPORE else PDPAJurisdiction.THAILAND
            
            return await wrapper.analyze_with_pdpa_compliance(
                text=text,
                individual_id=user_id,
                context=context,
                jurisdiction=jurisdiction
            )
        
        else:
            raise ValueError(f"Unsupported compliance framework: {framework}")
    
    def record_user_consent(
        self,
        user_id: str,
        frameworks: List[ComplianceFramework],
        consent_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Record user consent across multiple compliance frameworks."""
        
        consent_records = {}
        
        for framework in frameworks:
            try:
                if framework == ComplianceFramework.GDPR:
                    consent_id = self.gdpr_processor.record_consent(
                        data_subject_id=user_id,
                        consent_purposes=[GDPRPurpose.SENTIMENT_ANALYSIS],
                        consent_given=True,
                        consent_text=consent_context.get("consent_text", "Consent for sentiment analysis"),
                        consent_method=consent_context.get("method", "web_form")
                    )
                    consent_records[framework.value] = consent_id
                
                elif framework == ComplianceFramework.CCPA:
                    # CCPA doesn't require explicit consent, but we can record opt-in
                    if consent_context.get("ccpa_opt_in", False):
                        opt_in_id = self.ccpa_processor.record_opt_in(
                            consumer_id=user_id,
                            opt_in_method=consent_context.get("method", "web_form")
                        )
                        consent_records[framework.value] = opt_in_id
                
                elif framework in [ComplianceFramework.PDPA_SINGAPORE, ComplianceFramework.PDPA_THAILAND]:
                    processor = self.framework_processors[framework]
                    jurisdiction = PDPAJurisdiction.SINGAPORE if framework == ComplianceFramework.PDPA_SINGAPORE else PDPAJurisdiction.THAILAND
                    
                    from .pdpa_compliance import PDPAConsent
                    consent_id = processor.record_consent(
                        individual_id=user_id,
                        consent_type=PDPAConsent.EXPRESS_CONSENT,
                        purposes=[PDPAPurpose.SERVICE_PROVISION, PDPAPurpose.BUSINESS_IMPROVEMENT],
                        data_types=[PDPADataType.PERSONAL_DATA],
                        consent_method=consent_context.get("method", "web_form"),
                        jurisdiction=jurisdiction
                    )
                    consent_records[framework.value] = consent_id
                
            except Exception as e:
                logger.error(f"Failed to record consent for {framework.value}: {e}")
                consent_records[framework.value] = f"ERROR: {str(e)}"
        
        return consent_records
    
    def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,  # "access", "delete", "portability", "correction"
        frameworks: Optional[List[ComplianceFramework]] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Handle data subject requests across multiple frameworks."""
        
        if frameworks is None:
            # Determine applicable frameworks based on user context
            compliance_context = ComplianceContext()
            frameworks = compliance_context.get_applicable_frameworks()
        
        results = {}
        
        for framework in frameworks:
            try:
                if framework == ComplianceFramework.GDPR:
                    from .gdpr_compliance import DataSubjectRequest, DataSubjectRight
                    
                    # Map request types to GDPR rights
                    right_mapping = {
                        "access": DataSubjectRight.ACCESS,
                        "delete": DataSubjectRight.ERASURE,
                        "portability": DataSubjectRight.DATA_PORTABILITY,
                        "correction": DataSubjectRight.RECTIFICATION
                    }
                    
                    if request_type in right_mapping:
                        request = DataSubjectRequest(
                            request_id=f"gdpr_{datetime.now().timestamp()}",
                            data_subject_id=user_id,
                            request_type=right_mapping[request_type],
                            requested_at=datetime.now(),
                            description=request_context.get("description", f"GDPR {request_type} request")
                        )
                        
                        # In real implementation, would process this asynchronously
                        results[framework.value] = {
                            "status": "accepted",
                            "request_id": request.request_id,
                            "estimated_completion": "30 days"
                        }
                
                elif framework == ComplianceFramework.CCPA:
                    from .ccpa_compliance import CCPAConsumerRequest, CCPAConsumerRight
                    
                    # Map request types to CCPA rights
                    right_mapping = {
                        "access": CCPAConsumerRight.RIGHT_TO_KNOW,
                        "delete": CCPAConsumerRight.RIGHT_TO_DELETE,
                        "opt_out": CCPAConsumerRight.RIGHT_TO_OPT_OUT
                    }
                    
                    if request_type in right_mapping:
                        request = CCPAConsumerRequest(
                            request_id=f"ccpa_{datetime.now().timestamp()}",
                            consumer_id=user_id,
                            request_type=right_mapping[request_type],
                            requested_at=datetime.now(),
                            description=request_context.get("description", f"CCPA {request_type} request")
                        )
                        
                        results[framework.value] = {
                            "status": "accepted",
                            "request_id": request.request_id,
                            "estimated_completion": "45 days"
                        }
                
                elif framework in [ComplianceFramework.PDPA_SINGAPORE, ComplianceFramework.PDPA_THAILAND]:
                    from .pdpa_compliance import PDPAAccessRequest
                    jurisdiction = PDPAJurisdiction.SINGAPORE if framework == ComplianceFramework.PDPA_SINGAPORE else PDPAJurisdiction.THAILAND
                    
                    request = PDPAAccessRequest(
                        request_id=f"pdpa_{jurisdiction.value}_{datetime.now().timestamp()}",
                        individual_id=user_id,
                        jurisdiction=jurisdiction,
                        requested_at=datetime.now(),
                        request_type=request_type,
                        description=request_context.get("description", f"PDPA {request_type} request")
                    )
                    
                    results[framework.value] = {
                        "status": "accepted",
                        "request_id": request.request_id,
                        "estimated_completion": "30 days"
                    }
                
            except Exception as e:
                logger.error(f"Failed to handle request for {framework.value}: {e}")
                results[framework.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def run_comprehensive_compliance_check(self) -> Dict[str, Any]:
        """Run compliance checks across all frameworks."""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "COMPLIANT",
            "framework_results": {},
            "summary": {
                "total_frameworks": len(self.framework_processors),
                "compliant_frameworks": 0,
                "non_compliant_frameworks": 0,
                "failed_checks": []
            }
        }
        
        for framework, processor in self.framework_processors.items():
            try:
                if framework == ComplianceFramework.GDPR:
                    from .gdpr import get_gdpr_compliance
                    gdpr_compliance = get_gdpr_compliance()
                    framework_result = gdpr_compliance.run_compliance_check()
                
                elif framework == ComplianceFramework.CCPA:
                    from .ccpa_compliance import CCPACompliance
                    ccpa_compliance = CCPACompliance()
                    framework_result = ccpa_compliance.run_compliance_check()
                
                elif framework in [ComplianceFramework.PDPA_SINGAPORE, ComplianceFramework.PDPA_THAILAND]:
                    from .pdpa_compliance import PDPACompliance
                    jurisdiction = PDPAJurisdiction.SINGAPORE if framework == ComplianceFramework.PDPA_SINGAPORE else PDPAJurisdiction.THAILAND
                    pdpa_compliance = PDPACompliance(jurisdiction)
                    framework_result = pdpa_compliance.run_compliance_check()
                
                results["framework_results"][framework.value] = framework_result
                
                # Update summary
                if framework_result.get("overall_status") == "COMPLIANT":
                    results["summary"]["compliant_frameworks"] += 1
                else:
                    results["summary"]["non_compliant_frameworks"] += 1
                    results["overall_status"] = "NON_COMPLIANT"
                    
                    # Collect failed checks
                    failed_checks = [
                        f"{framework.value}: {check['name']}"
                        for check in framework_result.get("checks", [])
                        if check.get("status") == "FAIL"
                    ]
                    results["summary"]["failed_checks"].extend(failed_checks)
                
            except Exception as e:
                logger.error(f"Compliance check failed for {framework.value}: {e}")
                results["framework_results"][framework.value] = {
                    "error": str(e),
                    "overall_status": "ERROR"
                }
                results["summary"]["non_compliant_frameworks"] += 1
                results["overall_status"] = "NON_COMPLIANT"
        
        logger.info(f"Comprehensive compliance check completed: {results['overall_status']}")
        
        return results
    
    def generate_unified_privacy_policy(self) -> str:
        """Generate a unified privacy policy covering all supported frameworks."""
        
        policy = f"""
UNIFIED PRIVACY POLICY

Last Updated: {datetime.now().strftime('%B %d, %Y')}

This privacy policy describes how we collect, use, and protect your personal information
in compliance with applicable privacy laws including GDPR, CCPA, and PDPA.

1. INFORMATION WE COLLECT
We collect the following types of personal information:
• Text content for sentiment analysis
• Usage data and system logs
• Contact information for user accounts
• Technical data for system operation

2. HOW WE USE YOUR INFORMATION
We use your personal information for:
• Providing sentiment analysis services
• Improving system performance and reliability
• Ensuring system security and compliance
• Customer support and communication

3. LEGAL BASIS FOR PROCESSING
We process your data based on:
• Your consent (where required)
• Legitimate interests for service provision
• Legal obligations and compliance requirements

4. YOUR PRIVACY RIGHTS

GDPR Rights (EU Residents):
• Right to access your personal data
• Right to rectification of inaccurate data
• Right to erasure (right to be forgotten)
• Right to restrict processing
• Right to data portability
• Right to object to processing

CCPA Rights (California Residents):
• Right to know what personal information is collected
• Right to delete personal information
• Right to opt-out of sale of personal information
• Right to non-discriminatory treatment

PDPA Rights (Singapore/Thailand Residents):
• Right to access your personal data
• Right to correction of personal data
• Right to deletion of personal data
• Right to data portability
• Right to withdraw consent

5. HOW TO EXERCISE YOUR RIGHTS
• Email: privacy@healing-guard.com
• Subject: "[GDPR/CCPA/PDPA] Request - [Right Type]"
• Data Protection Officer: dpo@healing-guard.com

6. DATA RETENTION
We retain personal data for:
• Sentiment analysis data: 90-365 days
• System logs: 30-90 days
• Account data: Until account closure + legal requirements

7. DATA TRANSFERS
Data may be transferred to countries with adequate protection or appropriate safeguards
in compliance with applicable privacy laws.

8. CONTACT US
For privacy questions or to exercise your rights:
• Email: privacy@healing-guard.com
• Data Protection Officer: dpo@healing-guard.com
• Address: [Company Address]

This policy is designed to comply with:
• EU General Data Protection Regulation (GDPR)
• California Consumer Privacy Act (CCPA)
• Singapore Personal Data Protection Act (PDPA)
• Thailand Personal Data Protection Act (PDPA)
"""
        return policy.strip()


# Global unified compliance manager
unified_compliance_manager = UnifiedComplianceManager()