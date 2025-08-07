"""Compliance and regulatory frameworks for global deployment."""

from .gdpr import GDPRCompliance, DataProcessor
from .audit import ComplianceAuditor, AuditTrail
from .data_governance import DataGovernanceManager, DataClassification

# Enhanced compliance implementations
from .gdpr_compliance import (
    DataProcessingPurpose,
    LegalBasis,
    DataProcessingRecord,
    GDPRDataProcessor,
    SentimentAnalysisGDPRWrapper,
    gdpr_processor,
    gdpr_sentiment_wrapper
)

from .ccpa_compliance import (
    CCPAConsumerRight,
    CCPADataCategory,
    CCPABusinessPurpose,
    CCPADataProcessor,
    CCPACompliance,
    SentimentAnalysisCCPAWrapper,
    ccpa_processor,
    ccpa_compliance,
    ccpa_sentiment_wrapper
)

from .pdpa_compliance import (
    PDPAJurisdiction,
    PDPAConsent,
    PDPAPurpose,
    PDPADataType,
    PDPADataProcessor,
    PDPACompliance,
    SentimentAnalysisPDPAWrapper,
    pdpa_processor_singapore,
    pdpa_processor_thailand,
    pdpa_compliance_singapore,
    pdpa_compliance_thailand,
    pdpa_sentiment_wrapper_singapore,
    pdpa_sentiment_wrapper_thailand
)

from .unified_compliance import (
    ComplianceFramework,
    UserLocation,
    ComplianceContext,
    UnifiedComplianceResult,
    UnifiedComplianceManager,
    unified_compliance_manager
)

__all__ = [
    # Legacy exports
    "GDPRCompliance",
    "DataProcessor",
    "ComplianceAuditor", 
    "AuditTrail",
    "DataGovernanceManager",
    "DataClassification",
    
    # Enhanced GDPR exports
    "DataProcessingPurpose",
    "LegalBasis",
    "DataProcessingRecord",
    "GDPRDataProcessor",
    "SentimentAnalysisGDPRWrapper",
    "gdpr_processor",
    "gdpr_sentiment_wrapper",
    
    # CCPA exports
    "CCPAConsumerRight",
    "CCPADataCategory", 
    "CCPABusinessPurpose",
    "CCPADataProcessor",
    "CCPACompliance",
    "SentimentAnalysisCCPAWrapper",
    "ccpa_processor",
    "ccpa_compliance",
    "ccpa_sentiment_wrapper",
    
    # PDPA exports
    "PDPAJurisdiction",
    "PDPAConsent",
    "PDPAPurpose",
    "PDPADataType",
    "PDPADataProcessor",
    "PDPACompliance", 
    "SentimentAnalysisPDPAWrapper",
    "pdpa_processor_singapore",
    "pdpa_processor_thailand",
    "pdpa_compliance_singapore",
    "pdpa_compliance_thailand",
    "pdpa_sentiment_wrapper_singapore",
    "pdpa_sentiment_wrapper_thailand",
    
    # Unified compliance exports
    "ComplianceFramework",
    "UserLocation",
    "ComplianceContext",
    "UnifiedComplianceResult", 
    "UnifiedComplianceManager",
    "unified_compliance_manager"
]