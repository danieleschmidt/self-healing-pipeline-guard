"""API routes for sentiment analysis functionality."""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED, 
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY
)

from ..core.sentiment_analyzer import sentiment_analyzer, SentimentResult, SentimentLabel
from .middleware import get_rate_limit_key, check_rate_limit
from .exceptions import ValidationError, ResourceNotFoundError

# Initialize router
router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class PipelineEventRequest(BaseModel):
    """Request model for pipeline event sentiment analysis."""
    event_type: str = Field(..., description="Type of pipeline event")
    message: str = Field(..., min_length=1, max_length=10000, description="Event message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Event metadata")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        allowed_types = {
            'pipeline_failure', 'test_failure', 'build_error', 'deployment_error',
            'commit_message', 'pr_comment', 'issue_comment', 'code_review',
            'merge_request', 'release_notes', 'incident_report'
        }
        if v not in allowed_types:
            raise ValueError(f"Event type must be one of: {', '.join(allowed_types)}")
        return v


class BatchAnalysisRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Optional contexts for each text")
    
    @validator('texts')
    def validate_texts(cls, v):
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return [text.strip() for text in v]


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis results."""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    polarity: float = Field(..., ge=-1.0, le=1.0)
    subjectivity: float = Field(..., ge=0.0, le=1.0)
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    emotional_intensity: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str]
    context_factors: Dict[str, Any]
    processing_time_ms: float
    is_urgent: bool
    is_negative: bool
    is_frustrated: bool
    analyzed_at: datetime = Field(default_factory=datetime.now)
    
    @classmethod
    def from_sentiment_result(cls, result: SentimentResult) -> "SentimentResponse":
        """Create response from sentiment result."""
        return cls(
            label=result.label.value,
            confidence=result.confidence,
            polarity=result.polarity,
            subjectivity=result.subjectivity,
            urgency_score=result.urgency_score,
            emotional_intensity=result.emotional_intensity,
            keywords=result.keywords,
            context_factors=result.context_factors,
            processing_time_ms=result.processing_time_ms,
            is_urgent=result.is_urgent,
            is_negative=result.is_negative,
            is_frustrated=result.is_frustrated
        )


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    results: List[SentimentResponse]
    summary: Dict[str, Any]
    total_processing_time_ms: float
    analyzed_at: datetime = Field(default_factory=datetime.now)


@router.post("/analyze", 
             response_model=SentimentResponse,
             status_code=HTTP_200_OK,
             summary="Analyze text sentiment",
             description="Perform sentiment analysis on provided text with optional context")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    background_tasks: BackgroundTasks,
    rate_limit_key: str = Depends(get_rate_limit_key)
):
    """Analyze sentiment of provided text."""
    try:
        # Check rate limiting
        await check_rate_limit(rate_limit_key, max_requests=100, window_minutes=60)
        
        # Perform sentiment analysis
        result = await sentiment_analyzer.analyze_text(request.text, request.context)
        
        # Log significant events in background
        if result.is_urgent or result.is_frustrated:
            background_tasks.add_task(
                log_significant_sentiment,
                result.label.value,
                request.text[:100],  # Log first 100 chars
                result.confidence
            )
        
        return SentimentResponse.from_sentiment_result(result)
        
    except ValidationError as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )


@router.post("/analyze/pipeline-event",
             response_model=SentimentResponse,
             status_code=HTTP_200_OK,
             summary="Analyze pipeline event sentiment",
             description="Analyze sentiment of pipeline events with specialized context")
async def analyze_pipeline_event(
    request: PipelineEventRequest,
    background_tasks: BackgroundTasks,
    rate_limit_key: str = Depends(get_rate_limit_key)
):
    """Analyze sentiment for pipeline events."""
    try:
        # Check rate limiting
        await check_rate_limit(rate_limit_key, max_requests=200, window_minutes=60)
        
        # Perform specialized pipeline event analysis
        result = await sentiment_analyzer.analyze_pipeline_event(
            request.event_type,
            request.message,
            request.metadata
        )
        
        # Log critical pipeline sentiment events
        if result.is_urgent and request.event_type == 'pipeline_failure':
            background_tasks.add_task(
                alert_critical_pipeline_sentiment,
                request.event_type,
                result,
                request.metadata
            )
        
        return SentimentResponse.from_sentiment_result(result)
        
    except ValidationError as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to analyze pipeline event sentiment: {str(e)}"
        )


@router.post("/analyze/batch",
             response_model=BatchSentimentResponse,
             status_code=HTTP_200_OK,
             summary="Batch sentiment analysis",
             description="Analyze sentiment for multiple texts simultaneously")
async def analyze_batch_sentiment(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    rate_limit_key: str = Depends(get_rate_limit_key)
):
    """Perform batch sentiment analysis."""
    try:
        # Check rate limiting (higher limit for batch requests)
        await check_rate_limit(rate_limit_key, max_requests=10, window_minutes=60)
        
        start_time = datetime.now()
        
        # Perform batch analysis
        results = await sentiment_analyzer.analyze_batch(request.texts, request.contexts)
        
        # Convert results to response format
        response_results = [
            SentimentResponse.from_sentiment_result(result)
            for result in results
        ]
        
        # Generate summary
        summary = sentiment_analyzer.get_sentiment_summary(results)
        
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log concerning patterns in background
        if summary.get('concerning_pattern_detected', False):
            background_tasks.add_task(
                log_concerning_sentiment_pattern,
                summary,
                len(request.texts)
            )
        
        return BatchSentimentResponse(
            results=response_results,
            summary=summary,
            total_processing_time_ms=total_processing_time
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to perform batch sentiment analysis: {str(e)}"
        )


@router.get("/health",
            status_code=HTTP_200_OK,
            summary="Sentiment analyzer health check",
            description="Check if the sentiment analyzer is operational")
async def sentiment_health_check():
    """Check sentiment analyzer health."""
    try:
        # Perform a simple test analysis
        test_result = await sentiment_analyzer.analyze_text("test message")
        
        return {
            "status": "healthy",
            "analyzer_ready": True,
            "test_analysis_time_ms": test_result.processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sentiment analyzer health check failed: {str(e)}"
        )


@router.get("/stats",
            status_code=HTTP_200_OK,
            summary="Get sentiment analysis statistics",
            description="Retrieve statistics about sentiment analysis performance")
async def get_sentiment_stats():
    """Get sentiment analyzer statistics."""
    # In a real implementation, these would come from metrics storage
    return {
        "total_analyses_performed": 0,  # Would be tracked in metrics
        "average_processing_time_ms": 0.0,
        "most_common_sentiment": "neutral",
        "urgent_events_today": 0,
        "frustrated_events_today": 0,
        "analyzer_version": "1.0.0",
        "supported_languages": ["en"],  # Could be extended
        "uptime_hours": 0.0,
        "last_updated": datetime.now().isoformat()
    }


# Background task functions
async def log_significant_sentiment(sentiment_label: str, text_preview: str, confidence: float):
    """Log significant sentiment events for monitoring."""
    # In production, this would integrate with logging/monitoring systems
    print(f"SIGNIFICANT SENTIMENT: {sentiment_label} (confidence: {confidence:.2f}) - '{text_preview}...'")


async def alert_critical_pipeline_sentiment(event_type: str, result: SentimentResult, metadata: Optional[Dict]):
    """Alert on critical pipeline sentiment events."""
    # In production, this would integrate with alerting systems (Slack, PagerDuty, etc.)
    print(f"CRITICAL PIPELINE SENTIMENT ALERT: {event_type} - {result.label.value} (urgency: {result.urgency_score:.2f})")


async def log_concerning_sentiment_pattern(summary: Dict[str, Any], batch_size: int):
    """Log concerning sentiment patterns detected in batch analysis."""
    print(f"CONCERNING SENTIMENT PATTERN: {summary.get('urgent_events', 0)} urgent events in batch of {batch_size}")