"""Sentiment Analysis Engine for Pipeline Events and Developer Communications.

This module provides comprehensive sentiment analysis capabilities for:
- Pipeline failure messages and logs
- Commit messages and PR descriptions  
- Developer communications and feedback
- CI/CD event context analysis

Used to enhance healing decisions based on emotional context and urgency.
"""

import re
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .exceptions import SentimentAnalysisException, RetryableException
from ..security.input_validator import input_validator, validate_sentiment_input, validate_context_input
from ..monitoring.structured_logging import sentiment_logger, log_operation, logging_context
from ..performance.cache_manager import cache_sentiment_result, sentiment_cache
from ..performance.metrics_collector import collect_sentiment_metrics, sentiment_metrics
from ..performance.auto_scaling import auto_scaler, ResourceType

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    URGENT = "urgent"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    label: SentimentLabel
    confidence: float
    polarity: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    urgency_score: float  # 0.0 to 1.0
    emotional_intensity: float  # 0.0 to 1.0
    keywords: List[str]
    context_factors: Dict[str, Any]
    processing_time_ms: float
    
    @property
    def is_urgent(self) -> bool:
        """Determine if the sentiment indicates urgent attention needed."""
        return self.urgency_score > 0.7 or self.label == SentimentLabel.URGENT
    
    @property
    def is_negative(self) -> bool:
        """Determine if sentiment is negative."""
        return self.polarity < -0.1
    
    @property
    def is_frustrated(self) -> bool:
        """Determine if sentiment indicates frustration."""
        return self.label == SentimentLabel.FRUSTRATED or (
            self.polarity < -0.3 and self.emotional_intensity > 0.6
        )


class PipelineSentimentAnalyzer:
    """Advanced sentiment analyzer for CI/CD pipeline events and developer communications."""
    
    def __init__(self):
        """Initialize the sentiment analyzer with performance optimizations."""
        # Dynamic worker pool that can be scaled
        self._initial_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self._initial_workers)
        
        # Register auto-scaling callback
        auto_scaler.register_scaling_callback(
            ResourceType.SENTIMENT_ANALYZER_WORKERS,
            self._scale_worker_pool
        )
        
        # Performance tracking
        self._analysis_queue_size = 0
        self._queue_lock = asyncio.Lock()
        
        # Pre-compiled patterns and lexicons for faster processing
        self._load_sentiment_lexicons()
        self._compile_patterns()
        
        logger.info(f"Initialized PipelineSentimentAnalyzer with {self._initial_workers} workers")
    
    def _load_sentiment_lexicons(self):
        """Load sentiment lexicons and keywords for pipeline-specific analysis."""
        # Positive sentiment indicators
        self.positive_keywords = {
            # Success indicators
            "success", "passed", "completed", "fixed", "resolved", "working",
            "great", "excellent", "perfect", "awesome", "good", "nice",
            "clean", "stable", "reliable", "fast", "efficient", "optimized",
            
            # Developer satisfaction
            "happy", "pleased", "satisfied", "confident", "ready", "deployed",
            "merged", "approved", "reviewed", "tested", "validated",
            
            # Progress indicators  
            "improved", "enhanced", "upgraded", "modernized", "refactored",
            "implemented", "added", "created", "built", "developed"
        }
        
        # Negative sentiment indicators
        self.negative_keywords = {
            # Failure indicators
            "failed", "error", "broken", "crash", "timeout", "stuck",
            "blocked", "failing", "flaky", "unstable", "slow", "hanging",
            
            # Frustration indicators
            "frustrated", "annoying", "stupid", "ridiculous", "waste", "terrible",
            "awful", "horrible", "nightmare", "mess", "disaster", "broken",
            
            # Problem indicators
            "issue", "problem", "bug", "defect", "regression", "critical",
            "urgent", "blocker", "showstopper", "emergency", "outage"
        }
        
        # Urgency indicators
        self.urgency_keywords = {
            "urgent", "asap", "immediately", "now", "critical", "emergency",
            "blocker", "showstopper", "production", "outage", "down",
            "hotfix", "patch", "quick", "fast", "priority", "deadline"
        }
        
        # Developer emotion indicators
        self.emotion_keywords = {
            "frustrated": ["frustrated", "annoying", "irritating", "stuck"],
            "confident": ["confident", "ready", "solid", "stable", "good"],
            "worried": ["worried", "concerned", "unsure", "risky", "dangerous"],
            "excited": ["excited", "amazing", "awesome", "fantastic", "love"]
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for sentiment analysis."""
        # Pattern for detecting repeated characters (frustration indicator)
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
        # Pattern for detecting excessive punctuation (emphasis/emotion)
        self.excessive_punct_pattern = re.compile(r'[!?]{2,}')
        
        # Pattern for detecting all caps (shouting/urgency)
        self.caps_pattern = re.compile(r'\b[A-Z]{3,}\b')
        
        # Pattern for detecting time references (urgency context)
        self.time_urgency_pattern = re.compile(
            r'\b(?:today|tonight|tomorrow|asap|urgent|now|immediately|deadline)\b',
            re.IGNORECASE
        )
    
    @collect_sentiment_metrics("single")
    @cache_sentiment_result(ttl_seconds=1800)
    @log_operation("sentiment_analysis", log_args=False, performance_threshold_ms=500.0)
    @validate_sentiment_input(max_length=10000, allow_html=False)
    @validate_context_input()
    async def analyze_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> SentimentResult:
        """Analyze sentiment of text with pipeline-specific context."""
        analysis_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # Track queue size for auto-scaling
        await self._track_queue_size(increment=True)
        
        # Log analysis start
        sentiment_logger.log_analysis_start(
            text_preview=text,
            context=context,
            analysis_type="single"
        )
        
        try:
            if not text or not text.strip():
                empty_result = SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    polarity=0.0,
                    subjectivity=0.0,
                    urgency_score=0.0,
                    emotional_intensity=0.0,
                    keywords=[],
                    context_factors={},
                    processing_time_ms=0.0
                )
                
                sentiment_logger.log_analysis_complete(
                    analysis_id=analysis_id,
                    sentiment_label=empty_result.label.value,
                    confidence=empty_result.confidence,
                    processing_time_ms=empty_result.processing_time_ms,
                    text_length=0
                )
                
                await self._track_queue_size(increment=False)
                return empty_result
        
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            words = cleaned_text.lower().split()
            
            # Calculate base sentiment scores
            polarity, subjectivity = self._calculate_base_sentiment(words)
            
            # Calculate urgency score
            urgency_score = self._calculate_urgency(text, words, context)
            
            # Calculate emotional intensity
            emotional_intensity = self._calculate_emotional_intensity(text, words)
            
            # Extract relevant keywords
            keywords = self._extract_sentiment_keywords(words)
            
            # Determine primary sentiment label
            sentiment_label = self._classify_sentiment(
                polarity, urgency_score, emotional_intensity, keywords, context
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                polarity, subjectivity, urgency_score, emotional_intensity
            )
            
            # Extract context factors
            context_factors = self._extract_context_factors(text, context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = SentimentResult(
                label=sentiment_label,
                confidence=confidence,
                polarity=polarity,
                subjectivity=subjectivity,
                urgency_score=urgency_score,
                emotional_intensity=emotional_intensity,
                keywords=keywords,
                context_factors=context_factors,
                processing_time_ms=processing_time
            )
            
            # Log successful analysis
            sentiment_logger.log_analysis_complete(
                analysis_id=analysis_id,
                sentiment_label=result.label.value,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                text_length=len(text),
                urgency_score=result.urgency_score,
                is_urgent=result.is_urgent,
                is_frustrated=result.is_frustrated
            )
            
            await self._track_queue_size(increment=False)
            return result
            
        except Exception as e:
            # Track queue size on completion (even with error)
            await self._track_queue_size(increment=False)
            # Log analysis error
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            sentiment_logger.log_analysis_error(
                text_preview=text,
                error_message=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time
            )
            
            # Convert to appropriate exception type
            if isinstance(e, (ValueError, TypeError)):
                raise SentimentAnalysisException(
                    message=f"Analysis failed due to invalid input: {str(e)}",
                    text_preview=text[:100],
                    analysis_stage="processing",
                    details={"original_error": str(e)}
                )
            else:
                # Potentially retryable errors
                raise RetryableException(
                    message=f"Sentiment analysis failed: {str(e)}",
                    max_retries=2,
                    retry_delay=0.5,
                    details={
                        "text_preview": text[:100],
                        "analysis_stage": "processing",
                        "original_error": str(e)
                    }
                )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _scale_worker_pool(self, new_worker_count: int):
        """Scale the worker pool based on auto-scaling decisions."""
        try:
            # Shutdown old executor
            old_executor = self.executor
            old_executor.shutdown(wait=False)
            
            # Create new executor with new worker count
            self.executor = ThreadPoolExecutor(max_workers=new_worker_count)
            
            logger.info(f"Scaled sentiment analyzer worker pool to {new_worker_count} workers")
            
            # Update metrics
            sentiment_metrics.record_analysis_complete(
                duration_ms=0,  # No duration for scaling event
                sentiment_label="neutral",
                confidence=0,
                text_length=0,
                analysis_type="scaling",
                is_urgent=False,
                is_frustrated=False
            )
            
        except Exception as e:
            logger.error(f"Failed to scale worker pool: {e}")
    
    async def _track_queue_size(self, increment: bool = True):
        """Track analysis queue size for auto-scaling."""
        async with self._queue_lock:
            if increment:
                self._analysis_queue_size += 1
            else:
                self._analysis_queue_size = max(0, self._analysis_queue_size - 1)
            
            # Update metrics for auto-scaling
            sentiment_metrics.collector.set_gauge(
                "sentiment_analysis_queue_depth", 
                self._analysis_queue_size
            )
    
    def _calculate_base_sentiment(self, words: List[str]) -> Tuple[float, float]:
        """Calculate basic polarity and subjectivity scores."""
        positive_score = 0
        negative_score = 0
        subjective_score = 0
        
        for word in words:
            if word in self.positive_keywords:
                positive_score += 1
                subjective_score += 0.5
            elif word in self.negative_keywords:
                negative_score += 1
                subjective_score += 0.8
        
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words == 0:
            polarity = 0.0
            subjectivity = 0.0
        else:
            polarity = (positive_score - negative_score) / len(words)
            subjectivity = min(subjective_score / len(words), 1.0)
        
        return polarity, subjectivity
    
    def _calculate_urgency(self, text: str, words: List[str], context: Optional[Dict[str, Any]]) -> float:
        """Calculate urgency score based on text content and context."""
        urgency_score = 0.0
        
        # Check for urgency keywords
        urgency_word_count = sum(1 for word in words if word in self.urgency_keywords)
        urgency_score += min(urgency_word_count * 0.3, 1.0)
        
        # Check for time urgency patterns
        if self.time_urgency_pattern.search(text):
            urgency_score += 0.4
        
        # Check for excessive punctuation (indicates urgency/emotion)
        if self.excessive_punct_pattern.search(text):
            urgency_score += 0.2
        
        # Check for all caps (shouting/urgency)
        caps_matches = len(self.caps_pattern.findall(text))
        urgency_score += min(caps_matches * 0.1, 0.3)
        
        # Context-based urgency
        if context:
            if context.get('event_type') == 'pipeline_failure':
                urgency_score += 0.3
            if context.get('is_production', False):
                urgency_score += 0.4
            if context.get('consecutive_failures', 0) > 2:
                urgency_score += 0.3
        
        return min(urgency_score, 1.0)
    
    def _calculate_emotional_intensity(self, text: str, words: List[str]) -> float:
        """Calculate emotional intensity based on text patterns."""
        intensity = 0.0
        
        # Check for repeated characters (frustration)
        repeated_chars = len(self.repeated_char_pattern.findall(text))
        intensity += min(repeated_chars * 0.2, 0.6)
        
        # Check for excessive punctuation
        excessive_punct = len(self.excessive_punct_pattern.findall(text))
        intensity += min(excessive_punct * 0.3, 0.5)
        
        # Check for emotional keywords
        emotional_words = 0
        for emotion_type, keywords in self.emotion_keywords.items():
            for word in words:
                if word in keywords:
                    emotional_words += 1
        
        intensity += min(emotional_words * 0.2, 0.4)
        
        return min(intensity, 1.0)
    
    def _extract_sentiment_keywords(self, words: List[str]) -> List[str]:
        """Extract relevant sentiment keywords from text."""
        keywords = []
        
        all_sentiment_words = (
            self.positive_keywords | 
            self.negative_keywords | 
            self.urgency_keywords
        )
        
        for word in words:
            if word in all_sentiment_words:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _classify_sentiment(
        self,
        polarity: float,
        urgency_score: float,
        emotional_intensity: float,
        keywords: List[str],
        context: Optional[Dict[str, Any]]
    ) -> SentimentLabel:
        """Classify overall sentiment based on all factors."""
        
        # Check for urgent sentiment first
        if urgency_score > 0.7:
            return SentimentLabel.URGENT
        
        # Check for frustration indicators
        frustrated_keywords = {"frustrated", "annoying", "stuck", "broken", "terrible"}
        if (any(kw in keywords for kw in frustrated_keywords) and 
            polarity < -0.2 and emotional_intensity > 0.5):
            return SentimentLabel.FRUSTRATED
        
        # Check for confidence indicators
        confident_keywords = {"confident", "ready", "good", "stable", "working"}
        if (any(kw in keywords for kw in confident_keywords) and 
            polarity > 0.1):
            return SentimentLabel.CONFIDENT
        
        # Standard polarity-based classification
        if polarity > 0.3:
            return SentimentLabel.VERY_POSITIVE
        elif polarity > 0.1:
            return SentimentLabel.POSITIVE
        elif polarity < -0.3:
            return SentimentLabel.VERY_NEGATIVE
        elif polarity < -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_confidence(
        self,
        polarity: float,
        subjectivity: float,
        urgency_score: float,
        emotional_intensity: float
    ) -> float:
        """Calculate confidence score for the sentiment analysis."""
        
        # Base confidence on polarity strength
        polarity_confidence = min(abs(polarity) * 2, 1.0)
        
        # Boost confidence with subjectivity (more subjective = more confident)
        subjectivity_boost = subjectivity * 0.3
        
        # Boost confidence with clear urgency signals
        urgency_boost = urgency_score * 0.2
        
        # Boost confidence with emotional intensity
        emotion_boost = emotional_intensity * 0.2
        
        total_confidence = polarity_confidence + subjectivity_boost + urgency_boost + emotion_boost
        
        return min(total_confidence, 1.0)
    
    def _extract_context_factors(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract contextual factors that influence sentiment interpretation."""
        factors = {}
        
        # Text-based factors
        factors['text_length'] = len(text)
        factors['has_excessive_punctuation'] = bool(self.excessive_punct_pattern.search(text))
        factors['has_repeated_chars'] = bool(self.repeated_char_pattern.search(text))
        factors['has_caps_words'] = bool(self.caps_pattern.search(text))
        factors['has_time_urgency'] = bool(self.time_urgency_pattern.search(text))
        
        # Context-based factors
        if context:
            factors.update({
                'event_type': context.get('event_type'),
                'is_production': context.get('is_production', False),
                'consecutive_failures': context.get('consecutive_failures', 0),
                'time_of_day': context.get('time_of_day'),
                'author': context.get('author'),
                'repository': context.get('repository')
            })
        
        return factors
    
    @collect_sentiment_metrics("batch")
    @log_operation("batch_sentiment_analysis", performance_threshold_ms=5000.0)
    async def analyze_batch(
        self, 
        texts: List[str], 
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[SentimentResult]:
        """Analyze multiple texts in parallel with comprehensive error handling."""
        start_time = datetime.now()
        
        # Validate batch input
        batch_validation = input_validator.validate_batch_texts(texts)
        if not batch_validation.is_valid:
            raise SentimentAnalysisException(
                message="Batch validation failed",
                analysis_stage="validation",
                details={
                    "errors": batch_validation.errors,
                    "warnings": batch_validation.warnings,
                    "security_flags": batch_validation.security_flags
                }
            )
        
        if contexts is None:
            contexts = [None] * len(texts)
        
        # Use validated/sanitized texts
        sanitized_texts = batch_validation.sanitized_input
        
        sentiment_logger.log_analysis_start(
            text_preview=f"Batch of {len(sanitized_texts)} texts",
            context={"batch_size": len(sanitized_texts)},
            analysis_type="batch"
        )
        
        results = []
        successful_count = 0
        failed_count = 0
        
        try:
            # Create tasks for parallel processing with error handling
            async def analyze_with_fallback(text: str, context: Optional[Dict], index: int) -> Optional[SentimentResult]:
                try:
                    return await self.analyze_text(text, context)
                except Exception as e:
                    logger.warning(f"Failed to analyze text at index {index}: {e}")
                    return None
            
            tasks = [
                analyze_with_fallback(text, context, i)
                for i, (text, context) in enumerate(zip(sanitized_texts, contexts))
            ]
            
            # Execute all analyses in parallel
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and count successes/failures
            for i, result in enumerate(raw_results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Batch analysis failed for text {i}: {result}")
                elif result is not None:
                    results.append(result)
                    successful_count += 1
                else:
                    failed_count += 1
            
            # Calculate processing metrics
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate sentiment distribution
            sentiment_distribution = {}
            if results:
                for result in results:
                    label = result.label.value
                    sentiment_distribution[label] = sentiment_distribution.get(label, 0) + 1
            
            # Log batch completion
            sentiment_logger.log_batch_analysis(
                batch_size=len(sanitized_texts),
                total_processing_time_ms=total_processing_time,
                successful_analyses=successful_count,
                failed_analyses=failed_count,
                sentiment_distribution=sentiment_distribution
            )
            
            # Check for concerning patterns
            if results:
                summary = self.get_sentiment_summary(results)
                if summary.get('concerning_pattern_detected', False):
                    sentiment_logger.log_concerning_pattern(
                        pattern_type="batch_analysis",
                        urgency_count=summary.get('urgent_events', 0),
                        frustrated_count=summary.get('frustrated_events', 0),
                        negative_count=summary.get('negative_events', 0),
                        total_analyses=len(results),
                        context={"batch_size": len(sanitized_texts)}
                    )
            
            # If too many failures, raise an exception
            if failed_count > len(sanitized_texts) * 0.5:  # More than 50% failed
                raise SentimentAnalysisException(
                    message=f"Batch analysis failed for {failed_count}/{len(sanitized_texts)} texts",
                    analysis_stage="batch_processing",
                    details={
                        "successful_count": successful_count,
                        "failed_count": failed_count,
                        "total_count": len(sanitized_texts)
                    }
                )
            
            return results
            
        except Exception as e:
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            sentiment_logger.log_analysis_error(
                text_preview=f"Batch of {len(texts)} texts",
                error_message=str(e),
                error_type=type(e).__name__,
                processing_time_ms=total_processing_time
            )
            raise
    
    @collect_sentiment_metrics("pipeline_event")
    @cache_sentiment_result(ttl_seconds=600)  # Shorter TTL for pipeline events
    @log_operation("pipeline_event_sentiment_analysis", performance_threshold_ms=750.0)
    async def analyze_pipeline_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """Analyze sentiment for specific pipeline events with enhanced context."""
        try:
            # Validate pipeline event input
            validation_result = input_validator.validate_pipeline_event(event_type, message, metadata)
            if not validation_result.is_valid:
                raise SentimentAnalysisException(
                    message="Pipeline event validation failed",
                    analysis_stage="validation",
                    details={
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings,
                        "security_flags": validation_result.security_flags
                    }
                )
            
            # Use sanitized input
            sanitized_data = validation_result.sanitized_input
            sanitized_message = sanitized_data['message']
            sanitized_metadata = sanitized_data['metadata']
            
            context = {
                'event_type': event_type,
                'is_production': sanitized_metadata.get('environment') == 'production' if sanitized_metadata else False,
                'consecutive_failures': sanitized_metadata.get('consecutive_failures', 0) if sanitized_metadata else 0,
                'time_of_day': datetime.now().hour,
                'repository': sanitized_metadata.get('repository') if sanitized_metadata else None,
                'author': sanitized_metadata.get('author') if sanitized_metadata else None
            }
            
            with logging_context(
                event_type=event_type,
                repository=context.get('repository'),
                is_production=context.get('is_production')
            ):
                result = await self.analyze_text(sanitized_message, context)
                
                # Log significant sentiment events with enhanced context
                if result.is_urgent or result.is_frustrated:
                    sentiment_logger.log_concerning_pattern(
                        pattern_type="pipeline_event_concern",
                        urgency_count=1 if result.is_urgent else 0,
                        frustrated_count=1 if result.is_frustrated else 0,
                        negative_count=1 if result.is_negative else 0,
                        total_analyses=1,
                        context={
                            "event_type": event_type,
                            "sentiment_label": result.label.value,
                            "confidence": result.confidence,
                            "urgency_score": result.urgency_score,
                            "repository": context.get('repository'),
                            "is_production": context.get('is_production')
                        }
                    )
                
                return result
                
        except Exception as e:
            if not isinstance(e, SentimentAnalysisException):
                raise SentimentAnalysisException(
                    message=f"Pipeline event analysis failed: {str(e)}",
                    analysis_stage="pipeline_processing",
                    details={
                        "event_type": event_type,
                        "original_error": str(e)
                    }
                )
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Generate summary statistics from multiple sentiment results."""
        if not results:
            return {}
        
        # Calculate averages
        avg_polarity = sum(r.polarity for r in results) / len(results)
        avg_urgency = sum(r.urgency_score for r in results) / len(results)
        avg_intensity = sum(r.emotional_intensity for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Count sentiment labels
        label_counts = {}
        for result in results:
            label_counts[result.label.value] = label_counts.get(result.label.value, 0) + 1
        
        # Identify concerning patterns
        urgent_count = sum(1 for r in results if r.is_urgent)
        frustrated_count = sum(1 for r in results if r.is_frustrated)
        negative_count = sum(1 for r in results if r.is_negative)
        
        return {
            'total_analyses': len(results),
            'average_polarity': avg_polarity,
            'average_urgency': avg_urgency,
            'average_emotional_intensity': avg_intensity,
            'average_confidence': avg_confidence,
            'sentiment_distribution': label_counts,
            'urgent_events': urgent_count,
            'frustrated_events': frustrated_count,
            'negative_events': negative_count,
            'concerning_pattern_detected': urgent_count > 2 or frustrated_count > 1,
            'overall_sentiment_trend': (
                'positive' if avg_polarity > 0.1 else
                'negative' if avg_polarity < -0.1 else
                'neutral'
            )
        }


# Global sentiment analyzer instance
sentiment_analyzer = PipelineSentimentAnalyzer()