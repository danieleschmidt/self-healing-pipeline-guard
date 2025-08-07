"""High-performance caching system for sentiment analysis and healing operations."""

import json
import hashlib
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from functools import wraps

from ..core.config import settings
from ..core.exceptions import ResourceException, RetryableException
from ..monitoring.structured_logging import performance_logger

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Dynamic strategy based on usage patterns


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    tags: List[str] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def last_access_seconds(self) -> float:
        """Get seconds since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


class SentimentCache:
    """High-performance cache optimized for sentiment analysis results."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        memory_limit_mb: int = 100
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_size_bytes = 0
        
        # Strategy-specific data
        self._lru_order: List[str] = []  # For LRU strategy
        self._frequency_counter: Dict[str, int] = {}  # For LFU strategy
        
        logger.info(f"Initialized SentimentCache: max_size={max_size}, strategy={strategy.value}")
    
    def _generate_cache_key(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a unique cache key for text and context."""
        # Create hash of text and context for consistent key generation
        content = {"text": text}
        if context:
            # Sort context keys for consistent hashing
            content["context"] = {k: v for k, v in sorted(context.items()) if v is not None}
        
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value in bytes."""
        try:
            # Convert to JSON to estimate size
            json_str = json.dumps(value, default=str)
            return len(json_str.encode('utf-8'))
        except:
            # Fallback estimation
            return len(str(value)) * 2  # Rough estimate
    
    def get(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Retrieve value from cache."""
        cache_key = self._generate_cache_key(text, context)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(cache_key, reason="expired")
                self._misses += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Update strategy-specific data
            if self.strategy == CacheStrategy.LRU:
                self._lru_order.remove(cache_key)
                self._lru_order.append(cache_key)
            elif self.strategy == CacheStrategy.LFU:
                self._frequency_counter[cache_key] = self._frequency_counter.get(cache_key, 0) + 1
            
            self._hits += 1
            
            # Log cache hit for analysis
            performance_logger.log_resource_usage(
                resource_type="sentiment_cache_hit_rate",
                current_usage=self._hits,
                limit=self._hits + self._misses,
                usage_percentage=(self._hits / (self._hits + self._misses)) * 100 if (self._hits + self._misses) > 0 else 0
            )
            
            return entry.value
    
    def set(
        self, 
        text: str, 
        value: Any, 
        context: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Store value in cache."""
        cache_key = self._generate_cache_key(text, context)
        size_bytes = self._estimate_size(value)
        
        with self._lock:
            # Check memory limits
            if (self._current_size_bytes + size_bytes) > self.memory_limit_bytes:
                self._evict_to_make_space(size_bytes)
            
            # Check size limits
            if len(self._cache) >= self.max_size:
                self._evict_entries(1)
            
            # Create new entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl_seconds or self.default_ttl,
                tags=tags or [],
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if cache_key in self._cache:
                self._remove_entry(cache_key, reason="overwrite")
            
            # Add new entry
            self._cache[cache_key] = entry
            self._current_size_bytes += size_bytes
            
            # Update strategy-specific data
            if self.strategy == CacheStrategy.LRU:
                self._lru_order.append(cache_key)
            elif self.strategy == CacheStrategy.LFU:
                self._frequency_counter[cache_key] = 1
            
            logger.debug(f"Cached sentiment result: key={cache_key[:8]}..., size={size_bytes}B")
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all entries with specified tags."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if entry.tags and any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key, reason="tag_invalidation")
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._lru_order.clear()
            self._frequency_counter.clear()
            self._current_size_bytes = 0
            
            logger.info(f"Cache cleared: removed {count} entries")
    
    def _evict_entries(self, count: int):
        """Evict entries based on current strategy."""
        if not self._cache:
            return
        
        keys_to_evict = []
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            keys_to_evict = self._lru_order[:count]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            sorted_by_frequency = sorted(
                self._frequency_counter.items(),
                key=lambda x: x[1]
            )
            keys_to_evict = [key for key, _ in sorted_by_frequency[:count]]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            keys_to_evict.extend(expired_keys[:count])
            
            if len(keys_to_evict) < count:
                remaining = count - len(keys_to_evict)
                oldest_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                keys_to_evict.extend([
                    key for key in oldest_keys[:remaining]
                    if key not in keys_to_evict
                ])
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy: balance between LRU and LFU
            hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            
            if hit_rate > 0.8:  # High hit rate, use LFU
                sorted_by_frequency = sorted(
                    self._frequency_counter.items(),
                    key=lambda x: x[1]
                )
                keys_to_evict = [key for key, _ in sorted_by_frequency[:count]]
            else:  # Lower hit rate, use LRU
                keys_to_evict = self._lru_order[:count]
        
        # Perform evictions
        for key in keys_to_evict:
            if key in self._cache:
                self._remove_entry(key, reason="eviction")
    
    def _evict_to_make_space(self, required_bytes: int):
        """Evict entries to free up required memory."""
        freed_bytes = 0
        eviction_count = 0
        
        while (freed_bytes < required_bytes and 
               self._current_size_bytes > required_bytes and
               self._cache):
            
            # Find entry to evict based on strategy
            if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                key_to_evict = self._lru_order[0] if self._lru_order else next(iter(self._cache))
            else:  # LFU or TTL
                key_to_evict = min(
                    self._cache.keys(),
                    key=lambda k: self._frequency_counter.get(k, 0)
                )
            
            entry = self._cache.get(key_to_evict)
            if entry:
                freed_bytes += entry.size_bytes
                self._remove_entry(key_to_evict, reason="memory_pressure")
                eviction_count += 1
        
        logger.info(f"Evicted {eviction_count} entries to free {freed_bytes} bytes")
    
    def _remove_entry(self, key: str, reason: str = "unknown"):
        """Remove entry from cache and update metadata."""
        entry = self._cache.pop(key, None)
        if not entry:
            return
        
        self._current_size_bytes -= entry.size_bytes
        self._evictions += 1
        
        # Update strategy-specific data
        if key in self._lru_order:
            self._lru_order.remove(key)
        if key in self._frequency_counter:
            del self._frequency_counter[key]
        
        logger.debug(f"Removed cache entry: key={key[:8]}..., reason={reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self._current_size_bytes,
                "memory_limit_bytes": self.memory_limit_bytes,
                "memory_usage_percentage": (self._current_size_bytes / self.memory_limit_bytes) * 100,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "strategy": self.strategy.value,
                "average_entry_size": self._current_size_bytes / len(self._cache) if self._cache else 0
            }
    
    async def cleanup_expired(self):
        """Async cleanup of expired entries."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            with self._lock:
                self._remove_entry(key, reason="cleanup_expired")
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class HealingPlanCache:
    """Specialized cache for healing plans and actions."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 7200):  # 2 hours
        self.cache = SentimentCache(
            max_size=max_size,
            default_ttl=default_ttl,
            strategy=CacheStrategy.LRU,
            memory_limit_mb=50
        )
    
    def get_healing_plan(
        self, 
        failure_type: str, 
        failure_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached healing plan for failure type and context."""
        # Create a cache key from failure characteristics
        cache_key_data = {
            "failure_type": failure_type,
            "severity": failure_context.get("severity"),
            "repository": failure_context.get("repository"),
            "branch": failure_context.get("branch"),
            # Don't include job_id or timestamp in cache key
        }
        
        return self.cache.get(json.dumps(cache_key_data, sort_keys=True))
    
    def cache_healing_plan(
        self,
        failure_type: str,
        failure_context: Dict[str, Any],
        healing_plan: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ):
        """Cache healing plan for similar failures."""
        cache_key_data = {
            "failure_type": failure_type,
            "severity": failure_context.get("severity"),
            "repository": failure_context.get("repository"),
            "branch": failure_context.get("branch"),
        }
        
        # Add tags for invalidation
        tags = [
            f"failure_type:{failure_type}",
            f"repository:{failure_context.get('repository', 'unknown')}",
            f"severity:{failure_context.get('severity', 'unknown')}"
        ]
        
        self.cache.set(
            text=json.dumps(cache_key_data, sort_keys=True),
            value=healing_plan,
            ttl_seconds=ttl_seconds,
            tags=tags
        )


# Global cache instances
sentiment_cache = SentimentCache(
    max_size=settings.quantum_planner.max_parallel_tasks * 100,  # Scale with system capacity
    default_ttl=1800,  # 30 minutes for sentiment results
    strategy=CacheStrategy.ADAPTIVE
)

healing_plan_cache = HealingPlanCache(
    max_size=500,
    default_ttl=3600  # 1 hour for healing plans
)


# Caching decorators
def cache_sentiment_result(ttl_seconds: int = 1800):
    """Decorator to cache sentiment analysis results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract text and context from arguments
            text = args[1] if len(args) > 1 else kwargs.get('text')
            context = args[2] if len(args) > 2 else kwargs.get('context')
            
            if not text:
                # Can't cache without text
                return await func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = sentiment_cache.get(text, context)
            if cached_result is not None:
                logger.debug(f"Cache hit for sentiment analysis: {text[:50]}...")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Cache successful results
            if result and hasattr(result, 'confidence') and result.confidence > 0:
                # Add tags based on result characteristics
                tags = [f"sentiment:{result.label.value}"]
                if hasattr(result, 'is_urgent') and result.is_urgent:
                    tags.append("urgent")
                if context and context.get('repository'):
                    tags.append(f"repo:{context['repository']}")
                
                sentiment_cache.set(
                    text=text,
                    value=result,
                    context=context,
                    ttl_seconds=ttl_seconds,
                    tags=tags
                )
                logger.debug(f"Cached sentiment result: {text[:50]}...")
            
            return result
        return wrapper
    return decorator


def cache_healing_plan_result(ttl_seconds: int = 3600):
    """Decorator to cache healing plan results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract failure event from arguments
            failure_event = args[1] if len(args) > 1 else kwargs.get('failure_event')
            
            if not failure_event:
                return await func(*args, **kwargs)
            
            # Create context for caching
            failure_context = {
                "failure_type": failure_event.failure_type.value if hasattr(failure_event.failure_type, 'value') else str(failure_event.failure_type),
                "severity": failure_event.severity.value if hasattr(failure_event.severity, 'value') else str(failure_event.severity),
                "repository": getattr(failure_event, 'repository', None),
                "branch": getattr(failure_event, 'branch', None)
            }
            
            # Try to get from cache
            cached_plan = healing_plan_cache.get_healing_plan(
                failure_context["failure_type"],
                failure_context
            )
            
            if cached_plan is not None:
                logger.debug(f"Cache hit for healing plan: {failure_context['failure_type']}")
                return cached_plan
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Cache successful results
            if result and hasattr(result, 'actions') and result.actions:
                healing_plan_cache.cache_healing_plan(
                    failure_context["failure_type"],
                    failure_context,
                    result,
                    ttl_seconds
                )
                logger.debug(f"Cached healing plan: {failure_context['failure_type']}")
            
            return result
        return wrapper
    return decorator


# Background cache maintenance
class CacheManager:
    """Manages cache lifecycle and maintenance tasks."""
    
    def __init__(self):
        self.cleanup_interval = 300  # 5 minutes
        self.stats_interval = 600    # 10 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background cache maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
        logger.info("Cache manager started")
    
    async def stop(self):
        """Stop background cache maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()
        logger.info("Cache manager stopped")
    
    async def _cleanup_loop(self):
        """Background loop for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Cleanup expired entries
                await sentiment_cache.cleanup_expired()
                await healing_plan_cache.cache.cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _stats_loop(self):
        """Background loop for cache statistics logging."""
        while True:
            try:
                await asyncio.sleep(self.stats_interval)
                
                # Log sentiment cache stats
                sentiment_stats = sentiment_cache.get_stats()
                performance_logger.log_resource_usage(
                    resource_type="sentiment_cache_memory",
                    current_usage=sentiment_stats["memory_usage_bytes"] / (1024 * 1024),  # MB
                    limit=sentiment_stats["memory_limit_bytes"] / (1024 * 1024),
                    usage_percentage=sentiment_stats["memory_usage_percentage"]
                )
                
                # Log healing plan cache stats
                healing_stats = healing_plan_cache.cache.get_stats()
                performance_logger.log_resource_usage(
                    resource_type="healing_cache_memory",
                    current_usage=healing_stats["memory_usage_bytes"] / (1024 * 1024),  # MB
                    limit=healing_stats["memory_limit_bytes"] / (1024 * 1024),
                    usage_percentage=healing_stats["memory_usage_percentage"]
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache stats logging error: {e}")


# Global cache manager instance
cache_manager = CacheManager()