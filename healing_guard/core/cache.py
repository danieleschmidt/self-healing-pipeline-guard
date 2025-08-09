"""Advanced caching system with multiple backends and optimization strategies."""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from collections import OrderedDict
from threading import Lock

from contextlib import asynccontextmanager

# Optional dependencies - graceful degradation if not available
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all values from the cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
    
    def _evict_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and now > entry.expires_at
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Evict least recently used entries to maintain size limit."""
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
    
    def _remove_entry(self, key: str):
        """Remove an entry and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats["size_bytes"] -= entry.size_bytes
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode('utf-8'))
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the memory cache."""
        with self._lock:
            self._evict_expired()
            
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            entry.update_access()
            self._stats["hits"] += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the memory cache."""
        try:
            with self._lock:
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                
                # Create new entry
                size_bytes = self._calculate_size(value)
                expires_at = None
                if ttl or self.default_ttl:
                    expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )
                
                # Evict if necessary
                self._evict_lru()
                
                # Add new entry
                self._cache[key] = entry
                self._stats["size_bytes"] += size_bytes
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the memory cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all values from the memory cache."""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the memory cache."""
        with self._lock:
            self._evict_expired()
            return key in self._cache and not self._cache[key].is_expired
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern."""
        import fnmatch
        
        with self._lock:
            self._evict_expired()
            return [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "size_entries": len(self._cache),
                "size_bytes": self._stats["size_bytes"],
                "max_size": self.max_size
            }


class RedisCache(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "hg:"):
        if not HAS_REDIS:
            raise ImportError("Redis is not available. Install redis-py to use RedisCache.")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)
            
            data = redis_client.get(redis_key)
            if data is None:
                self._stats["misses"] += 1
                return None
            
            # Deserialize the data
            try:
                value = pickle.loads(data)
                self._stats["hits"] += 1
                return value
            except Exception as e:
                logger.error(f"Failed to deserialize cache data for key {key}: {e}")
                # Clean up corrupted data
                redis_client.delete(redis_key)
                self._stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis cache get error for key {key}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis cache."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)
            
            # Serialize the data
            try:
                data = pickle.dumps(value)
            except Exception as e:
                logger.error(f"Failed to serialize cache data for key {key}: {e}")
                return False
            
            # Set with TTL if specified
            if ttl:
                result = redis_client.setex(redis_key, ttl, data)
            else:
                result = redis_client.set(redis_key, data)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis cache set error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)
            
            result = redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis cache delete error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def clear(self) -> bool:
        """Clear all values from Redis cache."""
        try:
            redis_client = await self._get_redis()
            
            # Get all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = redis_client.keys(pattern)
            
            if keys:
                redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            self._stats["errors"] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)
            
            return bool(redis_client.exists(redis_key))
            
        except Exception as e:
            logger.error(f"Redis cache exists error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern."""
        try:
            redis_client = await self._get_redis()
            
            # Combine prefix with pattern
            redis_pattern = f"{self.key_prefix}{pattern}"
            redis_keys = redis_client.keys(redis_pattern)
            
            # Remove prefix from results
            prefix_len = len(self.key_prefix)
            return [key.decode('utf-8')[prefix_len:] for key in redis_keys]
            
        except Exception as e:
            logger.error(f"Redis cache keys error: {e}")
            self._stats["errors"] += 1
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "errors": self._stats["errors"],
            "backend": "redis"
        }


class MultiLevelCache(CacheBackend):
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers."""
    
    def __init__(
        self,
        l1_cache: Optional[MemoryCache] = None,
        l2_cache: Optional[CacheBackend] = None,
        l1_ttl: int = 300,  # 5 minutes
        l2_ttl: int = 3600  # 1 hour
    ):
        self.l1_cache = l1_cache or MemoryCache(max_size=500, default_ttl=l1_ttl)
        
        # Only use Redis if available
        if l2_cache is not None:
            self.l2_cache = l2_cache
        elif HAS_REDIS:
            try:
                self.l2_cache = RedisCache()
            except ImportError:
                logger.warning("Redis not available, using memory-only caching")
                self.l2_cache = MemoryCache(max_size=1000, default_ttl=l2_ttl)
        else:
            logger.warning("Redis not available, using memory-only caching")  
            self.l2_cache = MemoryCache(max_size=1000, default_ttl=l2_ttl)
            
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "l2_promotions": 0  # L2 hits promoted to L1
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from multi-level cache."""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self._stats["l1_hits"] += 1
            return value
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            self._stats["l2_hits"] += 1
            self._stats["l2_promotions"] += 1
            
            # Promote to L1 cache
            await self.l1_cache.set(key, value, self.l1_ttl)
            return value
        
        self._stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in multi-level cache."""
        # Set in both caches
        l1_ttl = min(ttl or self.l1_ttl, self.l1_ttl)
        l2_ttl = ttl or self.l2_ttl
        
        l1_success = await self.l1_cache.set(key, value, l1_ttl)
        l2_success = await self.l2_cache.set(key, value, l2_ttl)
        
        # Return True if at least one cache succeeded
        return l1_success or l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete a value from multi-level cache."""
        l1_result = await self.l1_cache.delete(key)
        l2_result = await self.l2_cache.delete(key)
        return l1_result or l2_result
    
    async def clear(self) -> bool:
        """Clear all values from multi-level cache."""
        l1_result = await self.l1_cache.clear()
        l2_result = await self.l2_cache.clear()
        return l1_result and l2_result
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in multi-level cache."""
        return await self.l1_cache.exists(key) or await self.l2_cache.exists(key)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern from both caches."""
        l1_keys = set(await self.l1_cache.keys(pattern))
        l2_keys = set(await self.l2_cache.keys(pattern))
        return list(l1_keys.union(l2_keys))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["misses"]
        
        return {
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "misses": self._stats["misses"],
            "l2_promotions": self._stats["l2_promotions"],
            "total_requests": total_requests,
            "l1_hit_rate": self._stats["l1_hits"] / total_requests if total_requests > 0 else 0,
            "overall_hit_rate": (self._stats["l1_hits"] + self._stats["l2_hits"]) / total_requests if total_requests > 0 else 0,
            "l1_cache_stats": self.l1_cache.get_stats(),
            "l2_cache_stats": self.l2_cache.get_stats()
        }


class CacheManager:
    """Advanced cache manager with smart caching strategies."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or MultiLevelCache()
        self._function_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_policies: Dict[str, Dict[str, Any]] = {}
        
    def _make_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Create a cache key from function arguments."""
        # Create a hash of the arguments for consistent keys
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get_cache(self, cache_name: str = None) -> CacheBackend:
        """Get cache backend instance."""
        # For now, return the main backend
        # In a full implementation, this could manage multiple named caches
        return self.backend
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return await self.backend.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        return await self.backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        return await self.backend.delete(key)
    
    async def clear(self) -> bool:
        """Clear all cache values."""
        return await self.backend.clear()
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        keys = await self.backend.keys(pattern)
        count = 0
        
        for key in keys:
            if await self.backend.delete(key):
                count += 1
        
        return count
    
    def cached(
        self,
        ttl: int = 3600,
        key_prefix: Optional[str] = None,
        invalidate_on: Optional[List[str]] = None,
        condition: Optional[Callable] = None
    ):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            prefix = key_prefix or f"func:{func.__module__}.{func.__name__}"
            
            async def async_wrapper(*args, **kwargs):
                # Check condition if provided
                if condition and not condition(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._make_cache_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.backend.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.backend.set(cache_key, result, ttl)
                
                return result
            
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to run in an event loop
                loop = asyncio.get_event_loop()
                
                # Check condition if provided
                if condition and not condition(*args, **kwargs):
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._make_cache_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = loop.run_until_complete(self.backend.get(cache_key))
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                loop.run_until_complete(self.backend.set(cache_key, result, ttl))
                
                return result
            
            # Store cache policy for management
            self._cache_policies[prefix] = {
                "ttl": ttl,
                "invalidate_on": invalidate_on or [],
                "function": func.__name__
            }
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    async def warm_cache(self, func: Callable, arg_sets: List[tuple], **common_kwargs):
        """Pre-warm cache with common function calls."""
        tasks = []
        
        for args in arg_sets:
            if asyncio.iscoroutinefunction(func):
                task = func(*args, **common_kwargs)
            else:
                # Wrap sync function for async execution
                task = asyncio.create_task(asyncio.to_thread(func, *args, **common_kwargs))
            
            tasks.append(task)
        
        # Execute all cache warming tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        info = {
            "backend_type": type(self.backend).__name__,
            "cache_policies": len(self._cache_policies),
            "policies": self._cache_policies
        }
        
        # Add backend-specific stats
        if hasattr(self.backend, 'get_stats'):
            info["stats"] = self.backend.get_stats()
        
        return info
    
    async def optimize_cache(self):
        """Perform cache optimization tasks."""
        logger.info("Starting cache optimization")
        
        # For memory cache, trigger cleanup
        if isinstance(self.backend, MemoryCache):
            # The memory cache automatically evicts expired entries on access
            # We can trigger a manual cleanup by checking some keys
            keys = await self.backend.keys()
            for key in keys[:10]:  # Check first 10 keys
                await self.backend.get(key)
        
        elif isinstance(self.backend, MultiLevelCache):
            # For multi-level cache, optimize both levels
            l1_keys = await self.backend.l1_cache.keys()
            for key in l1_keys[:10]:
                await self.backend.l1_cache.get(key)
        
        logger.info("Cache optimization completed")


# Global cache manager instance
cache_manager = CacheManager()