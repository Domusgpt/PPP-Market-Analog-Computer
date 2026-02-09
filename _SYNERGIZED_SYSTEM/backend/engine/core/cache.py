"""
Pattern Cache - Memoization for Encoding
========================================

Caching utilities to avoid redundant computation
for repeated encoding operations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import threading


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    hits: int = 0


class LRUCache:
    """
    Thread-safe LRU cache with size limit.

    Parameters
    ----------
    max_size_mb : float
        Maximum cache size in megabytes
    max_entries : int
        Maximum number of entries
    """

    def __init__(self, max_size_mb: float = 100.0, max_entries: int = 1000):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def _compute_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, dict):
            return sum(self._compute_size(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(self._compute_size(v) for v in value)
        else:
            return 64  # Default estimate

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(key)
                entry.hits += 1
                self._cache[key] = entry
                self._hits += 1
                return entry.value
            else:
                self._misses += 1
                return None

    def put(self, key: str, value: Any):
        """Put value in cache."""
        size = self._compute_size(value)

        with self._lock:
            # Remove if already exists
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_size -= old_entry.size_bytes

            # Evict until we have space
            while (self._current_size + size > self.max_size_bytes or
                   len(self._cache) >= self.max_entries):
                if not self._cache:
                    break
                # Remove oldest (first) entry
                _, old_entry = self._cache.popitem(last=False)
                self._current_size -= old_entry.size_bytes

            # Add new entry
            entry = CacheEntry(key=key, value=value, size_bytes=size)
            self._cache[key] = entry
            self._current_size += size

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0
            }


class PatternCache:
    """
    Specialized cache for moiré patterns and features.

    Caches:
    - Moiré patterns by angle
    - Coordinate meshes by grid size
    - Feature vectors by input hash

    Parameters
    ----------
    max_size_mb : float
        Maximum cache size
    """

    def __init__(self, max_size_mb: float = 100.0):
        self._pattern_cache = LRUCache(max_size_mb * 0.6, 500)
        self._mesh_cache = LRUCache(max_size_mb * 0.2, 50)
        self._feature_cache = LRUCache(max_size_mb * 0.2, 1000)

    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        """Compute hash of numpy array."""
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]

    def get_pattern(
        self,
        angle: float,
        grid_size: Tuple[int, int],
        state_hash: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get cached moiré pattern."""
        key = f"pattern_{angle:.4f}_{grid_size[0]}x{grid_size[1]}"
        if state_hash:
            key += f"_{state_hash}"
        return self._pattern_cache.get(key)

    def put_pattern(
        self,
        angle: float,
        grid_size: Tuple[int, int],
        pattern: np.ndarray,
        state_hash: Optional[str] = None
    ):
        """Cache moiré pattern."""
        key = f"pattern_{angle:.4f}_{grid_size[0]}x{grid_size[1]}"
        if state_hash:
            key += f"_{state_hash}"
        self._pattern_cache.put(key, pattern.copy())

    def get_mesh(
        self,
        grid_size: Tuple[int, int],
        field_size: Tuple[float, float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached coordinate mesh."""
        key = f"mesh_{grid_size}_{field_size}"
        return self._mesh_cache.get(key)

    def put_mesh(
        self,
        grid_size: Tuple[int, int],
        field_size: Tuple[float, float],
        X: np.ndarray,
        Y: np.ndarray
    ):
        """Cache coordinate mesh."""
        key = f"mesh_{grid_size}_{field_size}"
        self._mesh_cache.put(key, (X.copy(), Y.copy()))

    def get_features(self, input_hash: str) -> Optional[np.ndarray]:
        """Get cached features."""
        return self._feature_cache.get(f"features_{input_hash}")

    def put_features(self, input_hash: str, features: np.ndarray):
        """Cache features."""
        self._feature_cache.put(f"features_{input_hash}", features.copy())

    def clear(self):
        """Clear all caches."""
        self._pattern_cache.clear()
        self._mesh_cache.clear()
        self._feature_cache.clear()

    @property
    def stats(self) -> Dict:
        """Get combined cache statistics."""
        return {
            'patterns': self._pattern_cache.stats,
            'meshes': self._mesh_cache.stats,
            'features': self._feature_cache.stats
        }


def cached_computation(cache: LRUCache, key_func: Callable[..., str]):
    """
    Decorator for caching function results.

    Parameters
    ----------
    cache : LRUCache
        Cache instance to use
    key_func : Callable
        Function to generate cache key from arguments

    Example
    -------
    >>> cache = LRUCache()
    >>> @cached_computation(cache, lambda x, y: f"{x}_{y}")
    ... def expensive_function(x, y):
    ...     return x ** y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[PatternCache] = None


def get_global_cache() -> PatternCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PatternCache()
    return _global_cache


def clear_global_cache():
    """Clear global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
