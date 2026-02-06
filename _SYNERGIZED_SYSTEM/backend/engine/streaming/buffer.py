"""
Buffer Utilities - Ring Buffers for Streaming
=============================================

Memory-efficient ring buffers for streaming data.
"""

import numpy as np
from typing import Optional, Tuple, Generic, TypeVar
from threading import Lock
from collections import deque

T = TypeVar('T')


class RingBuffer:
    """
    Fixed-size ring buffer for streaming data.

    Thread-safe circular buffer that overwrites oldest
    data when full.

    Parameters
    ----------
    capacity : int
        Maximum number of elements
    dtype : np.dtype
        Data type for elements

    Example
    -------
    >>> buffer = RingBuffer(capacity=1000)
    >>> buffer.append(data)
    >>> recent = buffer.get_last(100)
    """

    def __init__(self, capacity: int, dtype: np.dtype = np.float32):
        self.capacity = capacity
        self.dtype = dtype

        self._buffer = np.zeros(capacity, dtype=dtype)
        self._write_idx = 0
        self._count = 0
        self._lock = Lock()

    def append(self, value: float):
        """Append single value."""
        with self._lock:
            self._buffer[self._write_idx] = value
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def extend(self, values: np.ndarray):
        """Append multiple values."""
        with self._lock:
            n = len(values)

            if n >= self.capacity:
                # Just keep the last capacity elements
                self._buffer[:] = values[-self.capacity:]
                self._write_idx = 0
                self._count = self.capacity
            else:
                # Write in chunks
                end_idx = self._write_idx + n
                if end_idx <= self.capacity:
                    self._buffer[self._write_idx:end_idx] = values
                else:
                    first_part = self.capacity - self._write_idx
                    self._buffer[self._write_idx:] = values[:first_part]
                    self._buffer[:n - first_part] = values[first_part:]

                self._write_idx = end_idx % self.capacity
                self._count = min(self._count + n, self.capacity)

    def get_last(self, n: int) -> np.ndarray:
        """Get last n elements."""
        with self._lock:
            n = min(n, self._count)
            if n == 0:
                return np.array([], dtype=self.dtype)

            start_idx = (self._write_idx - n) % self.capacity

            if start_idx < self._write_idx:
                return self._buffer[start_idx:self._write_idx].copy()
            else:
                return np.concatenate([
                    self._buffer[start_idx:],
                    self._buffer[:self._write_idx]
                ])

    def get_all(self) -> np.ndarray:
        """Get all elements in order."""
        return self.get_last(self._count)

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._write_idx = 0
            self._count = 0

    @property
    def count(self) -> int:
        """Number of elements in buffer."""
        return self._count

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._count >= self.capacity


class FrameBuffer:
    """
    Buffer for 2D frames (images/patterns).

    Parameters
    ----------
    capacity : int
        Maximum number of frames
    frame_shape : Tuple[int, int]
        Shape of each frame

    Example
    -------
    >>> buffer = FrameBuffer(capacity=30, frame_shape=(64, 64))
    >>> buffer.append(frame)
    >>> recent = buffer.get_last(10)  # Shape: (10, 64, 64)
    """

    def __init__(self, capacity: int, frame_shape: Tuple[int, int]):
        self.capacity = capacity
        self.frame_shape = frame_shape

        self._buffer = np.zeros((capacity, *frame_shape), dtype=np.float32)
        self._write_idx = 0
        self._count = 0
        self._lock = Lock()

    def append(self, frame: np.ndarray):
        """Append single frame."""
        with self._lock:
            self._buffer[self._write_idx] = frame
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def get_last(self, n: int) -> np.ndarray:
        """Get last n frames."""
        with self._lock:
            n = min(n, self._count)
            if n == 0:
                return np.zeros((0, *self.frame_shape), dtype=np.float32)

            start_idx = (self._write_idx - n) % self.capacity

            if start_idx < self._write_idx:
                return self._buffer[start_idx:self._write_idx].copy()
            else:
                return np.concatenate([
                    self._buffer[start_idx:],
                    self._buffer[:self._write_idx]
                ], axis=0)

    def get_current(self) -> Optional[np.ndarray]:
        """Get most recent frame."""
        with self._lock:
            if self._count == 0:
                return None
            idx = (self._write_idx - 1) % self.capacity
            return self._buffer[idx].copy()

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._write_idx = 0
            self._count = 0

    @property
    def count(self) -> int:
        return self._count


class FeatureBuffer:
    """
    Buffer for feature vectors with timestamps.

    Parameters
    ----------
    capacity : int
        Maximum number of feature vectors
    n_features : int
        Feature vector length
    """

    def __init__(self, capacity: int, n_features: int):
        self.capacity = capacity
        self.n_features = n_features

        self._features = np.zeros((capacity, n_features), dtype=np.float32)
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._write_idx = 0
        self._count = 0
        self._lock = Lock()

    def append(self, features: np.ndarray, timestamp: float):
        """Append feature vector with timestamp."""
        with self._lock:
            self._features[self._write_idx] = features
            self._timestamps[self._write_idx] = timestamp
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def get_last(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get last n feature vectors and timestamps."""
        with self._lock:
            n = min(n, self._count)
            if n == 0:
                return (np.zeros((0, self.n_features), dtype=np.float32),
                        np.zeros(0, dtype=np.float64))

            start_idx = (self._write_idx - n) % self.capacity

            if start_idx < self._write_idx:
                return (self._features[start_idx:self._write_idx].copy(),
                        self._timestamps[start_idx:self._write_idx].copy())
            else:
                return (np.concatenate([
                            self._features[start_idx:],
                            self._features[:self._write_idx]
                        ], axis=0),
                        np.concatenate([
                            self._timestamps[start_idx:],
                            self._timestamps[:self._write_idx]
                        ]))

    def get_time_range(self, start_time: float, end_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get features within time range."""
        features, timestamps = self.get_last(self._count)
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        return features[mask], timestamps[mask]

    def clear(self):
        with self._lock:
            self._features.fill(0)
            self._timestamps.fill(0)
            self._write_idx = 0
            self._count = 0
