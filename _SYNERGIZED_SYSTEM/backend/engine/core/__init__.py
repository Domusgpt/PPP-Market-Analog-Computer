"""
Core Module - High-Performance Encoder Components
=================================================

This module contains optimized, vectorized implementations
of the encoder components for production use.

Modules:
- fast_moire: Vectorized moiré pattern computation
- fast_cascade: Numba-accelerated reservoir dynamics
- batch_encoder: Batch processing API
- cache: Computation caching utilities
"""

from .fast_moire import FastMoireComputer
from .fast_cascade import FastCascadeSimulator
from .batch_encoder import BatchEncoder
from .cache import PatternCache

__all__ = [
    "FastMoireComputer",
    "FastCascadeSimulator",
    "BatchEncoder",
    "PatternCache"
]
