"""
HDC (Hyperdimensional Computing) Module
========================================

Brain-inspired computing for pattern encoding and matching.

Converts moiré patterns to high-dimensional binary vectors that:
- Naturally encode temporal sequences via permutation
- Enable fast similarity matching (SLAM loop closure)
- Support associative memory (one-shot learning)
- Run efficiently on edge devices (robotics)
"""

from .encoder import HDCEncoder, HDCConfig
from .memory import AssociativeMemory, MemoryResult
from .operations import (
    random_hypervector,
    bind,
    bundle,
    permute,
    similarity,
    threshold
)

__all__ = [
    "HDCEncoder",
    "HDCConfig",
    "AssociativeMemory",
    "MemoryResult",
    "random_hypervector",
    "bind",
    "bundle",
    "permute",
    "similarity",
    "threshold"
]
