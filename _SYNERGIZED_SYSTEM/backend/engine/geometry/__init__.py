"""
H4 Geometry Module - Four-dimensional polytope mathematics.

This module provides the geometric foundation for the H4 Constellation
prototype, implementing the 24-cell, 16-cell trilatic decomposition,
600-cell projection, and quaternion-based 4D rotation operations.
"""

from .h4_geometry import (
    H4Geometry,
    Polytope24Cell,
    Polytope16Cell,
    Polytope600Cell,
    Polytope120Cell,
    TrilaticDecomposition,
    LayerPlane,
)
from .quaternion_4d import (
    Quaternion4D,
    QuaternionRotation,
    IsoclinicRotation,
)
from .e8_projection import (
    E8ProjectionPipeline,
    E8Root,
    E8RootType,
    E8Projected,
    generate_e8_roots,
    project_e8_to_h4,
    galois_conjugate,
    icosian_norm,
    normalize_icosian,
    BAEZ_MATRIX,
    BAEZ_CONJUGATE_MATRIX,
)

__all__ = [
    "H4Geometry",
    "Polytope24Cell",
    "Polytope16Cell",
    "Polytope600Cell",
    "Polytope120Cell",
    "TrilaticDecomposition",
    "LayerPlane",
    "Quaternion4D",
    "QuaternionRotation",
    "IsoclinicRotation",
    "E8ProjectionPipeline",
    "E8Root",
    "E8RootType",
    "E8Projected",
    "generate_e8_roots",
    "project_e8_to_h4",
    "galois_conjugate",
    "icosian_norm",
    "normalize_icosian",
    "BAEZ_MATRIX",
    "BAEZ_CONJUGATE_MATRIX",
]
