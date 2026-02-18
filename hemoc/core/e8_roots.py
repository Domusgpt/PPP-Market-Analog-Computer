"""
E8 Root System Generator
=========================

Generates all 240 roots of the E8 lattice in 8-dimensional Euclidean space.

The E8 root system consists of two classes:

  Type 1 (112 roots):  All vectors with exactly two non-zero entries in
      {+1, -1}, the rest zero.  These are the permutations of
      (+-1, +-1, 0, 0, 0, 0, 0, 0).

  Type 2 (128 roots):  All vectors (+-1/2)^8 with an EVEN number of
      minus signs.

Every root has squared norm equal to 2.

References
----------
- Conway & Sloane, "Sphere Packings, Lattices and Groups" (3rd ed.), Ch. 8.
- Humphreys, "Reflection Groups and Coxeter Groups", Ch. 2.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List

import numpy as np


class E8RootType(Enum):
    """Classification of E8 root vectors."""
    PERMUTATION = "permutation"      # (+-1, +-1, 0, ..., 0) -- 112 roots
    HALF_INTEGER = "half_integer"    # (+-1/2)^8, even parity -- 128 roots


@dataclass(frozen=True)
class E8Root:
    """
    A single root of the E8 lattice.

    Attributes
    ----------
    coordinates : np.ndarray
        Length-8 vector in R^8.
    root_type : E8RootType
        Whether this root is a permutation type or half-integer type.
    """
    coordinates: np.ndarray
    root_type: E8RootType

    @property
    def norm_squared(self) -> float:
        """Squared Euclidean norm (should be exactly 2 for all E8 roots)."""
        return float(self.coordinates @ self.coordinates)

    def __eq__(self, other):
        if not isinstance(other, E8Root):
            return False
        return np.allclose(self.coordinates, other.coordinates, atol=1e-12)

    def __hash__(self):
        return hash(tuple(np.round(self.coordinates, 10)))


def generate_e8_roots() -> List[E8Root]:
    """
    Generate all 240 roots of the E8 lattice.

    Returns
    -------
    List[E8Root]
        Exactly 240 root vectors, each with ||r||^2 = 2.

    Notes
    -----
    The two construction classes yield 112 + 128 = 240 roots.  This matches
    the standard E8 root-system cardinality (Conway & Sloane, Ch. 8).
    """
    roots: List[E8Root] = []

    # --- Type 1: permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) ---
    # Choose 2 positions out of 8 = C(8,2) = 28 position pairs.
    # Each pair has 4 sign combinations -> 28 * 4 = 112 roots.
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    coords = np.zeros(8)
                    coords[i] = si
                    coords[j] = sj
                    roots.append(E8Root(
                        coordinates=coords,
                        root_type=E8RootType.PERMUTATION,
                    ))

    # --- Type 2: (+-1/2)^8 with even number of minus signs ---
    # 2^8 = 256 sign patterns, half have even parity -> 128 roots.
    for pattern in range(256):
        minus_count = bin(pattern).count('1')
        if minus_count % 2 == 0:
            coords = np.array([
                -0.5 if (pattern >> bit) & 1 else 0.5
                for bit in range(8)
            ])
            roots.append(E8Root(
                coordinates=coords,
                root_type=E8RootType.HALF_INTEGER,
            ))

    assert len(roots) == 240, f"Expected 240 E8 roots, got {len(roots)}"
    return roots


def verify_e8_roots(roots: List[E8Root]) -> dict:
    """
    Verify that a list of E8 roots satisfies the standard properties.

    Returns a dict with verification results (all should be True for
    a correctly generated root system).
    """
    coords = np.array([r.coordinates for r in roots])
    norms_sq = np.sum(coords ** 2, axis=1)

    n_perm = sum(1 for r in roots if r.root_type == E8RootType.PERMUTATION)
    n_half = sum(1 for r in roots if r.root_type == E8RootType.HALF_INTEGER)

    # Check that all roots sum to zero (centroid at origin)
    centroid = np.mean(coords, axis=0)

    return {
        "total_count": len(roots),
        "count_correct": len(roots) == 240,
        "permutation_count": n_perm,
        "half_integer_count": n_half,
        "type_counts_correct": n_perm == 112 and n_half == 128,
        "all_norm_sq_2": bool(np.allclose(norms_sq, 2.0, atol=1e-12)),
        "centroid_at_origin": bool(np.allclose(centroid, 0.0, atol=1e-12)),
        "centroid": centroid.tolist(),
    }
