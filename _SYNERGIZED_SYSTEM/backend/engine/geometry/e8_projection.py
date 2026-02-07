"""
E₈ → H₄ Projection Pipeline.

Implements the dimensional cascade from the E₈ root system (8D, 240 roots)
to 4D H₄ polytopes via the Baez 4×8 projection matrix.

Key mathematical structures:
- E₈: 240 roots in 8D, split into two φ-scaled 600-cells under projection
- H₄: Symmetry group of the 600-cell (order 14400)
- φ (golden ratio): (1 + √5) / 2 ≈ 1.618

The projection uses Galois conjugation (φ ↔ -1/φ) to produce two nested
600-cells: an outer (φ-scaled) and an inner (φ'-scaled).

Reference: John C. Baez, "From the Icosahedron to E₈"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

from .h4_geometry import PHI, PHI_INV, Vertex4D


# =============================================================================
# CONSTANTS
# =============================================================================

PHI_CONJUGATE = (1 - np.sqrt(5)) / 2  # = -1/φ ≈ -0.618

# Baez 4×8 projection matrix (each row divided by 2 for normalization)
# Projects E₈ roots onto H₄ (4D) via the icosian representation.
BAEZ_MATRIX = np.array([
    [1,   PHI, 0, -1,  PHI, 0,  0, 0],
    [PHI, 0,   1,  PHI, 0, -1,  0, 0],
    [0,   1, PHI,  0,  -1, PHI, 0, 0],
    [0,   0,   0,  0,   0,  0,  1, PHI],
]) / 2.0

# Conjugate matrix: replace φ with φ' = -1/φ = (1-√5)/2
BAEZ_CONJUGATE_MATRIX = np.array([
    [1,   PHI_CONJUGATE, 0, -1,  PHI_CONJUGATE, 0,  0, 0],
    [PHI_CONJUGATE, 0,   1,  PHI_CONJUGATE, 0, -1,  0, 0],
    [0,   1, PHI_CONJUGATE,  0,  -1, PHI_CONJUGATE, 0, 0],
    [0,   0,   0,  0,   0,  0,  1, PHI_CONJUGATE],
]) / 2.0


class E8RootType(Enum):
    """Classification of E₈ root vectors."""
    PERMUTATION = "permutation"    # (±1, ±1, 0, 0, 0, 0, 0, 0) — 112 roots
    HALF_INTEGER = "half_integer"  # (±½)^8 with even parity — 128 roots


@dataclass
class E8Root:
    """An E₈ root vector in 8D."""
    coordinates: np.ndarray  # shape (8,)
    root_type: E8RootType

    @property
    def norm_squared(self) -> float:
        """Squared norm (should be 2 for all E₈ roots)."""
        return float(np.dot(self.coordinates, self.coordinates))


@dataclass
class ProjectedPoint:
    """Result of projecting an E₈ root to 4D."""
    outer: np.ndarray        # 4D coords via φ-matrix
    inner: np.ndarray        # 4D coords via φ'-matrix
    source: E8Root
    outer_radius: float
    inner_radius: float


@dataclass
class E8Projected:
    """
    Result of projecting all 240 E₈ roots to 4D via both Baez matrices.

    The φ-matrix projects to `outer_vertices` (240 points at multiple radii).
    The φ'-matrix projects to `inner_vertices` (240 points at smaller radii).
    Together they encode the E₈ root system in the H₄ × H₄ decomposition.

    The outer projection has 8 concentric shells; the maximal-radius shell
    contains vertices related to the 600-cell geometry.
    """
    outer_vertices: np.ndarray   # (240, 4) — projected via φ-matrix
    inner_vertices: np.ndarray   # (240, 4) — projected via φ'-matrix
    all_projections: List[ProjectedPoint]
    outer_radii: np.ndarray      # (N,) unique outer shell radii
    inner_radii: np.ndarray      # (N,) unique inner shell radii


# =============================================================================
# E₈ ROOT GENERATION
# =============================================================================

def generate_e8_roots() -> List[E8Root]:
    """
    Generate all 240 roots of the E₈ lattice.

    Type 1 (112 roots): All vectors with two nonzero entries ±1, rest 0.
        These are permutations of (±1, ±1, 0, 0, 0, 0, 0, 0).

    Type 2 (128 roots): All vectors (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½)
        with an even number of minus signs.

    Every root has squared norm = 2.
    """
    roots: List[E8Root] = []

    # Type 1: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1, 1):
                for sj in (-1, 1):
                    coords = np.zeros(8)
                    coords[i] = si
                    coords[j] = sj
                    roots.append(E8Root(
                        coordinates=coords,
                        root_type=E8RootType.PERMUTATION,
                    ))

    # Type 2: (±½)^8 with even number of minus signs
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

    return roots


# =============================================================================
# PROJECTION
# =============================================================================

def project_root(root: E8Root) -> ProjectedPoint:
    """Project a single E₈ root to 4D via both Baez matrices."""
    outer = BAEZ_MATRIX @ root.coordinates
    inner = BAEZ_CONJUGATE_MATRIX @ root.coordinates
    return ProjectedPoint(
        outer=outer,
        inner=inner,
        source=root,
        outer_radius=float(np.linalg.norm(outer)),
        inner_radius=float(np.linalg.norm(inner)),
    )



def project_e8_to_h4() -> E8Projected:
    """
    Project all 240 E₈ roots to 4D via both Baez matrices.

    Returns an E8Projected containing:
    - outer_vertices: (240, 4) projected via φ-matrix
    - inner_vertices: (240, 4) projected via φ'-matrix
    - all_projections: full list of 240 ProjectedPoint objects
    - outer_radii / inner_radii: unique shell radii
    """
    roots = generate_e8_roots()
    projections = [project_root(r) for r in roots]

    outer = np.array([p.outer for p in projections])
    inner = np.array([p.inner for p in projections])

    outer_r = np.linalg.norm(outer, axis=1)
    inner_r = np.linalg.norm(inner, axis=1)
    outer_unique_r = np.unique(np.round(outer_r, 6))
    inner_unique_r = np.unique(np.round(inner_r, 6))

    return E8Projected(
        outer_vertices=outer,
        inner_vertices=inner,
        all_projections=projections,
        outer_radii=outer_unique_r,
        inner_radii=inner_unique_r,
    )


# =============================================================================
# GALOIS CONJUGATION
# =============================================================================

def galois_conjugate(v4: np.ndarray) -> np.ndarray:
    """
    Apply Galois conjugation to a 4D point.

    Maps φ ↔ -1/φ in the coordinate expressions, effectively
    scaling by (φ')²/φ² = 1/φ⁴. In practice the conjugate projection
    is obtained directly from BAEZ_CONJUGATE_MATRIX, so this function
    provides an approximate mapping between the two shells.
    """
    scale = 1.0 / (PHI * PHI)  # 1/φ²
    return v4 * scale


# =============================================================================
# ICOSIAN NORM
# =============================================================================

def icosian_norm(v4: np.ndarray) -> float:
    """Compute icosian norm: ||q||² = w² + x² + y² + z²."""
    return float(np.dot(v4, v4))


def normalize_icosian(v4: np.ndarray) -> np.ndarray:
    """Normalize a 4D vector to unit icosian norm."""
    n = np.sqrt(icosian_norm(v4))
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return v4 / n


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class E8ProjectionPipeline:
    """
    E₈ to H₄ Projection Pipeline.

    Manages the dimensional cascade from E₈ (8D) to H₄ (4D) polytopes.
    Provides access to the 240 E₈ roots, their 4D projections, and the
    resulting multi-shell structure in 4D.
    """

    def __init__(self):
        self._roots = generate_e8_roots()
        self._projected = project_e8_to_h4()

    @property
    def e8_roots(self) -> List[E8Root]:
        return self._roots

    @property
    def projected(self) -> E8Projected:
        return self._projected

    @property
    def outer_vertices(self) -> np.ndarray:
        """Projected via φ-matrix, shape (240, 4)."""
        return self._projected.outer_vertices

    @property
    def inner_vertices(self) -> np.ndarray:
        """Projected via φ'-matrix, shape (240, 4)."""
        return self._projected.inner_vertices

    def project(self, v8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project an arbitrary 8D vector to 4D.

        Returns:
            (outer_4d, inner_4d) tuple of 4D vectors.
        """
        outer = BAEZ_MATRIX @ v8
        inner = BAEZ_CONJUGATE_MATRIX @ v8
        return outer, inner

    def find_nearest_root(self, v8: np.ndarray) -> E8Root:
        """Find the E₈ root nearest to an 8D point."""
        best = self._roots[0]
        best_dist = np.inf
        for root in self._roots:
            d = np.sum((v8 - root.coordinates) ** 2)
            if d < best_dist:
                best_dist = d
                best = root
        return best

    def get_vertices_at_scale(self, scale: str = "both") -> np.ndarray:
        """
        Get projected vertices at a specific scale.

        Args:
            scale: "outer", "inner", or "both"
        """
        if scale == "outer":
            return self._projected.outer_vertices
        elif scale == "inner":
            return self._projected.inner_vertices
        else:
            return np.vstack([
                self._projected.outer_vertices,
                self._projected.inner_vertices,
            ])

    def get_shell(self, matrix: str = "outer",
                  radius: Optional[float] = None,
                  tol: float = 0.01) -> np.ndarray:
        """
        Get projected vertices at a specific shell radius.

        Args:
            matrix: "outer" (φ) or "inner" (φ')
            radius: target radius (None = all vertices)
            tol: matching tolerance

        Returns:
            (N, 4) array of vertices at the specified shell radius.
        """
        verts = self._projected.outer_vertices if matrix == "outer" \
            else self._projected.inner_vertices
        if radius is None:
            return verts
        radii = np.linalg.norm(verts, axis=1)
        mask = np.isclose(radii, radius, atol=tol)
        return verts[mask]

    def interpolate_scale(self, t: float) -> np.ndarray:
        """
        Interpolate between outer and inner projections.

        Args:
            t: 0.0 = outer (φ-matrix), 1.0 = inner (φ'-matrix)

        Returns:
            Interpolated vertex array, shape (240, 4).
        """
        return self._projected.outer_vertices * (1 - t) + \
            self._projected.inner_vertices * t

    def get_outer_as_vertex4d(self) -> List[Vertex4D]:
        """Return outer projected points as Vertex4D objects."""
        return [Vertex4D.from_array(v) for v in self._projected.outer_vertices]

    def get_inner_as_vertex4d(self) -> List[Vertex4D]:
        """Return inner projected points as Vertex4D objects."""
        return [Vertex4D.from_array(v) for v in self._projected.inner_vertices]
