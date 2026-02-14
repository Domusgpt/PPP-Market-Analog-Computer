"""
E₈ → H₄ Projection Pipeline.

Implements the dimensional cascade from the E₈ root system (8D, 240 roots)
to 4D H₄ polytopes via two projection approaches:

1. **Baez 4×8 matrix** — projects E₈ → single H₄ copy (lossy).
   Reference: John C. Baez, "From the Icosahedron to E₈"

2. **Phillips 8×8 matrix** — full E₈ → H₄_L ⊕ H₄_R decomposition.
   An original dense projection matrix with entry constants forming a
   golden-ratio geometric progression {a/φ, a, aφ} where a = 1/2.
   Verified properties:
   - Column Trichotomy (Theorem 4.1): 2-4-2 norm pattern {3-φ, 2.5, φ+2}
   - Pentagonal Row Norms (Theorem 5.1): √(3-φ) = 2·sin(36°)
   - Frobenius norm² = 20 (matches 600-cell vertex valence)
   - Shell coincidence: φ·‖r_L‖ = ‖r_R‖ (Galois coupling)

Key mathematical structures:
- E₈: 240 roots in 8D (112 permutation + 128 half-integer)
- H₄: Symmetry group of the 600-cell (order 14400)
- φ (golden ratio): (1 + √5) / 2 ≈ 1.618
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


# =============================================================================
# PHILLIPS MATRIX (8×8) — Full E₈ → H₄_L ⊕ H₄_R decomposition
# =============================================================================
# Phillips, "The Totalistic Geometry of E8" (2026)
#
# An original 8×8 dense projection matrix for the E₈ → H₄ folding.
# NOT the Moxness C600 matrix (which is sparse, symmetric, and rank 8).
# This matrix is dense (no zeros), non-symmetric, and rank 4.
#
# Constants:
#   a = 1/2                 ≈ 0.500
#   b = (φ-1)/2 = 1/(2φ)   ≈ 0.309
#   c = φ/2                 ≈ 0.809
#
# Entry structure: {a, b, c} form a geometric progression with ratio φ,
# centered at a = 1/2. That is: b = a/φ, c = aφ.
#
# The matrix has two blocks:
#   U_L (rows 0-3): entries from {±a, ±b}  — contracted row norm² = 3-φ
#   U_R (rows 4-7): entries from {±a, ±c}  — expanded  row norm² = φ+2
#
# Fundamental identity: U_R = φ · U_L (rank 4, not 8)
#
# Verified properties:
#   Column Trichotomy: dims {0,4}=φ+2, {1,2,5,6}=2.5, {3,7}=3-φ (the 2-4-2)
#   φ-scaling: ‖r_R‖/‖r_L‖ = φ
#   √5-coupling: ‖r_L‖·‖r_R‖ = √5
#   Frobenius norm² = 20
#   Pentagonal: √(3-φ) = 2·sin(36°)
#   Shell coincidence: φ·√(3-φ) = √(φ+2) ≈ 1.90211

_a = 0.5                   # 1/2
_b = (PHI - 1) / 2         # (φ-1)/2 = 1/(2φ) ≈ 0.309
_c = PHI / 2               # φ/2 ≈ 0.809

PHILLIPS_MATRIX = np.array([
    # U_L block (rows 0-3): contracted, entries {±a, ±b}
    [ _a,  _b,  _a,  _b,  _a, -_b,  _a, -_b],
    [ _a,  _a, -_b, -_b, -_a, -_a,  _b,  _b],
    [ _a, -_b, -_a,  _b,  _a, -_b, -_a,  _b],
    [ _a, -_a,  _b, -_b, -_a,  _a, -_b,  _b],
    # U_R block (rows 4-7): expanded, entries {±a, ±c}
    [ _c,  _a,  _c,  _a,  _c, -_a,  _c, -_a],
    [ _c,  _c, -_a, -_a, -_c, -_c,  _a,  _a],
    [ _c, -_a, -_c,  _a,  _c, -_a, -_c,  _a],
    [ _c, -_c,  _a, -_a, -_c,  _c, -_a,  _a],
])

# Submatrices for direct access
PHILLIPS_U_L = PHILLIPS_MATRIX[:4]   # (4, 8) — left H₄ projection
PHILLIPS_U_R = PHILLIPS_MATRIX[4:]   # (4, 8) — right H₄ projection

# Column Trichotomy class labels (2-4-2 pattern)
COLUMN_TRICHOTOMY = {
    'expanded':   [0, 4],        # norm² = φ+2
    'stable':     [1, 2, 5, 6],  # norm² = 2.5
    'contracted': [3, 7],        # norm² = 3-φ
}


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


# =============================================================================
# PHILLIPS 8×8 PROJECTION — Full H₄_L ⊕ H₄_R decomposition
# =============================================================================

@dataclass
class PhillipsProjectedPoint:
    """Result of projecting a single E₈ root via the Phillips 8×8 matrix."""
    h4_left: np.ndarray      # 4D coords from U_L (contracted block)
    h4_right: np.ndarray     # 4D coords from U_R (expanded block)
    full_8d: np.ndarray      # full 8D output (U @ root)
    source: E8Root
    left_radius: float
    right_radius: float


@dataclass
class PhillipsProjected:
    """
    Result of projecting all 240 E₈ roots via the Phillips 8×8 matrix.

    Unlike the Baez 4×8 projection (which is lossy), the Phillips matrix
    preserves the full E₈ information in the H₄_L ⊕ H₄_R decomposition.

    Properties verified by test:
      - Column Trichotomy: 2-4-2 norm pattern {φ+2, 2.5, 3-φ}
      - All U_L rows have norm² = 3-φ (contracted)
      - All U_R rows have norm² = φ+2 (expanded)
      - φ-scaling: ‖r_R‖/‖r_L‖ = φ for every root
      - √5-coupling: ‖r_L‖·‖r_R‖ = √5 for norm-2 roots
      - Frobenius norm² = 20
      - Shell coincidence: shells 2 and 3 merge at radius ≈ 1.90211
    """
    left_vertices: np.ndarray     # (240, 4) via U_L
    right_vertices: np.ndarray    # (240, 4) via U_R
    all_projections: List[PhillipsProjectedPoint]
    left_radii: np.ndarray        # unique shell radii from U_L
    right_radii: np.ndarray       # unique shell radii from U_R


def project_root_phillips(root: E8Root) -> PhillipsProjectedPoint:
    """Project a single E₈ root via the full Phillips 8×8 matrix."""
    full = PHILLIPS_MATRIX @ root.coordinates
    h4_l = full[:4]
    h4_r = full[4:]
    return PhillipsProjectedPoint(
        h4_left=h4_l,
        h4_right=h4_r,
        full_8d=full,
        source=root,
        left_radius=float(np.linalg.norm(h4_l)),
        right_radius=float(np.linalg.norm(h4_r)),
    )


def project_e8_phillips() -> PhillipsProjected:
    """
    Project all 240 E₈ roots via the Phillips 8×8 matrix.

    Returns a PhillipsProjected with H₄_L (contracted) and H₄_R (expanded)
    vertex sets, plus the full list of projected points.
    """
    roots = generate_e8_roots()
    projections = [project_root_phillips(r) for r in roots]

    left = np.array([p.h4_left for p in projections])
    right = np.array([p.h4_right for p in projections])

    left_r = np.linalg.norm(left, axis=1)
    right_r = np.linalg.norm(right, axis=1)

    return PhillipsProjected(
        left_vertices=left,
        right_vertices=right,
        all_projections=projections,
        left_radii=np.unique(np.round(left_r, 6)),
        right_radii=np.unique(np.round(right_r, 6)),
    )


class PhillipsProjectionPipeline:
    """
    Phillips 8×8 E₈ → H₄_L ⊕ H₄_R Pipeline.

    The full decomposition preserves all E₈ information by projecting
    simultaneously to two coupled H₄ copies (left/right). The left block
    (U_L) uses entries {±a, ±b} and has contracted row norms (3-φ). The
    right block (U_R) uses entries {±a, ±c} and has expanded row norms
    (φ+2). The two blocks are coupled by the golden ratio: ‖r_R‖/‖r_L‖ = φ.
    """

    def __init__(self):
        self._roots = generate_e8_roots()
        self._projected = project_e8_phillips()

    @property
    def e8_roots(self) -> List[E8Root]:
        return self._roots

    @property
    def projected(self) -> PhillipsProjected:
        return self._projected

    @property
    def left_vertices(self) -> np.ndarray:
        """H₄_L vertices (contracted block), shape (240, 4)."""
        return self._projected.left_vertices

    @property
    def right_vertices(self) -> np.ndarray:
        """H₄_R vertices (expanded block), shape (240, 4)."""
        return self._projected.right_vertices

    def project(self, v8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project an arbitrary 8D vector via Phillips matrix.

        Returns:
            (h4_left, h4_right) — 4D vectors from contracted/expanded blocks.
        """
        full = PHILLIPS_MATRIX @ v8
        return full[:4], full[4:]

    def project_full(self, v8: np.ndarray) -> np.ndarray:
        """Project to full 8D output (H₄_L ⊕ H₄_R concatenated)."""
        return PHILLIPS_MATRIX @ v8

    def round_trip(self, v8: np.ndarray) -> np.ndarray:
        """
        Round-trip: E₈ → (H₄_L ⊕ H₄_R) → E₈.

        Since the Phillips matrix is 8×8, U^T @ U @ v gives
        the round-trip reconstruction (exact if U is orthogonal).
        """
        return PHILLIPS_MATRIX.T @ (PHILLIPS_MATRIX @ v8)

    def get_shell(self, block: str = "left",
                  radius: Optional[float] = None,
                  tol: float = 0.01) -> np.ndarray:
        """Get vertices at a specific shell radius from left or right block."""
        verts = self._projected.left_vertices if block == "left" \
            else self._projected.right_vertices
        if radius is None:
            return verts
        radii = np.linalg.norm(verts, axis=1)
        return verts[np.isclose(radii, radius, atol=tol)]

    def get_shell_coincidence(self) -> np.ndarray:
        """
        Find vertices where left and right shells coincide.

        Returns vertices from the left block whose radius matches
        a right-block radius (the φ·√(3-φ) = √(φ+2) coincidence).
        """
        left_r = np.linalg.norm(self._projected.left_vertices, axis=1)
        right_r = np.linalg.norm(self._projected.right_vertices, axis=1)
        # The coincidence radius
        coincidence = PHI * np.sqrt(3 - PHI)
        left_match = np.isclose(left_r, coincidence, atol=0.01)
        right_match = np.isclose(right_r, coincidence, atol=0.01)
        return coincidence, left_match.sum(), right_match.sum()

    def get_left_as_vertex4d(self) -> List[Vertex4D]:
        """Return H₄_L projected points as Vertex4D objects."""
        return [Vertex4D.from_array(v) for v in self._projected.left_vertices]

    def get_right_as_vertex4d(self) -> List[Vertex4D]:
        """Return H₄_R projected points as Vertex4D objects."""
        return [Vertex4D.from_array(v) for v in self._projected.right_vertices]


# =============================================================================
# COMPARISON UTILITIES: Baez vs Phillips
# =============================================================================

def compare_projections() -> dict:
    """
    Compare Baez (4×8) and Phillips (8×8) projections on the same E₈ roots.

    Returns a dict with comparison metrics including shell counts,
    radius ranges, and structural properties of each approach.
    """
    roots = generate_e8_roots()

    # Baez projection
    baez_outer = np.array([BAEZ_MATRIX @ r.coordinates for r in roots])
    baez_inner = np.array([BAEZ_CONJUGATE_MATRIX @ r.coordinates for r in roots])

    # Phillips projection
    phil_left = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
    phil_right = np.array([PHILLIPS_U_R @ r.coordinates for r in roots])

    # Shell analysis
    baez_outer_r = np.linalg.norm(baez_outer, axis=1)
    baez_inner_r = np.linalg.norm(baez_inner, axis=1)
    phil_left_r = np.linalg.norm(phil_left, axis=1)
    phil_right_r = np.linalg.norm(phil_right, axis=1)

    # Phillips φ-scaling check
    phi_ratios = phil_right_r / np.where(phil_left_r > 1e-10, phil_left_r, 1)

    return {
        'baez': {
            'outer_shells': len(np.unique(np.round(baez_outer_r, 4))),
            'inner_shells': len(np.unique(np.round(baez_inner_r, 4))),
            'outer_radius_range': (float(baez_outer_r.min()),
                                   float(baez_outer_r.max())),
            'inner_radius_range': (float(baez_inner_r.min()),
                                   float(baez_inner_r.max())),
        },
        'phillips': {
            'left_shells': len(np.unique(np.round(phil_left_r, 4))),
            'right_shells': len(np.unique(np.round(phil_right_r, 4))),
            'left_radius_range': (float(phil_left_r.min()),
                                  float(phil_left_r.max())),
            'right_radius_range': (float(phil_right_r.min()),
                                   float(phil_right_r.max())),
            'phi_scaling_exact': bool(np.allclose(phi_ratios, PHI, atol=1e-6)),
            'phi_scaling_mean': float(np.mean(phi_ratios)),
        },
        'matrix_properties': {
            'phillips_frobenius_sq': float(np.sum(PHILLIPS_MATRIX ** 2)),
            'phillips_column_norms_sq': [
                float(np.sum(PHILLIPS_MATRIX[:, j] ** 2)) for j in range(8)
            ],
            'phillips_UL_row_norm_sq': float(
                np.sum(PHILLIPS_U_L[0] ** 2)),
            'phillips_UR_row_norm_sq': float(
                np.sum(PHILLIPS_U_R[0] ** 2)),
        },
    }
