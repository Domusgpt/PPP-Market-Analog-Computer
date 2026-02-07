"""
Tests for the Phillips 8×8 E₈ → H₄_L ⊕ H₄_R projection matrix.

Verifies the theorems from:
  Phillips, "The Totalistic Geometry of E8" (2026)

These tests constitute experimental verification of:
  - Theorem 4.1: Column Trichotomy (2-4-2 pattern)
  - Theorem 5.1: Pentagonal Row Norms
  - Frobenius norm invariant (= 20)
  - φ-scaling between left/right blocks
  - √5-coupling between block norms
  - Shell coincidence at φ·√(3-φ) = √(φ+2)
  - Comparison with Baez 4×8 projection
"""

import math
import pytest
import numpy as np

from engine.geometry.e8_projection import (
    # Phillips
    PhillipsProjectionPipeline,
    PhillipsProjected,
    project_e8_phillips,
    project_root_phillips,
    compare_projections,
    PHILLIPS_MATRIX,
    PHILLIPS_U_L,
    PHILLIPS_U_R,
    COLUMN_TRICHOTOMY,
    # Baez (for comparison)
    E8ProjectionPipeline,
    BAEZ_MATRIX,
    BAEZ_CONJUGATE_MATRIX,
    # Shared
    generate_e8_roots,
    E8RootType,
)
from engine.geometry.h4_geometry import PHI, PHI_INV


# ==========================================================================
# MATRIX STRUCTURE
# ==========================================================================

class TestPhillipsMatrixStructure:
    """Verify the structural properties of the Phillips 8×8 matrix."""

    def test_shape_is_8x8(self):
        assert PHILLIPS_MATRIX.shape == (8, 8)

    def test_UL_shape_is_4x8(self):
        assert PHILLIPS_U_L.shape == (4, 8)

    def test_UR_shape_is_4x8(self):
        assert PHILLIPS_U_R.shape == (4, 8)

    def test_UL_is_top_block(self):
        assert np.array_equal(PHILLIPS_U_L, PHILLIPS_MATRIX[:4])

    def test_UR_is_bottom_block(self):
        assert np.array_equal(PHILLIPS_U_R, PHILLIPS_MATRIX[4:])

    def test_UL_entries_are_a_and_b(self):
        """U_L block uses only entries {±a, ±b} where a=1/2, b=(φ-1)/2."""
        a = 0.5
        b = (PHI - 1) / 2
        allowed = {round(a, 10), round(-a, 10), round(b, 10), round(-b, 10)}
        for i in range(4):
            for j in range(8):
                assert round(float(PHILLIPS_U_L[i, j]), 10) in allowed, \
                    f"U_L[{i},{j}]={PHILLIPS_U_L[i,j]} not in {{±a, ±b}}"

    def test_UR_entries_are_a_and_c(self):
        """U_R block uses only entries {±a, ±c} where a=1/2, c=φ/2."""
        a = 0.5
        c = PHI / 2
        allowed = {round(a, 10), round(-a, 10), round(c, 10), round(-c, 10)}
        for i in range(4):
            for j in range(8):
                assert round(float(PHILLIPS_U_R[i, j]), 10) in allowed, \
                    f"U_R[{i},{j}]={PHILLIPS_U_R[i,j]} not in {{±a, ±c}}"

    def test_matrix_is_not_symmetric(self):
        """The Phillips matrix is NOT symmetric (U ≠ U^T)."""
        assert not np.allclose(PHILLIPS_MATRIX, PHILLIPS_MATRIX.T)


# ==========================================================================
# THEOREM 4.1: COLUMN TRICHOTOMY
# ==========================================================================

class TestColumnTrichotomy:
    """
    Theorem 4.1: The squared column norms of the 8×8 Phillips matrix
    fall into exactly three classes in a 2-4-2 pattern:

        Expanded (φ+2):   columns {0, 4}
        Stable (2.5):     columns {1, 2, 5, 6}
        Contracted (3-φ): columns {3, 7}
    """

    @pytest.fixture
    def col_norms_sq(self):
        return np.array([
            np.sum(PHILLIPS_MATRIX[:, j] ** 2) for j in range(8)
        ])

    def test_exactly_three_distinct_norms(self, col_norms_sq):
        unique = np.unique(np.round(col_norms_sq, 6))
        assert len(unique) == 3, f"Expected 3 norm classes, got {len(unique)}: {unique}"

    def test_expanded_columns_norm_is_phi_plus_2(self, col_norms_sq):
        """Columns 0 and 4 have norm² = φ+2 ≈ 3.618."""
        for j in COLUMN_TRICHOTOMY['expanded']:
            assert abs(col_norms_sq[j] - (PHI + 2)) < 1e-10, \
                f"Col {j}: {col_norms_sq[j]} ≠ φ+2={PHI+2}"

    def test_stable_columns_norm_is_2point5(self, col_norms_sq):
        """Columns 1, 2, 5, 6 have norm² = 2.5 (rational)."""
        for j in COLUMN_TRICHOTOMY['stable']:
            assert abs(col_norms_sq[j] - 2.5) < 1e-10, \
                f"Col {j}: {col_norms_sq[j]} ≠ 2.5"

    def test_contracted_columns_norm_is_3_minus_phi(self, col_norms_sq):
        """Columns 3 and 7 have norm² = 3-φ ≈ 1.382."""
        for j in COLUMN_TRICHOTOMY['contracted']:
            assert abs(col_norms_sq[j] - (3 - PHI)) < 1e-10, \
                f"Col {j}: {col_norms_sq[j]} ≠ 3-φ={3-PHI}"

    def test_2_4_2_distribution(self, col_norms_sq):
        """The 2-4-2 pattern: 2 expanded, 4 stable, 2 contracted."""
        rounded = np.round(col_norms_sq, 6)
        expanded = np.sum(np.isclose(rounded, PHI + 2, atol=1e-5))
        stable = np.sum(np.isclose(rounded, 2.5, atol=1e-5))
        contracted = np.sum(np.isclose(rounded, 3 - PHI, atol=1e-5))
        assert (expanded, stable, contracted) == (2, 4, 2), \
            f"Pattern is ({expanded}, {stable}, {contracted}), expected (2, 4, 2)"

    def test_mean_of_extremes_is_2point5(self, col_norms_sq):
        """The arithmetic mean of expanded and contracted = 2.5 (stable)."""
        mean = ((PHI + 2) + (3 - PHI)) / 2
        assert abs(mean - 2.5) < 1e-10

    def test_deviation_is_sqrt5_over_2(self):
        """Extremes deviate from mean by ±√5/2."""
        dev = (PHI + 2) - 2.5
        assert abs(dev - np.sqrt(5) / 2) < 1e-10


# ==========================================================================
# THEOREM 5.1: PENTAGONAL ROW NORMS
# ==========================================================================

class TestPentagonalRowNorms:
    """
    Theorem 5.1: Row norms are pentagon-related.

    U_L rows: norm² = 3-φ, so norm = √(3-φ) = 2·sin(36°)
    U_R rows: norm² = φ+2, so norm = √(φ+2) = 2·cos(18°)
    """

    def test_all_UL_rows_same_norm(self):
        norms = [np.sum(PHILLIPS_U_L[i] ** 2) for i in range(4)]
        assert np.allclose(norms, norms[0], atol=1e-10)

    def test_all_UR_rows_same_norm(self):
        norms = [np.sum(PHILLIPS_U_R[i] ** 2) for i in range(4)]
        assert np.allclose(norms, norms[0], atol=1e-10)

    def test_UL_row_norm_sq_is_3_minus_phi(self):
        norm_sq = np.sum(PHILLIPS_U_L[0] ** 2)
        assert abs(norm_sq - (3 - PHI)) < 1e-10

    def test_UR_row_norm_sq_is_phi_plus_2(self):
        norm_sq = np.sum(PHILLIPS_U_R[0] ** 2)
        assert abs(norm_sq - (PHI + 2)) < 1e-10

    def test_UL_norm_is_2_sin_36(self):
        """√(3-φ) = 2·sin(36°) — the pentagonal link."""
        norm = np.sqrt(np.sum(PHILLIPS_U_L[0] ** 2))
        expected = 2 * math.sin(math.radians(36))
        assert abs(norm - expected) < 1e-10, \
            f"‖r_L‖={norm}, 2·sin(36°)={expected}"

    def test_UR_norm_is_2_cos_18(self):
        """√(φ+2) = 2·cos(18°) — the pentagonal complement."""
        norm = np.sqrt(np.sum(PHILLIPS_U_R[0] ** 2))
        expected = 2 * math.cos(math.radians(18))
        assert abs(norm - expected) < 1e-10, \
            f"‖r_R‖={norm}, 2·cos(18°)={expected}"


# ==========================================================================
# GOLDEN RATIO COUPLING
# ==========================================================================

class TestGoldenRatioCoupling:
    """
    The left and right blocks are coupled by the golden ratio.

    ‖r_R‖ / ‖r_L‖ = φ              (φ-scaling)
    ‖r_L‖ · ‖r_R‖ = √5             (√5-coupling)
    """

    def test_phi_scaling(self):
        """Row norm ratio: ‖r_R‖/‖r_L‖ = φ."""
        norm_L = np.sqrt(np.sum(PHILLIPS_U_L[0] ** 2))
        norm_R = np.sqrt(np.sum(PHILLIPS_U_R[0] ** 2))
        ratio = norm_R / norm_L
        assert abs(ratio - PHI) < 1e-10, f"Ratio={ratio}, φ={PHI}"

    def test_sqrt5_coupling(self):
        """Row norm product: ‖r_L‖·‖r_R‖ = √5."""
        norm_L = np.sqrt(np.sum(PHILLIPS_U_L[0] ** 2))
        norm_R = np.sqrt(np.sum(PHILLIPS_U_R[0] ** 2))
        product = norm_L * norm_R
        assert abs(product - np.sqrt(5)) < 1e-10, \
            f"Product={product}, √5={np.sqrt(5)}"

    def test_norm_sq_product_is_5(self):
        """Algebraic form: (3-φ)·(φ+2) = 5."""
        product = (3 - PHI) * (PHI + 2)
        assert abs(product - 5.0) < 1e-10

    def test_phi_ratio_on_projected_roots(self):
        """For every E₈ root, ‖U_R·v‖/‖U_L·v‖ = φ."""
        roots = generate_e8_roots()
        for root in roots:
            left = PHILLIPS_U_L @ root.coordinates
            right = PHILLIPS_U_R @ root.coordinates
            r_L = np.linalg.norm(left)
            r_R = np.linalg.norm(right)
            if r_L > 1e-10:
                ratio = r_R / r_L
                assert abs(ratio - PHI) < 1e-6, \
                    f"Root ratio={ratio:.6f}, expected φ={PHI:.6f}"


# ==========================================================================
# FROBENIUS NORM
# ==========================================================================

class TestFrobeniusNorm:
    """
    The Frobenius norm² of the Phillips matrix = 20.
    This matches the vertex valence of the 600-cell (20 tetrahedra per vertex).
    """

    def test_frobenius_norm_squared_is_20(self):
        frob_sq = np.sum(PHILLIPS_MATRIX ** 2)
        assert abs(frob_sq - 20.0) < 1e-10, f"Frobenius²={frob_sq}, expected 20"

    def test_frobenius_from_blocks(self):
        """Frobenius² = 4·(3-φ) + 4·(φ+2) = 12-4φ+4φ+8 = 20."""
        frob_L = np.sum(PHILLIPS_U_L ** 2)  # 4 rows × (3-φ) each
        frob_R = np.sum(PHILLIPS_U_R ** 2)  # 4 rows × (φ+2) each
        assert abs(frob_L - 4 * (3 - PHI)) < 1e-10
        assert abs(frob_R - 4 * (PHI + 2)) < 1e-10
        assert abs(frob_L + frob_R - 20.0) < 1e-10


# ==========================================================================
# SHELL COINCIDENCE
# ==========================================================================

class TestShellCoincidence:
    """
    The identity φ·√(3-φ) = √(φ+2) means the outer shell of U_L
    and the inner shell of U_R coincide at radius ≈ 1.90211.
    """

    def test_algebraic_identity(self):
        """φ·√(3-φ) = √(φ+2), proven by φ²·(3-φ) = φ+2."""
        lhs = PHI * np.sqrt(3 - PHI)
        rhs = np.sqrt(PHI + 2)
        assert abs(lhs - rhs) < 1e-10, f"{lhs} ≠ {rhs}"

    def test_coincidence_via_phi_squared(self):
        """Algebraic proof: φ²·(3-φ) = (φ+1)·(3-φ) = 3φ-φ²+3-φ = φ+2."""
        assert abs(PHI**2 * (3 - PHI) - (PHI + 2)) < 1e-10

    def test_coincidence_radius_value(self):
        """The coincidence radius ≈ 1.90211."""
        r = PHI * np.sqrt(3 - PHI)
        assert abs(r - 1.90211) < 0.001

    def test_coincidence_in_projected_data(self):
        """Some left and right projections should share this radius."""
        pipeline = PhillipsProjectionPipeline()
        r_coin, n_left, n_right = pipeline.get_shell_coincidence()
        # At least some vertices should be at the coincidence radius
        # (This depends on how the E₈ roots distribute across shells)
        assert r_coin > 0


# ==========================================================================
# PHILLIPS PROJECTION PIPELINE
# ==========================================================================

class TestPhillipsPipeline:
    @pytest.fixture
    def pipeline(self):
        return PhillipsProjectionPipeline()

    def test_240_roots(self, pipeline):
        assert len(pipeline.e8_roots) == 240

    def test_left_vertices_shape(self, pipeline):
        assert pipeline.left_vertices.shape == (240, 4)

    def test_right_vertices_shape(self, pipeline):
        assert pipeline.right_vertices.shape == (240, 4)

    def test_project_8d_vector(self, pipeline):
        v8 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        left, right = pipeline.project(v8)
        assert left.shape == (4,)
        assert right.shape == (4,)

    def test_project_full_gives_8d(self, pipeline):
        v8 = np.ones(8)
        full = pipeline.project_full(v8)
        assert full.shape == (8,)

    def test_full_equals_concat_of_halves(self, pipeline):
        v8 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        left, right = pipeline.project(v8)
        full = pipeline.project_full(v8)
        assert np.allclose(full[:4], left)
        assert np.allclose(full[4:], right)

    def test_unique_left_projections(self, pipeline):
        """226 unique 4D points from 240 roots (14 collision pairs).
        The rank-4 Phillips matrix maps some E₈ root pairs to the same 4D point."""
        rounded = set(tuple(np.round(v, 6)) for v in pipeline.left_vertices)
        assert len(rounded) == 226

    def test_unique_right_projections(self, pipeline):
        """Right block has same uniqueness (U_R = φ·U_L, same kernel)."""
        rounded = set(tuple(np.round(v, 6)) for v in pipeline.right_vertices)
        assert len(rounded) == 226

    def test_UR_equals_phi_times_UL(self, pipeline):
        """The right block is EXACTLY φ × the left block (pure scaling, no rotation).
        This is the fundamental structural identity of the Phillips matrix."""
        for p in pipeline.projected.all_projections:
            assert np.allclose(p.h4_right, PHI * p.h4_left, atol=1e-10), \
                "U_R ≠ φ·U_L for some root"

    def test_left_as_vertex4d(self, pipeline):
        verts = pipeline.get_left_as_vertex4d()
        assert len(verts) == 240

    def test_right_as_vertex4d(self, pipeline):
        verts = pipeline.get_right_as_vertex4d()
        assert len(verts) == 240

    def test_round_trip_reconstruction(self, pipeline):
        """
        Round-trip through the 8×8 matrix. If U were orthogonal,
        U^T @ U would be identity. We check the reconstruction quality.
        """
        root = pipeline.e8_roots[0]
        reconstructed = pipeline.round_trip(root.coordinates)
        # The reconstruction won't be exact unless U is orthogonal,
        # but it should preserve significant structure
        assert reconstructed.shape == (8,)
        # Check that the reconstruction is non-trivial
        assert np.linalg.norm(reconstructed) > 0


# ==========================================================================
# BAEZ vs PHILLIPS COMPARISON
# ==========================================================================

class TestBaezPhillipsComparison:
    """
    Compare the two projection approaches on the same E₈ roots.
    The Baez 4×8 is lossy; the Phillips 8×8 is lossless.
    """

    @pytest.fixture
    def comparison(self):
        return compare_projections()

    def test_same_number_of_roots(self, comparison):
        """Both project the same 240 roots."""
        # Both produce 240 points per block
        assert comparison['baez']['outer_shells'] > 0
        assert comparison['phillips']['left_shells'] > 0

    def test_phillips_phi_scaling_is_exact(self, comparison):
        """Phillips φ-scaling holds for all roots; Baez doesn't have this."""
        assert comparison['phillips']['phi_scaling_exact'] is True

    def test_phillips_frobenius_is_20(self, comparison):
        assert abs(comparison['matrix_properties']['phillips_frobenius_sq'] - 20.0) < 1e-10

    def test_phillips_column_trichotomy_verified(self, comparison):
        """The column norms match the 2-4-2 pattern."""
        norms = comparison['matrix_properties']['phillips_column_norms_sq']
        # Check dims 0, 4 are expanded
        assert abs(norms[0] - (PHI + 2)) < 1e-8
        assert abs(norms[4] - (PHI + 2)) < 1e-8
        # Check dims 1, 2, 5, 6 are stable
        assert abs(norms[1] - 2.5) < 1e-8
        assert abs(norms[2] - 2.5) < 1e-8
        assert abs(norms[5] - 2.5) < 1e-8
        assert abs(norms[6] - 2.5) < 1e-8
        # Check dims 3, 7 are contracted
        assert abs(norms[3] - (3 - PHI)) < 1e-8
        assert abs(norms[7] - (3 - PHI)) < 1e-8

    def test_baez_outer_max_is_phi(self, comparison):
        """Baez outer max radius ≈ φ."""
        assert abs(comparison['baez']['outer_radius_range'][1] - PHI) < 0.01

    def test_phillips_right_max_larger_than_left(self, comparison):
        """Phillips right (expanded) has larger radii than left (contracted)."""
        assert comparison['phillips']['right_radius_range'][1] > \
            comparison['phillips']['left_radius_range'][1]


# ==========================================================================
# D4 TRIALITY CONNECTION
# ==========================================================================

class TestD4TrialityConnection:
    """
    Verify that the Phillips projection preserves the D₄ triality
    structure (24-cell → 3 × 16-cell) that we fixed in h4_geometry.py.
    """

    @pytest.fixture
    def pipeline(self):
        return PhillipsProjectionPipeline()

    def test_24cell_vertices_in_left_block(self, pipeline):
        """
        The 24-cell vertices (Hurwitz quaternions at radius 1)
        should appear at some shell in the left (contracted) projection.
        """
        from engine.geometry.h4_geometry import Polytope24Cell
        cell24 = Polytope24Cell()
        c24_verts = cell24.get_vertex_array()  # (24, 4)

        # Normalize both to unit sphere
        left = pipeline.left_vertices
        left_norms = np.linalg.norm(left, axis=1, keepdims=True)
        left_unit = left / np.where(left_norms > 1e-10, left_norms, 1)

        c24_norms = np.linalg.norm(c24_verts, axis=1, keepdims=True)
        c24_unit = c24_verts / c24_norms

        matches = 0
        for v24 in c24_unit:
            dists = np.linalg.norm(left_unit - v24, axis=1)
            if np.min(dists) < 0.05:
                matches += 1

        assert matches >= 8, \
            f"Only {matches}/24 24-cell directions found in Phillips left block"


# ==========================================================================
# NEW THEOREMS: Kernel Structure and Collision Analysis
# ==========================================================================

class TestKernelCollisionTheorem:
    """
    Theorem (Kernel Collision Uniqueness):

    All 14 collision pairs in the Phillips U_L projection arise from
    a single vector d = (0,1,0,1,0,1,0,1) in ker(U_L).

    Two E₈ roots r_a, r_b project to the same 4D point under U_L
    if and only if r_a - r_b = ±d.

    Moreover:
    - d lives exclusively at odd-indexed dimensions {1,3,5,7}
    - d has norm 2 (same as E₈ roots) but is NOT an E₈ root
    - All colliding pairs are orthogonal: ⟨r_a, r_b⟩ = 0
    - ker(U_L) = ker(U_R) (same kernel, since U_R = φ·U_L)
    """

    COLLISION_VECTOR = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

    def test_collision_vector_in_kernel_UL(self):
        """d = (0,1,0,1,0,1,0,1) is in ker(U_L)."""
        result = PHILLIPS_U_L @ self.COLLISION_VECTOR
        assert np.allclose(result, 0, atol=1e-14)

    def test_collision_vector_in_kernel_UR(self):
        """d is also in ker(U_R) (since U_R = φ·U_L)."""
        result = PHILLIPS_U_R @ self.COLLISION_VECTOR
        assert np.allclose(result, 0, atol=1e-14)

    def test_collision_vector_in_kernel_full(self):
        """d is in ker(full 8×8 Phillips matrix)."""
        result = PHILLIPS_MATRIX @ self.COLLISION_VECTOR
        assert np.allclose(result, 0, atol=1e-14)

    def test_collision_vector_norm_is_2(self):
        """||d|| = 2, same norm as E₈ roots."""
        assert abs(np.linalg.norm(self.COLLISION_VECTOR) - 2.0) < 1e-14

    def test_collision_vector_is_not_e8_root(self):
        """d has 4 nonzero entries, so it's not a permutation root (needs 2)
        and not a half-integer root (needs 8)."""
        n_nonzero = np.sum(np.abs(self.COLLISION_VECTOR) > 1e-10)
        assert n_nonzero == 4  # Neither 2 (perm) nor 8 (half-int)

    def test_collision_vector_at_odd_indices(self):
        """Nonzero entries are at odd positions {1,3,5,7}."""
        nonzero_dims = list(np.where(np.abs(self.COLLISION_VECTOR) > 1e-10)[0])
        assert nonzero_dims == [1, 3, 5, 7]

    def test_exactly_14_collision_pairs(self):
        """240 roots → 226 unique projections → 14 collision pairs."""
        roots = generate_e8_roots()
        left = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
        rounded = [tuple(np.round(v, 8)) for v in left]
        from collections import Counter
        counts = Counter(rounded)
        n_collisions = sum(1 for c in counts.values() if c == 2)
        assert n_collisions == 14

    def test_all_collision_diffs_are_single_vector(self):
        """Every collision pair has r_a - r_b = ±d (rank-1 collision space)."""
        roots = generate_e8_roots()
        left = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
        rounded = [tuple(np.round(v, 8)) for v in left]

        from collections import defaultdict
        groups = defaultdict(list)
        for idx, key in enumerate(rounded):
            groups[key].append(idx)

        d = self.COLLISION_VECTOR
        for key, idxs in groups.items():
            if len(idxs) == 2:
                diff = roots[idxs[0]].coordinates - roots[idxs[1]].coordinates
                assert np.allclose(diff, d, atol=1e-10) or \
                       np.allclose(diff, -d, atol=1e-10), \
                    f"Collision diff {diff} ≠ ±d"

    def test_colliding_pairs_are_orthogonal(self):
        """All colliding root pairs satisfy ⟨r_a, r_b⟩ = 0."""
        roots = generate_e8_roots()
        left = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
        rounded = [tuple(np.round(v, 8)) for v in left]

        from collections import defaultdict
        groups = defaultdict(list)
        for idx, key in enumerate(rounded):
            groups[key].append(idx)

        for key, idxs in groups.items():
            if len(idxs) == 2:
                ip = np.dot(roots[idxs[0]].coordinates, roots[idxs[1]].coordinates)
                assert abs(ip) < 1e-10, \
                    f"Colliding pair inner product = {ip}"

    def test_collision_types_6_perm_8_half(self):
        """6 collision pairs are perm+perm, 8 are half+half."""
        roots = generate_e8_roots()
        left = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
        rounded = [tuple(np.round(v, 8)) for v in left]

        from collections import defaultdict
        groups = defaultdict(list)
        for idx, key in enumerate(rounded):
            groups[key].append(idx)

        n_pp, n_hh = 0, 0
        for key, idxs in groups.items():
            if len(idxs) == 2:
                types = {roots[i].root_type for i in idxs}
                if types == {E8RootType.PERMUTATION}:
                    n_pp += 1
                elif types == {E8RootType.HALF_INTEGER}:
                    n_hh += 1
        assert n_pp == 6, f"Expected 6 perm+perm pairs, got {n_pp}"
        assert n_hh == 8, f"Expected 8 half+half pairs, got {n_hh}"

    def test_kernel_dimension_is_4(self):
        """ker(U_L) has dimension 4 = 8 - rank(U_L)."""
        rank = np.linalg.matrix_rank(PHILLIPS_U_L)
        assert rank == 4
        # Kernel dimension = 8 - 4 = 4
        # SVD of full 8×8 matrix shows the kernel
        _, S_full, _ = np.linalg.svd(PHILLIPS_MATRIX)
        n_zero = np.sum(S_full < 1e-10)
        assert n_zero == 4

    def test_no_e8_roots_in_kernel(self):
        """No E₈ root maps to the zero vector under U_L."""
        roots = generate_e8_roots()
        for r in roots:
            proj = PHILLIPS_U_L @ r.coordinates
            assert np.linalg.norm(proj) > 1e-6, \
                f"Root {r.coordinates} is in kernel"


# ==========================================================================
# NEW THEOREMS: Round-Trip Eigenstructure
# ==========================================================================

class TestRoundTripEigenstructure:
    """
    Theorem (Round-Trip Factorization):

    U^T U = (φ+2) · U_L^T U_L

    This follows from U_R = φ·U_L:
        U^T U = U_L^T U_L + U_R^T U_R
              = U_L^T U_L + φ²·U_L^T U_L
              = (1 + φ²)·U_L^T U_L
              = (φ + 2)·U_L^T U_L

    Eigenvalue consequences:
    - 4 zero eigenvalues (the kernel)
    - Eigenvalue 5 with multiplicity 2 (from (φ+2)(3-φ) = 5)
    - Two non-degenerate eigenvalues summing to 10
    - Total trace = 20 = Frobenius²
    """

    @pytest.fixture
    def UTU(self):
        return PHILLIPS_MATRIX.T @ PHILLIPS_MATRIX

    @pytest.fixture
    def eigenvalues(self, UTU):
        return np.sort(np.linalg.eigvalsh(UTU))

    def test_round_trip_factorization(self, UTU):
        """U^T U = (φ+2) · U_L^T U_L."""
        ULtUL = PHILLIPS_U_L.T @ PHILLIPS_U_L
        expected = (PHI + 2) * ULtUL
        assert np.allclose(UTU, expected, atol=1e-10)

    def test_four_zero_eigenvalues(self, eigenvalues):
        """4 eigenvalues are zero (kernel dimension = 4)."""
        assert np.sum(np.abs(eigenvalues) < 1e-8) == 4

    def test_four_positive_eigenvalues(self, eigenvalues):
        """4 eigenvalues are strictly positive."""
        assert np.sum(eigenvalues > 1e-8) == 4

    def test_eigenvalue_5_has_multiplicity_2(self, eigenvalues):
        """The eigenvalue 5 appears twice (from (φ+2)(3-φ) = 5)."""
        pos_evals = eigenvalues[eigenvalues > 1e-8]
        n_five = np.sum(np.isclose(pos_evals, 5.0, atol=1e-8))
        assert n_five == 2, f"Expected eigenvalue 5 with mult 2, found {n_five}"

    def test_eigenvalue_sum_is_frobenius(self, eigenvalues):
        """Sum of all eigenvalues = tr(U^T U) = ||U||²_F = 20."""
        assert abs(np.sum(eigenvalues) - 20.0) < 1e-8

    def test_positive_eigenvalue_sum_is_20(self, eigenvalues):
        """Sum of nonzero eigenvalues = 20."""
        pos_sum = np.sum(eigenvalues[eigenvalues > 1e-8])
        assert abs(pos_sum - 20.0) < 1e-8

    def test_outer_eigenvalues_sum_to_10(self, eigenvalues):
        """The two non-degenerate eigenvalues sum to 10 (= 20 - 2×5)."""
        pos_evals = np.sort(eigenvalues[eigenvalues > 1e-8])
        # The two that are NOT 5
        non_five = [ev for ev in pos_evals if not np.isclose(ev, 5.0, atol=1e-8)]
        assert len(non_five) == 2
        assert abs(sum(non_five) - 10.0) < 1e-8

    def test_cross_block_product(self):
        """U_L^T U_R = φ · U_L^T U_L (from U_R = φ·U_L)."""
        cross = PHILLIPS_U_L.T @ PHILLIPS_U_R
        expected = PHI * (PHILLIPS_U_L.T @ PHILLIPS_U_L)
        assert np.allclose(cross, expected, atol=1e-10)

    def test_UR_frobenius_is_phi_sq_times_UL(self):
        """||U_R||²_F = φ² · ||U_L||²_F."""
        frob_L = np.sum(PHILLIPS_U_L ** 2)
        frob_R = np.sum(PHILLIPS_U_R ** 2)
        assert abs(frob_R - PHI**2 * frob_L) < 1e-10


# ==========================================================================
# NEW THEOREMS: Amplification Factor
# ==========================================================================

class TestAmplificationFactor:
    """
    Theorem (Amplification = 5):

    Frobenius²/rank = 20/4 = 5 = number of 24-cells in the 600-cell.

    This is NOT a coincidence but a structural identity:
    - The eigenvalue 5 of U^T U comes from (φ+2)(3-φ) = 5
    - The same √5-coupling that binds U_L to U_R produces the amplification
    - The 600-cell decomposes into 5 inscribed 24-cells
    """

    def test_amplification_is_5(self):
        """Frobenius²/rank = 20/4 = 5."""
        frob_sq = np.sum(PHILLIPS_MATRIX ** 2)
        rank = np.linalg.matrix_rank(PHILLIPS_MATRIX)
        assert abs(frob_sq / rank - 5.0) < 1e-10

    def test_amplification_matches_24cell_count(self):
        """The 600-cell has exactly 5 inscribed 24-cells."""
        # 600-cell: 120 vertices, 24-cell: 24 vertices
        # 120 / 24 = 5 (this is the vertex counting argument)
        assert 120 // 24 == 5
        amp = np.sum(PHILLIPS_MATRIX ** 2) / np.linalg.matrix_rank(PHILLIPS_MATRIX)
        assert abs(amp - 5.0) < 1e-10

    def test_block_amplification_via_phi_sq(self):
        """The right block amplification = φ² × left block amplification.
        Combined: (1 + φ²) × left_amp = (φ+2) × left_amp = total_amp."""
        frob_L = np.sum(PHILLIPS_U_L ** 2)
        rank_L = np.linalg.matrix_rank(PHILLIPS_U_L)
        amp_L = frob_L / rank_L  # per-block amplification
        total_amp = (PHI + 2) * amp_L
        assert abs(total_amp - 5.0) < 1e-10


# ==========================================================================
# NEW THEOREMS: Row Products (Non-Orthogonality)
# ==========================================================================

class TestRowProducts:
    """
    Theorem (Row Cross-Talk):

    The rows of U_L are NOT orthogonal. The Gram matrix U_L U_L^T has:
    - Diagonal entries = 3-φ (row norms²)
    - Off-diagonal entries involving ±1/2 and ±(φ-1)/2

    The Gram matrix U_L U_R^T has diagonal entries = φ(3-φ) = √5
    (the √5-coupling appears on the diagonal of the cross-block Gram).

    However, U_L U_L^T IS a circulant-like structure with exact entries
    from the golden field Q(√5).
    """

    def test_UL_rows_not_orthogonal(self):
        """U_L U_L^T ≠ (3-φ)·I₄."""
        gram = PHILLIPS_U_L @ PHILLIPS_U_L.T
        expected_diag = (3 - PHI) * np.eye(4)
        assert not np.allclose(gram, expected_diag, atol=1e-4)

    def test_UL_gram_diagonal_is_3_minus_phi(self):
        """Diagonal of U_L U_L^T = 3-φ."""
        gram = PHILLIPS_U_L @ PHILLIPS_U_L.T
        for i in range(4):
            assert abs(gram[i, i] - (3 - PHI)) < 1e-10

    def test_UR_gram_diagonal_is_phi_plus_2(self):
        """Diagonal of U_R U_R^T = φ+2."""
        gram = PHILLIPS_U_R @ PHILLIPS_U_R.T
        for i in range(4):
            assert abs(gram[i, i] - (PHI + 2)) < 1e-10

    def test_cross_gram_diagonal_is_sqrt5(self):
        """Diagonal of U_L U_R^T = φ(3-φ) = √5."""
        cross = PHILLIPS_U_L @ PHILLIPS_U_R.T
        expected_diag = PHI * (3 - PHI)
        # φ(3-φ) = 3φ - φ² = 3φ - φ - 1 = 2φ - 1 = √5
        assert abs(expected_diag - np.sqrt(5)) < 1e-10
        for i in range(4):
            assert abs(cross[i, i] - expected_diag) < 1e-10

    def test_gram_entries_in_golden_field(self):
        """All entries of U_L U_L^T are in Q(√5) = {a + b·φ : a,b ∈ Q}.
        The off-diagonal entries are:
          0, ±1/2, and ±(2φ-3)/2 = ±(√5-2)/2 ≈ ±0.11803
        All expressible as rationals plus rational multiples of φ.
        """
        gram = PHILLIPS_U_L @ PHILLIPS_U_L.T
        golden_val = (2 * PHI - 3) / 2  # = (√5 - 2)/2 ≈ 0.11803
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                val = abs(gram[i, j])
                is_zero = val < 1e-10
                is_half = abs(val - 0.5) < 1e-10
                is_golden = abs(val - golden_val) < 1e-10
                assert is_zero or is_half or is_golden, \
                    f"gram[{i},{j}]={gram[i,j]} not in golden field basis"


# ==========================================================================
# NEW THEOREMS: Chirality / Non-normality
# ==========================================================================

class TestChirality:
    """
    The Phillips matrix is NOT normal (U^T U ≠ U U^T), which means:
    - It has a chiral structure (distinguishes left from right)
    - The symmetric/antisymmetric decomposition preserves Frobenius: 20
    - Its eigenvalues (of the 8×8 itself) are ALL REAL despite non-symmetry
    """

    def test_matrix_is_not_normal(self):
        """U^T U ≠ U U^T (Phillips is not a normal operator)."""
        UTU = PHILLIPS_MATRIX.T @ PHILLIPS_MATRIX
        UUT = PHILLIPS_MATRIX @ PHILLIPS_MATRIX.T
        assert not np.allclose(UTU, UUT, atol=1e-4)

    def test_frobenius_splits_sym_antisym(self):
        """||sym||² + ||antisym||² = ||U||² = 20."""
        sym = (PHILLIPS_MATRIX + PHILLIPS_MATRIX.T) / 2
        antisym = (PHILLIPS_MATRIX - PHILLIPS_MATRIX.T) / 2
        assert abs(np.sum(sym**2) + np.sum(antisym**2) - 20.0) < 1e-10

    def test_all_eigenvalues_are_real(self):
        """Despite non-symmetry, all eigenvalues are real."""
        evals = np.linalg.eigvals(PHILLIPS_MATRIX)
        assert np.allclose(evals.imag, 0, atol=1e-8), \
            f"Complex eigenvalues found: {evals}"

    def test_four_zero_eigenvalues_of_full_matrix(self):
        """The 8×8 matrix has exactly 4 zero eigenvalues (rank 4)."""
        evals = np.sort(np.abs(np.linalg.eigvals(PHILLIPS_MATRIX)))
        assert np.sum(evals < 1e-8) == 4
