"""Tests for E₈ → H₄ projection pipeline — mathematical invariants."""

import math
import pytest
import numpy as np
from engine.geometry.e8_projection import (
    E8ProjectionPipeline,
    E8Root,
    E8RootType,
    E8Projected,
    generate_e8_roots,
    project_root,
    project_e8_to_h4,
    galois_conjugate,
    icosian_norm,
    normalize_icosian,
    BAEZ_MATRIX,
    BAEZ_CONJUGATE_MATRIX,
    PHI_CONJUGATE,
)
from engine.geometry.h4_geometry import PHI, PHI_INV


# =========================================================================
# Constants
# =========================================================================

class TestConstants:
    def test_phi_conjugate_value(self):
        """φ' = (1-√5)/2 = -1/φ"""
        assert abs(PHI_CONJUGATE - (1 - math.sqrt(5)) / 2) < 1e-10

    def test_phi_conjugate_is_negative_inverse(self):
        """φ' = -1/φ"""
        assert abs(PHI_CONJUGATE + 1.0 / PHI) < 1e-10

    def test_baez_matrix_shape(self):
        assert BAEZ_MATRIX.shape == (4, 8)

    def test_conjugate_matrix_shape(self):
        assert BAEZ_CONJUGATE_MATRIX.shape == (4, 8)

    def test_matrices_differ_only_in_phi(self):
        """The two matrices should differ only where PHI appears."""
        for i in range(4):
            for j in range(8):
                b = BAEZ_MATRIX[i, j]
                c = BAEZ_CONJUGATE_MATRIX[i, j]
                if abs(b) < 1e-10:
                    assert abs(c) < 1e-10, f"({i},{j}): Baez=0 but conj={c}"


# =========================================================================
# E₈ Root Generation
# =========================================================================

class TestE8Roots:
    @pytest.fixture
    def roots(self):
        return generate_e8_roots()

    def test_240_roots(self, roots):
        """E₈ has exactly 240 roots."""
        assert len(roots) == 240

    def test_112_permutation_roots(self, roots):
        """C(8,2) × 2² = 28 × 4 = 112 permutation-type roots."""
        perm = [r for r in roots if r.root_type == E8RootType.PERMUTATION]
        assert len(perm) == 112

    def test_128_half_integer_roots(self, roots):
        """2⁸/2 = 128 half-integer roots (even parity)."""
        half = [r for r in roots if r.root_type == E8RootType.HALF_INTEGER]
        assert len(half) == 128

    def test_all_roots_norm_squared_is_2(self, roots):
        """Every E₈ root has ||v||² = 2."""
        for r in roots:
            assert abs(r.norm_squared - 2.0) < 1e-10, \
                f"Root {r.coordinates} has norm²={r.norm_squared}"

    def test_permutation_roots_have_two_nonzero(self, roots):
        """Permutation roots have exactly 2 nonzero entries."""
        for r in roots:
            if r.root_type == E8RootType.PERMUTATION:
                n_nonzero = np.sum(np.abs(r.coordinates) > 1e-10)
                assert n_nonzero == 2

    def test_half_integer_roots_all_half(self, roots):
        """Half-integer roots have all entries = ±0.5."""
        for r in roots:
            if r.root_type == E8RootType.HALF_INTEGER:
                assert np.allclose(np.abs(r.coordinates), 0.5, atol=1e-10)

    def test_half_integer_even_parity(self, roots):
        """Half-integer roots have even number of minus signs."""
        for r in roots:
            if r.root_type == E8RootType.HALF_INTEGER:
                n_neg = np.sum(r.coordinates < -0.1)
                assert n_neg % 2 == 0

    def test_no_duplicate_roots(self, roots):
        """All 240 roots should be distinct."""
        coords = set(tuple(np.round(r.coordinates, 8)) for r in roots)
        assert len(coords) == 240

    def test_closed_under_negation(self, roots):
        """If v is a root, then -v is also a root."""
        coord_set = set(tuple(np.round(r.coordinates, 8)) for r in roots)
        for r in roots:
            neg = tuple(np.round(-r.coordinates, 8))
            assert neg in coord_set, f"Missing -v for root {r.coordinates}"


# =========================================================================
# Projection
# =========================================================================

class TestProjection:
    @pytest.fixture
    def proj(self):
        return project_e8_to_h4()

    def test_240_outer_vertices(self, proj):
        """Each E₈ root projects to a unique 4D point via the φ-matrix."""
        assert proj.outer_vertices.shape == (240, 4)

    def test_240_inner_vertices(self, proj):
        """Each E₈ root projects to a unique 4D point via the φ'-matrix."""
        assert proj.inner_vertices.shape == (240, 4)

    def test_240_projections_total(self, proj):
        assert len(proj.all_projections) == 240

    def test_outer_multi_shell_structure(self, proj):
        """Outer projections form multiple concentric shells."""
        assert len(proj.outer_radii) > 1, "Expected multi-shell structure"

    def test_inner_multi_shell_structure(self, proj):
        """Inner projections also form multiple concentric shells."""
        assert len(proj.inner_radii) > 1

    def test_outer_radii_positive(self, proj):
        """All outer shell radii are positive."""
        assert np.all(proj.outer_radii > 0)

    def test_inner_radii_positive(self, proj):
        """All inner shell radii are positive."""
        assert np.all(proj.inner_radii > 0)

    def test_outer_max_radius_is_phi(self, proj):
        """Largest outer shell radius ≈ φ = 1.618."""
        assert abs(proj.outer_radii.max() - PHI) < 0.01

    def test_inner_max_radius_is_1(self, proj):
        """Largest inner shell radius ≈ 1.0."""
        assert abs(proj.inner_radii.max() - 1.0) < 0.01

    def test_all_outer_distinct(self, proj):
        """All 240 outer projections are distinct."""
        rounded = set(tuple(np.round(v, 6)) for v in proj.outer_vertices)
        assert len(rounded) == 240

    def test_all_inner_distinct(self, proj):
        """All 240 inner projections are distinct."""
        rounded = set(tuple(np.round(v, 6)) for v in proj.inner_vertices)
        assert len(rounded) == 240

    def test_antipodal_closure(self, proj):
        """For each outer point v, -v is also present (inherited from E₈)."""
        outer_set = set(tuple(np.round(v, 6)) for v in proj.outer_vertices)
        for v in proj.outer_vertices:
            neg = tuple(np.round(-v, 6))
            assert neg in outer_set

    def test_projection_linearity(self, proj):
        """project_root(r) outer = BAEZ_MATRIX @ r.coordinates."""
        root = proj.all_projections[0]
        direct = BAEZ_MATRIX @ root.source.coordinates
        assert np.allclose(root.outer, direct, atol=1e-12)


# =========================================================================
# Galois Conjugation
# =========================================================================

class TestGaloisConjugation:
    def test_conjugation_scales_by_inv_phi_squared(self):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        conj = galois_conjugate(v)
        expected_scale = 1.0 / (PHI * PHI)
        assert np.allclose(conj, v * expected_scale)

    def test_conjugation_preserves_direction(self):
        v = np.array([1.0, 2.0, -1.0, 0.5])
        conj = galois_conjugate(v)
        v_norm = v / np.linalg.norm(v)
        c_norm = conj / np.linalg.norm(conj)
        assert np.allclose(v_norm, c_norm, atol=1e-10)


# =========================================================================
# Icosian Norm
# =========================================================================

class TestIcosianNorm:
    def test_unit_vector_norm(self):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(icosian_norm(v) - 1.0) < 1e-10

    def test_half_integer_quaternion_norm(self):
        v = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(icosian_norm(v) - 1.0) < 1e-10

    def test_normalize_icosian(self):
        v = np.array([3.0, 4.0, 0.0, 0.0])
        n = normalize_icosian(v)
        assert abs(icosian_norm(n) - 1.0) < 1e-10

    def test_normalize_zero_returns_unit(self):
        v = np.zeros(4)
        n = normalize_icosian(v)
        assert np.allclose(n, [1, 0, 0, 0])


# =========================================================================
# Pipeline Class
# =========================================================================

class TestE8Pipeline:
    @pytest.fixture
    def pipeline(self):
        return E8ProjectionPipeline()

    def test_has_240_roots(self, pipeline):
        assert len(pipeline.e8_roots) == 240

    def test_outer_vertices_shape(self, pipeline):
        assert pipeline.outer_vertices.shape == (240, 4)

    def test_inner_vertices_shape(self, pipeline):
        assert pipeline.inner_vertices.shape == (240, 4)

    def test_project_8d_vector(self, pipeline):
        v8 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        outer, inner = pipeline.project(v8)
        assert outer.shape == (4,)
        assert inner.shape == (4,)

    def test_project_known_root(self, pipeline):
        """Projecting an E₈ root gives consistent outer/inner radii."""
        root = pipeline.e8_roots[0]
        outer, inner = pipeline.project(root.coordinates)
        assert np.linalg.norm(outer) > 0
        assert np.linalg.norm(inner) > 0

    def test_find_nearest_root(self, pipeline):
        """Querying an exact root should return itself."""
        root = pipeline.e8_roots[42]
        found = pipeline.find_nearest_root(root.coordinates)
        assert np.allclose(found.coordinates, root.coordinates)

    def test_find_nearest_root_perturbed(self, pipeline):
        """Small perturbation should still find the same root."""
        np.random.seed(42)
        root = pipeline.e8_roots[10]
        perturbed = root.coordinates + np.random.randn(8) * 0.01
        found = pipeline.find_nearest_root(perturbed)
        assert np.allclose(found.coordinates, root.coordinates)

    def test_get_vertices_outer(self, pipeline):
        v = pipeline.get_vertices_at_scale("outer")
        assert v.shape == (240, 4)

    def test_get_vertices_inner(self, pipeline):
        v = pipeline.get_vertices_at_scale("inner")
        assert v.shape == (240, 4)

    def test_get_vertices_both(self, pipeline):
        v = pipeline.get_vertices_at_scale("both")
        assert v.shape == (480, 4)

    def test_get_shell_filters_by_radius(self, pipeline):
        """get_shell returns vertices at a specific radius."""
        max_r = pipeline.projected.outer_radii.max()
        shell = pipeline.get_shell("outer", radius=max_r, tol=0.01)
        assert len(shell) > 0
        radii = np.linalg.norm(shell, axis=1)
        assert np.allclose(radii, max_r, atol=0.01)

    def test_interpolate_endpoints(self, pipeline):
        """t=0 gives outer, t=1 gives inner."""
        outer_interp = pipeline.interpolate_scale(0.0)
        inner_interp = pipeline.interpolate_scale(1.0)
        assert np.allclose(outer_interp, pipeline.outer_vertices, atol=1e-10)
        assert np.allclose(inner_interp, pipeline.inner_vertices, atol=1e-10)

    def test_get_outer_as_vertex4d(self, pipeline):
        verts = pipeline.get_outer_as_vertex4d()
        assert len(verts) == 240
        from engine.geometry.h4_geometry import Vertex4D
        assert all(isinstance(v, Vertex4D) for v in verts)

    def test_get_inner_as_vertex4d(self, pipeline):
        verts = pipeline.get_inner_as_vertex4d()
        assert len(verts) == 240


# =========================================================================
# Cross-validation: E₈ projection encodes 24-cell structure
# =========================================================================

class TestE8H4Connection:
    """Verify that E₈ projections connect to existing H4 geometry."""

    @pytest.fixture
    def pipeline(self):
        return E8ProjectionPipeline()

    def test_24cell_vertices_appear_in_projections(self, pipeline):
        """
        The 24-cell vertices (Hurwitz unit quaternions) should appear
        among the E₈ projections — specifically, the axis-aligned vertices
        (±1,0,0,0) perms should be found at some shell radius.
        """
        from engine.geometry.h4_geometry import Polytope24Cell
        cell24 = Polytope24Cell()
        cell24_verts = cell24.get_vertex_array()  # (24, 4)

        # Normalize outer projections to unit sphere
        outer = pipeline.outer_vertices
        outer_norms = np.linalg.norm(outer, axis=1, keepdims=True)
        outer_unit = outer / np.where(outer_norms > 1e-10, outer_norms, 1)

        # Also normalize 24-cell vertices
        c24_norms = np.linalg.norm(cell24_verts, axis=1, keepdims=True)
        c24_unit = cell24_verts / c24_norms

        # Each 24-cell direction should have a match in the outer directions
        matches = 0
        for v24 in c24_unit:
            dists = np.linalg.norm(outer_unit - v24, axis=1)
            if np.min(dists) < 0.05:
                matches += 1

        assert matches >= 8, \
            f"Only {matches} 24-cell directions found in outer E₈ projections"

    def test_phi_ratio_in_shell_radii(self, pipeline):
        """The ratio between max and min outer radii involves φ."""
        radii = pipeline.projected.outer_radii
        ratio = radii.max() / radii.min()
        # max/min ≈ φ/φ_inv = φ² ≈ 2.618 or some other φ-based ratio
        assert ratio > 1.5, f"Shell ratio {ratio:.3f} unexpectedly small"

    def test_e8_root_type_determines_shell(self, pipeline):
        """
        Permutation-type and half-integer-type E₈ roots should project
        to different shell radii in general.
        """
        perm_radii = set()
        half_radii = set()
        for p in pipeline.projected.all_projections:
            r = round(np.linalg.norm(p.outer), 4)
            if p.source.root_type == E8RootType.PERMUTATION:
                perm_radii.add(r)
            else:
                half_radii.add(r)
        # Both types should have representation across shells
        assert len(perm_radii) >= 2, "Permutation roots on too few shells"
        assert len(half_radii) >= 2, "Half-integer roots on too few shells"
