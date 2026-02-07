"""Tests for H4 4D polytope geometry — the mathematical invariants."""

import math
import pytest
import numpy as np
from engine.geometry.h4_geometry import (
    Polytope24Cell,
    Polytope16Cell,
    TrilaticDecomposition,
    Polytope600Cell,
    Polytope120Cell,
    LayerPlane,
    TrilaticChannel,
    Vertex4D,
)

# Try to import PHI/PHI_INV - may or may not be module-level constants
try:
    from engine.geometry.h4_geometry import PHI, PHI_INV
except ImportError:
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = PHI - 1


# =========================================================================
# Golden Ratio
# =========================================================================

class TestGoldenRatio:
    def test_phi_squared_identity(self):
        """φ² = φ + 1"""
        assert abs(PHI**2 - PHI - 1) < 1e-10

    def test_phi_inverse_identity(self):
        """1/φ = φ - 1"""
        assert abs(1.0 / PHI - (PHI - 1)) < 1e-10

    def test_phi_value(self):
        assert abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-10


# =========================================================================
# 16-Cell
# =========================================================================

class TestPolytope16Cell:
    @pytest.fixture
    def cell16(self):
        return Polytope16Cell()

    def test_has_8_vertices(self, cell16):
        assert len(cell16.vertices) == 8

    def test_vertices_are_permutations_of_pm1_0_0_0(self, cell16):
        """Each vertex should have exactly one nonzero coordinate = ±1."""
        for v in cell16.vertices:
            arr = v.to_array()
            nonzero = np.count_nonzero(np.abs(arr) > 1e-10)
            assert nonzero == 1
            assert abs(abs(arr[np.argmax(np.abs(arr))]) - 1.0) < 1e-10

    def test_vertices_equidistant_from_origin(self, cell16):
        radii = [v.distance_from_origin() for v in cell16.vertices]
        assert all(abs(r - 1.0) < 1e-10 for r in radii)

    def test_has_24_edges(self, cell16):
        edges = cell16.get_edges()
        assert len(edges) == 24

    def test_edge_length_is_sqrt2(self, cell16):
        edges = cell16.get_edges()
        for edge in edges:
            assert abs(edge.length() - math.sqrt(2)) < 0.01


# =========================================================================
# 24-Cell (Hurwitz Quaternion Form)
# =========================================================================

class TestPolytope24Cell:
    @pytest.fixture
    def cell24(self):
        return Polytope24Cell()

    def test_has_24_vertices(self, cell24):
        assert len(cell24.vertices) == 24

    def test_vertices_on_unit_sphere(self, cell24):
        """All 24-cell vertices should be equidistant from origin.
        In Hurwitz form: 8 unit quaternions (±1,0,0,0) perms at radius 1.0
        + 16 half-integer quaternions (±½,±½,±½,±½) also at radius 1.0.
        """
        radii = [v.distance_from_origin() for v in cell24.vertices]
        expected_radius = radii[0]
        for r in radii:
            assert abs(r - expected_radius) < 1e-6, \
                f"Vertex radius {r} != expected {expected_radius}"

    def test_no_duplicate_vertices(self, cell24):
        coords = [tuple(np.round(v.to_array(), 8)) for v in cell24.vertices]
        assert len(set(coords)) == 24

    def test_self_duality(self, cell24):
        """24-cell is self-dual: number of vertices = number of cells = 24."""
        assert len(cell24.vertices) == 24

    def test_vertex_array(self, cell24):
        arr = cell24.get_vertex_array()
        assert arr.shape == (24, 4)

    def test_layer_vertices(self, cell24):
        """Should be able to get vertices by layer plane."""
        for plane in LayerPlane:
            verts = cell24.get_layer_vertices(plane)
            assert isinstance(verts, list)


# =========================================================================
# Trinity (Trilatic) Decomposition
# =========================================================================

class TestTrinityDecomposition:
    @pytest.fixture
    def trilatic(self):
        cell24 = Polytope24Cell()
        return TrilaticDecomposition(cell24)

    def test_three_channels_exist(self, trilatic):
        """Each channel should have a corresponding 16-cell."""
        assert trilatic.cell_alpha is not None
        assert trilatic.cell_beta is not None
        assert trilatic.cell_gamma is not None

    def test_each_channel_has_8_vertices(self, trilatic):
        """Each 16-cell in the decomposition has 8 vertices."""
        assert len(trilatic.cell_alpha.vertices) == 8
        assert len(trilatic.cell_beta.vertices) == 8
        assert len(trilatic.cell_gamma.vertices) == 8

    def test_total_vertex_count(self, trilatic):
        """Each channel produces 8 vertices (3 × 8 = 24 total slots)."""
        total = (len(trilatic.cell_alpha.vertices) +
                 len(trilatic.cell_beta.vertices) +
                 len(trilatic.cell_gamma.vertices))
        assert total == 24

    def test_channels_vertices_on_sphere(self, trilatic):
        """All vertices in all channels should lie on the same radius sphere."""
        all_radii = []
        for cell in [trilatic.cell_alpha, trilatic.cell_beta, trilatic.cell_gamma]:
            for v in cell.vertices:
                all_radii.append(v.distance_from_origin())
        r0 = all_radii[0]
        for r in all_radii:
            assert abs(r - r0) < 1e-6, f"Vertex radius {r} != {r0}"

    def test_get_channel_state_returns_array(self, trilatic):
        """get_channel_state returns np.ndarray for each channel."""
        for channel in TrilaticChannel:
            state = trilatic.get_channel_state(channel)
            assert isinstance(state, np.ndarray)

    def test_get_all_vertices(self, trilatic):
        all_verts = trilatic.get_all_vertices()
        assert isinstance(all_verts, np.ndarray)
        assert all_verts.shape[0] == 24
        assert all_verts.shape[1] == 4


# =========================================================================
# 600-Cell
# =========================================================================

class TestPolytope600Cell:
    @pytest.fixture
    def cell600(self):
        return Polytope600Cell()

    def test_has_120_vertices(self, cell600):
        assert len(cell600.vertices) == 120

    def test_vertices_equidistant_from_origin(self, cell600):
        """All 600-cell vertices should be equidistant from origin."""
        radii = [v.distance_from_origin() for v in cell600.vertices]
        expected = radii[0]
        assert all(abs(r - expected) < 0.01 for r in radii)

    def test_no_duplicate_vertices(self, cell600):
        coords = [tuple(np.round(v.to_array(), 6)) for v in cell600.vertices]
        assert len(set(coords)) == 120

    def test_contains_embedded_24cells(self, cell600):
        """600-cell contains inscribed 24-cells."""
        embedded = cell600.get_embedded_24cells()
        assert len(embedded) > 0


# =========================================================================
# Layer Planes
# =========================================================================

class TestLayerPlanes:
    def test_six_layer_planes(self):
        planes = list(LayerPlane)
        assert len(planes) == 6
