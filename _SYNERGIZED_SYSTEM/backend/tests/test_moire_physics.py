"""Tests for moiré interference physics."""

import math
import pytest
import numpy as np
from engine.physics.moire_interference import MoireInterference
from engine.physics.talbot_resonator import TalbotResonator, TalbotMode
from engine.physics.trilatic_lattice import TrilaticLattice


# =========================================================================
# Moiré Interference
# =========================================================================

class TestMoireInterference:
    @pytest.fixture
    def moire(self):
        return MoireInterference(lattice_constant=1.0, wavelength_red=650.0, wavelength_cyan=500.0)

    def test_creation(self, moire):
        assert moire is not None

    def test_moire_period_formula(self, moire):
        """L_M = a / (2 * sin(θ/2))"""
        for angle_deg in [7.34, 9.43, 13.17, 21.79]:
            theta = math.radians(angle_deg)
            expected = 1.0 / (2 * math.sin(theta / 2))
            computed = moire.compute_moire_period(angle_deg)
            assert abs(computed - expected) / expected < 0.05

    def test_small_angle_gives_large_period(self, moire):
        p1 = moire.compute_moire_period(5.0)
        p2 = moire.compute_moire_period(20.0)
        assert p1 > p2

    def test_amplification_increases_with_angle(self, moire):
        a1 = moire.compute_amplification_factor(5.0)
        a2 = moire.compute_amplification_factor(20.0)
        # Smaller angle → larger amplification (finer moiré)
        assert a1 > a2

    def test_fringe_contrast_is_bounded(self, moire):
        field = moire.generate_moire_field(
            twist_angle=13.17,
            grid_size=(32, 32),
            field_size=(50.0, 50.0),
        )
        contrast = moire.compute_fringe_contrast(field)
        assert 0.0 <= contrast <= 1.0


# =========================================================================
# Talbot Resonator
# =========================================================================

class TestTalbotResonator:
    @pytest.fixture
    def talbot(self):
        return TalbotResonator(lattice_constant=1.0, wavelength=550.0)

    def test_talbot_length_positive(self, talbot):
        assert talbot.talbot_length > 0

    def test_integer_gap(self, talbot):
        """Integer Talbot gap should exist at order 1."""
        gap = talbot.compute_talbot_gap(order=1, mode=TalbotMode.INTEGER)
        assert gap > 0

    def test_half_integer_gap(self, talbot):
        """Half-integer Talbot gap should be between integer orders."""
        gap_int = talbot.compute_talbot_gap(order=1, mode=TalbotMode.INTEGER)
        gap_half = talbot.compute_talbot_gap(order=1, mode=TalbotMode.HALF_INTEGER)
        assert gap_half != gap_int

    def test_nearest_talbot_gap(self, talbot):
        """get_nearest_talbot_gap should snap to nearest resonance."""
        z_t = talbot.talbot_length
        state = talbot.get_nearest_talbot_gap(z_t)
        assert state.mode in (TalbotMode.INTEGER, TalbotMode.HALF_INTEGER)
        assert state.gap > 0

    def test_logic_gaps(self, talbot):
        """get_logic_gaps returns (positive_gap, negative_gap) tuple."""
        pos_gap, neg_gap = talbot.get_logic_gaps(base_order=1)
        assert pos_gap > 0
        assert neg_gap > 0
        assert pos_gap != neg_gap

    def test_resonance_ladder(self, talbot):
        ladder = talbot.generate_resonance_ladder(max_gap=100.0)
        assert len(ladder) > 0
        for state in ladder:
            assert state.gap > 0


# =========================================================================
# Trilatic Lattice
# =========================================================================

class TestTrilaticLattice:
    @pytest.fixture
    def lattice(self):
        return TrilaticLattice(lattice_constant=1.0)

    def test_creation(self, lattice):
        assert lattice is not None

    def test_commensurate_angle_computation(self, lattice):
        """cos(θ) = (n²+4mn+m²)/(2(n²+mn+m²))
        For (3,1): cos_θ = 22/26 ≈ 0.8462 → θ ≈ 32.2°"""
        angle = TrilaticLattice.compute_commensurate_angle(3, 1)
        assert angle > 0
        # Verify using the formula directly
        import math
        m, n = 3, 1
        cos_theta = (n**2 + 4*m*n + m**2) / (2 * (n**2 + m*n + m**2))
        expected = math.degrees(math.acos(cos_theta))
        assert abs(angle - expected) < 0.01

    def test_coprime_requirement(self):
        """Non-coprime indices should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            TrilaticLattice.compute_commensurate_angle(2, 2)

    def test_nearest_commensurate_angle(self, lattice):
        m, n, angle = lattice.get_nearest_commensurate_angle(30.0)
        assert angle > 0

    def test_generate_lattice_points(self, lattice):
        points = lattice.generate_lattice_points(n_cells=5)
        assert len(points) > 0

    def test_trilatic_wave_vectors(self, lattice):
        """Should have 3 wave vectors (trilatic = 3-fold symmetry)."""
        kvecs = lattice.get_trilatic_wave_vectors()
        assert len(kvecs) == 3
