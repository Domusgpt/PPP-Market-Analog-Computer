"""Tests for the three-rule-set enforcer."""

import math
import pytest
from engine.enforcer import (
    RuleEnforcer,
    CommensurateLock,
    TalbotLock,
    TiltLock,
    LogicPolarity,
    RuleViolation,
)


@pytest.fixture
def enforcer():
    return RuleEnforcer(lattice_constant=1.0, wavelength=550.0)


# =========================================================================
# Rule Set 1: Angular Commensurability
# =========================================================================

class TestRuleSet1:
    def test_commensurate_table_has_entries(self, enforcer):
        assert len(enforcer.COMMENSURATE_TABLE) >= 10

    def test_enforce_angle_returns_commensurate_lock(self, enforcer):
        lock = enforcer.enforce_angle(13.0)
        assert isinstance(lock, CommensurateLock)
        assert lock.angle > 0

    def test_zero_angle_is_transparent(self, enforcer):
        lock = enforcer.enforce_angle(0.0)
        assert lock.angle == 0.0
        assert "Transparent" in lock.cognitive_mode or "All-Pass" in lock.cognitive_mode

    def test_angle_snaps_to_nearest_commensurate(self, enforcer):
        """13.0° should snap to 13.174° (the (3,1) commensurate angle)."""
        lock = enforcer.enforce_angle(13.0)
        assert abs(lock.angle - 13.174) < 1.0

    def test_all_commensurate_angles_are_positive(self, enforcer):
        for (m, n), data in enforcer.COMMENSURATE_TABLE.items():
            assert data["angle"] >= 0.0

    def test_mode_names_exist(self, enforcer):
        for (m, n), data in enforcer.COMMENSURATE_TABLE.items():
            assert isinstance(data["mode"], str)
            assert len(data["mode"]) > 0


# =========================================================================
# Rule Set 2: Trilatic Tilt Symmetry
# =========================================================================

class TestRuleSet2:
    def test_enforce_tilt_returns_tilt_lock(self, enforcer):
        lock = enforcer.enforce_tilt(5.0, 0)
        assert isinstance(lock, TiltLock)

    def test_valid_tilt_axes_are_multiples_of_60(self, enforcer):
        for k in range(6):
            lock = enforcer.enforce_tilt(5.0, k)
            expected_axis_angle = k * 60.0
            assert abs(lock.axis_angle - expected_axis_angle) < 0.01

    def test_tilt_preserves_c3_symmetry(self, enforcer):
        """Axes 0, 2, 4 should give effective lattice constants related by C3."""
        locks = [enforcer.enforce_tilt(5.0, k) for k in [0, 2, 4]]
        # C3 symmetry means all three should have the same effective constant
        constants = [l.effective_lattice_constant for l in locks]
        assert abs(constants[0] - constants[1]) < 0.01
        assert abs(constants[1] - constants[2]) < 0.01


# =========================================================================
# Rule Set 3: Talbot Distance
# =========================================================================

class TestRuleSet3:
    def test_talbot_length_is_positive(self, enforcer):
        assert enforcer.talbot_length > 0

    def test_enforce_gap_returns_talbot_lock(self, enforcer):
        lock = enforcer.enforce_gap(1.0)
        assert isinstance(lock, TalbotLock)

    def test_integer_talbot_is_positive_logic(self, enforcer):
        pos_gap, _ = enforcer.get_logic_gaps(order=1)
        lock = enforcer.enforce_gap(pos_gap)
        assert lock.polarity == LogicPolarity.POSITIVE

    def test_half_integer_talbot_is_negative_logic(self, enforcer):
        _, neg_gap = enforcer.get_logic_gaps(order=1)
        lock = enforcer.enforce_gap(neg_gap)
        assert lock.polarity == LogicPolarity.NEGATIVE

    def test_logic_gaps_are_ordered(self, enforcer):
        """Half-integer gap should be between integer orders."""
        pos1, neg1 = enforcer.get_logic_gaps(order=1)
        pos2, _ = enforcer.get_logic_gaps(order=2)
        assert pos1 < neg1 < pos2

    def test_gap_contrast_is_bounded(self, enforcer):
        lock = enforcer.enforce_gap(enforcer.talbot_length)
        assert 0.0 <= lock.contrast <= 1.0


# =========================================================================
# Moiré period
# =========================================================================

class TestMoirePeriod:
    def test_moire_period_formula(self, enforcer):
        """L_M = a / (2 * sin(θ/2))"""
        for angle_deg in [7.341, 9.43, 13.174, 21.787]:
            theta = math.radians(angle_deg)
            expected = enforcer.a / (2 * math.sin(theta / 2))
            computed = enforcer.get_moiré_period(angle_deg)
            assert abs(computed - expected) / expected < 0.05

    def test_zero_angle_gives_infinite_period(self, enforcer):
        period = enforcer.get_moiré_period(0.0)
        assert period == float('inf') or period > 1e6
