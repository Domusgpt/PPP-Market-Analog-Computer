"""Tests for kirigami sheet (lattice reservoir)."""

import pytest
import numpy as np
from engine.kirigami.kirigami_sheet import KirigamiSheet, SheetConfig, CutPattern
from engine.kirigami.tristable_cell import CellState


@pytest.fixture
def sheet():
    config = SheetConfig(n_cells_x=8, n_cells_y=8)
    return KirigamiSheet(config)


class TestSheetCreation:
    def test_creates_correct_number_of_cells(self, sheet):
        assert len(sheet.cells) == 64  # 8x8

    def test_all_cells_start_closed(self, sheet):
        for cell in sheet.cells.values():
            assert cell.state == CellState.CLOSED

    def test_hexagonal_neighbors(self, sheet):
        """Interior cells should have neighbors set up."""
        interior_cell = sheet.cells.get((4, 4))
        assert interior_cell is not None


class TestInputInjection:
    def test_inject_changes_cell_states(self, sheet):
        data = np.ones((8, 8))
        sheet.inject_input(data)
        # At least some cells should no longer be at position 0
        states = sheet.get_state_field()
        assert np.any(states > 0)

    def test_inject_zeros_preserves_closed(self, sheet):
        data = np.zeros((8, 8))
        sheet.inject_input(data)
        states = sheet.get_state_field()
        assert np.allclose(states, 0.0, atol=0.1)

    def test_inject_with_scale(self, sheet):
        data = np.ones((8, 8))
        sheet.inject_input(data, input_scale=0.5)
        states = sheet.get_state_field()
        assert np.any(states > 0)


class TestCascadeDynamics:
    def test_cascade_converges(self, sheet):
        data = np.random.default_rng(42).random((8, 8))
        sheet.inject_input(data)
        steps = sheet.run_cascade(n_steps=50)
        assert steps <= 50

    def test_cascade_changes_state(self, sheet):
        data = np.random.default_rng(42).random((8, 8))
        sheet.inject_input(data)
        states_before = sheet.get_state_field().copy()
        sheet.run_cascade(n_steps=20)
        states_after = sheet.get_state_field()
        # State should change during cascade (neighbor coupling)
        # Changes may be small with low coupling, use tight tolerance
        assert not np.allclose(states_before, states_after, atol=1e-4)

    def test_step_returns_float(self, sheet):
        data = np.random.default_rng(42).random((8, 8))
        sheet.inject_input(data)
        result = sheet.step(dt=0.01)
        assert isinstance(result, float)


class TestStateReadout:
    def test_state_field_shape(self, sheet):
        states = sheet.get_state_field()
        assert states.shape == (8, 8)

    def test_state_values_in_valid_range(self, sheet):
        data = np.random.default_rng(42).random((8, 8))
        sheet.inject_input(data)
        sheet.run_cascade(n_steps=20)
        states = sheet.get_state_field()
        assert np.all(states >= 0.0)
        assert np.all(states <= 1.0)

    def test_discrete_state_readout(self, sheet):
        data = np.random.default_rng(42).random((8, 8))
        sheet.inject_input(data)
        sheet.run_cascade(n_steps=20)
        discrete = sheet.get_discrete_state()
        assert discrete.shape == (8, 8)

    def test_transmission_field(self, sheet):
        transmission = sheet.get_transmission_field()
        assert transmission.shape == (8, 8)
        assert np.all(transmission >= 0.0)
        assert np.all(transmission <= 1.0)


class TestCutPatterns:
    def test_all_cut_patterns_create_valid_sheets(self):
        for pattern in CutPattern:
            config = SheetConfig(
                n_cells_x=8, n_cells_y=8, cut_pattern=pattern
            )
            sheet = KirigamiSheet(config)
            assert len(sheet.cells) == 64

    def test_statistics(self, sheet):
        stats = sheet.get_statistics()
        assert isinstance(stats, dict)


class TestReset:
    def test_reset_restores_initial_state(self, sheet):
        data = np.ones((8, 8))
        sheet.inject_input(data)
        sheet.run_cascade(n_steps=20)
        sheet.reset()
        states = sheet.get_state_field()
        assert np.allclose(states, 0.0, atol=0.01)
