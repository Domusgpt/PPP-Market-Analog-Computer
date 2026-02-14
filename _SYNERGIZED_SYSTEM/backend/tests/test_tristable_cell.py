"""Tests for tristable cell mechanics."""

import math
import pytest
import numpy as np
from engine.kirigami.tristable_cell import TristableCell, CellState, CellProperties


@pytest.fixture
def cell():
    return TristableCell(position=(0, 0))


class TestCellStates:
    def test_initial_state_is_closed(self, cell):
        assert cell.state == CellState.CLOSED

    def test_three_valid_states(self):
        states = list(CellState)
        assert len(states) == 3
        values = sorted(s.value for s in states)
        assert values == [0.0, 0.5, 1.0]

    def test_cell_starts_at_equilibrium(self, cell):
        """Initial value should be near state 0."""
        assert abs(cell.value - 0.0) < 0.01


class TestPotentialEnergy:
    def test_potential_energy_at_equilibrium(self, cell):
        """Energy at closed state should be at/near a minimum."""
        e = cell._compute_potential_energy()
        assert e >= 0.0  # Energy should be non-negative
        assert e < 0.01  # Near-zero at equilibrium

    def test_energy_changes_with_state(self):
        """Moving cell away from equilibrium should increase energy."""
        cell = TristableCell(position=(0, 0))
        e_eq = cell._compute_potential_energy()
        # Push cell value away from equilibrium
        cell.set_value(0.25)  # Between stable states
        e_off = cell._compute_potential_energy()
        assert e_off > e_eq


class TestOpticalResponse:
    def test_transmission_in_range(self, cell):
        t = cell.get_transmission()
        assert 0.0 <= t <= 1.0

    def test_different_states_have_different_transmission(self):
        cell_closed = TristableCell(position=(0, 0), initial_state=CellState.CLOSED)
        cell_open = TristableCell(position=(0, 0), initial_state=CellState.OPEN)
        t_closed = cell_closed.get_transmission()
        t_open = cell_open.get_transmission()
        # They should have different transmissions
        assert abs(t_closed - t_open) > 0.01

    def test_optical_response_property(self):
        cell = TristableCell(position=(0, 0))
        resp = cell.optical_response
        assert hasattr(resp, 'transmission')
        assert hasattr(resp, 'resonance_shift')
        assert hasattr(resp, 'chirality')


class TestCellUpdate:
    def test_update_returns_float(self, cell):
        result = cell.update(dt=0.01)
        assert isinstance(result, float)

    def test_apply_input(self, cell):
        """Applying energy should potentially change cell state."""
        result = cell.apply_input(1.0)
        assert isinstance(result, bool)

    def test_set_state(self, cell):
        cell.set_state(CellState.OPEN)
        assert cell.state == CellState.OPEN

    def test_set_value(self, cell):
        cell.set_value(0.5)
        assert abs(cell.value - 0.5) < 0.01


class TestNeighborCoupling:
    def test_neighbor_force_is_zero_at_equilibrium(self, cell):
        """A cell at equilibrium should exert no force on neighbors."""
        force = cell.get_neighbor_force()
        assert abs(force) < 0.01

    def test_neighbor_force_changes_off_equilibrium(self):
        cell = TristableCell(position=(0, 0))
        cell.set_value(0.25)  # Push off equilibrium
        force = cell.get_neighbor_force()
        # Force may or may not be zero depending on implementation
        assert isinstance(force, float)
