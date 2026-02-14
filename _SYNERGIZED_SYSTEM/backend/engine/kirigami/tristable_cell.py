"""
Tristable Cell - Mechanical Memory Unit
======================================

Implements the tristable unit cell design from Section 3.1.

The three states (0, 0.5, 1) correspond to:
- State 0: Closed/Flat (Logic '0' / Null)
- State 0.5: Intermediate/Chiral (Logic '½' / Weight)
- State 1: Open/Extended (Logic '1' / Active)

Each state represents a local potential energy minimum,
allowing state maintenance without continuous power input.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
from enum import Enum


class CellState(Enum):
    """
    Tristable cell states as defined in Section 3.1.

    CLOSED (0): Planar configuration, minimized porosity
    INTERMEDIATE (0.5): Partial deformation, chiral symmetry breaking
    OPEN (1): Fully deployed, maximum transmission
    """
    CLOSED = 0.0
    INTERMEDIATE = 0.5
    OPEN = 1.0


@dataclass
class CellProperties:
    """
    Physical properties of a tristable kirigami cell.

    From Section 3.2: Hybrid cut patterns allow spatial
    variation in stiffness, acting as the weight matrix
    of the neural network.
    """
    stiffness: float = 1.0          # Local stiffness (spring constant)
    threshold_01: float = 0.3       # Energy threshold: 0 -> 0.5
    threshold_12: float = 0.7       # Energy threshold: 0.5 -> 1
    damping: float = 0.1            # Viscous damping coefficient
    coupling_strength: float = 0.5  # Neighbor coupling for cascades


@dataclass
class OpticalResponse:
    """Optical properties of the cell in each state."""
    transmission: float         # 0-1 transmission coefficient
    resonance_shift: float      # Plasmonic resonance shift (nm)
    chirality: float            # Optical activity parameter


class TristableCell:
    """
    Tristable kirigami unit cell for optical computing.

    This class models a single unit cell in the kirigami lattice.
    It implements:
    - Three stable states with potential energy wells
    - State transitions via snap-through buckling
    - Optical response (transmission, resonance) per state
    - Neighbor coupling for cascade propagation

    Parameters
    ----------
    position : Tuple[int, int]
        (i, j) indices in the lattice
    properties : CellProperties, optional
        Physical properties of this cell
    initial_state : CellState
        Starting state (default: CLOSED)
    """

    # Optical responses per state (from Section 3.1)
    OPTICAL_TABLE = {
        CellState.CLOSED: OpticalResponse(
            transmission=0.1,       # Minimal transmission
            resonance_shift=0.0,    # Base resonance
            chirality=0.0           # No chirality
        ),
        CellState.INTERMEDIATE: OpticalResponse(
            transmission=0.5,       # Partial transmission
            resonance_shift=-20.0,  # Blue-shift from coupling
            chirality=0.3           # Induced chirality
        ),
        CellState.OPEN: OpticalResponse(
            transmission=0.9,       # Maximum transmission
            resonance_shift=50.0,   # Red-shift from extension
            chirality=0.0           # Restored symmetry
        )
    }

    def __init__(
        self,
        position: Tuple[int, int],
        properties: Optional[CellProperties] = None,
        initial_state: CellState = CellState.CLOSED
    ):
        self.position = position
        self.properties = properties or CellProperties()
        self._state = initial_state
        self._continuous_value = float(initial_state.value)

        # Dynamic state for reservoir computing
        self._velocity = 0.0
        self._force = 0.0
        self._energy = self._compute_potential_energy()

        # Neighbor references (set by KirigamiSheet)
        self.neighbors: List['TristableCell'] = []

    @property
    def state(self) -> CellState:
        """Get the current discrete state."""
        return self._state

    @property
    def value(self) -> float:
        """Get the continuous state value (0 to 1)."""
        return self._continuous_value

    @property
    def optical_response(self) -> OpticalResponse:
        """Get the optical response for current state."""
        return self.OPTICAL_TABLE[self._state]

    def _compute_potential_energy(self) -> float:
        """
        Compute potential energy for tristable landscape.

        The energy landscape has three minima at 0, 0.5, and 1,
        with barriers at thresholds defined in properties.

        Uses a triple-well potential:
        U(x) = (x - 0)(x - 0.5)(x - 1) * k
        """
        x = self._continuous_value
        k = self.properties.stiffness

        # Triple-well potential with minima at 0, 0.5, 1
        U = k * ((x - 0) * (x - 0.5) * (x - 1)) ** 2

        return U

    def _compute_force(self) -> float:
        """
        Compute restoring force from potential gradient.

        F = -dU/dx (negative gradient)
        """
        x = self._continuous_value
        k = self.properties.stiffness
        dx = 0.001  # Numerical differentiation step

        U_plus = k * ((x + dx) * (x + dx - 0.5) * (x + dx - 1)) ** 2
        U_minus = k * ((x - dx) * (x - dx - 0.5) * (x - dx - 1)) ** 2

        force = -(U_plus - U_minus) / (2 * dx)
        return force

    def apply_input(self, input_energy: float) -> bool:
        """
        Apply external input energy to the cell.

        If input exceeds threshold, triggers state transition.
        Returns True if state changed.

        Parameters
        ----------
        input_energy : float
            Energy input (normalized, 0-1 scale)

        Returns
        -------
        bool
            True if state transition occurred
        """
        old_state = self._state

        # Directly blend input into cell value (more responsive)
        # This simulates strong optical/mechanical forcing
        blend_factor = 0.8  # How much input affects cell
        self._continuous_value = (1 - blend_factor) * self._continuous_value + blend_factor * input_energy

        # Also add velocity for dynamics
        self._velocity += input_energy * 0.5

        # Check for state transitions based on current value
        if self._continuous_value < 0.25:
            self._state = CellState.CLOSED
        elif self._continuous_value < 0.75:
            self._state = CellState.INTERMEDIATE
        else:
            self._state = CellState.OPEN

        return self._state != old_state

    def update(self, dt: float = 0.01, external_force: float = 0.0) -> float:
        """
        Update cell dynamics for one timestep.

        Implements damped oscillator dynamics with tristable potential.
        This enables the "cascading changes" for reservoir computing.

        Parameters
        ----------
        dt : float
            Time step
        external_force : float
            External forcing (from input or neighbors)

        Returns
        -------
        float
            Energy transferred to neighbors (for cascading)
        """
        # Compute total force
        internal_force = self._compute_force()
        damping_force = -self.properties.damping * self._velocity
        total_force = internal_force + damping_force + external_force

        # Velocity Verlet integration
        self._velocity += total_force * dt
        self._continuous_value += self._velocity * dt

        # Clamp to valid range
        self._continuous_value = np.clip(self._continuous_value, 0.0, 1.0)

        # Update discrete state
        if self._continuous_value < 0.25:
            self._state = CellState.CLOSED
        elif self._continuous_value < 0.75:
            self._state = CellState.INTERMEDIATE
        else:
            self._state = CellState.OPEN

        # Update energy
        old_energy = self._energy
        self._energy = self._compute_potential_energy()

        # Energy available for neighbor coupling
        energy_release = max(0, old_energy - self._energy)

        return energy_release * self.properties.coupling_strength

    def get_neighbor_force(self) -> float:
        """
        Compute force contribution from neighbors.

        Implements mechanical coupling for cascade propagation.
        The coupling creates "strain propagation" as described
        in Section 7.1.

        Returns
        -------
        float
            Total force from neighbor interactions
        """
        if not self.neighbors:
            return 0.0

        total_force = 0.0
        for neighbor in self.neighbors:
            # Spring-like coupling
            delta = neighbor.value - self._continuous_value
            coupling = self.properties.coupling_strength
            total_force += coupling * delta

        return total_force

    def set_state(self, state: CellState):
        """Directly set cell state (for initialization)."""
        self._state = state
        self._continuous_value = float(state.value)
        self._velocity = 0.0
        self._energy = self._compute_potential_energy()

    def set_value(self, value: float):
        """Set continuous value directly."""
        self._continuous_value = np.clip(value, 0.0, 1.0)

        if self._continuous_value < 0.25:
            self._state = CellState.CLOSED
        elif self._continuous_value < 0.75:
            self._state = CellState.INTERMEDIATE
        else:
            self._state = CellState.OPEN

        self._energy = self._compute_potential_energy()

    def get_transmission(self) -> float:
        """
        Get interpolated transmission based on continuous value.

        Returns smoother optical response than discrete states.
        """
        x = self._continuous_value

        # Interpolate between state responses
        t0 = self.OPTICAL_TABLE[CellState.CLOSED].transmission
        t05 = self.OPTICAL_TABLE[CellState.INTERMEDIATE].transmission
        t1 = self.OPTICAL_TABLE[CellState.OPEN].transmission

        if x < 0.5:
            # Interpolate between 0 and 0.5
            t = 2 * x
            return t0 * (1 - t) + t05 * t
        else:
            # Interpolate between 0.5 and 1
            t = 2 * (x - 0.5)
            return t05 * (1 - t) + t1 * t

    def __repr__(self) -> str:
        return f"TristableCell(pos={self.position}, state={self._state.name}, value={self._continuous_value:.3f})"
