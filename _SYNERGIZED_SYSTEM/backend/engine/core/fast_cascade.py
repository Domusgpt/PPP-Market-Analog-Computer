"""
Fast Cascade Simulator - Vectorized Reservoir Dynamics
======================================================

High-performance implementation of kirigami reservoir cascade
dynamics using NumPy vectorization and optional Numba JIT.

Key optimizations:
- Convolution-based neighbor coupling (replaces cell iteration)
- Vectorized state updates
- Pre-allocated arrays
- Optional Numba parallel execution
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

# Try to import numba
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import scipy for convolution
try:
    from scipy.signal import convolve2d
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class CellState(IntEnum):
    """Tristable cell states as integers for fast computation."""
    CLOSED = 0
    INTERMEDIATE = 1
    OPEN = 2


@dataclass
class CascadeResult:
    """Result of cascade simulation."""
    final_state: np.ndarray       # Final cell values
    discrete_state: np.ndarray    # Quantized to 0, 0.5, 1
    history: Optional[np.ndarray] # Evolution history if recorded
    steps_taken: int              # Actual steps before convergence
    total_energy: float           # Total energy released


class FastCascadeSimulator:
    """
    Vectorized reservoir cascade simulator.

    Simulates the cascading dynamics of a hexagonal cell grid
    using array operations instead of cell-by-cell iteration.

    The cascade implements:
    - Tristable potential (3 stable states)
    - Hexagonal neighbor coupling
    - Damped dynamics with energy dissipation
    - Configurable stiffness maps (attention weights)

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Grid dimensions (ny, nx)
    coupling_strength : float
        Neighbor coupling coefficient (0-1)
    damping : float
        Damping coefficient for energy dissipation
    stiffness : float or np.ndarray
        Base stiffness or spatial stiffness map

    Example
    -------
    >>> sim = FastCascadeSimulator((64, 64), coupling_strength=0.3)
    >>> result = sim.run(input_field, n_steps=50)
    >>> plt.imshow(result.final_state)
    """

    # Hexagonal neighbor kernel (6 neighbors with offset for hex grid)
    # This is an approximation using a 3x3 kernel
    HEX_KERNEL = np.array([
        [0.5, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 0.5]
    ]) / 6.0

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        coupling_strength: float = 0.3,
        damping: float = 0.1,
        stiffness: float = 1.0
    ):
        self.ny, self.nx = grid_size
        self.coupling = coupling_strength
        self.damping = damping

        # State arrays (pre-allocated)
        self.values = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.velocities = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Stiffness map
        if isinstance(stiffness, np.ndarray):
            self.stiffness = stiffness.astype(np.float64)
        else:
            self.stiffness = np.full((self.ny, self.nx), stiffness, dtype=np.float64)

        # Potential well positions (tristable)
        self.wells = np.array([0.0, 0.5, 1.0])

    def reset(self, initial_value: float = 0.0):
        """Reset all cells to initial value."""
        self.values.fill(initial_value)
        self.velocities.fill(0.0)

    def inject_input(self, input_field: np.ndarray, scale: float = 1.0, blend: float = 0.5):
        """
        Inject input into the reservoir.

        Parameters
        ----------
        input_field : np.ndarray
            2D input array (will be resized if needed)
        scale : float
            Scaling factor for velocity perturbation
        blend : float
            How much to directly blend input into values (0-1)
            KEY FIX: Direct value injection ensures tristable states are used
        """
        # Resize if needed
        if input_field.shape != (self.ny, self.nx):
            from scipy.ndimage import zoom
            factors = (self.ny / input_field.shape[0],
                      self.nx / input_field.shape[1])
            input_field = zoom(input_field, factors, order=1)

        # Normalize to [0, 1]
        input_min = np.min(input_field)
        input_max = np.max(input_field)
        if input_max > input_min:
            input_field = (input_field - input_min) / (input_max - input_min)

        # KEY FIX: Directly blend input into VALUES (not just velocity)
        # This ensures cells actually reach different wells based on input
        self.values = (1 - blend) * self.values + blend * input_field

        # Also add velocity for dynamics
        self.velocities += input_field * scale * 0.3

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_potential_force_numba(values: np.ndarray, wells: np.ndarray) -> np.ndarray:
        """
        Compute force from tristable potential using Numba.

        Potential: U(x) = (x - 0)(x - 0.5)(x - 1) type function
        Force: F = -dU/dx
        """
        ny, nx = values.shape
        force = np.zeros_like(values)

        for j in range(ny):
            for i in range(nx):
                x = values[j, i]
                # Derivative of triple-well potential
                # U = (x-0)^2 * (x-0.5)^2 * (x-1)^2 simplified
                # F = -dU/dx approximated as gradient toward nearest well

                # Find nearest well
                d0 = abs(x - 0.0)
                d1 = abs(x - 0.5)
                d2 = abs(x - 1.0)

                if d0 <= d1 and d0 <= d2:
                    target = 0.0
                elif d1 <= d2:
                    target = 0.5
                else:
                    target = 1.0

                # Force toward nearest well with nonlinear spring
                diff = target - x
                force[j, i] = 2.0 * diff * (1.0 + abs(diff))

        return force

    def _compute_potential_force_numpy(self, values: np.ndarray) -> np.ndarray:
        """
        Compute force from tristable potential using NumPy.

        Vectorized computation of force toward nearest stable state.
        """
        # Distance to each well
        d0 = np.abs(values - 0.0)
        d1 = np.abs(values - 0.5)
        d2 = np.abs(values - 1.0)

        # Find nearest well (vectorized argmin)
        distances = np.stack([d0, d1, d2], axis=0)
        nearest_idx = np.argmin(distances, axis=0)

        # Target values
        targets = np.choose(nearest_idx, self.wells)

        # Force toward target (nonlinear spring)
        diff = targets - values
        force = 2.0 * diff * (1.0 + np.abs(diff))

        return force

    def _compute_neighbor_coupling(self, values: np.ndarray) -> np.ndarray:
        """
        Compute neighbor coupling forces using convolution.

        This replaces the cell-by-cell neighbor iteration with
        a single convolution operation.
        """
        if HAS_SCIPY:
            # Convolve with hex kernel to get neighbor average
            neighbor_avg = convolve2d(values, self.HEX_KERNEL, mode='same', boundary='wrap')
        else:
            # Fallback: simple 3x3 average
            neighbor_avg = np.zeros_like(values)
            neighbor_avg[1:-1, 1:-1] = (
                values[:-2, 1:-1] + values[2:, 1:-1] +
                values[1:-1, :-2] + values[1:-1, 2:] +
                values[:-2, :-2] + values[2:, 2:]
            ) / 6.0

        # Coupling force
        return self.coupling * (neighbor_avg - values)

    def step(self, dt: float = 0.02) -> float:
        """
        Advance simulation by one timestep.

        Returns total energy change (for convergence check).
        """
        # Potential force (toward stable states)
        if HAS_NUMBA:
            F_potential = self._compute_potential_force_numba(self.values, self.wells)
        else:
            F_potential = self._compute_potential_force_numpy(self.values)

        # Neighbor coupling force
        F_coupling = self._compute_neighbor_coupling(self.values)

        # Total force (scaled by stiffness)
        F_total = self.stiffness * F_potential + F_coupling

        # Damped velocity update
        self.velocities = (1.0 - self.damping) * self.velocities + dt * F_total

        # Position update
        old_values = self.values.copy()
        self.values = np.clip(self.values + dt * self.velocities, 0.0, 1.0)

        # Energy released (sum of squared changes)
        energy = np.sum((self.values - old_values) ** 2)

        return energy

    def run(
        self,
        input_field: Optional[np.ndarray] = None,
        n_steps: int = 50,
        dt: float = 0.02,
        convergence_threshold: float = 1e-5,
        record_history: bool = False
    ) -> CascadeResult:
        """
        Run full cascade simulation.

        Parameters
        ----------
        input_field : np.ndarray, optional
            Input to inject before cascade
        n_steps : int
            Maximum number of steps
        dt : float
            Time step
        convergence_threshold : float
            Energy threshold for early stopping
        record_history : bool
            Record state at each step

        Returns
        -------
        CascadeResult
            Simulation results
        """
        # Inject input if provided
        if input_field is not None:
            self.inject_input(input_field)

        # History recording
        history = [] if record_history else None
        total_energy = 0.0

        # Run cascade
        for step in range(n_steps):
            if record_history:
                history.append(self.values.copy())

            energy = self.step(dt)
            total_energy += energy

            if energy < convergence_threshold:
                break

        # Quantize final state
        discrete = np.zeros_like(self.values)
        discrete[self.values < 0.25] = 0.0
        discrete[(self.values >= 0.25) & (self.values < 0.75)] = 0.5
        discrete[self.values >= 0.75] = 1.0

        return CascadeResult(
            final_state=self.values.copy(),
            discrete_state=discrete,
            history=np.array(history) if history else None,
            steps_taken=step + 1,
            total_energy=total_energy
        )

    def set_stiffness_map(self, stiffness: np.ndarray):
        """
        Set spatial stiffness map (attention weights).

        Parameters
        ----------
        stiffness : np.ndarray
            2D stiffness values (will be resized if needed)
        """
        if stiffness.shape != (self.ny, self.nx):
            from scipy.ndimage import zoom
            factors = (self.ny / stiffness.shape[0],
                      self.nx / stiffness.shape[1])
            stiffness = zoom(stiffness, factors, order=1)

        self.stiffness = stiffness.astype(np.float64)

    def set_radial_attention(self, center: Tuple[float, float] = (0.5, 0.5), falloff: float = 0.5):
        """
        Set radial attention pattern (soft center, stiff edges or vice versa).

        Parameters
        ----------
        center : Tuple[float, float]
            Center of attention in normalized coords (0-1)
        falloff : float
            Falloff rate (positive = soft center, negative = stiff center)
        """
        y = np.linspace(0, 1, self.ny)
        x = np.linspace(0, 1, self.nx)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Distance from center
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        dist = dist / np.max(dist)  # Normalize

        # Stiffness profile
        if falloff > 0:
            # Soft center (low stiffness = more responsive)
            self.stiffness = 0.5 + falloff * dist
        else:
            # Stiff center
            self.stiffness = 0.5 - falloff * (1 - dist)

    def get_transmission(self) -> np.ndarray:
        """
        Get optical transmission map from current state.

        Maps cell values to transmission coefficients.
        """
        # Sigmoid-like transmission curve
        return 0.2 + 0.8 * self.values


class BatchCascadeSimulator:
    """
    Batch cascade simulator for processing multiple inputs.

    Reuses single simulator instance for memory efficiency.
    """

    def __init__(self, grid_size: Tuple[int, int] = (64, 64), **kwargs):
        self.simulator = FastCascadeSimulator(grid_size, **kwargs)
        self.grid_size = grid_size

    def run_batch(
        self,
        inputs: List[np.ndarray],
        n_steps: int = 50,
        **kwargs
    ) -> List[CascadeResult]:
        """
        Process batch of inputs.

        Parameters
        ----------
        inputs : List[np.ndarray]
            List of input arrays
        n_steps : int
            Steps per cascade
        **kwargs
            Additional arguments for run()

        Returns
        -------
        List[CascadeResult]
            Results for each input
        """
        results = []

        for input_field in inputs:
            self.simulator.reset()
            result = self.simulator.run(input_field, n_steps, **kwargs)
            results.append(result)

        return results

    def run_batch_array(
        self,
        inputs: np.ndarray,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Process batch and return just final states.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of inputs, shape (batch, ny, nx)
        n_steps : int
            Steps per cascade

        Returns
        -------
        np.ndarray
            Final states, shape (batch, ny, nx)
        """
        batch_size = inputs.shape[0]
        outputs = np.zeros((batch_size, *self.grid_size), dtype=np.float32)

        for i in range(batch_size):
            self.simulator.reset()
            result = self.simulator.run(inputs[i], n_steps)
            outputs[i] = result.final_state

        return outputs
