"""
Multi-Scale Reservoir - Multiple Timescale Processing
====================================================

Reservoir with fast and slow dynamics for temporal processing.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from ..core.fast_cascade import FastCascadeSimulator, CascadeResult


@dataclass
class MultiScaleResult:
    """Result from multi-scale reservoir."""
    fast_state: np.ndarray      # Fast layer final state
    slow_state: np.ndarray      # Slow layer final state
    combined_state: np.ndarray  # Merged state
    fast_history: Optional[np.ndarray]
    slow_history: Optional[np.ndarray]


class MultiScaleReservoir:
    """
    Multi-timescale reservoir for rich temporal processing.

    Implements two coupled reservoirs with different time constants:
    - Fast layer: Quick response, short memory
    - Slow layer: Slow response, long memory

    Cross-layer coupling enables hierarchical temporal processing.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Grid dimensions
    fast_coupling : float
        Coupling strength for fast layer
    slow_coupling : float
        Coupling strength for slow layer
    cross_coupling : float
        Coupling between layers
    fast_damping : float
        Damping for fast layer (higher = faster decay)
    slow_damping : float
        Damping for slow layer (lower = slower decay)

    Example
    -------
    >>> reservoir = MultiScaleReservoir((64, 64))
    >>> result = reservoir.process(input_sequence)
    >>> combined = result.combined_state
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        fast_coupling: float = 0.5,
        slow_coupling: float = 0.2,
        cross_coupling: float = 0.1,
        fast_damping: float = 0.3,
        slow_damping: float = 0.05
    ):
        self.grid_size = grid_size
        self.cross_coupling = cross_coupling

        # Fast reservoir (short memory)
        self.fast = FastCascadeSimulator(
            grid_size,
            coupling_strength=fast_coupling,
            damping=fast_damping
        )

        # Slow reservoir (long memory)
        self.slow = FastCascadeSimulator(
            grid_size,
            coupling_strength=slow_coupling,
            damping=slow_damping
        )

    def reset(self):
        """Reset both reservoirs."""
        self.fast.reset()
        self.slow.reset()

    def step(self, input_field: Optional[np.ndarray] = None, dt: float = 0.02) -> Tuple[float, float]:
        """
        Single timestep update of both layers.

        Parameters
        ----------
        input_field : np.ndarray, optional
            External input
        dt : float
            Time step

        Returns
        -------
        Tuple[float, float]
            Energy from fast and slow layers
        """
        # Inject input into fast layer
        if input_field is not None:
            self.fast.inject_input(input_field, scale=0.5)

        # Cross-coupling: fast -> slow
        cross_signal = self.fast.values * self.cross_coupling
        self.slow.velocities += cross_signal * dt

        # Update fast layer
        fast_energy = self.fast.step(dt)

        # Update slow layer (with slower time scale)
        slow_energy = self.slow.step(dt * 0.5)

        return fast_energy, slow_energy

    def process(
        self,
        input_sequence: List[np.ndarray],
        steps_per_input: int = 10,
        record_history: bool = False
    ) -> MultiScaleResult:
        """
        Process sequence of inputs.

        Parameters
        ----------
        input_sequence : List[np.ndarray]
            Sequence of input arrays
        steps_per_input : int
            Cascade steps per input
        record_history : bool
            Record state history

        Returns
        -------
        MultiScaleResult
            Processing results
        """
        fast_history = [] if record_history else None
        slow_history = [] if record_history else None

        for input_field in input_sequence:
            # Inject input
            self.fast.inject_input(input_field, scale=0.3)

            # Run cascade steps
            for _ in range(steps_per_input):
                self.step(dt=0.02)

                if record_history:
                    fast_history.append(self.fast.values.copy())
                    slow_history.append(self.slow.values.copy())

        # Combine states
        combined = 0.6 * self.fast.values + 0.4 * self.slow.values

        return MultiScaleResult(
            fast_state=self.fast.values.copy(),
            slow_state=self.slow.values.copy(),
            combined_state=combined,
            fast_history=np.array(fast_history) if fast_history else None,
            slow_history=np.array(slow_history) if slow_history else None
        )

    def process_single(
        self,
        input_field: np.ndarray,
        n_steps: int = 30
    ) -> MultiScaleResult:
        """
        Process single input.

        Parameters
        ----------
        input_field : np.ndarray
            Single input
        n_steps : int
            Total cascade steps

        Returns
        -------
        MultiScaleResult
            Results
        """
        return self.process([input_field], steps_per_input=n_steps)

    def get_memory_content(self) -> Dict[str, float]:
        """
        Estimate memory content in each layer.

        Returns statistics about how much information
        is retained in each timescale.

        Returns
        -------
        Dict[str, float]
            Memory metrics
        """
        fast_activity = np.mean(np.abs(self.fast.values - 0.5))
        slow_activity = np.mean(np.abs(self.slow.values - 0.5))

        fast_variance = np.var(self.fast.values)
        slow_variance = np.var(self.slow.values)

        return {
            'fast_activity': fast_activity,
            'slow_activity': slow_activity,
            'fast_variance': fast_variance,
            'slow_variance': slow_variance,
            'memory_ratio': slow_activity / (fast_activity + 1e-8)
        }

    def set_timescales(self, fast_damping: float, slow_damping: float):
        """Adjust timescale separation."""
        self.fast.damping = fast_damping
        self.slow.damping = slow_damping
