"""
Criticality Analyzer - Edge-of-Chaos Dynamics
============================================

Tools for analyzing and controlling reservoir criticality.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from ..core.fast_cascade import FastCascadeSimulator


@dataclass
class CriticalityMetrics:
    """Criticality analysis results."""
    lyapunov_estimate: float      # Largest Lyapunov exponent estimate
    branching_ratio: float        # Avalanche branching ratio
    correlation_length: float     # Spatial correlation length
    is_critical: bool             # Near critical point?
    regime: str                   # "subcritical", "critical", "supercritical"


class CriticalityAnalyzer:
    """
    Analyzer for reservoir criticality.

    Measures how close the reservoir operates to the
    "edge of chaos" - the critical point between ordered
    and chaotic dynamics where computation is optimal.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Reservoir grid size

    Example
    -------
    >>> analyzer = CriticalityAnalyzer((64, 64))
    >>> metrics = analyzer.analyze(coupling=0.3, damping=0.1)
    >>> print(f"Regime: {metrics.regime}")
    >>> optimal_coupling = analyzer.find_critical_point()
    """

    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        self.grid_size = grid_size

    def analyze(
        self,
        coupling: float,
        damping: float,
        n_trials: int = 10,
        n_steps: int = 100
    ) -> CriticalityMetrics:
        """
        Analyze criticality for given parameters.

        Parameters
        ----------
        coupling : float
            Coupling strength
        damping : float
            Damping coefficient
        n_trials : int
            Number of analysis trials
        n_steps : int
            Steps per trial

        Returns
        -------
        CriticalityMetrics
            Criticality analysis results
        """
        # Create simulator
        sim = FastCascadeSimulator(
            self.grid_size,
            coupling_strength=coupling,
            damping=damping
        )

        # Lyapunov estimate via trajectory divergence
        lyapunov = self._estimate_lyapunov(sim, n_steps)

        # Branching ratio from avalanche statistics
        branching = self._compute_branching_ratio(sim, n_trials, n_steps)

        # Correlation length
        corr_length = self._compute_correlation_length(sim)

        # Classify regime
        if branching < 0.9:
            regime = "subcritical"
        elif branching > 1.1:
            regime = "supercritical"
        else:
            regime = "critical"

        is_critical = 0.9 <= branching <= 1.1

        return CriticalityMetrics(
            lyapunov_estimate=lyapunov,
            branching_ratio=branching,
            correlation_length=corr_length,
            is_critical=is_critical,
            regime=regime
        )

    def _estimate_lyapunov(
        self,
        sim: FastCascadeSimulator,
        n_steps: int
    ) -> float:
        """Estimate largest Lyapunov exponent."""
        # Run two trajectories with small initial difference
        sim.reset()
        sim.values = np.random.rand(*self.grid_size) * 0.5
        initial1 = sim.values.copy()

        # Perturbed initial condition
        epsilon = 1e-6
        initial2 = initial1 + epsilon * np.random.randn(*self.grid_size)

        # Run both trajectories
        trajectory1 = []
        sim.values = initial1.copy()
        for _ in range(n_steps):
            sim.step()
            trajectory1.append(sim.values.copy())

        trajectory2 = []
        sim.values = initial2.copy()
        sim.reset()
        sim.values = initial2.copy()
        for _ in range(n_steps):
            sim.step()
            trajectory2.append(sim.values.copy())

        # Compute divergence
        divergences = []
        for t1, t2 in zip(trajectory1, trajectory2):
            div = np.linalg.norm(t1 - t2)
            if div > 0:
                divergences.append(div)

        if len(divergences) < 2:
            return 0.0

        # Lyapunov exponent estimate
        divergences = np.array(divergences)
        lyapunov = np.mean(np.log(divergences[1:] / divergences[:-1] + 1e-10))

        return lyapunov

    def _compute_branching_ratio(
        self,
        sim: FastCascadeSimulator,
        n_trials: int,
        n_steps: int
    ) -> float:
        """Compute avalanche branching ratio."""
        ratios = []

        for _ in range(n_trials):
            sim.reset()

            # Trigger single cell
            center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
            sim.values[center] = 1.0

            # Count active cells over time
            active_counts = []
            threshold = 0.1

            for _ in range(n_steps):
                sim.step()
                active = np.sum(np.abs(sim.values - 0.5) > threshold)
                active_counts.append(active)

            # Branching ratio: ratio of activity generation to generation
            for i in range(1, len(active_counts)):
                if active_counts[i-1] > 0:
                    ratio = active_counts[i] / active_counts[i-1]
                    ratios.append(ratio)

        if not ratios:
            return 1.0

        return np.median(ratios)

    def _compute_correlation_length(
        self,
        sim: FastCascadeSimulator
    ) -> float:
        """Compute spatial correlation length."""
        # Generate random activity
        sim.reset()
        sim.values = np.random.rand(*self.grid_size)
        sim.run(n_steps=50)

        state = sim.values

        # Compute autocorrelation
        fft = np.fft.fft2(state - np.mean(state))
        power = np.abs(fft) ** 2
        autocorr = np.real(np.fft.ifft2(power))
        autocorr = autocorr / autocorr[0, 0]

        # Find correlation length (distance to 1/e)
        ny, nx = self.grid_size
        center_y, center_x = ny // 2, nx // 2

        for r in range(1, min(center_y, center_x)):
            # Average correlation at distance r
            y, x = np.ogrid[-center_y:ny-center_y, -center_x:nx-center_x]
            mask = (np.sqrt(x**2 + y**2) >= r-0.5) & (np.sqrt(x**2 + y**2) < r+0.5)

            if np.any(mask):
                avg_corr = np.mean(autocorr[mask])
                if avg_corr < 1/np.e:
                    return float(r)

        return float(min(center_y, center_x))

    def find_critical_point(
        self,
        coupling_range: Tuple[float, float] = (0.1, 0.9),
        n_search: int = 10
    ) -> float:
        """
        Find coupling value near critical point.

        Parameters
        ----------
        coupling_range : Tuple[float, float]
            Search range for coupling
        n_search : int
            Number of search points

        Returns
        -------
        float
            Optimal coupling value
        """
        couplings = np.linspace(*coupling_range, n_search)
        best_coupling = couplings[len(couplings)//2]
        best_distance = float('inf')

        for coupling in couplings:
            metrics = self.analyze(coupling, damping=0.1, n_trials=5, n_steps=50)

            # Distance from branching ratio = 1
            distance = abs(metrics.branching_ratio - 1.0)

            if distance < best_distance:
                best_distance = distance
                best_coupling = coupling

        return best_coupling

    def sweep_parameters(
        self,
        coupling_range: Tuple[float, float] = (0.1, 0.8),
        damping_range: Tuple[float, float] = (0.05, 0.3),
        n_points: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Sweep parameter space and compute criticality metrics.

        Parameters
        ----------
        coupling_range : Tuple[float, float]
            Coupling range to sweep
        damping_range : Tuple[float, float]
            Damping range to sweep
        n_points : int
            Points per dimension

        Returns
        -------
        Dict[str, np.ndarray]
            Grid of metrics
        """
        couplings = np.linspace(*coupling_range, n_points)
        dampings = np.linspace(*damping_range, n_points)

        branching_grid = np.zeros((n_points, n_points))
        lyapunov_grid = np.zeros((n_points, n_points))

        for i, coupling in enumerate(couplings):
            for j, damping in enumerate(dampings):
                metrics = self.analyze(coupling, damping, n_trials=3, n_steps=30)
                branching_grid[i, j] = metrics.branching_ratio
                lyapunov_grid[i, j] = metrics.lyapunov_estimate

        return {
            'couplings': couplings,
            'dampings': dampings,
            'branching_ratio': branching_grid,
            'lyapunov': lyapunov_grid
        }
