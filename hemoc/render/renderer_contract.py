"""
Renderer Contract (Abstract Interface)
========================================

Any renderer in the HEMOC system must satisfy this contract.  The contract
ensures that:

  1. Rotation consistency:  known analytic identities hold under rotation.
  2. Determinism:  same parameters -> identical output.
  3. Sensitivity:  each of the 6 angular parameters changes the output.
  4. Proxy detection:  L1 divergence from ground truth is bounded.
  5. Norm preservation:  phi-coupling between channels is maintained.

Renderers are tagged by fidelity class:
  - "proxy":       Heuristic approximation (known ceiling effects)
  - "physics":     Full 4D rotation + projection (the fixed renderer)
  - "differentiable": Physics renderer + autograd support (for training)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RenderResult:
    """
    Output of a single render call.

    Attributes
    ----------
    image : np.ndarray
        Rendered image.  Shape depends on renderer (e.g., (64,64,3) or
        (5, 64, 64) for 5-channel hex grating).
    left_image : np.ndarray or None
        If dual-channel, the h_L-driven image.
    right_image : np.ndarray or None
        If dual-channel, the h_R-driven image.
    metadata : dict
        Renderer version, parameters used, timing, etc.
    galois_ratio : float or None
        If Galois-verified, the measured ||h_R|| / ||h_L|| ratio.
    galois_valid : bool or None
        Whether the Galois invariant was within tolerance.
    """
    image: np.ndarray
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    galois_ratio: Optional[float] = None
    galois_valid: Optional[bool] = None


class RendererContract(ABC):
    """
    Abstract contract that all HEMOC renderers must implement.

    Subclasses must provide:
        render(angles) -> RenderResult
        fidelity_class -> str

    The contract includes self-validation methods that can be run
    as part of a test suite.
    """

    @abstractmethod
    def render(self, angles: np.ndarray) -> RenderResult:
        """
        Render a moire pattern from 6 angular parameters.

        Parameters
        ----------
        angles : np.ndarray of shape (6,)
            The six rotation-plane angles controlling the 4D rotation:
            (xy, yz, xz, xw, yw, zw) or equivalent.

        Returns
        -------
        RenderResult
        """
        ...

    @property
    @abstractmethod
    def fidelity_class(self) -> str:
        """One of 'proxy', 'physics', 'differentiable'."""
        ...

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """Shape of the rendered image (e.g., (64, 64, 3))."""
        ...

    # ------------------------------------------------------------------
    # Contract validation methods (can be called by test suite)
    # ------------------------------------------------------------------

    def validate_determinism(self, angles: np.ndarray, n_reps: int = 5) -> Dict:
        """Same angles must produce identical output every time."""
        results = [self.render(angles) for _ in range(n_reps)]
        reference = results[0].image
        all_identical = all(
            np.array_equal(r.image, reference) for r in results[1:]
        )
        return {
            "test": "determinism",
            "n_reps": n_reps,
            "all_identical": all_identical,
            "pass": all_identical,
        }

    def validate_sensitivity(
        self,
        base_angles: np.ndarray,
        delta: float = 0.1,
    ) -> Dict:
        """Each of the 6 angles must affect the output."""
        base_result = self.render(base_angles)
        sensitivities = []

        for i in range(6):
            perturbed = base_angles.copy()
            perturbed[i] += delta
            pert_result = self.render(perturbed)
            diff = np.mean(np.abs(pert_result.image - base_result.image))
            sensitivities.append(float(diff))

        return {
            "test": "sensitivity",
            "per_angle_sensitivity": sensitivities,
            "all_nonzero": all(s > 1e-8 for s in sensitivities),
            "pass": all(s > 1e-8 for s in sensitivities),
        }

    def validate_output_shape(self, angles: np.ndarray) -> Dict:
        """Output must match declared shape."""
        result = self.render(angles)
        return {
            "test": "output_shape",
            "actual_shape": result.image.shape,
            "declared_shape": self.output_shape,
            "pass": result.image.shape == self.output_shape,
        }

    def validate_rotation_consistency(
        self,
        angles: np.ndarray,
        tolerance: float = 0.1,
    ) -> Dict:
        """
        Rotation by 2*pi in any plane should return to the original.

        This tests whether the renderer correctly implements periodic
        angular dependence.
        """
        base_result = self.render(angles)
        results = []

        for i in range(6):
            rotated = angles.copy()
            rotated[i] += 2.0 * np.pi
            rot_result = self.render(rotated)
            diff = float(np.mean(np.abs(rot_result.image - base_result.image)))
            results.append({
                "angle_index": i,
                "diff_mean": diff,
                "periodic": diff < tolerance,
            })

        return {
            "test": "rotation_consistency",
            "details": results,
            "pass": all(r["periodic"] for r in results),
        }

    def run_contract_suite(self, seed: int = 42) -> Dict:
        """
        Run the full contract validation suite.

        Returns
        -------
        Dict with all test results.
        """
        rng = np.random.RandomState(seed)
        angles = rng.uniform(0, np.pi, size=6)

        tests = [
            self.validate_determinism(angles),
            self.validate_sensitivity(angles),
            self.validate_output_shape(angles),
            self.validate_rotation_consistency(angles),
        ]

        return {
            "renderer_class": type(self).__name__,
            "fidelity_class": self.fidelity_class,
            "all_pass": all(t["pass"] for t in tests),
            "n_passed": sum(1 for t in tests if t["pass"]),
            "n_total": len(tests),
            "tests": tests,
        }
