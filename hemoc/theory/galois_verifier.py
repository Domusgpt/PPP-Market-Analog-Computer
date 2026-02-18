"""
Galois Dual-Channel Verifier
==============================

Every projection through the Phillips matrix runs through BOTH the left
(contracted) and right (expanded) blocks.  The Galois automorphism
phi <-> -1/phi implies:

    ||U_R x|| / ||U_L x|| = phi     for ALL x in R^8  (ratio check)
    ||U_L x|| * ||U_R x|| = phi * ||U_L x||^2          (product check)

If these identities are violated at runtime, an error has occurred in
the computation pipeline.  This is FREE error detection: the redundancy
is built into the algebraic structure, not added as overhead.

This module provides both single-vector and batch verification, plus
a diagnostic mode that returns deviation distributions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from hemoc.core.phillips_matrix import PHILLIPS_U_L, PHILLIPS_U_R, PHI, SQRT5
from hemoc.core.e8_roots import generate_e8_roots


class GaloisDualVerifier:
    """
    Dual-channel computation with phi-coupled error detection.

    Parameters
    ----------
    tolerance : float
        Maximum allowed deviation from phi-coupling before flagging.
    """

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        self._checks = 0
        self._errors = 0
        self._max_deviation = 0.0

    def verify_single(self, v8: np.ndarray) -> Dict:
        """
        Verify phi-coupling for a single 8D vector.

        Returns
        -------
        Dict with ratio, deviation, validity flag.
        """
        v8 = np.asarray(v8, dtype=np.float64)
        h_L = PHILLIPS_U_L @ v8
        h_R = PHILLIPS_U_R @ v8

        norm_L = float(np.linalg.norm(h_L))
        norm_R = float(np.linalg.norm(h_R))

        self._checks += 1

        if norm_L < 1e-14:
            return {
                "h4_left": h_L, "h4_right": h_R,
                "ratio": float("nan"),
                "deviation": 0.0,
                "valid": True,
                "note": "zero vector (trivially valid)",
            }

        ratio = norm_R / norm_L
        deviation = abs(ratio - PHI)
        self._max_deviation = max(self._max_deviation, deviation)

        valid = deviation < self.tolerance
        if not valid:
            self._errors += 1

        return {
            "h4_left": h_L,
            "h4_right": h_R,
            "norm_left": norm_L,
            "norm_right": norm_R,
            "ratio": ratio,
            "expected_ratio": float(PHI),
            "deviation": deviation,
            "valid": valid,
        }

    def verify_batch(self, vectors: np.ndarray) -> Dict:
        """
        Batch verification for an (N, 8) array of vectors.

        Returns
        -------
        Dict with aggregate statistics.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        left_norms = np.linalg.norm(vectors @ PHILLIPS_U_L.T, axis=1)
        right_norms = np.linalg.norm(vectors @ PHILLIPS_U_R.T, axis=1)

        safe = left_norms > 1e-14
        ratios = np.full(len(vectors), np.nan)
        ratios[safe] = right_norms[safe] / left_norms[safe]
        deviations = np.abs(ratios - PHI)
        deviations[~safe] = 0.0

        self._checks += len(vectors)
        n_invalid = int(np.sum(deviations[safe] >= self.tolerance))
        self._errors += n_invalid

        return {
            "n_vectors": len(vectors),
            "all_valid": n_invalid == 0,
            "n_invalid": n_invalid,
            "max_deviation": float(np.nanmax(deviations)) if safe.any() else 0.0,
            "mean_deviation": float(np.nanmean(deviations[safe])) if safe.any() else 0.0,
            "deviation_std": float(np.nanstd(deviations[safe])) if safe.any() else 0.0,
        }

    def verify_e8_roots(self) -> Dict:
        """Verify phi-coupling across all 240 E8 roots (should be exact)."""
        roots = generate_e8_roots()
        coords = np.array([r.coordinates for r in roots])
        return self.verify_batch(coords)

    def sqrt5_row_norm_check(self) -> Dict:
        """
        Verify the sqrt(5) identity on the matrix itself:
          sqrt(row_norm_L^2) * sqrt(row_norm_R^2) = sqrt(5).
        """
        row_norm_L = np.sqrt(np.sum(PHILLIPS_U_L[0] ** 2))
        row_norm_R = np.sqrt(np.sum(PHILLIPS_U_R[0] ** 2))
        product = row_norm_L * row_norm_R
        return {
            "row_norm_left": float(row_norm_L),
            "row_norm_right": float(row_norm_R),
            "product": float(product),
            "expected": float(SQRT5),
            "deviation": abs(product - SQRT5),
            "pass": abs(product - SQRT5) < self.tolerance,
        }

    @property
    def cumulative_error_rate(self) -> float:
        if self._checks == 0:
            return 0.0
        return self._errors / self._checks

    @property
    def max_observed_deviation(self) -> float:
        return self._max_deviation

    def reset_counters(self):
        self._checks = 0
        self._errors = 0
        self._max_deviation = 0.0
