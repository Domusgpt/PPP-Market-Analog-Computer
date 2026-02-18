"""
Renderer Validation Test Suite
================================

Standalone functions that validate ANY renderer implementing the
RendererContract.  These are designed to be called from pytest or
from a Colab notebook.

Tests
-----
  test_contract_compliance()   -- Full contract suite
  test_proxy_vs_physics()      -- Detect proxy renderers by L1 divergence
  test_parameter_sensitivity() -- Per-angle derivative distribution
  test_galois_preservation()   -- Dual-channel phi-coupling after render
"""

from typing import Dict
import numpy as np

from hemoc.render.renderer_contract import RendererContract
from hemoc.core.phillips_matrix import PHI


def test_contract_compliance(
    renderer: RendererContract,
    seed: int = 42,
) -> Dict:
    """
    Run the full contract validation suite on a renderer.

    Returns
    -------
    Dict with all test results and overall pass/fail.
    """
    return renderer.run_contract_suite(seed=seed)


def test_proxy_vs_physics(
    renderer_a: RendererContract,
    renderer_b: RendererContract,
    n_samples: int = 50,
    l1_threshold: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Compare two renderers and measure L1 divergence.

    If L1 > threshold, the renderers are producing materially different
    outputs, which indicates one may be a proxy.

    Parameters
    ----------
    renderer_a, renderer_b : RendererContract
        The two renderers to compare.
    n_samples : int
        Number of random angle configurations to test.
    l1_threshold : float
        Maximum acceptable mean L1 divergence.
    seed : int
        Random seed.

    Returns
    -------
    Dict with per-sample and aggregate L1 statistics.
    """
    rng = np.random.RandomState(seed)
    l1_values = []

    for _ in range(n_samples):
        angles = rng.uniform(0, np.pi, size=6)
        result_a = renderer_a.render(angles)
        result_b = renderer_b.render(angles)

        # Normalize shapes if needed
        img_a = result_a.image
        img_b = result_b.image
        if img_a.shape != img_b.shape:
            # Cannot compare different output shapes
            return {
                "comparable": False,
                "reason": f"Shape mismatch: {img_a.shape} vs {img_b.shape}",
            }

        l1 = float(np.mean(np.abs(img_a - img_b)))
        l1_values.append(l1)

    mean_l1 = float(np.mean(l1_values))
    return {
        "comparable": True,
        "n_samples": n_samples,
        "mean_l1": mean_l1,
        "std_l1": float(np.std(l1_values)),
        "max_l1": float(np.max(l1_values)),
        "min_l1": float(np.min(l1_values)),
        "threshold": l1_threshold,
        "likely_proxy": mean_l1 > l1_threshold,
        "pass": mean_l1 <= l1_threshold,
    }


def test_parameter_sensitivity(
    renderer: RendererContract,
    n_samples: int = 20,
    delta: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Measure per-angle sensitivity as a derivative distribution.

    For each angle, computes the finite-difference derivative of the
    mean image value.  Angles with near-zero sensitivity are flagged.

    Returns
    -------
    Dict with per-angle sensitivity statistics.
    """
    rng = np.random.RandomState(seed)
    sensitivities = [[] for _ in range(6)]

    for _ in range(n_samples):
        base_angles = rng.uniform(0, np.pi, size=6)
        base_result = renderer.render(base_angles)
        base_mean = np.mean(base_result.image)

        for i in range(6):
            pert_angles = base_angles.copy()
            pert_angles[i] += delta
            pert_result = renderer.render(pert_angles)
            pert_mean = np.mean(pert_result.image)
            deriv = abs(pert_mean - base_mean) / delta
            sensitivities[i].append(deriv)

    results = {}
    for i in range(6):
        s = sensitivities[i]
        results[f"angle_{i}"] = {
            "mean_sensitivity": float(np.mean(s)),
            "std_sensitivity": float(np.std(s)),
            "min_sensitivity": float(np.min(s)),
            "nonzero": float(np.mean(s)) > 1e-6,
        }

    all_nonzero = all(r["nonzero"] for r in results.values())

    return {
        "n_samples": n_samples,
        "delta": delta,
        "per_angle": results,
        "all_angles_sensitive": all_nonzero,
        "pass": all_nonzero,
    }


def test_galois_preservation(
    renderer: RendererContract,
    n_samples: int = 50,
    tolerance: float = 1e-2,
    seed: int = 42,
) -> Dict:
    """
    Check that the Galois phi-coupling is preserved in render outputs.

    For dual-channel renderers, verify that the left and right image
    energies maintain a phi-related ratio.

    Returns
    -------
    Dict with Galois coupling statistics.
    """
    rng = np.random.RandomState(seed)
    ratios = []
    valid_count = 0

    for _ in range(n_samples):
        angles = rng.uniform(0, np.pi, size=6)
        result = renderer.render(angles)

        if result.galois_ratio is not None:
            ratios.append(result.galois_ratio)
            if result.galois_valid:
                valid_count += 1
        elif result.left_image is not None and result.right_image is not None:
            energy_L = np.sum(result.left_image ** 2)
            energy_R = np.sum(result.right_image ** 2)
            if energy_L > 1e-12:
                ratio = np.sqrt(energy_R / energy_L)
                ratios.append(ratio)

    if not ratios:
        return {
            "n_samples": n_samples,
            "galois_supported": False,
            "note": "Renderer does not produce dual-channel or Galois metadata",
            "pass": True,  # Not applicable
        }

    deviations = [abs(r - PHI) for r in ratios]
    return {
        "n_samples": n_samples,
        "galois_supported": True,
        "mean_ratio": float(np.mean(ratios)),
        "expected_ratio": float(PHI),
        "mean_deviation": float(np.mean(deviations)),
        "max_deviation": float(np.max(deviations)),
        "valid_count": valid_count,
        "valid_rate": valid_count / len(ratios) if ratios else 0.0,
        "pass": float(np.mean(deviations)) < tolerance,
    }
