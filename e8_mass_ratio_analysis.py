#!/usr/bin/env python3
"""
E8 → H4 PARTICLE MASS RATIO ANALYSIS
=====================================

Tests whether the E8 folding framework can predict particle mass ratios
better than random chance or standard parameterizations.

Key hypothesis: The golden ratio (φ) structure embedded in H4/600-cell
geometry may encode fundamental mass relationships.

Author: E8 Three-Body Research
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple
import json

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# =============================================================================
# MEASURED PARTICLE MASSES (PDG 2024 values in MeV/c²)
# =============================================================================

LEPTON_MASSES = {
    'electron': 0.51099895,
    'muon': 105.6583755,
    'tau': 1776.86,
}

QUARK_MASSES = {
    # Current quark masses (MS-bar scheme at 2 GeV for light quarks)
    'up': 2.16,        # MeV
    'down': 4.67,      # MeV
    'strange': 93.4,   # MeV
    'charm': 1270,     # MeV (at mc scale)
    'bottom': 4180,    # MeV (at mb scale)
    'top': 172760,     # MeV
}

BOSON_MASSES = {
    'W': 80377,        # MeV
    'Z': 91187.6,      # MeV
    'Higgs': 125250,   # MeV
}

# Key dimensionless ratios that Standard Model doesn't predict
FUNDAMENTAL_RATIOS = {
    'proton/electron': 1836.15267343,
    'tau/muon': 16.8167,
    'muon/electron': 206.7682830,
    'tau/electron': 3477.23,
    'top/bottom': 41.33,
    'bottom/charm': 3.291,
    'charm/strange': 13.60,
    'strange/down': 20.0,
    'down/up': 2.162,
    'W/Z': 0.8815,
    'Higgs/W': 1.559,
    'Higgs/Z': 1.373,
}


# =============================================================================
# E8 / H4 MATHEMATICAL FRAMEWORK
# =============================================================================

def create_moxness_matrix() -> np.ndarray:
    """Corrected orthogonal Moxness matrix (det=1, rank=8)."""
    matrix = np.array([
        [ 0.2628656, -0.2628656, -0.2628656,  0.2628656,  0.6424204, -0.5330869, -0.1217464,  0.1090828],
        [ 0.2628656, -0.2628656,  0.2628656, -0.2628656,  0.3841082,  0.4628869, -0.4013878, -0.4479857],
        [ 0.2628656,  0.2628656, -0.2628656, -0.2628656,  0.3841082,  0.4628869,  0.4013878,  0.4479857],
        [ 0.2628656,  0.2628656,  0.2628656,  0.2628656,  0.1257960, -0.1043868,  0.6217397, -0.5570685],
        [ 0.4253254, -0.4253254, -0.4253254,  0.4253254, -0.3970376,  0.3294658,  0.0752434, -0.0674169],
        [ 0.4253254, -0.4253254,  0.4253254, -0.4253254, -0.2373919, -0.2860798,  0.2480713,  0.2768704],
        [ 0.4253254,  0.4253254, -0.4253254, -0.4253254, -0.2373919, -0.2860798, -0.2480713, -0.2768704],
        [ 0.4253254,  0.4253254,  0.4253254,  0.4253254, -0.0777462,  0.0645146, -0.3842563,  0.3442873],
    ])
    return matrix


def generate_e8_roots() -> np.ndarray:
    """Generate all 240 roots of the E8 lattice."""
    roots = []

    # Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
    for i in range(8):
        for j in range(i+1, 8):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    root = np.zeros(8)
                    root[i] = si
                    root[j] = sj
                    roots.append(root)

    # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs - 128 roots
    for bits in range(256):
        signs = [1 if (bits >> i) & 1 else -1 for i in range(8)]
        if signs.count(-1) % 2 == 0:
            roots.append(np.array(signs) * 0.5)

    return np.array(roots)


def fold_e8_to_h4(roots: np.ndarray, moxness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold E8 roots to H4 using Moxness matrix.
    Returns left and right H4 projections.
    """
    folded = roots @ moxness.T
    left_h4 = folded[:, :4]   # First 4 components
    right_h4 = folded[:, 4:]  # Last 4 components
    return left_h4, right_h4


def compute_h4_magnitudes(h4_points: np.ndarray) -> np.ndarray:
    """Compute magnitudes of H4 vectors."""
    return np.linalg.norm(h4_points, axis=1)


def generate_phi_cascade(n_levels: int = 12) -> np.ndarray:
    """
    Generate φ-based cascade of values.
    φ^n for n = -6 to +6 gives a natural hierarchy.
    """
    exponents = np.arange(-n_levels//2, n_levels//2 + 1)
    return PHI ** exponents


def generate_600_cell_vertices() -> np.ndarray:
    """Generate all 120 vertices of the 600-cell."""
    vertices = []

    # 8 vertices: (±1, 0, 0, 0) permutations
    for i in range(4):
        for s in [-1, 1]:
            v = np.zeros(4)
            v[i] = s
            vertices.append(v)

    # 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
    for signs in range(16):
        v = np.array([(1 if (signs >> i) & 1 else -1) * 0.5 for i in range(4)])
        vertices.append(v)

    # 96 vertices: even permutations of (±φ/2, ±1/2, ±1/(2φ), 0)
    a, b, c = PHI/2, 0.5, 1/(2*PHI)
    base_perms = [
        [a, b, c, 0], [a, c, 0, b], [a, 0, b, c],
        [b, a, 0, c], [b, c, a, 0], [b, 0, c, a],
        [c, a, b, 0], [c, b, 0, a], [c, 0, a, b],
        [0, a, c, b], [0, b, a, c], [0, c, b, a],
    ]

    for perm in base_perms:
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    if perm[3] == 0:
                        v = np.array([s0*perm[0], s1*perm[1], s2*perm[2], 0])
                        vertices.append(v)

    return np.unique(np.round(np.array(vertices), 10), axis=0)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_eigenvalue_spectrum(matrix: np.ndarray) -> Dict:
    """Analyze eigenvalue spectrum of the Moxness matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

    # Look for φ relationships
    ratios = []
    for i in range(len(eigenvalues)-1):
        if eigenvalues[i+1] > 1e-10:
            ratios.append(eigenvalues[i] / eigenvalues[i+1])

    return {
        'eigenvalues': eigenvalues.tolist(),
        'successive_ratios': ratios,
        'phi_deviations': [abs(r - PHI) for r in ratios],
        'determinant': float(np.linalg.det(matrix)),
    }


def analyze_h4_magnitude_spectrum(magnitudes: np.ndarray) -> Dict:
    """Analyze the spectrum of H4 magnitudes."""
    unique_mags = np.unique(np.round(magnitudes, 8))
    unique_mags = unique_mags[unique_mags > 1e-10]  # Remove zeros
    unique_mags = np.sort(unique_mags)

    # Compute ratios between unique magnitudes
    ratios = []
    for i in range(len(unique_mags)-1):
        ratios.append(unique_mags[i+1] / unique_mags[i])

    return {
        'unique_magnitudes': unique_mags.tolist(),
        'magnitude_ratios': ratios,
        'count_per_magnitude': {str(m): int(np.sum(np.abs(magnitudes - m) < 1e-6))
                                for m in unique_mags},
    }


def find_best_phi_match(target_ratio: float, max_power: int = 20) -> Tuple[str, float]:
    """
    Find the best φ^n approximation to a target ratio.
    Returns the expression and the relative error.
    """
    best_expr = None
    best_error = float('inf')

    # Try φ^n
    for n in range(-max_power, max_power + 1):
        val = PHI ** n
        error = abs(val - target_ratio) / target_ratio
        if error < best_error:
            best_error = error
            best_expr = f"φ^{n}"

    # Try φ^n * sqrt(k) for small k
    for n in range(-max_power//2, max_power//2 + 1):
        for k in [2, 3, 5]:
            val = (PHI ** n) * np.sqrt(k)
            error = abs(val - target_ratio) / target_ratio
            if error < best_error:
                best_error = error
                best_expr = f"φ^{n}·√{k}"

    # Try (φ^n + φ^m) / 2 combinations
    for n in range(-10, 10):
        for m in range(-10, 10):
            val = (PHI**n + PHI**m) / 2
            error = abs(val - target_ratio) / target_ratio
            if error < best_error:
                best_error = error
                best_expr = f"(φ^{n}+φ^{m})/2"

    return best_expr, best_error


def predict_mass_ratios_from_e8(h4_magnitudes: np.ndarray) -> Dict:
    """
    Attempt to match E8/H4 magnitude ratios to particle mass ratios.
    """
    unique_mags = np.unique(np.round(h4_magnitudes, 8))
    unique_mags = unique_mags[unique_mags > 1e-10]
    unique_mags = np.sort(unique_mags)

    # Generate all possible ratios from H4 magnitudes
    h4_ratios = []
    for i, m1 in enumerate(unique_mags):
        for j, m2 in enumerate(unique_mags):
            if i != j and m2 > 0:
                h4_ratios.append(m1 / m2)

    h4_ratios = np.array(h4_ratios)

    # Also include powers of these ratios
    extended_ratios = []
    for r in h4_ratios:
        for power in [1, 2, 3, 4, 0.5, 1/3, 1/4]:
            if r > 0:
                extended_ratios.append(r ** power)

    extended_ratios = np.array(extended_ratios)

    # Match to fundamental ratios
    predictions = {}
    for name, measured in FUNDAMENTAL_RATIOS.items():
        # Find closest H4-derived ratio
        if len(extended_ratios) > 0:
            idx = np.argmin(np.abs(extended_ratios - measured))
            predicted = extended_ratios[idx]
            error = abs(predicted - measured) / measured * 100
        else:
            predicted = 0
            error = 100

        # Also find best φ expression
        phi_expr, phi_error = find_best_phi_match(measured)

        predictions[name] = {
            'measured': measured,
            'h4_predicted': float(predicted),
            'h4_error_pct': float(error),
            'phi_expression': phi_expr,
            'phi_error_pct': float(phi_error * 100),
        }

    return predictions


def generate_phi_mass_hierarchy() -> Dict:
    """
    Generate a φ-based mass hierarchy and compare to measured masses.

    Hypothesis: Masses follow m_n = m_0 * φ^(α*n) for some base mass and exponent.
    """
    # Lepton masses in MeV
    m_e = LEPTON_MASSES['electron']
    m_mu = LEPTON_MASSES['muon']
    m_tau = LEPTON_MASSES['tau']

    # Test: do leptons follow φ hierarchy?
    # If m_mu/m_e = φ^a and m_tau/m_mu = φ^b, what are a and b?

    a_mu_e = np.log(m_mu / m_e) / np.log(PHI)   # ≈ 11.1
    a_tau_mu = np.log(m_tau / m_mu) / np.log(PHI)  # ≈ 5.85
    a_tau_e = np.log(m_tau / m_e) / np.log(PHI)   # ≈ 16.95

    # Idealized: if they were EXACTLY φ powers
    ideal_mu = m_e * PHI ** round(a_mu_e)
    ideal_tau = m_e * PHI ** round(a_tau_e)

    return {
        'lepton_analysis': {
            'electron_MeV': m_e,
            'muon_MeV': m_mu,
            'tau_MeV': m_tau,
            'phi_exponent_mu/e': float(a_mu_e),
            'phi_exponent_tau/mu': float(a_tau_mu),
            'phi_exponent_tau/e': float(a_tau_e),
            'nearest_integer_exponents': {
                'mu/e': int(round(a_mu_e)),
                'tau/e': int(round(a_tau_e)),
            },
            'ideal_phi_predictions': {
                'muon_if_phi^11': float(m_e * PHI**11),
                'tau_if_phi^17': float(m_e * PHI**17),
                'muon_error_pct': float(abs(m_e * PHI**11 - m_mu) / m_mu * 100),
                'tau_error_pct': float(abs(m_e * PHI**17 - m_tau) / m_tau * 100),
            }
        },
        'proton_electron_analysis': {
            'measured_ratio': FUNDAMENTAL_RATIOS['proton/electron'],
            'phi_11': float(PHI**11),
            'phi_12': float(PHI**12),
            '6*phi_9': float(6 * PHI**9),
            'phi_11_error_pct': float(abs(PHI**11 - 1836.15) / 1836.15 * 100),
            '4*phi^10_error_pct': float(abs(4 * PHI**10 - 1836.15) / 1836.15 * 100),
        }
    }


def statistical_significance_test(h4_predictions: Dict, n_random_trials: int = 10000) -> Dict:
    """
    Test statistical significance: are E8/H4 predictions better than random?

    Null hypothesis: Random numbers could achieve same match quality.
    """
    # Calculate mean error of our predictions
    our_errors = [v['h4_error_pct'] for v in h4_predictions.values()]
    our_mean_error = np.mean(our_errors)

    # Monte Carlo: generate random "predictions" and compute errors
    measured_values = [v['measured'] for v in h4_predictions.values()]

    random_mean_errors = []
    for _ in range(n_random_trials):
        # Generate random ratios in similar range
        random_ratios = 10 ** (np.random.uniform(-1, 5, size=100))

        errors = []
        for measured in measured_values:
            # Find closest random ratio
            closest = random_ratios[np.argmin(np.abs(random_ratios - measured))]
            error = abs(closest - measured) / measured * 100
            errors.append(error)

        random_mean_errors.append(np.mean(errors))

    random_mean_errors = np.array(random_mean_errors)

    # P-value: fraction of random trials that did as well or better
    p_value = np.mean(random_mean_errors <= our_mean_error)

    return {
        'our_mean_error_pct': float(our_mean_error),
        'random_mean_error_pct': float(np.mean(random_mean_errors)),
        'random_std_error': float(np.std(random_mean_errors)),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'significant_at_0.01': bool(p_value < 0.01),
        'n_trials': n_random_trials,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("E8 → H4 PARTICLE MASS RATIO ANALYSIS")
    print("=" * 70)
    print()

    # Step 1: Generate E8 and fold to H4
    print("[1/5] Generating E8 roots and Moxness folding...")
    moxness = create_moxness_matrix()
    e8_roots = generate_e8_roots()
    left_h4, right_h4 = fold_e8_to_h4(e8_roots, moxness)

    print(f"      E8 roots: {len(e8_roots)}")
    print(f"      Moxness det: {np.linalg.det(moxness):.6f}")
    print()

    # Step 2: Analyze eigenvalue spectrum
    print("[2/5] Analyzing Moxness eigenvalue spectrum...")
    eigen_analysis = analyze_eigenvalue_spectrum(moxness)
    print(f"      Eigenvalues: {[f'{e:.4f}' for e in eigen_analysis['eigenvalues']]}")
    print(f"      Successive ratios: {[f'{r:.4f}' for r in eigen_analysis['successive_ratios']]}")
    print()

    # Step 3: Analyze H4 magnitude spectrum
    print("[3/5] Analyzing H4 magnitude spectrum...")
    left_mags = compute_h4_magnitudes(left_h4)
    right_mags = compute_h4_magnitudes(right_h4)
    combined_mags = np.concatenate([left_mags, right_mags])

    mag_analysis = analyze_h4_magnitude_spectrum(combined_mags)
    print(f"      Unique magnitudes: {[f'{m:.4f}' for m in mag_analysis['unique_magnitudes'][:10]]}")
    if mag_analysis['magnitude_ratios']:
        print(f"      Magnitude ratios: {[f'{r:.4f}' for r in mag_analysis['magnitude_ratios'][:5]]}")
    print()

    # Step 4: Predict mass ratios
    print("[4/5] Matching to fundamental mass ratios...")
    mass_predictions = predict_mass_ratios_from_e8(combined_mags)

    print()
    print("      " + "-" * 62)
    print(f"      {'Ratio':<20} {'Measured':>12} {'φ-expr':>15} {'φ-err%':>10}")
    print("      " + "-" * 62)

    for name, data in sorted(mass_predictions.items(), key=lambda x: x[1]['phi_error_pct']):
        print(f"      {name:<20} {data['measured']:>12.4f} {data['phi_expression']:>15} {data['phi_error_pct']:>10.2f}")
    print("      " + "-" * 62)
    print()

    # Step 5: φ hierarchy analysis
    print("[5/5] Golden ratio mass hierarchy analysis...")
    phi_hierarchy = generate_phi_mass_hierarchy()

    lepton = phi_hierarchy['lepton_analysis']
    print()
    print("      LEPTON φ-HIERARCHY:")
    print(f"      m_μ/m_e = φ^{lepton['phi_exponent_mu/e']:.2f} (nearest integer: {lepton['nearest_integer_exponents']['mu/e']})")
    print(f"      m_τ/m_e = φ^{lepton['phi_exponent_tau/e']:.2f} (nearest integer: {lepton['nearest_integer_exponents']['tau/e']})")
    print()
    print(f"      If m_μ = m_e × φ^11: {lepton['ideal_phi_predictions']['muon_if_phi^11']:.2f} MeV (error: {lepton['ideal_phi_predictions']['muon_error_pct']:.2f}%)")
    print(f"      If m_τ = m_e × φ^17: {lepton['ideal_phi_predictions']['tau_if_phi^17']:.2f} MeV (error: {lepton['ideal_phi_predictions']['tau_error_pct']:.2f}%)")
    print()

    proton = phi_hierarchy['proton_electron_analysis']
    print("      PROTON/ELECTRON RATIO:")
    print(f"      Measured: {proton['measured_ratio']:.4f}")
    print(f"      φ^11 = {proton['phi_11']:.4f} (error: {proton['phi_11_error_pct']:.2f}%)")
    print(f"      4×φ^10 = {4 * PHI**10:.4f} (error: {proton['4*phi^10_error_pct']:.2f}%)")
    print()

    # Statistical significance
    print("[STAT] Running Monte Carlo significance test...")
    significance = statistical_significance_test(mass_predictions)
    print(f"      Our mean error: {significance['our_mean_error_pct']:.2f}%")
    print(f"      Random baseline: {significance['random_mean_error_pct']:.2f}% ± {significance['random_std_error']:.2f}%")
    print(f"      P-value: {significance['p_value']:.4f}")
    print(f"      Significant at α=0.05: {significance['significant_at_0.05']}")
    print()

    # Summary
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    # Find best φ predictions
    best_matches = sorted(mass_predictions.items(), key=lambda x: x[1]['phi_error_pct'])[:5]

    print()
    print("TOP 5 φ-BASED PREDICTIONS (lowest error):")
    for name, data in best_matches:
        print(f"  • {name}: {data['phi_expression']} → {data['measured']:.4f} ({data['phi_error_pct']:.2f}% error)")

    print()
    print("KEY FINDINGS:")
    print(f"  • Lepton mass ratios follow φ^n pattern with {lepton['ideal_phi_predictions']['muon_error_pct']:.1f}% / {lepton['ideal_phi_predictions']['tau_error_pct']:.1f}% error")
    print(f"  • Statistical significance: p = {significance['p_value']:.4f}")

    if significance['p_value'] < 0.05:
        print("  • ✓ Results are statistically significant (p < 0.05)")
    else:
        print("  • ✗ Results not statistically significant at α=0.05")

    # Save results
    results = {
        'eigenvalue_analysis': eigen_analysis,
        'magnitude_analysis': mag_analysis,
        'mass_predictions': mass_predictions,
        'phi_hierarchy': phi_hierarchy,
        'statistical_significance': significance,
    }

    with open('e8_mass_ratio_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: e8_mass_ratio_results.json")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
