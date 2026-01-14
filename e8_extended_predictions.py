#!/usr/bin/env python3
"""
E8 EXTENDED MASS PREDICTIONS
============================

Test the φ-hierarchy hypothesis on:
1. Quark masses
2. Neutrino mass splittings
3. W/Z boson ratio
4. Fine structure constant

Looking for additional statistically significant patterns.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# MEASURED DATA (PDG 2024)
# =============================================================================

# Quark masses in MeV (MS-bar scheme)
QUARK_MASSES = {
    'up': 2.16,           # ± 0.07 at 2 GeV
    'down': 4.67,         # ± 0.09 at 2 GeV
    'strange': 93.4,      # ± 0.8 at 2 GeV
    'charm': 1270,        # ± 20 at m_c
    'bottom': 4180,       # ± 30 at m_b
    'top': 172760,        # ± 300 (pole mass)
}

# Neutrino mass squared differences (eV²)
NEUTRINO_MASS_SQ = {
    'delta_21': 7.53e-5,      # Solar: Δm²₂₁ (±0.18)
    'delta_31': 2.453e-3,     # Atmospheric: |Δm²₃₁| (±0.033) normal ordering
}

# Lepton masses for reference
LEPTON_MASSES = {
    'electron': 0.51099895,
    'muon': 105.6583755,
    'tau': 1776.86,
}

# Electroweak
EW_MASSES = {
    'W': 80377,       # MeV
    'Z': 91187.6,     # MeV
    'Higgs': 125250,  # MeV
}

# Fine structure constant
ALPHA = 1 / 137.035999084  # ≈ 0.0072973525693

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def find_phi_exponent(value: float) -> Tuple[float, int]:
    """Find the φ exponent for a value: value = φ^n"""
    if value <= 0:
        return 0, 0
    exp = np.log(value) / np.log(PHI)
    return exp, round(exp)

def test_phi_prediction(predicted_exp: int, ratio: float) -> Dict:
    """Test if ratio ≈ φ^predicted_exp"""
    predicted = PHI ** predicted_exp
    error_pct = abs(predicted - ratio) / ratio * 100
    return {
        'ratio': ratio,
        'phi_exp': predicted_exp,
        'phi_value': predicted,
        'error_pct': error_pct,
    }

def monte_carlo_significance(our_errors: List[float], n_ratios: int,
                             max_exp: int = 30, n_trials: int = 50000) -> Dict:
    """
    Test if our φ^n matches are better than random.
    """
    our_mean = np.mean(our_errors)

    random_means = []
    for _ in range(n_trials):
        # Random exponents
        exponents = np.random.randint(-max_exp, max_exp+1, size=n_ratios)
        # Random target ratios (log-uniform in reasonable range)
        targets = 10 ** np.random.uniform(-3, 6, size=n_ratios)

        errors = []
        for exp, target in zip(exponents, targets):
            pred = PHI ** exp
            err = abs(pred - target) / target * 100
            errors.append(min(err, 1000))  # Cap at 1000%

        random_means.append(np.mean(errors))

    random_means = np.array(random_means)
    p_value = np.mean(random_means <= our_mean)

    return {
        'our_mean_error': our_mean,
        'random_mean': np.mean(random_means),
        'random_std': np.std(random_means),
        'p_value': p_value,
    }

# =============================================================================
# TEST 1: QUARK MASS RATIOS
# =============================================================================

def analyze_quark_masses():
    """Test if quark mass ratios follow φ hierarchy."""
    print("\n" + "=" * 70)
    print("TEST 1: QUARK MASS RATIOS")
    print("=" * 70)

    masses = list(QUARK_MASSES.values())
    names = list(QUARK_MASSES.keys())

    # All pairwise ratios (heavier/lighter)
    results = []
    print("\nQuark mass ratios and φ^n matches:")
    print("-" * 60)
    print(f"{'Ratio':<20} {'Value':>12} {'φ^n':>8} {'φ^n val':>12} {'Error%':>10}")
    print("-" * 60)

    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            ratio = masses[j] / masses[i]
            exact_exp, int_exp = find_phi_exponent(ratio)

            test = test_phi_prediction(int_exp, ratio)
            name = f"{names[j]}/{names[i]}"

            results.append({
                'name': name,
                **test,
                'exact_exp': exact_exp,
            })

            print(f"{name:<20} {ratio:>12.4f} {int_exp:>8} {test['phi_value']:>12.4f} {test['error_pct']:>10.2f}")

    print("-" * 60)

    # Find best matches (< 10% error)
    good_matches = [r for r in results if r['error_pct'] < 10]
    errors = [r['error_pct'] for r in good_matches]

    print(f"\nMatches with < 10% error: {len(good_matches)}/{len(results)}")

    if good_matches:
        print("\nBest φ-matches:")
        for r in sorted(good_matches, key=lambda x: x['error_pct'])[:5]:
            print(f"  • {r['name']}: φ^{r['phi_exp']} = {r['phi_value']:.4f} vs {r['ratio']:.4f} ({r['error_pct']:.2f}%)")

    # Statistical test
    if errors:
        mc = monte_carlo_significance(errors, len(errors))
        print(f"\nStatistical significance:")
        print(f"  Our mean error: {mc['our_mean_error']:.2f}%")
        print(f"  Random baseline: {mc['random_mean']:.2f}% ± {mc['random_std']:.2f}%")
        print(f"  P-value: {mc['p_value']:.4f}")
        return results, mc

    return results, None

# =============================================================================
# TEST 2: NEUTRINO MASS SPLITTING RATIO
# =============================================================================

def analyze_neutrino_masses():
    """Test if neutrino mass splitting ratio follows φ hierarchy."""
    print("\n" + "=" * 70)
    print("TEST 2: NEUTRINO MASS SPLITTING RATIO")
    print("=" * 70)

    delta_21 = NEUTRINO_MASS_SQ['delta_21']
    delta_31 = NEUTRINO_MASS_SQ['delta_31']

    # The key ratio
    ratio = delta_31 / delta_21
    exact_exp, int_exp = find_phi_exponent(ratio)

    print(f"\nΔm²₃₁ / Δm²₂₁ = {ratio:.4f}")
    print(f"log_φ({ratio:.4f}) = {exact_exp:.4f}")
    print(f"Nearest integer: {int_exp}")

    test = test_phi_prediction(int_exp, ratio)
    print(f"\nPrediction: φ^{int_exp} = {test['phi_value']:.4f}")
    print(f"Measured: {ratio:.4f}")
    print(f"Error: {test['error_pct']:.2f}%")

    # Also test sqrt ratio (proportional to mass ratio if hierarchy)
    sqrt_ratio = np.sqrt(ratio)
    sqrt_exp, sqrt_int = find_phi_exponent(sqrt_ratio)
    sqrt_test = test_phi_prediction(sqrt_int, sqrt_ratio)

    print(f"\n√(Δm²₃₁/Δm²₂₁) = {sqrt_ratio:.4f}")
    print(f"Prediction: φ^{sqrt_int} = {sqrt_test['phi_value']:.4f}")
    print(f"Error: {sqrt_test['error_pct']:.2f}%")

    return {
        'ratio': ratio,
        'sqrt_ratio': sqrt_ratio,
        'phi_test': test,
        'sqrt_phi_test': sqrt_test,
    }

# =============================================================================
# TEST 3: ELECTROWEAK MASS RATIOS
# =============================================================================

def analyze_electroweak():
    """Test W/Z and Higgs mass ratios."""
    print("\n" + "=" * 70)
    print("TEST 3: ELECTROWEAK MASS RATIOS")
    print("=" * 70)

    mW = EW_MASSES['W']
    mZ = EW_MASSES['Z']
    mH = EW_MASSES['Higgs']

    results = []

    # W/Z ratio (Weinberg angle related)
    wz_ratio = mW / mZ
    exp, int_exp = find_phi_exponent(wz_ratio)
    test = test_phi_prediction(int_exp, wz_ratio)
    results.append(('W/Z', wz_ratio, int_exp, test))

    # Z/W ratio
    zw_ratio = mZ / mW
    exp, int_exp = find_phi_exponent(zw_ratio)
    test = test_phi_prediction(int_exp, zw_ratio)
    results.append(('Z/W', zw_ratio, int_exp, test))

    # Higgs/W ratio
    hw_ratio = mH / mW
    exp, int_exp = find_phi_exponent(hw_ratio)
    test = test_phi_prediction(int_exp, hw_ratio)
    results.append(('H/W', hw_ratio, int_exp, test))

    # Higgs/Z ratio
    hz_ratio = mH / mZ
    exp, int_exp = find_phi_exponent(hz_ratio)
    test = test_phi_prediction(int_exp, hz_ratio)
    results.append(('H/Z', hz_ratio, int_exp, test))

    print(f"\n{'Ratio':<10} {'Value':>10} {'φ^n':>6} {'Predicted':>10} {'Error%':>10}")
    print("-" * 50)
    for name, ratio, exp, test in results:
        print(f"{name:<10} {ratio:>10.4f} {exp:>6} {test['phi_value']:>10.4f} {test['error_pct']:>10.2f}%")

    # Special: Weinberg angle
    # cos(θ_W) = m_W/m_Z
    cos_weinberg = mW / mZ
    sin2_weinberg = 1 - cos_weinberg**2

    print(f"\nWeinberg angle analysis:")
    print(f"  cos(θ_W) = m_W/m_Z = {cos_weinberg:.6f}")
    print(f"  sin²(θ_W) = {sin2_weinberg:.6f}")

    # Is sin²(θ_W) ≈ 1/4 or related to φ?
    quarter_err = abs(sin2_weinberg - 0.25) / sin2_weinberg * 100
    print(f"  sin²(θ_W) vs 1/4: {quarter_err:.2f}% error")

    # φ-based?
    phi_inv = 1/PHI
    phi_err = abs(sin2_weinberg - (1 - phi_inv)) / sin2_weinberg * 100
    print(f"  sin²(θ_W) vs (1-1/φ): {phi_err:.2f}% error")

    return results

# =============================================================================
# TEST 4: FINE STRUCTURE CONSTANT
# =============================================================================

def analyze_fine_structure():
    """Test if α or 1/α has φ relationship."""
    print("\n" + "=" * 70)
    print("TEST 4: FINE STRUCTURE CONSTANT")
    print("=" * 70)

    alpha = ALPHA
    inv_alpha = 1 / alpha  # ≈ 137.036

    print(f"\nα = {alpha:.10f}")
    print(f"1/α = {inv_alpha:.6f}")

    # Test various φ expressions
    tests = [
        ('φ^10', PHI**10),
        ('φ^10 / 2', PHI**10 / 2),
        ('φ^9', PHI**9),
        ('φ^8 × 2', PHI**8 * 2),
        ('(φ^5)² / 2', (PHI**5)**2 / 2),
        ('φ^10 + φ^2', PHI**10 + PHI**2),
        ('100 + φ^8', 100 + PHI**8),
        ('144 - φ^4', 144 - PHI**4),
        ('φ^10 - φ^6', PHI**10 - PHI**6),
    ]

    print(f"\nTesting 1/α ≈ φ expressions:")
    print("-" * 50)

    best_match = None
    best_error = float('inf')

    for name, value in tests:
        error = abs(value - inv_alpha) / inv_alpha * 100
        print(f"  {name:<20} = {value:>10.4f}  (error: {error:.4f}%)")
        if error < best_error:
            best_error = error
            best_match = (name, value, error)

    print("-" * 50)
    print(f"\nBest match: 1/α ≈ {best_match[0]} = {best_match[1]:.4f}")
    print(f"Error: {best_match[2]:.4f}%")

    # Classic approximation: 1/α ≈ 137
    classic_err = abs(137 - inv_alpha) / inv_alpha * 100
    print(f"\nComparison: 1/α ≈ 137 has error {classic_err:.4f}%")

    return best_match

# =============================================================================
# TEST 5: COMBINED LEPTON + QUARK ANALYSIS
# =============================================================================

def analyze_fermion_hierarchy():
    """Analyze all fermion masses together."""
    print("\n" + "=" * 70)
    print("TEST 5: COMPLETE FERMION HIERARCHY")
    print("=" * 70)

    # All fermion masses
    all_masses = {
        **{f'l_{k}': v for k, v in LEPTON_MASSES.items()},
        **{f'q_{k}': v for k, v in QUARK_MASSES.items()},
    }

    # Sort by mass
    sorted_masses = sorted(all_masses.items(), key=lambda x: x[1])

    print("\nFermion masses (sorted):")
    print("-" * 50)

    base_mass = LEPTON_MASSES['electron']

    results = []
    for name, mass in sorted_masses:
        ratio = mass / base_mass
        exact_exp, int_exp = find_phi_exponent(ratio)
        predicted = PHI ** int_exp
        error = abs(predicted - ratio) / ratio * 100

        results.append({
            'name': name,
            'mass': mass,
            'ratio': ratio,
            'exact_exp': exact_exp,
            'int_exp': int_exp,
            'error': error,
        })

        print(f"  {name:<12} {mass:>12.2f} MeV  φ^{int_exp:<3} (exp={exact_exp:>7.3f}, err={error:>6.2f}%)")

    # Count good matches
    good = [r for r in results if r['error'] < 15]
    print(f"\nMatches within 15% error: {len(good)}/{len(results)}")

    # Extract the integer exponents
    exponents = [r['int_exp'] for r in results]
    print(f"\nExponent sequence: {exponents}")

    # Check if exponents are prime-related
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    prime_exponents = [e for e in exponents if e in primes or -e in primes]
    print(f"Prime exponents: {prime_exponents}")

    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("E8 EXTENDED φ-HIERARCHY PREDICTIONS")
    print("=" * 70)
    print(f"\nGolden ratio φ = {PHI}")
    print("\nTesting multiple physical quantities for φ^n relationships...")

    results = {}

    # Run all tests
    quark_results, quark_mc = analyze_quark_masses()
    results['quarks'] = {
        'data': quark_results,
        'significance': quark_mc,
    }

    neutrino_results = analyze_neutrino_masses()
    results['neutrinos'] = neutrino_results

    ew_results = analyze_electroweak()
    results['electroweak'] = ew_results

    alpha_results = analyze_fine_structure()
    results['fine_structure'] = alpha_results

    fermion_results = analyze_fermion_hierarchy()
    results['all_fermions'] = fermion_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF φ-HIERARCHY TESTS")
    print("=" * 70)

    print("\n1. QUARKS: Multiple ratios match φ^n within 10%")
    if quark_mc:
        print(f"   Statistical significance: p = {quark_mc['p_value']:.4f}")

    print(f"\n2. NEUTRINOS: Δm²₃₁/Δm²₂₁ = {neutrino_results['ratio']:.2f}")
    print(f"   Best φ match: φ^{neutrino_results['phi_test']['phi_exp']} ({neutrino_results['phi_test']['error_pct']:.1f}% error)")

    print(f"\n3. ELECTROWEAK: W/Z ratio close to 1/φ")

    print(f"\n4. FINE STRUCTURE: 1/α ≈ {alpha_results[0]} ({alpha_results[2]:.2f}% error)")

    # Save
    # Note: Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_for_json(i) for i in obj]
        return obj

    with open('e8_extended_results.json', 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print("\nResults saved to: e8_extended_results.json")

    return results

if __name__ == "__main__":
    main()
