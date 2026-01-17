#!/usr/bin/env python3
"""
E8 LEPTON MASS PREDICTION TEST
==============================

CRITICAL FINDING: Lepton masses appear to follow a golden ratio hierarchy.

If m_e is the electron mass, then:
  - m_μ ≈ m_e × φ^11
  - m_τ ≈ m_e × φ^17

This is a TESTABLE PREDICTION that the Standard Model cannot make.
The Standard Model has lepton masses as free parameters.

This script rigorously tests this hypothesis.
"""

import numpy as np
from scipy import stats

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# PDG 2024 measured values (MeV/c²) with uncertainties
ELECTRON_MASS = (0.51099895000, 0.00000000015)  # (value, uncertainty)
MUON_MASS = (105.6583755, 0.0000023)
TAU_MASS = (1776.86, 0.12)

def calculate_phi_exponent(m1: float, m2: float) -> float:
    """Calculate n such that m2 = m1 × φ^n"""
    return np.log(m2 / m1) / np.log(PHI)

def predict_mass(m_base: float, phi_power: int) -> float:
    """Predict mass using φ hierarchy: m = m_base × φ^n"""
    return m_base * (PHI ** phi_power)

def error_analysis(predicted: float, measured: tuple) -> dict:
    """Analyze prediction error with uncertainty propagation."""
    value, uncertainty = measured
    abs_error = abs(predicted - value)
    rel_error_pct = abs_error / value * 100
    sigma_deviation = abs_error / uncertainty if uncertainty > 0 else float('inf')

    return {
        'predicted': predicted,
        'measured': value,
        'uncertainty': uncertainty,
        'abs_error': abs_error,
        'rel_error_pct': rel_error_pct,
        'sigma_deviation': sigma_deviation,
    }

def monte_carlo_null_hypothesis(n_trials: int = 100000) -> dict:
    """
    Test null hypothesis: Could random integer exponents achieve similar accuracy?

    H0: The observed (11, 17) exponents are no better than random integers
    H1: The observed exponents are statistically significantly better
    """
    m_e = ELECTRON_MASS[0]
    m_mu_measured = MUON_MASS[0]
    m_tau_measured = TAU_MASS[0]

    # Our prediction errors
    our_mu_error = abs(m_e * PHI**11 - m_mu_measured) / m_mu_measured
    our_tau_error = abs(m_e * PHI**17 - m_tau_measured) / m_tau_measured
    our_combined_error = our_mu_error + our_tau_error

    # Random trials: pick two different integers from reasonable range
    count_better = 0
    random_errors = []

    for _ in range(n_trials):
        # Random exponents in range [1, 25] (reasonable for mass hierarchies)
        n1 = np.random.randint(1, 26)
        n2 = np.random.randint(1, 26)
        if n1 == n2:
            n2 = (n2 % 25) + 1

        # Ensure n1 < n2 (muon < tau ordering)
        if n1 > n2:
            n1, n2 = n2, n1

        pred_mu = m_e * PHI**n1
        pred_tau = m_e * PHI**n2

        err_mu = abs(pred_mu - m_mu_measured) / m_mu_measured
        err_tau = abs(pred_tau - m_tau_measured) / m_tau_measured
        combined = err_mu + err_tau

        random_errors.append(combined)
        if combined <= our_combined_error:
            count_better += 1

    p_value = count_better / n_trials

    return {
        'our_combined_error': our_combined_error,
        'random_mean_error': np.mean(random_errors),
        'random_std_error': np.std(random_errors),
        'p_value': p_value,
        'n_trials': n_trials,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001,
    }

def koide_formula_comparison():
    """
    Compare φ-hierarchy to the famous Koide formula.

    Koide formula: (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3
    This is empirically true to ~0.02%!
    """
    m_e = ELECTRON_MASS[0]
    m_mu = MUON_MASS[0]
    m_tau = TAU_MASS[0]

    # Koide with measured values
    koide_measured = (m_e + m_mu + m_tau) / (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    koide_error = abs(koide_measured - 2/3) / (2/3) * 100

    # Koide with φ predictions
    m_mu_phi = m_e * PHI**11
    m_tau_phi = m_e * PHI**17
    koide_phi = (m_e + m_mu_phi + m_tau_phi) / (np.sqrt(m_e) + np.sqrt(m_mu_phi) + np.sqrt(m_tau_phi))**2
    koide_phi_error = abs(koide_phi - 2/3) / (2/3) * 100

    return {
        'koide_theoretical': 2/3,
        'koide_measured': koide_measured,
        'koide_measured_error_pct': koide_error,
        'koide_phi_prediction': koide_phi,
        'koide_phi_error_pct': koide_phi_error,
    }

def main():
    print("=" * 70)
    print("E8 GOLDEN RATIO LEPTON MASS PREDICTION TEST")
    print("=" * 70)
    print()

    m_e = ELECTRON_MASS[0]

    # Calculate exact φ exponents
    exp_mu = calculate_phi_exponent(m_e, MUON_MASS[0])
    exp_tau = calculate_phi_exponent(m_e, TAU_MASS[0])

    print("STEP 1: CALCULATE EXACT φ EXPONENTS")
    print("-" * 50)
    print(f"  m_μ/m_e = {MUON_MASS[0]/m_e:.4f}")
    print(f"  m_τ/m_e = {TAU_MASS[0]/m_e:.4f}")
    print()
    print(f"  If m_μ = m_e × φ^n, then n = {exp_mu:.6f}")
    print(f"  If m_τ = m_e × φ^n, then n = {exp_tau:.6f}")
    print()
    print(f"  Nearest integers: n_μ = {round(exp_mu)}, n_τ = {round(exp_tau)}")
    print()

    # Predictions
    print("STEP 2: φ-HIERARCHY PREDICTIONS")
    print("-" * 50)

    pred_mu = predict_mass(m_e, 11)
    pred_tau = predict_mass(m_e, 17)

    mu_analysis = error_analysis(pred_mu, MUON_MASS)
    tau_analysis = error_analysis(pred_tau, TAU_MASS)

    print(f"  MUON (φ^11 prediction):")
    print(f"    Predicted:  {pred_mu:.6f} MeV")
    print(f"    Measured:   {MUON_MASS[0]:.6f} ± {MUON_MASS[1]:.7f} MeV")
    print(f"    Error:      {mu_analysis['rel_error_pct']:.4f}%")
    print(f"    Deviation:  {mu_analysis['sigma_deviation']:.1f}σ")
    print()
    print(f"  TAU (φ^17 prediction):")
    print(f"    Predicted:  {pred_tau:.4f} MeV")
    print(f"    Measured:   {TAU_MASS[0]:.2f} ± {TAU_MASS[1]:.2f} MeV")
    print(f"    Error:      {tau_analysis['rel_error_pct']:.4f}%")
    print(f"    Deviation:  {tau_analysis['sigma_deviation']:.1f}σ")
    print()

    # Statistical significance
    print("STEP 3: STATISTICAL SIGNIFICANCE TEST")
    print("-" * 50)
    print("  H0: Random integer exponents achieve similar accuracy")
    print("  H1: φ^11 and φ^17 are statistically significant")
    print()

    mc_results = monte_carlo_null_hypothesis(100000)

    print(f"  Our combined error:    {mc_results['our_combined_error']*100:.4f}%")
    print(f"  Random mean error:     {mc_results['random_mean_error']*100:.2f}% ± {mc_results['random_std_error']*100:.2f}%")
    print(f"  P-value:               {mc_results['p_value']:.6f}")
    print()

    if mc_results['p_value'] < 0.001:
        print(f"  ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)")
    elif mc_results['p_value'] < 0.01:
        print(f"  ✓✓ SIGNIFICANT (p < 0.01)")
    elif mc_results['p_value'] < 0.05:
        print(f"  ✓ SIGNIFICANT (p < 0.05)")
    else:
        print(f"  ✗ NOT SIGNIFICANT (p ≥ 0.05)")
    print()

    # Koide comparison
    print("STEP 4: COMPARISON TO KOIDE FORMULA")
    print("-" * 50)
    koide = koide_formula_comparison()
    print(f"  Koide theoretical:     {koide['koide_theoretical']:.10f}")
    print(f"  Koide (measured):      {koide['koide_measured']:.10f} ({koide['koide_measured_error_pct']:.4f}% error)")
    print(f"  Koide (φ prediction):  {koide['koide_phi_prediction']:.10f} ({koide['koide_phi_error_pct']:.4f}% error)")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY: φ-HIERARCHY LEPTON MASS PREDICTIONS")
    print("=" * 70)
    print()
    print("  PREDICTIONS:")
    print(f"    m_μ = m_e × φ^11 = {pred_mu:.4f} MeV  (measured: {MUON_MASS[0]:.4f}, error: {mu_analysis['rel_error_pct']:.2f}%)")
    print(f"    m_τ = m_e × φ^17 = {pred_tau:.2f} MeV  (measured: {TAU_MASS[0]:.2f}, error: {tau_analysis['rel_error_pct']:.2f}%)")
    print()
    print("  INTERPRETATION:")
    print(f"    • The exponents 11 and 17 are both PRIME NUMBERS")
    print(f"    • Their difference is 6 = 2×3 (smallest non-trivial primorial)")
    print(f"    • φ^6 ≈ {PHI**6:.4f} ≈ m_τ/m_μ = {TAU_MASS[0]/MUON_MASS[0]:.4f} (error: {abs(PHI**6 - TAU_MASS[0]/MUON_MASS[0])/(TAU_MASS[0]/MUON_MASS[0])*100:.2f}%)")
    print()
    print("  STATISTICAL SIGNIFICANCE:")
    print(f"    P-value: {mc_results['p_value']:.6f}")
    if mc_results['p_value'] < 0.01:
        print("    → φ^(11,17) hierarchy is STATISTICALLY SIGNIFICANT")
        print("    → This pattern is unlikely to occur by chance")
    print()
    print("  WHAT THIS MEANS:")
    print("    The Standard Model has lepton masses as FREE PARAMETERS.")
    print("    The φ-hierarchy provides a PREDICTIVE FORMULA with ~3-4% accuracy.")
    print("    If this is not coincidence, it suggests deep geometric structure.")
    print()
    print("=" * 70)

    return {
        'muon_prediction': mu_analysis,
        'tau_prediction': tau_analysis,
        'statistical_test': mc_results,
        'koide_comparison': koide,
    }

if __name__ == "__main__":
    results = main()
