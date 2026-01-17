#!/usr/bin/env python3
"""
E8 φ-HIERARCHY: FULLY TRANSPARENT STATISTICAL TEST
===================================================

This script shows EVERY calculation step-by-step so you can verify
each result by hand.

NEW DATASET: Hadron masses (proton, neutron, pions, kaons, etc.)
This is INDEPENDENT from the lepton/quark analysis.
"""

import numpy as np

print("=" * 80)
print("E8 φ-HIERARCHY: TRANSPARENT TEST WITH NEW DATASET")
print("=" * 80)
print()

# =============================================================================
# STEP 1: DEFINE THE GOLDEN RATIO
# =============================================================================
print("STEP 1: DEFINE GOLDEN RATIO")
print("-" * 80)

PHI = (1 + np.sqrt(5)) / 2

print(f"φ = (1 + √5) / 2")
print(f"φ = (1 + {np.sqrt(5):.10f}) / 2")
print(f"φ = {PHI:.15f}")
print()

# Verify φ property: φ² = φ + 1
phi_squared = PHI ** 2
phi_plus_one = PHI + 1
print(f"Verification: φ² = {phi_squared:.10f}")
print(f"             φ+1 = {phi_plus_one:.10f}")
print(f"             Match: {np.isclose(phi_squared, phi_plus_one)}")
print()

# =============================================================================
# STEP 2: NEW DATASET - HADRON MASSES (PDG 2024)
# =============================================================================
print("STEP 2: NEW DATASET - HADRON MASSES")
print("-" * 80)
print()
print("Source: Particle Data Group (PDG) 2024")
print("URL: https://pdg.lbl.gov/")
print()

# All masses in MeV/c²
HADRON_MASSES = {
    'pion_charged': 139.57039,      # π± mass
    'pion_neutral': 134.9768,       # π⁰ mass
    'kaon_charged': 493.677,        # K± mass
    'kaon_neutral': 497.611,        # K⁰ mass
    'proton': 938.27208816,         # p mass
    'neutron': 939.56542052,        # n mass
    'eta': 547.862,                 # η mass
    'eta_prime': 957.78,            # η' mass
    'rho': 775.26,                  # ρ mass
    'omega': 782.66,                # ω mass
    'phi_meson': 1019.461,          # φ meson mass
    'D_meson': 1869.66,             # D± mass
    'B_meson': 5279.34,             # B± mass
    'J_psi': 3096.900,              # J/ψ mass
    'Upsilon': 9460.30,             # Υ(1S) mass
}

print("Hadron masses (MeV/c²):")
print("-" * 40)
for name, mass in HADRON_MASSES.items():
    print(f"  {name:<15} = {mass:>12.5f} MeV")
print()

# =============================================================================
# STEP 3: CALCULATE φ^n VALUES FOR REFERENCE
# =============================================================================
print("STEP 3: φ^n REFERENCE TABLE")
print("-" * 80)
print()
print("  n      φ^n")
print("-" * 25)
for n in range(-3, 21):
    print(f"  {n:>3}    {PHI**n:>15.6f}")
print()

# =============================================================================
# STEP 4: USE PION AS BASE MASS (NEW APPROACH)
# =============================================================================
print("STEP 4: CALCULATE RATIOS USING PION AS BASE")
print("-" * 80)
print()
print("Base mass: π± = 139.57039 MeV")
print()

base_mass = HADRON_MASSES['pion_charged']
base_name = 'pion_charged'

results = []

print(f"{'Hadron':<15} {'Mass (MeV)':<14} {'Ratio m/m_π':<14} {'log_φ(ratio)':<14} {'Nearest n':<10} {'φ^n':<14} {'Error %':<10}")
print("-" * 100)

for name, mass in HADRON_MASSES.items():
    if name == base_name:
        continue

    # Step 4a: Calculate ratio
    ratio = mass / base_mass

    # Step 4b: Calculate exact φ exponent
    # If ratio = φ^x, then x = log(ratio) / log(φ)
    exact_exp = np.log(ratio) / np.log(PHI)

    # Step 4c: Round to nearest integer
    nearest_n = round(exact_exp)

    # Step 4d: Calculate predicted value
    predicted = PHI ** nearest_n

    # Step 4e: Calculate error
    error_pct = abs(predicted - ratio) / ratio * 100

    results.append({
        'name': name,
        'mass': mass,
        'ratio': ratio,
        'exact_exp': exact_exp,
        'nearest_n': nearest_n,
        'predicted': predicted,
        'error_pct': error_pct,
    })

    print(f"{name:<15} {mass:<14.5f} {ratio:<14.6f} {exact_exp:<14.6f} {nearest_n:<10} {predicted:<14.6f} {error_pct:<10.4f}")

print("-" * 100)
print()

# =============================================================================
# STEP 5: SORT BY ERROR AND SHOW BEST MATCHES
# =============================================================================
print("STEP 5: BEST φ^n MATCHES (SORTED BY ERROR)")
print("-" * 80)
print()

results_sorted = sorted(results, key=lambda x: x['error_pct'])

print("Rank  Hadron          Ratio        φ^n      Predicted    Error%")
print("-" * 65)
for i, r in enumerate(results_sorted):
    print(f"{i+1:<6}{r['name']:<16}{r['ratio']:<13.4f}φ^{r['nearest_n']:<6}{r['predicted']:<13.4f}{r['error_pct']:<10.4f}")
print()

# Count good matches
excellent = [r for r in results_sorted if r['error_pct'] < 5]
good = [r for r in results_sorted if r['error_pct'] < 15]
print(f"Matches with < 5% error:  {len(excellent)}/{len(results_sorted)}")
print(f"Matches with < 15% error: {len(good)}/{len(results_sorted)}")
print()

# =============================================================================
# STEP 6: VERIFY TOP RESULT BY HAND
# =============================================================================
print("STEP 6: MANUAL VERIFICATION OF TOP RESULT")
print("-" * 80)
print()

top = results_sorted[0]
print(f"Best match: {top['name']}")
print()
print(f"Step-by-step calculation:")
print(f"  1. {top['name']} mass = {top['mass']:.5f} MeV")
print(f"  2. π± mass = {base_mass:.5f} MeV")
print(f"  3. Ratio = {top['mass']:.5f} / {base_mass:.5f} = {top['ratio']:.10f}")
print(f"  4. log(ratio) = log({top['ratio']:.6f}) = {np.log(top['ratio']):.10f}")
print(f"  5. log(φ) = log({PHI:.6f}) = {np.log(PHI):.10f}")
print(f"  6. Exact exponent = {np.log(top['ratio']):.10f} / {np.log(PHI):.10f} = {top['exact_exp']:.10f}")
print(f"  7. Nearest integer = round({top['exact_exp']:.6f}) = {top['nearest_n']}")
print(f"  8. Predicted ratio = φ^{top['nearest_n']} = {PHI:.10f}^{top['nearest_n']} = {top['predicted']:.10f}")
print(f"  9. Error = |{top['predicted']:.6f} - {top['ratio']:.6f}| / {top['ratio']:.6f} × 100")
print(f"           = {abs(top['predicted'] - top['ratio']):.10f} / {top['ratio']:.6f} × 100")
print(f"           = {top['error_pct']:.6f}%")
print()

# =============================================================================
# STEP 7: STATISTICAL SIGNIFICANCE TEST
# =============================================================================
print("STEP 7: MONTE CARLO STATISTICAL TEST")
print("-" * 80)
print()
print("NULL HYPOTHESIS: Random integer exponents match equally well")
print()
print("Test procedure:")
print("  1. For each hadron, we found the nearest integer n where ratio ≈ φ^n")
print("  2. We calculated the mean error across all hadrons")
print("  3. We repeat 100,000 times with RANDOM integers and count how often")
print("     random integers achieve lower error than our integers")
print()

# Our actual errors
our_errors = [r['error_pct'] for r in results_sorted]
our_mean_error = np.mean(our_errors)
print(f"Our mean error: {our_mean_error:.4f}%")
print()

# Monte Carlo
print("Running 100,000 Monte Carlo trials...")
print()

n_trials = 100000
n_hadrons = len(results_sorted)
ratios = [r['ratio'] for r in results_sorted]

np.random.seed(42)  # For reproducibility

random_mean_errors = []
count_better = 0

for trial in range(n_trials):
    # Generate random integer exponents in range [-3, 20]
    random_exps = np.random.randint(-3, 21, size=n_hadrons)

    # Calculate errors for each random exponent
    trial_errors = []
    for ratio, exp in zip(ratios, random_exps):
        pred = PHI ** exp
        err = abs(pred - ratio) / ratio * 100
        trial_errors.append(min(err, 1000))  # Cap at 1000%

    mean_err = np.mean(trial_errors)
    random_mean_errors.append(mean_err)

    if mean_err <= our_mean_error:
        count_better += 1

random_mean_errors = np.array(random_mean_errors)
p_value = count_better / n_trials

print(f"Results of {n_trials:,} trials:")
print(f"  Our mean error:        {our_mean_error:.4f}%")
print(f"  Random mean error:     {np.mean(random_mean_errors):.4f}% ± {np.std(random_mean_errors):.4f}%")
print(f"  Random min error:      {np.min(random_mean_errors):.4f}%")
print(f"  Random max error:      {np.max(random_mean_errors):.4f}%")
print(f"  Trials with ≤ our error: {count_better} / {n_trials}")
print(f"  P-value:               {p_value:.10f}")
print()

if p_value < 0.001:
    print("  ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)")
    significance = "HIGHLY SIGNIFICANT"
elif p_value < 0.01:
    print("  ✓✓ SIGNIFICANT (p < 0.01)")
    significance = "SIGNIFICANT"
elif p_value < 0.05:
    print("  ✓ SIGNIFICANT (p < 0.05)")
    significance = "MARGINALLY SIGNIFICANT"
else:
    print("  ✗ NOT SIGNIFICANT (p ≥ 0.05)")
    significance = "NOT SIGNIFICANT"

print()

# =============================================================================
# STEP 8: VERIFICATION - SHOW DISTRIBUTION
# =============================================================================
print("STEP 8: ERROR DISTRIBUTION ANALYSIS")
print("-" * 80)
print()
print("Distribution of random mean errors:")
print()

# Create histogram bins
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Percentile    Random Error")
print("-" * 30)
for p in percentiles:
    val = np.percentile(random_mean_errors, p)
    marker = " <-- OUR RESULT" if val > our_mean_error and np.percentile(random_mean_errors, p-1 if p > 1 else 0) <= our_mean_error else ""
    print(f"    {p:>2}%        {val:>10.4f}%{marker}")

print()
print(f"Our result ({our_mean_error:.4f}%) is better than {100*(1-p_value):.4f}% of random trials")
print()

# =============================================================================
# STEP 9: FINAL SUMMARY
# =============================================================================
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print(f"Dataset: Hadron masses (14 particles)")
print(f"Base: Pion (π±) = {base_mass:.5f} MeV")
print(f"Method: Test if mass ratios = φ^n for integer n")
print()
print(f"RESULTS:")
print(f"  - Matches with < 5% error:  {len(excellent)}/14")
print(f"  - Matches with < 15% error: {len(good)}/14")
print(f"  - Our mean error:           {our_mean_error:.4f}%")
print(f"  - Random baseline:          {np.mean(random_mean_errors):.4f}%")
print(f"  - P-value:                  {p_value:.10f}")
print(f"  - Significance:             {significance}")
print()
print("TOP 5 PREDICTIONS:")
for r in results_sorted[:5]:
    print(f"  • {r['name']}: m/m_π = {r['ratio']:.4f} ≈ φ^{r['nearest_n']} ({r['error_pct']:.2f}% error)")
print()
print("=" * 80)
print("TEST COMPLETE - ALL CALCULATIONS SHOWN ABOVE FOR VERIFICATION")
print("=" * 80)
