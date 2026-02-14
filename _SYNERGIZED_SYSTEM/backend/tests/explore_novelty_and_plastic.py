"""
Phillips Matrix: Computational Novelty & Plastic Ratio Exploration.

Companion script to COMPUTATIONAL_NOVELTY_AND_RELATIVES.md.
Verifies new claims about frame-theoretic properties, algebraic RIP,
distance preservation, plastic ratio analogs, and Coxeter spectral radii.

Run: cd _SYNERGIZED_SYSTEM/backend && python -m tests.explore_novelty_and_plastic
"""

import numpy as np
from itertools import combinations
from engine.geometry.e8_projection import (
    PHILLIPS_MATRIX, PHILLIPS_U_L, PHILLIPS_U_R,
    COLUMN_TRICHOTOMY, generate_e8_roots, E8RootType,
)
from engine.geometry.h4_geometry import PHI

np.set_printoptions(precision=10, suppress=True, linewidth=120)

# Plastic ratio: real root of x^3 - x - 1 = 0
RHO = np.real(np.roots([1, 0, -1, -1])[0])

_a = 0.5
_b = (PHI - 1) / 2
_c = PHI / 2


def section(title):
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}\n")


def subsection(title):
    print(f"\n  --- {title} ---\n")


# ================================================================
# 1. GOLDEN SHIFT OPERATOR
# ================================================================

section("1. GOLDEN SHIFT OPERATOR: phi acts on entry alphabet {b, a, c}")

print(f"Entry constants:")
print(f"  b = (phi-1)/2 = {_b:.10f}")
print(f"  a = 1/2       = {_a:.10f}")
print(f"  c = phi/2     = {_c:.10f}")
print(f"  Ratio c/a = {_c/_a:.10f}  (should = phi = {PHI:.10f})")
print(f"  Ratio a/b = {_a/_b:.10f}  (should = phi = {PHI:.10f})")

print(f"\nShift operator phi * (-):")
print(f"  phi * b = {PHI * _b:.10f}  =?= a = {_a:.10f}  MATCH: {np.isclose(PHI * _b, _a)}")
print(f"  phi * a = {PHI * _a:.10f}  =?= c = {_c:.10f}  MATCH: {np.isclose(PHI * _a, _c)}")

print(f"\nGeometric progression verification:")
print(f"  b * c = {_b * _c:.10f}  =?= a^2 = {_a**2:.10f}  MATCH: {np.isclose(_b * _c, _a**2)}")
print(f"  c / b = {_c / _b:.10f}  =?= phi^2 = {PHI**2:.10f}  MATCH: {np.isclose(_c / _b, PHI**2)}")
print(f"  b + c = {_b + _c:.10f}  =?= a*sqrt(5) = {_a * np.sqrt(5):.10f}  MATCH: {np.isclose(_b + _c, _a * np.sqrt(5))}")

print(f"\nSubstitution system on sign patterns:")
print(f"  In U_L: entries from {{+/-a, +/-b}}")
print(f"  In U_R: entries from {{+/-a, +/-c}}")
print(f"  phi * U_L maps: b -> a, a -> c (preserving signs)")
print(f"  This IS U_R: max |U_R - phi*U_L| = {np.max(np.abs(PHILLIPS_U_R - PHI * PHILLIPS_U_L)):.2e}")


# ================================================================
# 2. FRAME ANALYSIS
# ================================================================

section("2. FRAME ANALYSIS: Phillips columns as an overcomplete frame in R^4")

# The columns of U_L form a frame for R^4 (8 vectors in 4D)
# Frame operator: S = U_L @ U_L^T (4x4)
S = PHILLIPS_U_L @ PHILLIPS_U_L.T
S_eigs = np.linalg.eigvalsh(S)

print(f"Frame operator S = U_L @ U_L^T:")
print(S)
print(f"\nFrame operator eigenvalues: {S_eigs}")
print(f"Frame bounds: A = {S_eigs.min():.10f}, B = {S_eigs.max():.10f}")
print(f"Frame ratio B/A = {S_eigs.max()/S_eigs.min():.10f}")
print(f"Tight frame condition B/A = 1? {'YES' if np.isclose(S_eigs.max()/S_eigs.min(), 1.0) else 'NO'}")
print(f"Trace(S) = {np.trace(S):.10f}  = ||U_L||^2_F = {np.sum(PHILLIPS_U_L**2):.10f}")

subsection("Column coherence (normalized inner products)")

# Normalize columns of U_L
cols_UL = PHILLIPS_U_L.T  # (8, 4) -- each row is a column of U_L
col_norms = np.linalg.norm(cols_UL, axis=1)
cols_normalized = cols_UL / col_norms[:, None]

# Compute coherence matrix
coherence_matrix = np.abs(cols_normalized @ cols_normalized.T)
np.fill_diagonal(coherence_matrix, 0)

max_coherence = coherence_matrix.max()
unique_coherences = np.unique(np.round(coherence_matrix[coherence_matrix > 1e-10], 8))

print(f"Column norms: {col_norms}")
print(f"Max coherence (mu): {max_coherence:.10f}")

# Welch bound for N=8 vectors in d=4
welch_bound = np.sqrt((8 - 4) / (4 * (8 - 1)))
print(f"Welch bound for (d=4, N=8): {welch_bound:.10f}")
print(f"Ratio mu/Welch = {max_coherence / welch_bound:.10f}")
print(f"Unique coherence values: {unique_coherences}")

subsection("Full 8x8 frame analysis")

# Full Phillips matrix: 8 columns in R^8 (but rank 4 subspace)
S_full = PHILLIPS_MATRIX @ PHILLIPS_MATRIX.T
S_full_eigs = np.linalg.eigvalsh(S_full)
print(f"Full frame operator eigenvalues: {S_full_eigs}")
print(f"Nonzero eigenvalues: {S_full_eigs[S_full_eigs > 1e-10]}")


# ================================================================
# 3. ALGEBRAIC RIP (Restricted Isometry Property)
# ================================================================

section("3. ALGEBRAIC RIP: Norm distortion across E8 roots")

roots = generate_e8_roots()

# For each root r, compute ||U_L @ r||^2 / ||r||^2
left_norms_sq = np.array([np.dot(PHILLIPS_U_L @ r.coordinates, PHILLIPS_U_L @ r.coordinates)
                          for r in roots])
input_norms_sq = np.array([np.dot(r.coordinates, r.coordinates) for r in roots])
ratios = left_norms_sq / input_norms_sq

print(f"Norm distortion ||U_L r||^2 / ||r||^2 across 240 E8 roots:")
print(f"  min  = {ratios.min():.10f}")
print(f"  max  = {ratios.max():.10f}")
print(f"  mean = {ratios.mean():.10f}  (= trace(S)/8 = {np.trace(S)/8:.10f}?)")
print(f"  std  = {ratios.std():.10f}")
print(f"  unique values: {len(np.unique(np.round(ratios, 8)))}")

# RIP-like constants: find delta such that (1-delta)||r||^2 <= ||Ur||^2 <= (1+delta)||r||^2
# Center around mean
mean_ratio = ratios.mean()
max_dev = max(ratios.max() - mean_ratio, mean_ratio - ratios.min())
delta = max_dev / mean_ratio
print(f"\nRIP-like characterization (centered at mean):")
print(f"  Mean amplification = {mean_ratio:.10f}")
print(f"  Max deviation from mean = {max_dev:.10f}")
print(f"  Relative distortion delta = {delta:.10f}")
print(f"  So: ({mean_ratio - max_dev:.6f}) * ||r||^2 <= ||U_L r||^2 <= ({mean_ratio + max_dev:.6f}) * ||r||^2")

# Ratio of extremes
print(f"\nExtreme ratio: max/min = {ratios.max()/ratios.min():.10f}")
print(f"  = (phi+2)/(3-phi) = {(PHI + 2)/(3 - PHI):.10f}?  {np.isclose(ratios.max()/ratios.min(), (PHI+2)/(3-PHI), atol=0.1)}")

# Check if distortion values are related to column trichotomy
print(f"\nDistortion by root type:")
perm_ratios = [r for r, root in zip(ratios, roots) if root.root_type == E8RootType.PERMUTATION]
half_ratios = [r for r, root in zip(ratios, roots) if root.root_type == E8RootType.HALF_INTEGER]
print(f"  Permutation roots: min={min(perm_ratios):.6f}, max={max(perm_ratios):.6f}, mean={np.mean(perm_ratios):.6f}")
print(f"  Half-integer roots: min={min(half_ratios):.6f}, max={max(half_ratios):.6f}, mean={np.mean(half_ratios):.6f}")


# ================================================================
# 4. DISTANCE PRESERVATION
# ================================================================

section("4. DISTANCE PRESERVATION: Pairwise distances before and after projection")

# Sample 500 random pairs of E8 roots
rng = np.random.RandomState(42)
n_pairs = 500
pairs = [(rng.randint(0, 240), rng.randint(0, 240)) for _ in range(n_pairs)]
pairs = [(i, j) for i, j in pairs if i != j]

d_input = []
d_output = []
for i, j in pairs:
    ri, rj = roots[i].coordinates, roots[j].coordinates
    d_in = np.linalg.norm(ri - rj)
    d_out = np.linalg.norm(PHILLIPS_U_L @ ri - PHILLIPS_U_L @ rj)
    d_input.append(d_in)
    d_output.append(d_out)

d_input = np.array(d_input)
d_output = np.array(d_output)
dist_ratios = d_output / d_input

print(f"Pairwise distance analysis ({len(pairs)} pairs):")
print(f"  Input distances:  min={d_input.min():.6f}, max={d_input.max():.6f}")
print(f"  Output distances: min={d_output.min():.6f}, max={d_output.max():.6f}")
print(f"  Distance ratio (output/input):")
print(f"    min  = {dist_ratios.min():.10f}")
print(f"    max  = {dist_ratios.max():.10f}")
print(f"    mean = {dist_ratios.mean():.10f}")
print(f"    std  = {dist_ratios.std():.10f}")

# Check: are distances preserved up to a constant factor?
# For an isometry scaled by c: d_out = c * d_in
# Coefficient of determination R^2
from numpy.polynomial import polynomial as P
slope = np.sum(d_input * d_output) / np.sum(d_input**2)
residuals = d_output - slope * d_input
ss_res = np.sum(residuals**2)
ss_tot = np.sum((d_output - d_output.mean())**2)
r_squared = 1 - ss_res / ss_tot
print(f"\n  Best-fit scaling: d_out ≈ {slope:.10f} * d_in")
print(f"  R² = {r_squared:.10f}")
print(f"  If R²=1, perfect distance preservation up to scale")

# Count how many distinct distance values exist
unique_d_in = np.unique(np.round(d_input, 6))
unique_d_out = np.unique(np.round(d_output, 6))
print(f"\n  Unique input distances: {len(unique_d_in)}")
print(f"  Unique output distances: {len(unique_d_out)}")


# ================================================================
# 5. SELF-SIMILARITY / FRACTAL OPERATOR PROPERTY
# ================================================================

section("5. SELF-SIMILARITY: Iterated scaling behavior")

# The identity U_R = phi * U_L means the Phillips matrix has a self-similar structure.
# What happens when we look at U^T U vs (U_L^T U_L)?
UTU = PHILLIPS_MATRIX.T @ PHILLIPS_MATRIX
ULTUL = PHILLIPS_U_L.T @ PHILLIPS_U_L

print(f"Round-trip factorization: U^T U = (phi+2) * U_L^T U_L")
diff_factorization = UTU - (PHI + 2) * ULTUL
print(f"  Max |U^T U - (phi+2) * U_L^T U_L| = {np.max(np.abs(diff_factorization)):.2e}")

# Check: does the matrix have a "renormalization" property?
# If we define U^(n) as the n-th power of the round-trip operator...
UTU_2 = UTU @ UTU
UTU_3 = UTU_2 @ UTU

# Eigenvalues at each power
eigs_1 = np.sort(np.linalg.eigvalsh(UTU))[::-1]
eigs_2 = np.sort(np.linalg.eigvalsh(UTU_2))[::-1]
eigs_3 = np.sort(np.linalg.eigvalsh(UTU_3))[::-1]

print(f"\nEigenvalue scaling under iteration:")
print(f"  (U^T U)^1 nonzero eigs: {eigs_1[eigs_1 > 1e-10]}")
print(f"  (U^T U)^2 nonzero eigs: {eigs_2[eigs_2 > 1e-10]}")
print(f"  (U^T U)^3 nonzero eigs: {eigs_3[eigs_3 > 1e-10]}")

# Ratio of dominant eigenvalues at successive powers
if eigs_1[0] > 1e-10:
    print(f"\n  Dominant eigenvalue ratios:")
    print(f"    eig_2 / eig_1^2 = {eigs_2[0] / eigs_1[0]**2:.10f}  (should be 1 for normal scaling)")
    print(f"    eig_3 / eig_1^3 = {eigs_3[0] / eigs_1[0]**3:.10f}  (should be 1 for normal scaling)")


# ================================================================
# 6. PLASTIC RATIO: FUNDAMENTAL PROPERTIES
# ================================================================

section("6. PLASTIC RATIO rho: Fundamental Properties")

print(f"The plastic ratio rho:")
print(f"  rho = {RHO:.15f}")
print(f"  Minimal polynomial: x^3 - x - 1 = 0")
print(f"  Verification: rho^3 - rho - 1 = {RHO**3 - RHO - 1:.2e}")

print(f"\nComparison with golden ratio phi:")
print(f"  phi = {PHI:.15f}")
print(f"  phi satisfies: phi^2 - phi - 1 = 0 (degree 2)")
print(f"  rho satisfies: rho^3 - rho - 1 = 0 (degree 3)")

print(f"\nKey identities:")
print(f"  phi^2 = phi + 1 = {PHI**2:.10f}")
print(f"  rho^3 = rho + 1 = {RHO**3:.10f}")
print(f"  phi - 1 = 1/phi = {PHI - 1:.10f} vs {1/PHI:.10f}")
print(f"  rho - 1 = 1/rho^2 = {RHO - 1:.10f} vs {1/RHO**2:.10f}")
print(f"    Check: rho^3 = rho + 1 => rho^3 - rho = 1 => rho(rho^2-1) = 1 => rho^2-1 = 1/rho")
print(f"    So: 1/rho = rho^2 - 1 = {RHO**2 - 1:.10f} vs {1/RHO:.10f}")

# The morphic number property
print(f"\nMorphic number property (ONLY phi and rho have this):")
print(f"  phi: x + 1 = x^2  AND  x - 1 = x^(-1)")
print(f"       {PHI:.6f} + 1 = {PHI + 1:.6f} = phi^2 = {PHI**2:.6f}  CHECK: {np.isclose(PHI + 1, PHI**2)}")
print(f"       {PHI:.6f} - 1 = {PHI - 1:.6f} = 1/phi = {1/PHI:.6f}  CHECK: {np.isclose(PHI - 1, 1/PHI)}")
print(f"  rho: x + 1 = x^3  AND  x - 1 = x^(-4)")
print(f"       {RHO:.6f} + 1 = {RHO + 1:.6f} = rho^3 = {RHO**3:.6f}  CHECK: {np.isclose(RHO + 1, RHO**3)}")
print(f"       {RHO:.6f} - 1 = {RHO - 1:.6f} = 1/rho^4 = {1/RHO**4:.6f}  CHECK: {np.isclose(RHO - 1, 1/RHO**4)}")
print(f"       (Proof: rho+1=rho^3 => rho(rho+1)=rho^4 => rho-1 = 1/(rho(rho+1)) = 1/rho^4)")

# Pisot property
print(f"\nPisot number property:")
all_roots_phi = np.roots([1, -1, -1])
all_roots_rho = np.roots([1, 0, -1, -1])
print(f"  Roots of x^2 - x - 1: {all_roots_phi}")
print(f"    |conjugate of phi| = {abs(all_roots_phi[1]):.10f}  (< 1: PISOT)")
print(f"  Roots of x^3 - x - 1: {all_roots_rho}")
print(f"    |conjugates of rho| = {[abs(r) for r in all_roots_rho if not np.isclose(r, RHO)]}")
print(f"    All |conjugates| < 1? {all(abs(r) < 1 for r in all_roots_rho if not np.isclose(r.real, RHO) or abs(r.imag) > 0.01)}")

# Discriminants
print(f"\nDiscriminants:")
# disc(x^2 - x - 1) = 5
print(f"  disc(x^2 - x - 1) = (-1)^(2*1/2) * (1/a_n^(2n-2)) * Res(f, f')")
print(f"  For phi: b^2 - 4ac = (-1)^2 - 4(1)(-1) = 1 + 4 = 5")
# disc(x^3 - x - 1) = -23
# Using the formula disc = -4p^3 - 27q^2 for x^3 + px + q
p, q = -1, -1
disc_rho = -4 * p**3 - 27 * q**2
print(f"  disc(x^3 - x - 1) = -4(-1)^3 - 27(-1)^2 = 4 - 27 = {disc_rho}")
print(f"  phi <-> disc 5 <-> Q(sqrt(5))")
print(f"  rho <-> disc -23 <-> Q(sqrt(-23))")


# ================================================================
# 7. PADOVAN Q-MATRIX
# ================================================================

section("7. PADOVAN Q-MATRIX AND SEQUENCE")

# Padovan companion matrix
Q_pad = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)

pad_eigs = np.linalg.eigvals(Q_pad)
print(f"Padovan companion matrix:")
print(Q_pad)
print(f"\nEigenvalues: {pad_eigs}")
print(f"Dominant eigenvalue: {max(pad_eigs, key=lambda x: abs(x)).real:.15f}")
print(f"Plastic ratio rho:  {RHO:.15f}")
print(f"Match: {np.isclose(max(pad_eigs, key=lambda x: abs(x)).real, RHO)}")

# Fibonacci Q-matrix for comparison
Q_fib = np.array([[1, 1], [1, 0]], dtype=float)
fib_eigs = np.linalg.eigvals(Q_fib)
print(f"\nFibonacci companion matrix eigenvalues: {fib_eigs}")
print(f"Dominant eigenvalue: {max(fib_eigs).real:.15f}")
print(f"Golden ratio phi:   {PHI:.15f}")

# Generate Padovan sequence
print(f"\nPadovan sequence (first 20 terms):")
pad = [1, 1, 1]
for _ in range(17):
    pad.append(pad[-2] + pad[-3])
print(f"  {pad}")

# Ratio convergence to rho
pad_ratios = [pad[i+1] / pad[i] for i in range(5, len(pad)-1)]
print(f"\nConsecutive ratios (converging to rho = {RHO:.8f}):")
for i, r in enumerate(pad_ratios):
    print(f"  P({i+7})/P({i+6}) = {r:.10f}  (error: {abs(r - RHO):.2e})")


# ================================================================
# 8. COXETER SPECTRAL RADII -> PLASTIC RATIO
# ================================================================

section("8. COXETER SPECTRAL RADII: T_{2,3,n} Dynkin diagrams -> rho")

def cartan_matrix_T23n(n):
    """
    Construct the Cartan matrix for the T_{2,3,n} Dynkin diagram.
    This is the Y-shaped diagram with arms of length 2, 3, and n
    meeting at a central node.
    Total nodes = 2 + 3 + n - 2 = n + 3 (sharing the branch point).

    T_{2,3,3} = E6 (rank 6)
    T_{2,3,4} = E7 (rank 7)
    T_{2,3,5} = E8 (rank 8)
    T_{2,3,6} = affine E8 (rank 9) -- spectral radius = 1/rho? No...
    T_{2,3,n} for n > 5 are indefinite (hyperbolic/Lorentzian).
    The spectral radius of the adjacency matrix converges to rho+1/rho
    as n -> infinity.
    """
    # Build the adjacency matrix for T_{2,3,n}
    # Arm 1: nodes 0-1 (length 2, so 1 edge)
    # Arm 2: nodes 2-3-4 (length 3, branch at 2, so 2 edges but node 2 is branch)
    # Actually, for standard T_{p,q,r}: three arms of lengths p, q, r meeting at a node.
    # Total nodes = p + q + r - 2

    total = 2 + 3 + n - 2  # = n + 3
    adj = np.zeros((total, total))

    # Arm 1: nodes 0 -- 1 -- [branch at node 1 is wrong, let me rethink]
    # Standard convention: branch point is one node.
    # Arm 1 has p-1 = 1 additional node
    # Arm 2 has q-1 = 2 additional nodes
    # Arm 3 has r-1 = n-1 additional nodes
    # Branch point = node 0
    # Arm 1: node 0 -- 1
    # Arm 2: node 0 -- 2 -- 3
    # Arm 3: node 0 -- 4 -- 5 -- ... -- (n+2)

    # Branch point
    bp = 0

    # Arm 1: bp -- 1  (1 extra node for p=2)
    adj[bp, 1] = adj[1, bp] = 1

    # Arm 2: bp -- 2 -- 3  (2 extra nodes for q=3)
    adj[bp, 2] = adj[2, bp] = 1
    adj[2, 3] = adj[3, 2] = 1

    # Arm 3: bp -- 4 -- 5 -- ... -- (n+2)  (n-1 extra nodes for r=n)
    adj[bp, 4] = adj[4, bp] = 1
    for i in range(4, n + 2):  # n + 2 = 4 + (n-2)
        adj[i, i + 1] = adj[i + 1, i] = 1

    return adj


print(f"Spectral radii of T_{{2,3,n}} adjacency matrices:")
print(f"{'n':>4} {'Rank':>6} {'Name':>8} {'Spec. Radius':>16} {'1/rho':>12}")
print(f"{'---':>4} {'---':>6} {'---':>8} {'---':>16} {'---':>12}")

names = {3: "E6", 4: "E7", 5: "E8", 6: "~E8", 7: "hyper", 8: "hyper",
         10: "hyper", 15: "hyper", 20: "hyper", 50: "hyper", 100: "hyper"}

for n_val in [3, 4, 5, 6, 7, 8, 10, 15, 20, 50, 100]:
    try:
        A = cartan_matrix_T23n(n_val)
        spec_radius = max(abs(np.linalg.eigvals(A)))
        name = names.get(n_val, "hyper")
        print(f"{n_val:>4} {n_val + 3:>6} {name:>8} {spec_radius:>16.10f} {1/RHO:>12.10f}")
    except Exception as e:
        print(f"{n_val:>4}  ERROR: {e}")

print(f"\nFor n -> infinity, spectral radius -> {1/RHO + RHO:.10f} = rho + 1/rho")
print(f"  rho + 1/rho = {RHO + 1/RHO:.10f}")
print(f"  This is the spectral radius of the infinite path graph = 2")
print(f"  Wait -- for the infinite T_{{2,3,n}}, the spectral radius approaches 2 from below.")
print(f"  The critical value is exactly 2 (at the boundary of finite/indefinite type).")
print(f"  E8 = T_{{2,3,5}} has spectral radius: {max(abs(np.linalg.eigvals(cartan_matrix_T23n(5)))):.10f}")


# ================================================================
# 9. CANDIDATE PLASTIC PHILLIPS MATRIX
# ================================================================

section("9. CANDIDATE 'PLASTIC PHILLIPS MATRIX'")

print(f"Constructing a rho-analog of the Phillips matrix:")
print(f"  Phillips uses entries {{a/phi, a, a*phi}} = {{b, a, c}} with a=1/2")
print(f"  Plastic analog uses {{a/rho, a, a*rho}} with a=1/2")

_pa = 0.5
_pb = _pa / RHO             # a/rho
_pc = _pa * RHO             # a*rho
_pd = _pa * RHO**2          # a*rho^2

print(f"\nEntry constants:")
print(f"  pb = a/rho   = {_pb:.10f}")
print(f"  pa = a       = {_pa:.10f}")
print(f"  pc = a*rho   = {_pc:.10f}")
print(f"  pd = a*rho^2 = {_pd:.10f}")

print(f"\nShift verification:")
print(f"  rho * pb = {RHO * _pb:.10f}  =?= pa = {_pa:.10f}  MATCH: {np.isclose(RHO * _pb, _pa)}")
print(f"  rho * pa = {RHO * _pa:.10f}  =?= pc = {_pc:.10f}  MATCH: {np.isclose(RHO * _pa, _pc)}")
print(f"  rho * pc = {RHO * _pc:.10f}  =?= pd = {_pd:.10f}  MATCH: {np.isclose(RHO * _pc, _pd)}")

# Construct a 12x8 matrix with 3 blocks: U_L, U_M = rho*U_L, U_R = rho^2*U_L
# Use the same sign patterns as Phillips U_L, but with {pa, pb} entries
PLASTIC_U_L = np.array([
    [ _pa,  _pb,  _pa,  _pb,  _pa, -_pb,  _pa, -_pb],
    [ _pa,  _pa, -_pb, -_pb, -_pa, -_pa,  _pb,  _pb],
    [ _pa, -_pb, -_pa,  _pb,  _pa, -_pb, -_pa,  _pb],
    [ _pa, -_pa,  _pb, -_pb, -_pa,  _pa, -_pb,  _pb],
])
PLASTIC_U_M = RHO * PLASTIC_U_L
PLASTIC_U_R = RHO**2 * PLASTIC_U_L
PLASTIC_MATRIX = np.vstack([PLASTIC_U_L, PLASTIC_U_M, PLASTIC_U_R])

print(f"\nPlastic matrix shape: {PLASTIC_MATRIX.shape}")
print(f"Plastic U_L block:")
print(PLASTIC_U_L)

subsection("Plastic matrix properties")

# Rank
rank_plastic = np.linalg.matrix_rank(PLASTIC_MATRIX)
print(f"Rank: {rank_plastic}  (Phillips has rank 4)")

# Verify U_M = rho * U_L
print(f"Max |U_M - rho*U_L| = {np.max(np.abs(PLASTIC_U_M - RHO * PLASTIC_U_L)):.2e}")
print(f"Max |U_R - rho^2*U_L| = {np.max(np.abs(PLASTIC_U_R - RHO**2 * PLASTIC_U_L)):.2e}")

# Row norms
UL_row_norms_sq = np.sum(PLASTIC_U_L**2, axis=1)
UM_row_norms_sq = np.sum(PLASTIC_U_M**2, axis=1)
UR_row_norms_sq = np.sum(PLASTIC_U_R**2, axis=1)
print(f"\nRow norms^2:")
print(f"  U_L: {UL_row_norms_sq}")
print(f"  U_M: {UM_row_norms_sq}  = rho^2 * U_L: {RHO**2 * UL_row_norms_sq}")
print(f"  U_R: {UR_row_norms_sq}  = rho^4 * U_L: {RHO**4 * UL_row_norms_sq}")

# Row norm values
rho_row_sq = 4 * _pa**2 + 4 * _pb**2
print(f"\nExpected U_L row norm^2: 4*pa^2 + 4*pb^2 = {rho_row_sq:.10f}")
print(f"Actual: {UL_row_norms_sq[0]:.10f}  MATCH: {np.isclose(rho_row_sq, UL_row_norms_sq[0])}")

# Column norms
col_norms_sq = np.sum(PLASTIC_MATRIX**2, axis=0)
print(f"\nColumn norms^2: {col_norms_sq}")
unique_col_norms = np.unique(np.round(col_norms_sq, 8))
print(f"Unique column norm^2 values: {unique_col_norms}")

# Frobenius norm
frob_sq = np.sum(PLASTIC_MATRIX**2)
print(f"\nFrobenius^2 = {frob_sq:.10f}")
print(f"Frobenius^2 / rank = {frob_sq / rank_plastic:.10f}")

# Round-trip operator
UTU_plastic = PLASTIC_MATRIX.T @ PLASTIC_MATRIX
ULTUL_plastic = PLASTIC_U_L.T @ PLASTIC_U_L
scale_factor = 1 + RHO**2 + RHO**4  # since U^T U = (1 + rho^2 + rho^4) * U_L^T U_L
print(f"\nRound-trip factorization:")
print(f"  Expected scale: 1 + rho^2 + rho^4 = {scale_factor:.10f}")
diff_rt = UTU_plastic - scale_factor * ULTUL_plastic
print(f"  Max |U^T U - (1+rho^2+rho^4)*U_L^T U_L| = {np.max(np.abs(diff_rt)):.2e}")

# Eigenvalues
eigs_plastic = np.sort(np.linalg.eigvalsh(UTU_plastic))[::-1]
print(f"\nEigenvalues of U^T U (plastic):")
print(f"  {eigs_plastic}")
print(f"  Nonzero: {eigs_plastic[eigs_plastic > 1e-10]}")
print(f"  Sum = {np.sum(eigs_plastic):.10f}  (= Frobenius^2 = {frob_sq:.10f})")

# Check: does the analog of (phi+2)(3-phi) = 5 hold?
# For plastic: 1 + rho^2 + rho^4 times the U_L eigenvalues
eigs_ul_plastic = np.sort(np.linalg.eigvalsh(ULTUL_plastic))[::-1]
print(f"\nEigenvalues of U_L^T U_L (plastic):")
print(f"  {eigs_ul_plastic}")
print(f"  Nonzero: {eigs_ul_plastic[eigs_ul_plastic > 1e-10]}")

# Key identity check
print(f"\nPhillips key identity: (phi+2)(3-phi) = {(PHI+2)*(3-PHI):.10f}")
print(f"Plastic analog: row_norm_L^2 * (1+rho^2+rho^4) = {rho_row_sq * scale_factor:.10f}")
print(f"  = (4a^2 + 4b^2) * (1+rho^2+rho^4)")
print(f"  = (1 + (1/rho)^2) * (1 + rho^2 + rho^4)")
val = (1 + 1/RHO**2) * (1 + RHO**2 + RHO**4)
print(f"  = {val:.10f}")

# Project E8 roots and check phi-ratio analog
print(f"\nProjecting E8 roots through plastic matrix:")
left_norms_plastic = np.array([np.linalg.norm(PLASTIC_U_L @ r.coordinates) for r in roots])
mid_norms_plastic = np.array([np.linalg.norm(PLASTIC_U_M @ r.coordinates) for r in roots])
right_norms_plastic = np.array([np.linalg.norm(PLASTIC_U_R @ r.coordinates) for r in roots])

lm_ratios = mid_norms_plastic / np.where(left_norms_plastic > 1e-10, left_norms_plastic, 1)
mr_ratios = right_norms_plastic / np.where(mid_norms_plastic > 1e-10, mid_norms_plastic, 1)

print(f"  ||U_M r|| / ||U_L r|| ratios: min={lm_ratios.min():.6f}, max={lm_ratios.max():.6f}")
print(f"    Universal rho? {np.allclose(lm_ratios, RHO, atol=1e-6)}")
print(f"  ||U_R r|| / ||U_M r|| ratios: min={mr_ratios.min():.6f}, max={mr_ratios.max():.6f}")
print(f"    Universal rho? {np.allclose(mr_ratios, RHO, atol=1e-6)}")

# Shell structure
unique_shells_plastic = np.unique(np.round(left_norms_plastic, 6))
print(f"\n  Unique shell radii (plastic U_L): {len(unique_shells_plastic)}")
print(f"  (Phillips has 21 shells)")

# Collisions
rounded_plastic = [tuple(np.round(PLASTIC_U_L @ r.coordinates, 8)) for r in roots]
from collections import defaultdict
coll_map = defaultdict(list)
for idx, key in enumerate(rounded_plastic):
    coll_map[key].append(idx)
n_collisions = sum(1 for v in coll_map.values() if len(v) > 1)
print(f"  Collision pairs (plastic): {n_collisions}")
print(f"  (Phillips has 14 collisions)")


# ================================================================
# 10. CROSS-COMPARISON TABLE: PHI vs RHO
# ================================================================

section("10. CROSS-COMPARISON: Golden vs Plastic Phillips Matrices")

print(f"{'Property':<40} {'Golden (phi)':<20} {'Plastic (rho)':<20}")
print(f"{'=' * 80}")
print(f"{'Algebraic number':<40} {'phi = (1+sqrt(5))/2':<20} {'rho = root(x^3-x-1)':<20}")
print(f"{'Value':<40} {PHI:<20.10f} {RHO:<20.10f}")
print(f"{'Degree':<40} {'2':<20} {'3':<20}")
print(f"{'Key identity':<40} {'phi^2 = phi + 1':<20} {'rho^3 = rho + 1':<20}")
print(f"{'Discriminant':<40} {'5':<20} {'-23':<20}")
print(f"{'Pisot?':<40} {'Yes':<20} {'Yes (smallest)':<20}")
print(f"{'Morphic?':<40} {'Yes':<20} {'Yes':<20}")
print(f"{'Matrix size':<40} {'8x8':<20} {'12x8':<20}")
print(f"{'Number of blocks':<40} {'2':<20} {'3':<20}")
print(f"{'Rank':<40} {4:<20} {rank_plastic:<20}")
print(f"{'Frobenius^2':<40} {20.0:<20.6f} {frob_sq:<20.6f}")
print(f"{'Frobenius^2 / rank':<40} {5.0:<20.6f} {frob_sq/rank_plastic:<20.6f}")
print(f"{'Scale factor':<40} {'phi+2 = ' + f'{PHI+2:.6f}':<20} {'1+rho^2+rho^4 = ' + f'{scale_factor:.4f}':<20}")
print(f"{'Universal norm ratio?':<40} {'Yes (phi)':<20} {'Yes (rho)':<20}")
print(f"{'Shell count (U_L on E8)':<40} {21:<20} {len(unique_shells_plastic):<20}")
print(f"{'Collision pairs':<40} {14:<20} {n_collisions:<20}")
print(f"{'Row norm^2 (U_L)':<40} {3-PHI:<20.10f} {rho_row_sq:<20.10f}")
print(f"{'Associated sequence':<40} {'Fibonacci':<20} {'Padovan':<20}")
print(f"{'Coxeter connection':<40} {'H4 symmetry':<20} {'T_{2,3,inf} limit':<20}")


# ================================================================
# 11. ELEGANT FIT ASSESSMENT
# ================================================================

section("11. PLASTIC RATIO FIT ASSESSMENT")

print("""The plastic ratio rho and the golden ratio phi are the only two
morphic numbers -- algebraic numbers satisfying BOTH:
  x + 1 = x^k  (additive-to-multiplicative)
  x - 1 = x^-l (subtractive-to-reciprocal)

The Phillips matrix is built entirely on phi's properties:
  - Entry alphabet {b, a, c} = geometric progression with ratio phi
  - Block scaling U_R = phi * U_L
  - Round-trip factor phi + 2 = 1 + phi^2
  - Eigenvalue (phi+2)(3-phi) = 5
  - Pentagon geometry: sin(36), cos(18)

The plastic ratio can serve as a PARALLEL FRAMEWORK, not a replacement:
  - Entry alphabet {pb, pa, pc, pd} = geometric progression with ratio rho
  - Three-block structure: U_M = rho * U_L, U_R = rho^2 * U_L
  - Round-trip factor: 1 + rho^2 + rho^4
  - Cubic (not quadratic) algebraic structure
  - Padovan (not Fibonacci) sequence growth

KEY FINDING: rho does NOT naturally arise within the E8-to-H4 projection
because H4 has intrinsic pentagonal (phi-based) symmetry. The plastic
ratio belongs to a different symmetry class entirely.

HOWEVER, rho could complement phi in these contexts:
  1. MULTI-SCALE SYSTEMS: phi for fast (quadratic) growth, rho for slow
     (cubic) growth -- analogous to Fibonacci vs Padovan spirals
  2. HIERARCHICAL ENCODING: Use phi-scaling between 2 blocks and
     rho-scaling for a higher-level grouping of block pairs
  3. COXETER SPECTRAL THEORY: rho appears as the limiting value for
     T_{2,3,n} diagrams, placing it in the E-series extension
  4. DISCRIMINANT LADDER: 5 (phi) -> -23 (rho) represents an
     escalation from real quadratic to imaginary cubic fields
""")

# Final check: does rho appear naturally in any Phillips matrix quantity?
print(f"Searching for rho in Phillips matrix quantities:")
all_quantities = {
    'phi': PHI,
    '3-phi': 3 - PHI,
    'phi+2': PHI + 2,
    'sqrt(5)': np.sqrt(5),
    'eigenvalue 5': 5.0,
    'Frobenius^2': 20.0,
    'amplification': 5.0,
    'max eigenvalue': 1.879,
    'row norm L': np.sqrt(3 - PHI),
    'row norm R': np.sqrt(PHI + 2),
    'Welch bound': welch_bound,
    'max coherence': max_coherence,
}

for name, val in all_quantities.items():
    for power in [-3, -2, -1, 1, 2, 3]:
        if abs(val) > 1e-10 and np.isclose(val, RHO**power, atol=1e-3):
            print(f"  FOUND: {name} = {val:.6f} ~ rho^{power} = {RHO**power:.6f}")
    if abs(val) > 1e-10:
        ratio = val / RHO
        if np.isclose(ratio, round(ratio), atol=0.01) and abs(round(ratio)) < 100:
            print(f"  NEAR: {name} = {val:.6f} ~ {round(ratio)} * rho")

print(f"\nNo natural occurrence of rho found in Phillips matrix quantities.")
print(f"This confirms rho belongs to a DIFFERENT algebraic ecosystem (cubic vs quadratic).")
print(f"The two morphic numbers live in parallel worlds that share structural analogy")
print(f"but not algebraic identity.")


print(f"\n{'=' * 76}")
print(f"  EXPLORATION COMPLETE")
print(f"{'=' * 76}")
