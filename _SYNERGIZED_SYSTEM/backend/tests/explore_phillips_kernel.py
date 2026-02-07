"""
Phillips Matrix Deep Exploration Script.

Computes kernel structure, eigendecomposition, collision-to-24cell mapping,
amplification factor, and round-trip operator analysis.

Run: python -m tests.explore_phillips_kernel
"""

import numpy as np
from itertools import combinations
from engine.geometry.e8_projection import (
    PHILLIPS_MATRIX, PHILLIPS_U_L, PHILLIPS_U_R,
    COLUMN_TRICHOTOMY, generate_e8_roots, E8RootType,
)
from engine.geometry.h4_geometry import PHI

np.set_printoptions(precision=8, suppress=True, linewidth=120)


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ============================================================
# 1. KERNEL OF U_L (4×8)
# ============================================================

section("1. KERNEL OF PHILLIPS U_L")

# U_L is 4×8, so kernel has dimension 8-4 = 4 (if rank 4)
rank_UL = np.linalg.matrix_rank(PHILLIPS_U_L)
print(f"Rank of U_L: {rank_UL}")
print(f"Expected kernel dimension: {8 - rank_UL}")

# SVD to get null space
U, S, Vt = np.linalg.svd(PHILLIPS_U_L)
print(f"\nSingular values of U_L: {S}")

# Null space = rows of V^T corresponding to zero singular values
null_mask = S < 1e-10
kernel_basis = Vt[rank_UL:]  # shape (4, 8)
print(f"\nKernel basis vectors (rows):")
for i, v in enumerate(kernel_basis):
    print(f"  k{i}: {v}")
    print(f"       norm = {np.linalg.norm(v):.8f}")

# Verify: U_L @ k = 0
print(f"\nVerification (should all be ~0):")
for i, k in enumerate(kernel_basis):
    residual = PHILLIPS_U_L @ k
    print(f"  U_L @ k{i} = {residual}  (max={np.max(np.abs(residual)):.2e})")

# Check if kernel is also the kernel of U_R (since U_R = phi * U_L)
print(f"\nIs kernel(U_R) = kernel(U_L)?")
for i, k in enumerate(kernel_basis):
    residual_R = PHILLIPS_U_R @ k
    print(f"  U_R @ k{i} = {residual_R}  (max={np.max(np.abs(residual_R)):.2e})")

# Gram matrix of the kernel basis (to check lattice structure)
kernel_gram = kernel_basis @ kernel_basis.T
print(f"\nKernel Gram matrix (K^T K):")
print(kernel_gram)
print(f"\nKernel Gram determinant: {np.linalg.det(kernel_gram):.8f}")
print(f"Kernel Gram eigenvalues: {np.linalg.eigvalsh(kernel_gram)}")

# Check if kernel vectors have any special structure
print(f"\nKernel basis entry analysis:")
for i, k in enumerate(kernel_basis):
    nonzero = np.abs(k) > 1e-10
    print(f"  k{i}: nonzero entries at dims {list(np.where(nonzero)[0])}")
    unique_abs = np.unique(np.round(np.abs(k[nonzero]), 6))
    print(f"       unique |entries|: {unique_abs}")

# Check: does kernel align with contracted/stable/expanded dims?
print(f"\nColumn Trichotomy vs kernel:")
for name, dims in COLUMN_TRICHOTOMY.items():
    projections = kernel_basis[:, dims]
    norms = np.linalg.norm(projections, axis=0)
    print(f"  {name} dims {dims}: kernel projections norms = {norms}")


# ============================================================
# 2. U^T @ U EIGENSTRUCTURE (Round-trip operator)
# ============================================================

section("2. ROUND-TRIP OPERATOR U^T @ U")

UTU = PHILLIPS_MATRIX.T @ PHILLIPS_MATRIX
print(f"U^T U shape: {UTU.shape}")
print(f"U^T U matrix:")
print(UTU)

eigenvalues, eigenvectors = np.linalg.eigh(UTU)
print(f"\nEigenvalues of U^T U: {eigenvalues}")
print(f"Sum of eigenvalues: {np.sum(eigenvalues):.8f}  (= Frobenius² = 20)")
print(f"Nonzero eigenvalues: {eigenvalues[eigenvalues > 1e-10]}")
print(f"Count nonzero: {np.sum(eigenvalues > 1e-10)}")

print(f"\nEigenvectors (columns):")
for i in range(8):
    ev = eigenvalues[i]
    vec = eigenvectors[:, i]
    print(f"  λ_{i} = {ev:.8f}  v = {vec}")

# Check if nonzero eigenvalues have phi-structure
nonzero_evals = eigenvalues[eigenvalues > 1e-10]
print(f"\nNonzero eigenvalue analysis:")
for ev in nonzero_evals:
    print(f"  λ = {ev:.8f}")
    print(f"    λ/phi   = {ev/PHI:.8f}")
    print(f"    λ/phi²  = {ev/(PHI**2):.8f}")
    print(f"    λ/(3-phi) = {ev/(3-PHI):.8f}")
    print(f"    λ/(phi+2) = {ev/(PHI+2):.8f}")
    print(f"    λ/5     = {ev/5:.8f}")

# U_L^T @ U_L specifically
ULtUL = PHILLIPS_U_L.T @ PHILLIPS_U_L
print(f"\nU_L^T @ U_L eigenvalues: {np.linalg.eigvalsh(ULtUL)}")
print(f"U_L^T @ U_L trace: {np.trace(ULtUL):.8f}  (should = 4*(3-phi) = {4*(3-PHI):.8f})")

URtUR = PHILLIPS_U_R.T @ PHILLIPS_U_R
print(f"\nU_R^T @ U_R eigenvalues: {np.linalg.eigvalsh(URtUR)}")
print(f"U_R^T @ U_R trace: {np.trace(URtUR):.8f}  (should = 4*(phi+2) = {4*(PHI+2):.8f})")

# Cross term
ULtUR = PHILLIPS_U_L.T @ PHILLIPS_U_R
print(f"\nU_L^T @ U_R eigenvalues: {np.linalg.eigvalsh(ULtUR.T @ ULtUR)}")
print(f"Is U_L^T @ U_R = phi * U_L^T @ U_L? ")
diff = ULtUR - PHI * ULtUL
print(f"  Max diff: {np.max(np.abs(diff)):.2e}")


# ============================================================
# 3. COLLISION PAIRS vs 24-CELL 5-COMPOUND
# ============================================================

section("3. COLLISION PAIRS vs 600-CELL / 24-CELL STRUCTURE")

roots = generate_e8_roots()

# Find collisions in U_L projection
left_verts = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
rounded = [tuple(np.round(v, 8)) for v in left_verts]

# Build collision map
from collections import defaultdict
collision_map = defaultdict(list)
for idx, key in enumerate(rounded):
    collision_map[key].append(idx)

collision_pairs = [(idxs[0], idxs[1]) for idxs in collision_map.values() if len(idxs) == 2]
print(f"Total collision pairs: {len(collision_pairs)}")

# Characterize each collision pair
print(f"\nCollision pair analysis:")
for i, (a, b) in enumerate(collision_pairs):
    ra, rb = roots[a], roots[b]
    diff = ra.coordinates - rb.coordinates
    print(f"\n  Pair {i+1}: root[{a}] ({ra.root_type.value}) vs root[{b}] ({rb.root_type.value})")
    print(f"    root_a = {ra.coordinates}")
    print(f"    root_b = {rb.coordinates}")
    print(f"    diff   = {diff}")
    print(f"    diff nonzero dims: {list(np.where(np.abs(diff) > 1e-10)[0])}")

# Now check: what 24-cell does each colliding root belong to?
# The 600-cell has 120 vertices that decompose into 5 copies of the 24-cell
# The 24-cell has 24 vertices. We need to identify which 24-cell each projected
# vertex falls into.

# For the 24-cell 5-compound, we use the trilatic decomposition knowledge:
# A 24-cell decomposes into 3 16-cells (alpha, beta, gamma)
# The 600-cell has 5 inscribed 24-cells.

# Let's look at this from the E8 root type perspective
perm_roots_in_collisions = []
half_roots_in_collisions = []
for a, b in collision_pairs:
    for idx in [a, b]:
        if roots[idx].root_type == E8RootType.PERMUTATION:
            perm_roots_in_collisions.append(idx)
        else:
            half_roots_in_collisions.append(idx)

print(f"\nRoots involved in collisions:")
print(f"  Permutation roots: {len(perm_roots_in_collisions)} (out of 112)")
print(f"  Half-integer roots: {len(half_roots_in_collisions)} (out of 128)")

# Check the inner product structure of collision pairs
print(f"\nInner products within collision pairs:")
for i, (a, b) in enumerate(collision_pairs):
    ip = np.dot(roots[a].coordinates, roots[b].coordinates)
    print(f"  Pair {i+1}: <r_a, r_b> = {ip:.8f}")

# Check the projected 4D radii of collision pairs
print(f"\nProjected left radii of collision pairs:")
for i, (a, b) in enumerate(collision_pairs):
    ra = np.linalg.norm(left_verts[a])
    rb = np.linalg.norm(left_verts[b])
    print(f"  Pair {i+1}: r_a = {ra:.8f}, r_b = {rb:.8f}")


# ============================================================
# 4. AMPLIFICATION FACTOR: Frobenius²/rank = 5 = #24-cells
# ============================================================

section("4. AMPLIFICATION FACTOR")

frob_sq = float(np.sum(PHILLIPS_MATRIX ** 2))
rank = np.linalg.matrix_rank(PHILLIPS_MATRIX)
amp = frob_sq / rank

print(f"Frobenius² = {frob_sq:.8f}")
print(f"Rank = {rank}")
print(f"Amplification = Frobenius²/rank = {amp:.8f}")
print(f"Number of 24-cells in 600-cell = 5")
print(f"Match: {abs(amp - 5.0) < 1e-10}")

# Break down by blocks
frob_L = float(np.sum(PHILLIPS_U_L ** 2))
frob_R = float(np.sum(PHILLIPS_U_R ** 2))
print(f"\nBlock decomposition:")
print(f"  ||U_L||²_F = {frob_L:.8f}  = 4*(3-phi) = {4*(3-PHI):.8f}")
print(f"  ||U_R||²_F = {frob_R:.8f}  = 4*(phi+2) = {4*(PHI+2):.8f}")
print(f"  Sum = {frob_L + frob_R:.8f} = 20")
print(f"  ||U_L||²_F * phi² = {frob_L * PHI**2:.8f}  (should = ||U_R||²_F = {frob_R:.8f})")

# The 5 = amplification interpretation:
# Each E8 root's energy is amplified by 5x through the round-trip
print(f"\nRound-trip energy test on E8 roots:")
energies = []
for r in roots[:10]:
    v = r.coordinates
    UTUv = PHILLIPS_MATRIX.T @ (PHILLIPS_MATRIX @ v)
    input_energy = np.dot(v, v)
    output_energy = np.dot(v, UTUv)
    ratio = output_energy / input_energy
    energies.append(ratio)
    print(f"  root norm²={input_energy:.1f}  <v, U^TUv>/<v,v> = {ratio:.8f}")

# Is the ratio constant across all roots? Or does it vary?
print(f"\nFull 240-root energy amplification:")
all_ratios = []
for r in roots:
    v = r.coordinates
    UTUv = UTU @ v
    ratio = np.dot(v, UTUv) / np.dot(v, v)
    all_ratios.append(ratio)
all_ratios = np.array(all_ratios)
print(f"  min = {all_ratios.min():.8f}")
print(f"  max = {all_ratios.max():.8f}")
print(f"  mean = {all_ratios.mean():.8f}")
print(f"  unique values: {np.unique(np.round(all_ratios, 6))}")


# ============================================================
# 5. CHIRALITY: U vs U^T
# ============================================================

section("5. CHIRALITY ANALYSIS")

UUT = PHILLIPS_MATRIX @ PHILLIPS_MATRIX.T
print(f"U @ U^T (8×8):")
print(UUT)
print(f"\nEigenvalues of U @ U^T: {np.linalg.eigvalsh(UUT)}")

# Check if U is normal (U^TU = UU^T means normal)
diff_normal = UTU - UUT
print(f"\nU^TU - UU^T max diff: {np.max(np.abs(diff_normal)):.8e}")
print(f"Is Phillips matrix normal? {np.allclose(UTU, UUT, atol=1e-10)}")

# Asymmetry: U - U^T
asym = PHILLIPS_MATRIX - PHILLIPS_MATRIX.T
print(f"\nU - U^T:")
print(asym)
print(f"||U - U^T||_F = {np.linalg.norm(asym):.8f}")

# Decompose into symmetric + antisymmetric parts
sym = (PHILLIPS_MATRIX + PHILLIPS_MATRIX.T) / 2
antisym = (PHILLIPS_MATRIX - PHILLIPS_MATRIX.T) / 2
print(f"\n||sym(U)||_F = {np.linalg.norm(sym):.8f}")
print(f"||antisym(U)||_F = {np.linalg.norm(antisym):.8f}")
print(f"||sym||² + ||antisym||² = {np.sum(sym**2) + np.sum(antisym**2):.8f}  (should = 20)")

# Eigenvalues of the full matrix (may be complex since not symmetric)
evals_full = np.linalg.eigvals(PHILLIPS_MATRIX)
print(f"\nFull eigenvalues of Phillips matrix (may be complex):")
for ev in evals_full:
    if np.abs(ev.imag) < 1e-10:
        print(f"  {ev.real:.8f}")
    else:
        print(f"  {ev.real:.8f} + {ev.imag:.8f}i  (|λ| = {abs(ev):.8f})")


# ============================================================
# 6. KERNEL LATTICE IDENTIFICATION
# ============================================================

section("6. KERNEL LATTICE STRUCTURE")

# The kernel of U_L is a 4D sublattice of Z^8/2.
# Check if it's a known lattice: D4, A4, or scaled version.

# First, compute inner products of kernel basis vectors
print(f"Kernel basis inner products:")
for i in range(len(kernel_basis)):
    for j in range(i, len(kernel_basis)):
        ip = np.dot(kernel_basis[i], kernel_basis[j])
        print(f"  <k{i}, k{j}> = {ip:.8f}")

# Check if the kernel contains any E8 roots
print(f"\nDoes the kernel contain any E8 roots?")
kernel_proj = np.array([PHILLIPS_U_L @ r.coordinates for r in roots])
kernel_roots = []
for idx, r in enumerate(roots):
    if np.linalg.norm(kernel_proj[idx]) < 1e-10:
        kernel_roots.append(idx)
        print(f"  Root {idx}: {r.coordinates} ({r.root_type.value}) → {kernel_proj[idx]}")

if not kernel_roots:
    print(f"  No E8 roots lie in the kernel.")

# Check which E8 roots have small projections
print(f"\nSmallest U_L projection norms:")
proj_norms = np.linalg.norm(kernel_proj, axis=1)
sorted_idx = np.argsort(proj_norms)
for idx in sorted_idx[:10]:
    print(f"  root[{idx}] norm={proj_norms[idx]:.8f} type={roots[idx].root_type.value}")

# Check if the kernel basis vectors, when doubled, form a D4 lattice
# D4 root system: vectors with 2 nonzero ±1 entries in 4D
print(f"\nKernel basis vectors scaled to check for lattice:")
for i, k in enumerate(kernel_basis):
    # Check various scalings
    for scale_name, scale in [("x1", 1), ("x√2", np.sqrt(2)), ("x2", 2)]:
        scaled = k * scale
        rounded_entries = np.round(scaled, 6)
        is_integer = np.allclose(rounded_entries, np.round(rounded_entries), atol=1e-4)
        print(f"  k{i} * {scale_name} = {rounded_entries}  {'INTEGER!' if is_integer else ''}")


# ============================================================
# 7. COLLISION DIFFERENCE VECTORS and KERNEL
# ============================================================

section("7. COLLISION DIFFERENCES vs KERNEL")

print("Are collision difference vectors in the kernel of U_L?")
for i, (a, b) in enumerate(collision_pairs):
    diff = roots[a].coordinates - roots[b].coordinates
    proj_diff = PHILLIPS_U_L @ diff
    in_kernel = np.linalg.norm(proj_diff) < 1e-10
    print(f"  Pair {i+1}: ||U_L @ diff|| = {np.linalg.norm(proj_diff):.2e}  {'IN KERNEL' if in_kernel else 'NOT in kernel'}")
    print(f"           diff = {diff}")

# Express collision diffs in terms of kernel basis
print(f"\nCollision differences expressed in kernel basis:")
for i, (a, b) in enumerate(collision_pairs):
    diff = roots[a].coordinates - roots[b].coordinates
    # Solve: diff ≈ kernel_basis.T @ coeffs
    coeffs, residual, _, _ = np.linalg.lstsq(kernel_basis.T, diff, rcond=None)
    recon = kernel_basis.T @ coeffs
    err = np.linalg.norm(diff - recon)
    print(f"  Pair {i+1}: coeffs = {coeffs}  residual = {err:.2e}")

# Do the collision diffs form a sublattice of the kernel?
print(f"\nCollision diff inner product matrix:")
diffs = []
for a, b in collision_pairs:
    diffs.append(roots[a].coordinates - roots[b].coordinates)
diffs = np.array(diffs)
diff_gram = diffs @ diffs.T
print(f"  Shape: {diff_gram.shape}")
print(f"  Rank: {np.linalg.matrix_rank(diff_gram, tol=1e-8)}")
print(f"  Eigenvalues: {np.linalg.eigvalsh(diff_gram)}")

# Unique difference vectors (up to sign)
unique_diffs = []
for d in diffs:
    is_dup = False
    for ud in unique_diffs:
        if np.allclose(d, ud, atol=1e-10) or np.allclose(d, -ud, atol=1e-10):
            is_dup = True
            break
    if not is_dup:
        unique_diffs.append(d)
print(f"\nUnique diff vectors (up to sign): {len(unique_diffs)}")
for i, d in enumerate(unique_diffs):
    print(f"  d{i}: {d}  norm={np.linalg.norm(d):.8f}")


# ============================================================
# 8. COLUMN SPACE ANALYSIS
# ============================================================

section("8. COLUMN SPACE / IMAGE ANALYSIS")

# The image of U_L is a 4D subspace of R^4 (it IS R^4 since rank=4)
# But the image of the FULL Phillips matrix is a 4D subspace of R^8
# Let's characterize it

_, S_full, _ = np.linalg.svd(PHILLIPS_MATRIX)
print(f"Singular values of full Phillips 8×8: {S_full}")

# Column space of U_L
U_L_col, S_L, _ = np.linalg.svd(PHILLIPS_U_L)
print(f"\nLeft singular vectors of U_L (columns of U):")
for i in range(4):
    print(f"  u{i} (σ={S_L[i]:.8f}): {U_L_col[:, i]}")

# Is U_L orthogonal? U_L @ U_L^T = ?
UL_ULT = PHILLIPS_U_L @ PHILLIPS_U_L.T
print(f"\nU_L @ U_L^T:")
print(UL_ULT)
print(f"Is U_L @ U_L^T = (3-phi) * I_4? max diff = {np.max(np.abs(UL_ULT - (3-PHI)*np.eye(4))):.8e}")
# If this is true, U_L has orthogonal rows (but not orthonormal)

UR_URT = PHILLIPS_U_R @ PHILLIPS_U_R.T
print(f"\nU_R @ U_R^T:")
print(UR_URT)
print(f"Is U_R @ U_R^T = (phi+2) * I_4? max diff = {np.max(np.abs(UR_URT - (PHI+2)*np.eye(4))):.8e}")

# Cross block
UL_URT = PHILLIPS_U_L @ PHILLIPS_U_R.T
print(f"\nU_L @ U_R^T:")
print(UL_URT)
print(f"Is U_L @ U_R^T = phi*(3-phi) * I_4?")
expected = PHI * (3 - PHI)
print(f"  phi*(3-phi) = {expected:.8f}")
print(f"  max diff = {np.max(np.abs(UL_URT - expected*np.eye(4))):.8e}")


print(f"\n{'='*72}")
print(f"  EXPLORATION COMPLETE")
print(f"{'='*72}")
