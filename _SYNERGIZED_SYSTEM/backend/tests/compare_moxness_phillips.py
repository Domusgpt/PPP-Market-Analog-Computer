"""
Definitive comparison: Moxness C600 vs Phillips matrix.

Checks every reasonable transformation to determine if one
can be derived from the other.
"""
import numpy as np

PHI = (1 + np.sqrt(5)) / 2

# =====================================================================
# MOXNESS C600 — exact entries from viXra:1411.0130v1 (2014)
# =====================================================================
C600 = np.array([
    [PHI**2,    0,     0,     0,   1/PHI,    0,     0,     0  ],
    [   0,      1,   PHI,     0,      0,    -1,   PHI,     0  ],
    [   0,    PHI,     0,     1,      0,   PHI,     0,    -1  ],
    [   0,      0,     1,   PHI,      0,     0,    -1,   PHI  ],
    [1/PHI,    0,     0,     0,   PHI**2,    0,     0,     0  ],
    [   0,     -1,   PHI,     0,      0,     1,   PHI,     0  ],
    [   0,    PHI,     0,    -1,      0,   PHI,     0,     1  ],
    [   0,      0,    -1,   PHI,      0,     0,     1,   PHI  ],
])

# =====================================================================
# PHILLIPS — from our e8_projection.py
# =====================================================================
_a = 0.5
_b = (PHI - 1) / 2
_c = PHI / 2

PHILLIPS = np.array([
    [ _a,  _b,  _a,  _b,  _a, -_b,  _a, -_b],
    [ _a,  _a, -_b, -_b, -_a, -_a,  _b,  _b],
    [ _a, -_b, -_a,  _b,  _a, -_b, -_a,  _b],
    [ _a, -_a,  _b, -_b, -_a,  _a, -_b,  _b],
    [ _c,  _a,  _c,  _a,  _c, -_a,  _c, -_a],
    [ _c,  _c, -_a, -_a, -_c, -_c,  _a,  _a],
    [ _c, -_a, -_c,  _a,  _c, -_a, -_c,  _a],
    [ _c, -_c,  _a, -_a, -_c,  _c, -_a,  _a],
])

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("="*72)
print("MOXNESS C600 (2014 viXra)")
print("="*72)
print(C600)
print(f"\nSymmetric: {np.allclose(C600, C600.T)}")
print(f"Rank: {np.linalg.matrix_rank(C600)}")
print(f"Det: {np.linalg.det(C600):.4f}")
print(f"Frobenius²: {np.sum(C600**2):.6f}")
print(f"Trace: {np.trace(C600):.6f}")
print(f"Nonzero entries: {np.sum(np.abs(C600) > 1e-10)}/64")
print(f"Unique |entries|: {sorted(np.unique(np.round(np.abs(C600[C600 != 0]), 8)))}")

print(f"\n{'='*72}")
print("PHILLIPS (our code)")
print("="*72)
print(PHILLIPS)
print(f"\nSymmetric: {np.allclose(PHILLIPS, PHILLIPS.T)}")
print(f"Rank: {np.linalg.matrix_rank(PHILLIPS)}")
print(f"Det: {np.linalg.det(PHILLIPS):.4f}")
print(f"Frobenius²: {np.sum(PHILLIPS**2):.6f}")
print(f"Trace: {np.trace(PHILLIPS):.6f}")
print(f"Nonzero entries: {np.sum(np.abs(PHILLIPS) > 1e-10)}/64")
print(f"Unique |entries|: {sorted(np.unique(np.round(np.abs(PHILLIPS), 8)))}")

# =====================================================================
# CHECK 1: Direct equality
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 1: Are they the same matrix?")
print("="*72)
print(f"Max difference: {np.max(np.abs(C600 - PHILLIPS)):.6f}")
print(f"Same? NO" if not np.allclose(C600, PHILLIPS) else "Same? YES")

# =====================================================================
# CHECK 2: Scalar multiples
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 2: Is Phillips = alpha * C600 for some alpha?")
print("="*72)

# Can't be a scalar multiple if one is sparse and other is dense
print(f"C600 has {np.sum(np.abs(C600) < 1e-10)} zeros")
print(f"Phillips has {np.sum(np.abs(PHILLIPS) < 1e-10)} zeros")
print("Different sparsity patterns → cannot be scalar multiples")

# Try C600/2
C600_half = C600 / 2
print(f"\nC600/2 entries: {sorted(np.unique(np.round(np.abs(C600_half[np.abs(C600_half) > 1e-10]), 8)))}")
print(f"Phillips entries: {sorted(np.unique(np.round(np.abs(PHILLIPS), 8)))}")
print(f"C600/2 == Phillips? {np.allclose(C600_half, PHILLIPS)}")

# Unimodular: C600 / (2*sqrt(phi))
C600_uni = C600 / (2 * np.sqrt(PHI))
print(f"\nC600/(2√φ) entries: {sorted(np.unique(np.round(np.abs(C600_uni[np.abs(C600_uni) > 1e-10]), 8)))}")
print(f"C600/(2√φ) == Phillips? {np.allclose(C600_uni, PHILLIPS)}")

# =====================================================================
# CHECK 3: Do they project E8 roots to the same point sets?
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 3: Do they produce the same projected point cloud?")
print("="*72)

from engine.geometry.e8_projection import generate_e8_roots
roots = generate_e8_roots()
coords = np.array([r.coordinates for r in roots])

# Project through top 4 rows of each
C600_top4 = C600[:4]
PHIL_top4 = PHILLIPS[:4]

proj_C600 = coords @ C600_top4.T  # (240, 4)
proj_PHIL = coords @ PHIL_top4.T  # (240, 4)

# Normalize both to unit sphere and compare
def normalize_cloud(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.where(norms > 1e-10, norms, 1)

proj_C600_unit = normalize_cloud(proj_C600)
proj_PHIL_unit = normalize_cloud(proj_PHIL)

# Count unique points in each
unique_C600 = len(set(tuple(np.round(v, 6)) for v in proj_C600))
unique_PHIL = len(set(tuple(np.round(v, 6)) for v in proj_PHIL))
print(f"Unique projections via C600 top-4: {unique_C600}")
print(f"Unique projections via Phillips top-4: {unique_PHIL}")

# Check if one cloud is a rotation/scaling of the other
# Compute radius distributions
radii_C600 = sorted(np.unique(np.round(np.linalg.norm(proj_C600, axis=1), 6)))
radii_PHIL = sorted(np.unique(np.round(np.linalg.norm(proj_PHIL, axis=1), 6)))
print(f"\nC600 top-4 shell radii ({len(radii_C600)}): {radii_C600}")
print(f"Phillips top-4 shell radii ({len(radii_PHIL)}): {radii_PHIL}")

# =====================================================================
# CHECK 4: Row space comparison
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 4: Do they span the same row space?")
print("="*72)

# If they span the same 4D subspace, one is a linear transform of the other
combined = np.vstack([C600_top4, PHIL_top4])  # (8, 8)
rank_combined = np.linalg.matrix_rank(combined, tol=1e-8)
print(f"Rank of [C600_top4; Phillips_top4] stacked: {rank_combined}")
print(f"  If = 4: same row space (one is a 4x4 transform of the other)")
print(f"  If > 4: different row spaces")

if rank_combined == 4:
    # Find the 4x4 transform: Phillips_top4 = M @ C600_top4
    M = PHIL_top4 @ np.linalg.pinv(C600_top4)
    print(f"\nTransform M such that Phillips = M @ C600:")
    print(M)
    print(f"Verify: max residual = {np.max(np.abs(PHIL_top4 - M @ C600_top4)):.2e}")

# =====================================================================
# CHECK 5: Full 8x8 row space comparison
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 5: Full 8x8 comparison")
print("="*72)

combined_full = np.vstack([C600, PHILLIPS])  # (16, 8)
rank_full = np.linalg.matrix_rank(combined_full, tol=1e-8)
print(f"Rank of [C600; Phillips] stacked: {rank_full}")
print(f"C600 rank: {np.linalg.matrix_rank(C600)}, Phillips rank: {np.linalg.matrix_rank(PHILLIPS)}")

# =====================================================================
# CHECK 6: Eigenvalue comparison
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 6: Eigenvalue comparison")
print("="*72)

evals_C = sorted(np.linalg.eigvalsh(C600))  # symmetric so real
evals_P = sorted(np.real(np.linalg.eigvals(PHILLIPS)))
print(f"C600 eigenvalues:    {[f'{v:.6f}' for v in evals_C]}")
print(f"Phillips eigenvalues: {[f'{v:.6f}' for v in evals_P]}")

# =====================================================================
# CHECK 7: Does the BAEZ matrix relate to C600?
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 7: Our Baez 4x8 vs Moxness C600 top-4 rows")
print("="*72)

from engine.geometry.e8_projection import BAEZ_MATRIX

print(f"Baez matrix (4x8):")
print(BAEZ_MATRIX)
print(f"\nC600 top 4 rows:")
print(C600_top4)
print(f"\nC600 top 4 / 2:")
print(C600_top4 / 2)
print(f"\nBaez == C600/2 top-4? {np.allclose(BAEZ_MATRIX, C600_top4/2)}")

# Check row by row
for i in range(4):
    match = np.allclose(BAEZ_MATRIX[i], C600_top4[i]/2)
    print(f"  Row {i}: {'MATCH' if match else 'DIFFER'}")
    if not match:
        print(f"    Baez:  {BAEZ_MATRIX[i]}")
        print(f"    C600/2: {C600_top4[i]/2}")

# =====================================================================
# CHECK 8: Moxness bottom-4 vs top-4 relationship
# =====================================================================
print(f"\n{'='*72}")
print("CHECK 8: Does Moxness have U_R = phi * U_L?")
print("="*72)

C600_bot4 = C600[4:]
ratio = C600_bot4 / np.where(np.abs(C600_top4) > 1e-10, C600_top4, 1)
print(f"C600 bottom/top ratio (nonzero entries):")
nonzero_mask = np.abs(C600_top4) > 1e-10
ratios = C600_bot4[nonzero_mask] / C600_top4[nonzero_mask]
print(f"  Unique ratios: {np.unique(np.round(ratios, 6))}")
print(f"  Is C600_bot = phi * C600_top? {np.allclose(C600_bot4, PHI * C600_top4)}")
print(f"  Is C600_bot = C600_top? (symmetric blocks?) Check...")
print(f"  Max diff C600_bot - C600_top: {np.max(np.abs(C600_bot4 - C600_top4)):.6f}")

# What IS the relationship between C600 blocks?
print(f"\nC600 top-left 4x4:")
print(C600[:4, :4])
print(f"\nC600 top-right 4x4:")
print(C600[:4, 4:])
print(f"\nC600 bottom-left 4x4:")
print(C600[4:, :4])
print(f"\nC600 bottom-right 4x4:")
print(C600[4:, 4:])

print(f"\nTL == BR? {np.allclose(C600[:4,:4], C600[4:,4:])}")
print(f"TR == BL? {np.allclose(C600[:4,4:], C600[4:,:4])}")

print(f"\n{'='*72}")
print("CONCLUSION")
print("="*72)
