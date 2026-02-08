"""
Definitive comparison using Moxness's ACTUAL matrix from his U-analysis.pdf.

This replaces the earlier comparison which used a C600 from the published
viXra paper (which Moxness noted has typos in rows 3&5, columns 4&5).

The PDF shows two forms:
  1. φ𝕌 (unscaled, NOT det=1) — the one we analyze here
  2. 𝕌/Det1f (scaled to Det=1) — the volume-preserving rotation

Both are: Symmetric, Traceless, Invertible, CentroSymmetric
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/PPP-Market-Analog-Computer/_SYNERGIZED_SYSTEM/backend')

PHI = (1 + np.sqrt(5)) / 2

# =====================================================================
# MOXNESS φ𝕌 — EXACT from U-analysis.pdf (corrected, authoritative)
# =====================================================================
# Symbolic entries: {0, ±1, ±φ, ±1/φ, ±φ²}
MOXNESS_PHI_U = np.array([
    [-1/PHI,    0,     0,     0,     0,     0,     0,  -PHI**2],
    [   0,     -1,   PHI,     0,     0,   PHI,     1,     0   ],
    [   0,    PHI,     0,    -1,     1,     0,   PHI,     0   ],
    [   0,      0,    -1,   PHI,   PHI,     1,     0,     0   ],
    [   0,      0,     1,   PHI,   PHI,    -1,     0,     0   ],
    [   0,    PHI,     0,     1,    -1,     0,   PHI,     0   ],
    [   0,      1,   PHI,     0,     0,   PHI,    -1,     0   ],
    [-PHI**2,   0,     0,     0,     0,     0,     0,  -1/PHI ],
])

# OLD C600 from viXra:1411.0130v1 (may have typos per Moxness)
OLD_C600 = np.array([
    [PHI**2,    0,     0,     0,   1/PHI,    0,     0,     0  ],
    [   0,      1,   PHI,     0,      0,    -1,   PHI,     0  ],
    [   0,    PHI,     0,     1,      0,   PHI,     0,    -1  ],
    [   0,      0,     1,   PHI,      0,     0,    -1,   PHI  ],
    [1/PHI,    0,     0,     0,   PHI**2,    0,     0,     0  ],
    [   0,     -1,   PHI,     0,      0,     1,   PHI,     0  ],
    [   0,    PHI,     0,    -1,      0,   PHI,     0,     1  ],
    [   0,      0,    -1,   PHI,      0,     0,     1,   PHI  ],
])

# PHILLIPS — from e8_projection.py
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

# =====================================================================
# SECTION 1: Verify the PDF matrix properties that Moxness checks
# =====================================================================
print("=" * 72)
print("MOXNESS φ𝕌 (from U-analysis.pdf) — Property Verification")
print("=" * 72)
M = MOXNESS_PHI_U
print(f"Symmetric (m = mᵀ):          {np.allclose(M, M.T)}")
print(f"Traceless (Tr = 0):           {np.isclose(np.trace(M), 0)} (trace = {np.trace(M):.10f})")
print(f"Invertible (det ≠ 0):         {not np.isclose(np.linalg.det(M), 0)} (det = {np.linalg.det(M):.6f})")
print(f"Rank:                         {np.linalg.matrix_rank(M)}")

# Volume preserving = |det| = 1
det_M = np.linalg.det(M)
print(f"VolPreserving (|det| = 1):    {np.isclose(abs(det_M), 1)} (|det| = {abs(det_M):.6f})")

# CentroSymmetric: AJ = JA (where J is the reversal matrix)
J = np.fliplr(np.eye(8))
print(f"CentroSymmetric (AJ = JA):    {np.allclose(M @ J, J @ M)}")
print(f"SkewCentroSymm (AJ = -JA):    {np.allclose(M @ J, -J @ M)}")

# Orthogonal: m.mᵀ = I
print(f"Orthogonal (m·mᵀ = I):        {np.allclose(M @ M.T, np.eye(8))}")

# Eigenvalues
evals_M = sorted(np.linalg.eigvalsh(M))
print(f"Eigenvalues: {[f'{v:.6f}' for v in evals_M]}")
print(f"Sum of eigenvalues:           {sum(evals_M):.10f}")

# =====================================================================
# SECTION 2: Compare PDF matrix vs our OLD C600
# =====================================================================
print(f"\n{'=' * 72}")
print("PDF φ𝕌 vs OLD C600 (from viXra paper)")
print("=" * 72)
print(f"Same matrix? {np.allclose(M, OLD_C600)}")
print(f"Max difference: {np.max(np.abs(M - OLD_C600)):.6f}")

# Check if related by permutation or sign change
print(f"\nPDF nonzero pattern (row, col):")
pdf_nz = list(zip(*np.where(np.abs(M) > 1e-10)))
old_nz = list(zip(*np.where(np.abs(OLD_C600) > 1e-10)))
print(f"  PDF: {len(pdf_nz)} nonzero entries")
print(f"  OLD: {len(old_nz)} nonzero entries")
print(f"  Same positions? {set(pdf_nz) == set(old_nz)}")

# Check if OLD C600 has the typos Moxness mentioned (rows 3&5, cols 4&5)
print(f"\nMoxness mentioned typos in rows 3 & 5 / columns 4 & 5:")
print(f"  PDF row 3 (0-indexed 2), cols 3-4: {M[2, 3]:.3f}, {M[2, 4]:.3f}")
print(f"  OLD row 3 (0-indexed 2), cols 3-4: {OLD_C600[2, 3]:.3f}, {OLD_C600[2, 4]:.3f}")
print(f"  PDF row 5 (0-indexed 4), cols 3-4: {M[4, 3]:.3f}, {M[4, 4]:.3f}")
print(f"  OLD row 5 (0-indexed 4), cols 3-4: {OLD_C600[4, 3]:.3f}, {OLD_C600[4, 4]:.3f}")

# Check if they're related by a column/row permutation
print(f"\nSymmetry comparison:")
print(f"  PDF: Symmetric={np.allclose(M, M.T)}, Det={np.linalg.det(M):.4f}")
print(f"  OLD: Symmetric={np.allclose(OLD_C600, OLD_C600.T)}, Det={np.linalg.det(OLD_C600):.4f}")

# =====================================================================
# SECTION 3: Moxness φ𝕌 (PDF) vs Phillips — definitive comparison
# =====================================================================
print(f"\n{'=' * 72}")
print("DEFINITIVE: Moxness φ𝕌 (PDF, corrected) vs Phillips")
print("=" * 72)

P = PHILLIPS

print(f"\n--- Basic Properties ---")
props = [
    ("Size", f"{M.shape}", f"{P.shape}"),
    ("Rank", f"{np.linalg.matrix_rank(M)}", f"{np.linalg.matrix_rank(P)}"),
    ("Symmetric", f"{np.allclose(M, M.T)}", f"{np.allclose(P, P.T)}"),
    ("Determinant", f"{np.linalg.det(M):.4f}", f"{np.linalg.det(P):.6f}"),
    ("Trace", f"{np.trace(M):.6f}", f"{np.trace(P):.6f}"),
    ("Frobenius²", f"{np.sum(M**2):.6f}", f"{np.sum(P**2):.6f}"),
    ("Zero entries", f"{np.sum(np.abs(M) < 1e-10)}/64", f"{np.sum(np.abs(P) < 1e-10)}/64"),
]
print(f"{'Property':<20} {'Moxness φ𝕌':<25} {'Phillips':<25}")
print("-" * 70)
for name, mv, pv in props:
    print(f"{name:<20} {mv:<25} {pv:<25}")

# Entry values
m_entries = sorted(np.unique(np.round(np.abs(M[np.abs(M) > 1e-10]), 8)))
p_entries = sorted(np.unique(np.round(np.abs(P), 8)))
print(f"\nUnique |entry| values:")
print(f"  Moxness: {m_entries}")
print(f"  Phillips: {p_entries}")
print(f"  Moxness symbolic: {{1/φ={1/PHI:.6f}, 1={1:.6f}, φ={PHI:.6f}, φ²={PHI**2:.6f}}}")
print(f"  Phillips symbolic: {{(φ-1)/2={_b:.6f}, 1/2={_a:.6f}, φ/2={_c:.6f}}}")

# =====================================================================
# SECTION 4: Block structure comparison
# =====================================================================
print(f"\n{'=' * 72}")
print("Block Structure Analysis")
print("=" * 72)

# Moxness: check his block properties
M_TL = M[:4, :4]
M_TR = M[:4, 4:]
M_BL = M[4:, :4]
M_BR = M[4:, 4:]

print(f"\n--- Moxness φ𝕌 2x2 block structure ---")
print(f"TL == BR?  {np.allclose(M_TL, M_BR)}  (symmetric → yes)")
print(f"TR == BL?  {np.allclose(M_TR, M_BL)}  (symmetric → yes)")

# What's the top-bottom relationship for Moxness?
M_top = M[:4]
M_bot = M[4:]

# Check if bottom rows are related to top by reversal + sign change
J4 = np.fliplr(np.eye(8))  # 8x8 column reversal
print(f"\nM_bot == M_top @ J (column reversal)?  {np.allclose(M_bot, M_top @ J4)}")
print(f"M_bot == -M_top (negation)?             {np.allclose(M_bot, -M_top)}")

# Check CentroSymmetric relationship: rows reverse
J8 = np.fliplr(np.eye(8))
M_row_rev = M[::-1, :]
print(f"Row-reversed M == M?                     {np.allclose(M_row_rev, M)}")
M_col_rev = M[:, ::-1]
print(f"Col-reversed M == M?                     {np.allclose(M_col_rev, M)}")
print(f"180° rotated M == M (centrosymmetric)?   {np.allclose(M[::-1, ::-1], M)}")

# Phillips: check U_R = φ·U_L
P_top = P[:4]
P_bot = P[4:]
print(f"\n--- Phillips block structure ---")
print(f"U_R = φ·U_L?  {np.allclose(P_bot, PHI * P_top)}")
print(f"Max error:     {np.max(np.abs(P_bot - PHI * P_top)):.2e}")

# Moxness: does U_R = φ·U_L hold?
print(f"\n--- Does Moxness φ𝕌 have U_R = φ·U_L? ---")
print(f"M_bot = φ * M_top?  {np.allclose(M_bot, PHI * M_top)}")
print(f"Max error:          {np.max(np.abs(M_bot - PHI * M_top)):.6f}")

# What IS the bottom-to-top relationship in Moxness?
print(f"\nMoxness top 4 rows:")
print(M_top)
print(f"\nMoxness bottom 4 rows:")
print(M_bot)
print(f"\nMoxness top rows 2-4 vs bottom rows 5-7 (inner 6x6):")
print(f"  Row 2 (idx 1): {M[1]}")
print(f"  Row 7 (idx 6): {M[6]}")
print(f"  Diff:          {M[1] - M[6]}")
print(f"  Sum:           {M[1] + M[6]}")
print(f"\n  Row 3 (idx 2): {M[2]}")
print(f"  Row 6 (idx 5): {M[5]}")
print(f"  Diff:          {M[2] - M[5]}")
print(f"\n  Row 4 (idx 3): {M[3]}")
print(f"  Row 5 (idx 4): {M[4]}")
print(f"  Diff:          {M[3] - M[4]}")

# =====================================================================
# SECTION 5: E8 root projections
# =====================================================================
print(f"\n{'=' * 72}")
print("E8 Root Projections")
print("=" * 72)

from engine.geometry.e8_projection import generate_e8_roots
roots = generate_e8_roots()
coords = np.array([r.coordinates for r in roots])

# Project through FULL 8x8 of each
proj_M = coords @ M.T   # (240, 8) for Moxness
proj_P = coords @ P.T   # (240, 8) for Phillips

print(f"\n--- Full 8x8 projection ---")
print(f"Moxness output unique points: {len(set(tuple(np.round(v, 6)) for v in proj_M))}")
print(f"Phillips output unique points: {len(set(tuple(np.round(v, 6)) for v in proj_P))}")

# For Moxness: top-4 rows give L copy, bottom-4 give R copy
M_L = coords @ M[:4].T  # (240, 4)
M_R = coords @ M[4:].T  # (240, 4)
P_L = coords @ P[:4].T  # (240, 4)
P_R = coords @ P[4:].T  # (240, 4)

print(f"\n--- 4D projections (top-4 = Left, bottom-4 = Right) ---")
unique_ML = len(set(tuple(np.round(v, 6)) for v in M_L))
unique_MR = len(set(tuple(np.round(v, 6)) for v in M_R))
unique_PL = len(set(tuple(np.round(v, 6)) for v in P_L))
unique_PR = len(set(tuple(np.round(v, 6)) for v in P_R))
print(f"Moxness Left unique:   {unique_ML}")
print(f"Moxness Right unique:  {unique_MR}")
print(f"Phillips Left unique:  {unique_PL}")
print(f"Phillips Right unique: {unique_PR}")

# Shell radii
radii_ML = sorted(np.unique(np.round(np.linalg.norm(M_L, axis=1), 6)))
radii_MR = sorted(np.unique(np.round(np.linalg.norm(M_R, axis=1), 6)))
radii_PL = sorted(np.unique(np.round(np.linalg.norm(P_L, axis=1), 6)))
radii_PR = sorted(np.unique(np.round(np.linalg.norm(P_R, axis=1), 6)))

print(f"\nShell radii:")
print(f"  Moxness Left  ({len(radii_ML)} shells): {radii_ML}")
print(f"  Moxness Right ({len(radii_MR)} shells): {radii_MR}")
print(f"  Phillips Left ({len(radii_PL)} shells): {radii_PL}")
print(f"  Phillips Right ({len(radii_PR)} shells): {radii_PR}")

# Check φ-scaling between L and R
print(f"\n--- φ-scaling between L and R copies ---")
ML_norms = np.linalg.norm(M_L, axis=1)
MR_norms = np.linalg.norm(M_R, axis=1)
PL_norms = np.linalg.norm(P_L, axis=1)
PR_norms = np.linalg.norm(P_R, axis=1)

# Phillips: ratio should be exactly φ
P_ratios = PR_norms / np.where(PL_norms > 1e-10, PL_norms, 1)
M_ratios = MR_norms / np.where(ML_norms > 1e-10, ML_norms, 1)

print(f"Phillips R/L ratios: {sorted(np.unique(np.round(P_ratios, 6)))}")
print(f"  All exactly φ? {np.allclose(P_ratios, PHI)}")

print(f"Moxness R/L ratios: {sorted(np.unique(np.round(M_ratios, 6))[:10])}...")
print(f"  All exactly φ? {np.allclose(M_ratios, PHI)}")
print(f"  Unique ratio count: {len(np.unique(np.round(M_ratios, 6)))}")

# =====================================================================
# SECTION 6: Row space overlap
# =====================================================================
print(f"\n{'=' * 72}")
print("Row Space Comparison")
print("=" * 72)

# Do they span the same 4D subspace?
combined_top = np.vstack([M[:4], P[:4]])  # (8, 8)
rank_combined = np.linalg.matrix_rank(combined_top, tol=1e-8)
print(f"Rank of [Moxness_top4; Phillips_top4] stacked: {rank_combined}")
print(f"  = 4 means same row space; > 4 means different subspaces")

# Full 8x8 comparison
combined_full = np.vstack([M, P])  # (16, 8)
rank_full = np.linalg.matrix_rank(combined_full, tol=1e-8)
print(f"Rank of [Moxness_8x8; Phillips_8x8] stacked: {rank_full}")

# =====================================================================
# SECTION 7: Moxness's L/R palindromic structure
# =====================================================================
print(f"\n{'=' * 72}")
print("Moxness L/R Palindromic Structure (his key insight)")
print("=" * 72)

# He says: "each snub 24-cell left row has a right entry that can be
# found the other table. These L/R entries are palindromically reversed,
# except for +/- sign changes on φ^1/2 elements and inversion on
# φ^(+/-3/2) elements"
print(f"\nFor Moxness, comparing row i with row (7-i) (palindromic partner):")
for i in range(4):
    j = 7 - i
    row_i = M[i]
    row_j = M[j]
    row_j_rev = M[j, ::-1]  # column-reversed
    print(f"\n  Row {i+1}: {np.round(row_i, 4)}")
    print(f"  Row {j+1}: {np.round(row_j, 4)}")
    print(f"  Row {j+1} col-reversed: {np.round(row_j_rev, 4)}")
    print(f"  Row {i+1} == Row {j+1} reversed? {np.allclose(row_i, row_j_rev)}")

# =====================================================================
# SECTION 8: Det=1 normalization
# =====================================================================
print(f"\n{'=' * 72}")
print("Det=1 Normalization")
print("=" * 72)

det_raw = np.linalg.det(M)
scale = abs(det_raw) ** (1/8)
M_det1 = M / scale

print(f"Raw |det| = {abs(det_raw):.6f}")
print(f"Scale factor = |det|^(1/8) = {scale:.6f}")
print(f"Normalized det = {np.linalg.det(M_det1):.10f}")
print(f"Normalized Frobenius² = {np.sum(M_det1**2):.6f}")

# Check unique entry values of normalized version
m_det1_entries = sorted(np.unique(np.round(np.abs(M_det1[np.abs(M_det1) > 1e-10]), 6)))
print(f"Normalized |entry| values: {m_det1_entries}")

# Compare with PDF's Det=1 values
print(f"\nPDF Det=1 inner entries: ±0.393076, ±0.63601")
print(f"Our computed: √(1/φ)/2 = {np.sqrt(1/PHI)/2:.6f}")
print(f"              1/(2√(1/φ)) = {1/(2*np.sqrt(1/PHI)):.6f}")

# =====================================================================
# CONCLUSION
# =====================================================================
print(f"\n{'=' * 72}")
print("CONCLUSION")
print("=" * 72)
print("""
The Moxness φ𝕌 matrix (from U-analysis.pdf) and the Phillips matrix
are CONFIRMED to be entirely different mathematical objects:

  Moxness φ𝕌                          Phillips
  ─────────────────────────────────    ──────────────────────────────
  Symmetric                            Non-symmetric
  Rank 8, Invertible                   Rank 4, Singular
  Traceless (Tr = 0)                   Non-traceless
  CentroSymmetric                      Not centrosymmetric
  Entries: {0, ±1, ±φ, ±1/φ, ±φ²}     Entries: {±1/2, ±(φ-1)/2, ±φ/2}
  Palindromic L/R (row reversal)       U_R = φ · U_L (pure scaling)
  Det ≈ ±42.36 (or ±1 when scaled)    Det = 0
  Multiple R/L norm ratios             Universal R/L ratio = φ

Both project E₈ → H₄, but through fundamentally different mechanisms.
Moxness's is a full-rank rotation; Phillips's is a rank-deficient
golden-ratio projection with spectral properties (eigenvalue 5,
amplification factor 5, single collision vector) that do not arise
in the full-rank case.
""")
