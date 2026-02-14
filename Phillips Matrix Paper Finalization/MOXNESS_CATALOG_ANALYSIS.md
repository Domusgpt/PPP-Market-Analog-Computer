# Complete Moxness Matrix Catalog vs Phillips Matrix
## Definitive Computational Verification — February 14, 2026

---

## Executive Summary

Five distinct Moxness matrices were extracted from primary sources spanning 2013-2023,
plus the corrected phi-U matrix from personal communication (U-analysis.pdf, 2026).
Each was computationally verified and cross-compared with the Phillips Matrix.

**The Phillips Matrix is a fundamentally different mathematical object from every
published Moxness matrix.** They cannot be related by any basis change, share
essentially no null/image space overlap, and produce structurally different
projections of E8 roots.

Critically, **Moxness's own matrices are inconsistent across papers** -- M2 (2014)
and M3 (2018) differ in 28 of 64 entries despite his claims of continuity.

---

## The Moxness Matrix Catalog

### M1: 2013 Blog Post (Original 4x8 Fold)

**Source:** http://theoryofeverything.org/theToE/2013/11/18/1454/

| Property | Value |
|----------|-------|
| Size | 4x8 |
| Rank | 4 |
| Zeros | 18/32 entries |
| Row norms^2 | All 7.236 |
| Frobenius^2 | 28.944 |
| 600-cell test | **CLEAN: 120+120 pts, ratio = phi** |

**This is the ONLY Moxness matrix that produces a clean H4+phi*H4 from its own rows.**

**Key finding: M1/sqrt(2) = Baez matrix exactly.** Null/image space overlap = 4/4.

### M2: 2014 viXra:1411.0130, Eq. (1) -- "H4fold" (8x8)

| Property | Value |
|----------|-------|
| Size | 8x8 |
| Rank | 8 |
| Symmetric | Yes |
| Determinant | 73.889 |
| Trace | 6.854 |
| Zeros | 36/64 |
| Frobenius^2 | 45.489 |
| Eigenvalues | {-3.236, -2.0, 0.191, 1.427, 2.0, 2.0, 3.236, 3.236} |
| 600-cell test (top-4) | 9 norm classes -- NOT clean H4+phi*H4 |

### M3: 2018 viXra:1808.0107, Eq. (3) -- "H4rot" (8x8)

| Property | Value |
|----------|-------|
| Size | 8x8 |
| Rank | 8 |
| Symmetric | Yes |
| Determinant | -38.111 |
| Trace | 4.000 |
| Zeros | 36/64 |
| Frobenius^2 | 22.011 |
| Eigenvalues | {-2.0, -1.236, -1.236, 1.236, 1.309, 1.927, 2.0, 2.0} |
| 600-cell test (top-4) | 9 norm classes -- NOT clean H4+phi*H4 |

### Corrected phi*U: U-analysis.pdf (Moxness, personal communication 2026)

| Property | Value |
|----------|-------|
| Size | 8x8 |
| Rank | 8 |
| Symmetric | Yes |
| Determinant | 1754.650 |
| Trace | 0.000 (traceless) |
| Zeros | 36/64 |
| Frobenius^2 | 57.889 |
| CentroSymmetric | Yes |
| Palindromic | Yes |
| Eigenvalues | {-3.236, -3.236, -2.0, -2.0, 2.0, 2.0, 3.236, 3.236} |
| 600-cell test (top-4) | **CLEAN: 120+120 pts, ratio = phi** |
| All row norms^2 | 7.236 (uniform) |
| All col norms^2 | 7.236 (uniform) |
| Entry values | {0, +/-1, +/-phi, +/-1/phi, +/-phi^2} |

The corrected matrix is the authoritative Moxness version. Unlike M2 and M3, its
top-4 rows produce a clean H4+phi*H4 decomposition. It has the beautiful property
of uniform row and column norms, tracelessness, centrosymmetry, and palindromic
row structure.

---

## Cross-Comparison: Moxness Internal Consistency

### M2 (2014) vs M3 (2018): NOT THE SAME

Moxness claims M3 is the same matrix as M2 with different notation. **This is false.**

- Max |M2 - M3| = 1.618
- 28 of 64 entries differ
- Ratio M3/M2 is NOT constant: ratios include {-0.618, 0.5, 0.618, 2.0}

The relationship involves both scaling AND sign changes -- a non-trivial
transformation, not a simple notation change.

### M1 (2013 4x8) vs M2 top-4 (2014): DIFFERENT

- Max |M1 - M2_top4| = 3.618
- Completely different sparsity patterns
- Basis change reconstruction error: 1.45 -- not related by any 4x4 transform

### M1 (2013) vs Baez: IDENTICAL (up to normalization)

- M1 / sqrt(2) = Baez matrix to machine precision (error < 1.1e-16)
- Null space overlap = 4/4, Image space overlap = 4/4
- This is the standard E8->H4 projection from the literature

### Corrected phi*U vs M2 and M3: ALL DIFFERENT

- phi*U vs M2: max diff = 2.618
- phi*U vs M3: max diff = 2.618
- Not related by any simple scaling ({x1, x2, x*sqrt(phi), x*2*sqrt(phi)} all fail)

---

## Phillips Matrix vs ALL Moxness Matrices

### Phillips Matrix Properties

| Property | Value |
|----------|-------|
| Size | 8x8 |
| Rank | 4 |
| Symmetric | No |
| Determinant | 0 |
| Trace | -0.118 |
| Zeros | 0/64 (completely dense) |
| Frobenius^2 | 20.000 |
| U_R = phi * U_L | Yes (error < 1.1e-16) |
| Entry values | {+/-1/2, +/-(phi-1)/2, +/-phi/2} |
| Column norms^2 | 3 classes: {phi+2, 5/2, 3-phi} in 2-4-2 pattern |
| Row norms^2 | 2 classes: {3-phi, phi+2} |
| Unique 4D projections | 226 (14 collisions from single vector d) |
| Shell radii | 21 classes |

### The Definitive Non-Relationship

**Basis change test: Phillips = T x Moxness for ANY 4x4 matrix T?**

| Test | Reconstruction Error | Verdict |
|------|---------------------|---------|
| Phillips = T x M1 (2013)? | 0.724 | **FAILS** |
| Phillips = T x M2 top-4? | 0.557 | **FAILS** |
| Phillips = T x M3 top-4? | 0.585 | **FAILS** |
| Phillips = T x Corrected phi*U top-4? | 0.543 | **FAILS** |
| Phillips = T x Baez? | 0.724 | **FAILS** |

All reconstruction errors are O(1) -- these are not small numerical artifacts but
fundamental structural incompatibility.

**Null space / Image space overlap (4 = identical, 0 = orthogonal):**

| Pair | Null Overlap | Image Overlap |
|------|-------------|---------------|
| Phillips ^ M1 | 0 | 0 |
| Phillips ^ M2 | 1 | 1 |
| Phillips ^ M3 | 0 | 0 |
| Phillips ^ Corrected phi*U | 0 | 0 |
| Phillips ^ Baez | 0 | 0 |

Nearly complete orthogonality. The single dimension of overlap with M2 is anomalous
and likely coincidental.

**Row space analysis (stacked rank):**

| Stacking | Combined Rank |
|----------|--------------|
| [Phillips; M1] | 8 |
| [Phillips; M2 top-4] | 7 |
| [Phillips; M3 top-4] | 8 |
| [Phillips; Corrected phi*U top-4] | 8 |
| [Phillips; Baez] | 8 |
| [Phillips 8x8; M2 8x8] | 8 |
| [Phillips 8x8; M3 8x8] | 8 |
| [Phillips 8x8; Corrected phi*U 8x8] | 8 |

When stacking fills all 8 dimensions, the matrices project into completely
non-overlapping 4D subspaces of R^8.

---

## Block Structure Comparison

### Phillips: Pure phi-scaling

```
U_R = phi * U_L  (verified to machine precision)
```

Every entry of the bottom 4 rows is exactly phi times the corresponding entry in
the top 4 rows. This means:
- For ALL vectors v in R^8: ||U_R * v|| / ||U_L * v|| = phi
- The 8x8 matrix carries no more information than U_L (rank 4)

### Moxness (M2, M3): Cayley-Dickson blocks

```
[A, B; B, A]  -- top-left == bottom-right, top-right == bottom-left
```

L/R norm ratios vary across roots:
- M2: 9 distinct ratio classes (range 0.476 to 2.103)
- M3: 9 distinct ratio classes (range 0.608 to 1.645)

### Corrected phi*U: Palindromic structure

```
Row i <-> reverse of row (7-i)  -- column-reversed palindrome
```

CentroSymmetric (180-degree rotation invariant).
L/R norm ratios: exactly 3 classes {1/phi, 1, phi}.

---

## E8 Root Projection Comparison

| Projector | Unique Pts | Collisions | Norm Classes | Range |
|-----------|-----------|------------|-------------|-------|
| Phillips U_L | 226 | 14 | 21 | [0.382, 1.819] |
| M1 (2013) | 240 | 0 | 2 | [2.000, 3.236] |
| M2 top-4 | 240 | 0 | 9 | [0.191, 3.236] |
| M3 top-4 | 240 | 0 | 9 | [1.207, 2.012] |
| Corrected phi*U top-4 | 240 | 0 | 2 | [2.000, 3.236] |
| Baez | 240 | 0 | 2 | [1.414, 2.288] |

Key observations:
- Only M1, Corrected phi*U, and Baez produce clean 2-shell H4+phi*H4
- M2 and M3 produce 9-shell distributions -- NOT the expected H4 geometry
- Phillips produces 21-shell distribution with 14 collisions (unique behavior)

---

## Conclusions for Publication Strategy

1. **The Phillips Matrix is computationally verified to be a distinct mathematical
   object** from every published Moxness matrix. No confusion is justified by
   the mathematics.

2. **Moxness's own catalog has internal inconsistencies** -- his 2014 and 2018
   matrices differ beyond notation (28/64 entries), his corrected matrix differs
   from both, and only M1 (2013) and the corrected phi*U produce clean H4 geometry.

3. **M1 (2013) = Baez / sqrt(2)** -- this is the standard projection from the
   literature, not an original Moxness contribution.

4. **The Phillips Matrix has genuinely unique properties:**
   - Universal phi-ratio (||U_R*v|| / ||U_L*v|| = phi for ALL v in E8)
   - Complete density (no zero entries)
   - 2-4-2 Column Trichotomy
   - Entry set {1/2, (phi-1)/2, phi/2} = geometric progression under phi
   - Rank 4 with golden rank deficiency (U_R = phi * U_L)
   - Single collision vector d = (0,1,0,1,0,1,0,1)
   - Eigenvalue 5 with multiplicity 2: (phi+2)(3-phi) = 5
   - Amplification factor = Frobenius^2/rank = 20/4 = 5 = #24-cells in 600-cell

5. **These properties cannot be obtained from any Moxness matrix by basis change**
   -- the null/image spaces are nearly orthogonal (overlap dimension 0-1 out of 4).

---

## Computational Methodology

All results verified with:
- NumPy (linear algebra, SVD, eigendecomposition)
- All 240 E8 roots (112 permutation + 128 half-integer)
- Machine precision tolerances (1e-10 or better)
- Script: `_SYNERGIZED_SYSTEM/backend/tests/complete_moxness_catalog_vs_phillips.py`
- Existing test suite: 281 automated tests, 0 failures

---

*Generated: 2026-02-14 | Builds on work from sessions dating to 2026-01-13*
*Previous comparisons: compare_moxness_phillips.py, compare_moxness_pdf_vs_phillips.py*
