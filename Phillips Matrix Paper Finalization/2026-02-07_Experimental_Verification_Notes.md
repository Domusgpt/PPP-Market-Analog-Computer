# Phillips Matrix: Experimental Verification Notes
**Date:** 2026-02-07
**Author:** Claude (coding agent), in collaboration with Paul Phillips
**Repository:** PPP-Market-Analog-Computer
**Branch:** claude/design-coding-agent-prompt-3TfBK
**Test Suite:** `_SYNERGIZED_SYSTEM/backend/tests/test_phillips_matrix.py` (50 tests, all pass)

---

## 1. What Was Built and Tested

The Phillips 8x8 matrix was implemented in Python alongside the existing Baez 4x8 matrix
in `_SYNERGIZED_SYSTEM/backend/engine/geometry/e8_projection.py`. Both pipelines share the
same E8 root generation (240 roots: 112 permutation + 128 half-integer), allowing direct
comparison on identical input data.

### The Phillips Matrix (exact values used in code)

```
      a = 1/2           = 0.500000
      b = (phi-1)/2     = 0.309017   (= 1/(2*phi))
      c = phi/2         = 0.809017
      phi = (1+sqrt(5))/2 = 1.618034

U = [ a   b   a   b   a  -b   a  -b ]   <- U_L block (rows 0-3)
    [ a   a  -b  -b  -a  -a   b   b ]      entries from {+/-a, +/-b}
    [ a  -b  -a   b   a  -b  -a   b ]
    [ a  -a   b  -b  -a   a  -b   b ]
    [ c   a   c   a   c  -a   c  -a ]   <- U_R block (rows 4-7)
    [ c   c  -a  -a  -c  -c   a   a ]      entries from {+/-a, +/-c}
    [ c  -a  -c   a   c  -a  -c   a ]
    [ c  -c   a  -a  -c   c  -a   a ]
```

### The Baez Matrix (for comparison)

```
BAEZ = (1/2) * [ 1    phi  0   -1   phi  0    0   0  ]
               [ phi  0    1    phi  0   -1    0   0  ]
               [ 0    1    phi  0   -1    phi  0   0  ]
               [ 0    0    0    0    0    0    1   phi ]
```

---

## 2. Theorems Verified to Machine Precision

All tests pass to floating-point precision (atol < 1e-10 unless noted).

### Theorem 4.1: Column Trichotomy (2-4-2 Pattern)

**VERIFIED.** The 8 column norms^2 of the Phillips matrix fall into exactly 3 classes:

| Column | Norm^2      | Class       | Value    |
|--------|-------------|-------------|----------|
| 0      | phi + 2     | Expanded    | 3.618034 |
| 1      | 2.5         | Stable      | 2.500000 |
| 2      | 2.5         | Stable      | 2.500000 |
| 3      | 3 - phi     | Contracted  | 1.381966 |
| 4      | phi + 2     | Expanded    | 3.618034 |
| 5      | 2.5         | Stable      | 2.500000 |
| 6      | 2.5         | Stable      | 2.500000 |
| 7      | 3 - phi     | Contracted  | 1.381966 |

**Distribution:** 2 expanded, 4 stable, 2 contracted = the "2-4-2" pattern.

**Conservation property:** Mean of extremes = ((phi+2) + (3-phi))/2 = 5/2 = 2.5 = stable norm.
**Deviation:** Extremes deviate from mean by exactly +/- sqrt(5)/2.

### Theorem 5.1: Pentagonal Row Norms

**VERIFIED.** All U_L rows have identical norm; all U_R rows have identical norm.

| Block | Row norm^2 | Row norm       | Pentagon identity      |
|-------|-----------|----------------|------------------------|
| U_L   | 3 - phi   | sqrt(3-phi)    | = 2*sin(36deg)         |
| U_R   | phi + 2   | sqrt(phi+2)    | = 2*cos(18deg)         |

The pentagon link: sin(36deg) and cos(18deg) are the fundamental lengths of the regular
pentagon (side and apothem relationships). The row norms ARE pentagon constructible lengths.

### Frobenius Norm

**VERIFIED.** ||U||_F^2 = sum of all entries^2 = 20.000000 (exact).

Decomposition: 4*(3-phi) + 4*(phi+2) = 12-4*phi + 4*phi+8 = 20.

The number 20 = the vertex valence of the 600-cell (20 tetrahedra meet at each vertex).

### Golden Ratio Coupling

**VERIFIED** on all 240 E8 roots individually (not just row norms):

- phi-scaling: ||U_R @ v|| / ||U_L @ v|| = phi for EVERY E8 root v (atol < 1e-6)
- sqrt(5)-coupling: (3-phi)*(phi+2) = 5 exactly
- Product of row norms: sqrt(3-phi) * sqrt(phi+2) = sqrt(5)

### Shell Coincidence

**VERIFIED.** The algebraic identity phi * sqrt(3-phi) = sqrt(phi+2) holds exactly.

Proof: phi^2 * (3-phi) = (phi+1)(3-phi) = 3*phi - phi^2 + 3 - phi = phi + 2. QED.

Coincidence radius = phi * sqrt(3-phi) = sqrt(phi+2) approx 1.90211.

---

## 3. Critical Discovery: The Phillips Matrix Has Rank 4

**This was NOT predicted by the papers and is a new experimental finding.**

```
Matrix rank:   4  (not 8)
Determinant:   0.000000
```

### What this means

The 8x8 Phillips matrix projects R^8 to a 4-dimensional subspace, not to R^8. The two
blocks (U_L and U_R) project to the SAME 4D subspace, related by pure scaling:

```
U_R = phi * I_4 * U_L
```

Where I_4 is the 4x4 identity matrix. This was verified to machine precision:

```
M = U_R @ pinv(U_L) = phi * [[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]
```

Max reconstruction error: 7.77e-16 (essentially zero).

### Implications

1. **The right block carries NO independent information.** Given U_L @ v, you can
   compute U_R @ v = phi * (U_L @ v). The phi-scaling is structural, not statistical.

2. **14 collision pairs.** Of 240 E8 roots, 28 roots (14 pairs) project to the same
   4D point under U_L, yielding 226 unique projected vertices.

3. **The Phillips matrix is a rank-4 operator with built-in phi-scaling.** This is a
   fundamentally different object from the Baez matrix, which is rank 4 by construction
   (being 4x8). The Phillips matrix LOOKS 8x8 but IS 4x8 in information content.

4. **The "lossless" claim needs qualification.** The paper's claim that the Phillips matrix
   provides "lossless" R^8 -> R^4 + R^4 decomposition is misleading because U_R is
   entirely determined by U_L. The decomposition is R^8 -> R^4 (with phi-shadow).

5. **However, the rank-4 structure is itself a theorem.** The fact that U_R = phi * U_L
   is not obvious from the matrix entries. It's a consequence of the specific choice of
   {a, b, c} = {1/2, (phi-1)/2, phi/2} and the sign patterns. This identity should
   be stated as a theorem in the paper.

---

## 4. Baez vs Phillips: Comparison on Same E8 Roots

| Property                    | Baez (4x8)      | Phillips (8x8)      |
|-----------------------------|-----------------|---------------------|
| Matrix shape                | 4x8             | 8x8                 |
| Effective rank              | 4               | 4                   |
| Unique projected points     | 240 (all)       | 226 (14 collisions) |
| Number of shell radii       | 8 (outer)       | varies by block     |
| Max outer/left radius       | phi (1.618)     | varies              |
| phi-scaling per root        | NO              | YES (exact)         |
| Column Trichotomy           | N/A (4 rows)    | YES (2-4-2)         |
| Pentagonal Row Norms        | NO              | YES (2*sin(36))     |
| Frobenius^2                 | ~5.24 (4x8)     | 20.0 (exact)        |
| Galois conjugate pair       | Separate matrix | Built into U_R=phi*U_L |
| Contains negative entries   | Yes (-1)        | Yes (-a,-b,-c)      |
| Contains zeros              | Yes (many)      | No (all entries nonzero) |

### Key insight

The Baez matrix produces 240 distinct 4D points at 8 radii because its entries include
zeros and integer values that create more diverse projections. The Phillips matrix
produces 226 unique points because its rank-4 structure with the b=(phi-1)/2 entry causes
14 root pairs to collapse. But Phillips has the phi-scaling guarantee on every single root,
which Baez does not.

The two matrices are complementary, not competing:
- **Baez**: Better for individual root analysis (all 240 distinct)
- **Phillips**: Better for structural analysis (Column Trichotomy, pentagon, Frobenius=20)

---

## 5. What This Means for the Paper

### Claims that are experimentally verified (and should be stated as theorems with proofs):

1. Column Trichotomy: exactly 3 norm classes in 2-4-2 pattern
2. Pentagonal Row Norms: sqrt(3-phi) = 2*sin(36deg)
3. Frobenius = 20 (600-cell vertex valence)
4. phi-scaling on all E8 roots
5. sqrt(5)-coupling between blocks
6. Shell coincidence identity: phi*sqrt(3-phi) = sqrt(phi+2)
7. **NEW: U_R = phi * U_L** (rank-4 structure)

### Claims that need refinement:

1. "Lossless R^8 -> R^4 + R^4 decomposition" -- the matrix is rank 4, so R^4+R^4 = R^4
   with a phi-scaled copy. Should say "R^8 -> R^4 with intrinsic phi-shadow."

2. "Two nested 600-cells" -- the projection produces 226 unique 4D points (not 120 or 240).
   The shell structure is more complex than two clean 600-cells.

3. The 14 collision pairs should be characterized: which E8 root pairs collide?
   This may reveal information about the kernel structure.

### Experimental data available for the paper:

All test results are reproducible from the code at:
  `_SYNERGIZED_SYSTEM/backend/tests/test_phillips_matrix.py` (50 tests)
  `_SYNERGIZED_SYSTEM/backend/tests/test_e8_projection.py` (50 tests)

The comparison function `compare_projections()` in `e8_projection.py` generates a
structured dict with all metrics for both matrices.

Total test suite: 248 tests, 0 failures.

---

## 6. Trilatic Decomposition: Independently Verified

The W(D4) coset decomposition of the 24-cell into 3 disjoint 16-cells was independently
fixed and verified in this codebase:

- **Alpha:** 8 axis-aligned vertices (perms of (+/-1, 0, 0, 0))
- **Beta:** 8 half-integer vertices with EVEN count of negative signs
- **Gamma:** 8 half-integer vertices with ODD count of negative signs

30 tests verify: disjointness, coverage (8+8+8=24), cross-polytope structure (4 antipodal
pairs, 24 edges of length sqrt(2)), and sign parity classification.

This decomposition is the "Trinity" at the heart of the paper's dialectical logic model
(Alpha=thesis, Beta=antithesis, Gamma=synthesis), and it is mathematically correct as
implemented.

---

## 7. Feature Extraction Pipeline: Verified Working

The claim that "pixel patterns encode computational results" was tested and initially
FAILED (cosine similarity 0.999+ between different pattern classes). After integrating
Gabor filter banks, spectral analyzers, and HOG descriptors into MoireFeatureExtractor
(466 features), classification accuracy went from 33% to 100%.

This validates the core PPP concept that visual interference patterns carry discriminative
information, but demonstrates that extracting that information requires sophisticated
feature extraction, not simple pixel sampling.

---

## 8. Files and Locations

| File | Description |
|------|-------------|
| `_SYNERGIZED_SYSTEM/backend/engine/geometry/e8_projection.py` | Phillips + Baez implementations |
| `_SYNERGIZED_SYSTEM/backend/tests/test_phillips_matrix.py` | 50 Phillips verification tests |
| `_SYNERGIZED_SYSTEM/backend/tests/test_e8_projection.py` | 50 Baez verification tests |
| `_SYNERGIZED_SYSTEM/backend/engine/geometry/h4_geometry.py` | Trilatic decomposition (fixed) |
| `_SYNERGIZED_SYSTEM/backend/tests/test_h4_geometry.py` | 30 H4 geometry tests |
| `_SYNERGIZED_SYSTEM/backend/engine/reservoir/readout.py` | Feature extraction (fixed) |

Total verified test count: **248 tests, 0 failures.**
