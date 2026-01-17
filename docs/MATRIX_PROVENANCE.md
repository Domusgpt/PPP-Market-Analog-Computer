# Matrix Provenance: PPP Matrix vs Moxness Matrix

**Date:** January 2026
**Status:** Critical Clarification

---

## Executive Summary

The 8×8 "Moxness matrix" used in the PPP codebase is **NOT** the actual Moxness folding matrix from the published literature. This document clarifies the distinction and provides proper attribution.

---

## The Actual Moxness Matrix

### Source
- **Moxness, J.G. (2013-2014)** "E8 folding to H4+H4/φ"
- Available at: http://theoryofeverything.org/theToE/2013/11/18/1454/
- Also: viXra:1411.0130, viXra:1808.0107

### Structure (4×8 Folding Matrix)

```
x = (1,  φ,  0, -1,  φ,  0,  0,    0)
y = (φ,  0,  1,  φ,  0, -1,  0,    0)
z = (0,  1,  φ,  0, -1,  φ,  0,    0)
w = (0,  0,  0,  0,  0,  0,  φ²,  1/φ)

where φ = (1+√5)/2 ≈ 1.618
```

### Coefficients Used
- **{0, 1, -1, φ, 1/φ, φ²}**
- Sparse structure (many zeros)
- Related to icosahedral/quaternionic geometry

### Properties (from Moxness's papers)
- Symmetric: C600 = Transpose[C600]
- Unimodular version exists with det = 1
- Projects E8 (240 roots) → H4 (120 vertices) × 2 scales
- Produces four chiral copies: H4_L ⊕ φH4_L ⊕ H4_R ⊕ φH4_R

---

## The PPP Codebase Matrix

### Source
- **Internal PPP development** (origin unclear)
- File: `lib/topology/E8H4Folding.ts:107-156`

### Structure (8×8 Matrix)

```typescript
const a = 0.5;                    // 1/2
const b = 0.5 * (PHI - 1);        // (φ-1)/2 ≈ 0.309
const c = 0.5 * PHI;              // φ/2 ≈ 0.809

// Hadamard-like sign pattern
Row 0: [a,  a,  a,  a,  b,  b, -b, -b]
Row 1: [a,  a, -a, -a,  b, -b,  b, -b]
Row 2: [a, -a,  a, -a,  b, -b, -b,  b]
Row 3: [a, -a, -a,  a,  b,  b, -b, -b]
Row 4: [c,  c,  c,  c, -a, -a,  a,  a]
Row 5: [c,  c, -c, -c, -a,  a, -a,  a]
Row 6: [c, -c,  c, -c, -a,  a,  a, -a]
Row 7: [c, -c, -c,  c, -a, -a,  a,  a]
```

### Coefficients Used
- **{0.5, 0.309, 0.809}** (normalized φ-scaled values)
- Dense structure (no zeros)
- Hadamard-like sign pattern in each 4×4 block

### Properties (verified in PPP)
- **NOT symmetric**: M[0][4] = b ≠ c = M[4][0]
- **Singular**: det = 0, rank = 7
- **Null space**: [0, 0, 0, 0, 1, 1, 1, 1]ᵀ
- **√5-Coupling**: √(3-φ) × √(φ+2) = √5

---

## Side-by-Side Comparison

| Property | Moxness Original | PPP Codebase |
|----------|------------------|--------------|
| **Coefficients** | 0, ±1, φ, 1/φ, φ² | 0.5, (φ-1)/2, φ/2 |
| **Sparsity** | Sparse (many zeros) | Dense (all non-zero) |
| **Symmetry** | Symmetric | NOT symmetric |
| **Determinant** | 1 (unimodular version) | 0 (singular) |
| **Rank** | 8 | 7 |
| **Sign pattern** | Irregular | Hadamard-like |
| **Source** | Published (viXra) | Internal development |

---

## What the PPP Matrix Actually Is

The PPP matrix appears to be a **φ-coupled Hadamard-like construction** that:

1. Uses normalized golden ratio coefficients (a, b, c) where b/a = 1/φ and c/a = φ
2. Applies a Hadamard sign pattern to create orthogonal-like blocks
3. Has interesting algebraic properties (√5 coupling, rank-7 anomaly)
4. Was likely developed for visualization/encoding purposes within PPP

### Verified Properties of PPP Matrix

| Property | Status | Evidence |
|----------|--------|----------|
| √5-Coupling Theorem | ✓ Verified | `validate_paper_claims.ts` |
| Rank 7 | ✓ Verified | Algebraic computation |
| Null space [0,0,0,0,1,1,1,1]ᵀ | ✓ Verified | Direct calculation |
| φ-family scaling | ✓ Verified | `track_a_investigation.ts` |
| Derived from Moxness | ✗ FALSE | Different coefficients entirely |

---

## Implications

### For Documentation
All references to "Moxness matrix" in PPP should be clarified:
- The PPP matrix is **inspired by** E8→H4 folding concepts
- It is **not** a direct implementation of Moxness's published work
- It has its **own** interesting mathematical properties worth studying

### For Research
- Track A findings about the PPP matrix (√5 coupling, rank-7, φ-family) remain valid
- These are properties of the **PPP construction**, not the Moxness matrix
- A separate investigation could compare PPP matrix behavior to the actual Moxness matrix

### Recommended Terminology
- Call it the **"PPP projection matrix"** or **"φ-coupled Hadamard matrix"**
- Reserve **"Moxness matrix"** for the actual published construction
- When referencing E8→H4 folding, cite Moxness but clarify the PPP implementation differs

---

## References

### Moxness's Actual Work
1. Moxness, J.G. "E8 folding to H4+H4/φ" (2013)
   http://theoryofeverything.org/theToE/2013/11/18/1454/

2. Moxness, J.G. "The 3D Visualization of E8 using an H4 Folding Matrix" (2014)
   viXra:1411.0130

3. Moxness, J.G. "Mapping the fourfold H4 600-cells emerging from E8" (2018)
   viXra:1808.0107

4. Moxness, J.G. "Unimodular rotation of E8 to H4 600-cells" (2019)
   ResearchGate

### Mathematical Background
5. Denney et al. "The geometry of H4 polytopes" (2020)
   *Advances in Geometry* 20(3), 433-444
   https://www.degruyterbrill.com/document/doi/10.1515/advgeom-2020-0005/html

6. Baez, J. "The 600-Cell" blog series (2020)
   https://johncarlosbaez.wordpress.com/2020/11/30/the-600-cell-part-4/

7. Dechant, P.-P. "Clifford algebra unveils a surprising geometric significance of quaternionic root systems of Coxeter groups" (2013)
   *Adv. Appl. Clifford Algebras* 23, 301-321

8. Conway & Sloane. *Sphere Packings, Lattices and Groups*
   Springer (icosian ring = E8 lattice)

---

*Document created to clarify matrix provenance after source investigation.*
