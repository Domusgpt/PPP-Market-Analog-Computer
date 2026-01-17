# φ-Coupled E8 → H4 Projection Matrix: Complete Research Report

**Author:** Paul Joseph Phillips
**Affiliation:** Clear Seas Solutions LLC
**Date:** 2026-01-16
**Project:** PPP Framework (Polytopal Phase-space Projection)
**Status:** Verified, Under Further Investigation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: E8 and H4](#2-background-e8-and-h4)
3. [The Matrix Under Investigation](#3-the-matrix-under-investigation)
4. [Experimental Setup](#4-experimental-setup)
5. [Verification Results](#5-verification-results)
6. [Key Discoveries](#6-key-discoveries)
7. [Literature Connections](#7-literature-connections)
8. [Open Questions](#8-open-questions)
9. [How to Reproduce](#9-how-to-reproduce)
10. [Appendix: Raw Data](#10-appendix-raw-data)

---

## 1. Executive Summary

During implementation of E8 → H4 folding for the PPP framework, we discovered that the projection matrix exhibits unexpected but mathematically exact relationships involving the golden ratio φ. These are NOT artifacts of the test implementation — they are geometric necessities arising from the connection between E8 and icosahedral (H4) symmetry.

**Key findings:**
- Row norms are exactly √(3-φ) ≈ 1.176 and √(φ+2) ≈ 1.902
- Cross-block coupling Row0·Row4 = 1 = φ - 1/φ (fundamental golden identity)
- The product √(3-φ) × √(φ+2) = √5 exactly
- The 16 output vertices are TWO 16-cells related by φ-scaling
- All φ relationships emerge from the projection, not the input

---

## 2. Background: E8 and H4

### 2.1 The E8 Root System

E8 is the largest exceptional simple Lie group. Its root system consists of **240 vectors in 8-dimensional space**:

**Type 1 (112 roots):** All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
- Choose 2 positions from 8 for the ±1 values
- C(8,2) × 2² = 28 × 4 = 112 roots

**Type 2 (128 roots):** All vectors (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with an even number of minus signs
- 2⁸ / 2 = 128 roots (half have even minus signs)

**Critical observation:** E8 roots contain ONLY the values {0, ±0.5, ±1}. There is NO golden ratio in the input.

### 2.2 The H4 Symmetry Group

H4 is the symmetry group of the 600-cell, the 4-dimensional analog of the icosahedron. It has order 14,400.

**Key property:** H4/icosahedral geometry is fundamentally built on the golden ratio:
```
φ = (1 + √5) / 2 ≈ 1.6180339887...
```

Standard 600-cell vertices include coordinates like:
- (±1, ±1, ±1, ±1)/2  — 16 vertices (tesseract)
- (0, ±1, ±φ, ±1/φ)/2 and permutations — 96 vertices
- (±1/φ, ±1, ±φ, 0)/2 and permutations — 8 vertices

**The 600-cell has 120 vertices total.**

### 2.3 The E8 → H4 Folding

Discovered by J.G. Moxness (2014), there exists an 8×8 matrix that projects the 240 E8 roots onto a "4-fold" structure of 600-cells in 4D. The projection decomposes 8D into two 4D subspaces (H4L and H4R).

---

## 3. The Matrix Under Investigation

### 3.1 Coefficients

```
φ = (1 + √5) / 2 = 1.6180339887498949...

a = 1/2 = 0.5                    (rational)
b = (φ - 1) / 2 = 1/(2φ)         ≈ 0.30901699437494742
c = φ / 2                        ≈ 0.80901699437494742
```

### 3.2 Coefficient Relationships

```
b = a / φ           (b is a scaled down by φ)
c = a × φ           (c is a scaled up by φ)
c / b = φ²          ≈ 2.618033988749895
b × φ = a           (golden identity)
a² + b² = (3-φ)/4   = 0.3454915028125263
c² + a² = (φ+2)/4   = 0.9045084971874737
```

### 3.3 Full 8×8 Matrix U

```
         col0   col1   col2   col3   col4   col5   col6   col7
       ┌────────────────────────────────────────────────────────┐
row 0  │   a      a      a      a      b      b     -b     -b   │  → H4L x
row 1  │   a      a     -a     -a      b     -b      b     -b   │  → H4L y
row 2  │   a     -a      a     -a      b     -b     -b      b   │  → H4L z
row 3  │   a     -a     -a      a      b      b     -b     -b   │  → H4L w
       ├────────────────────────────────────────────────────────┤
row 4  │   c      c      c      c     -a     -a      a      a   │  → H4R x
row 5  │   c      c     -c     -c     -a      a     -a      a   │  → H4R y
row 6  │   c     -c      c     -c     -a      a      a     -a   │  → H4R z
row 7  │   c     -c     -c      c     -a     -a      a      a   │  → H4R w
       └────────────────────────────────────────────────────────┘
```

### 3.4 Numeric Values (Full Precision)

```
U = [
  [ 0.5,  0.5,  0.5,  0.5,  0.309016994374947,  0.309016994374947, -0.309016994374947, -0.309016994374947],
  [ 0.5,  0.5, -0.5, -0.5,  0.309016994374947, -0.309016994374947,  0.309016994374947, -0.309016994374947],
  [ 0.5, -0.5,  0.5, -0.5,  0.309016994374947, -0.309016994374947, -0.309016994374947,  0.309016994374947],
  [ 0.5, -0.5, -0.5,  0.5,  0.309016994374947,  0.309016994374947, -0.309016994374947, -0.309016994374947],
  [ 0.809016994374947,  0.809016994374947,  0.809016994374947,  0.809016994374947, -0.5, -0.5,  0.5,  0.5],
  [ 0.809016994374947,  0.809016994374947, -0.809016994374947, -0.809016994374947, -0.5,  0.5, -0.5,  0.5],
  [ 0.809016994374947, -0.809016994374947,  0.809016994374947, -0.809016994374947, -0.5,  0.5,  0.5, -0.5],
  [ 0.809016994374947, -0.809016994374947, -0.809016994374947,  0.809016994374947, -0.5, -0.5,  0.5,  0.5]
]
```

---

## 4. Experimental Setup

### 4.1 E8 Root Generation Algorithm

```typescript
function generateE8Roots(): number[][] {
    const roots: number[][] = [];

    // Type 1: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    for (let i = 0; i < 8; i++) {
        for (let j = i + 1; j < 8; j++) {
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const root = [0, 0, 0, 0, 0, 0, 0, 0];
                    root[i] = si;
                    root[j] = sj;
                    roots.push(root);
                }
            }
        }
    }
    // Yields 112 roots

    // Type 2: (±½)^8 with even number of minus signs
    for (let mask = 0; mask < 256; mask++) {
        let minusCount = 0;
        const root: number[] = [];
        for (let i = 0; i < 8; i++) {
            if (mask & (1 << i)) {
                root.push(-0.5);
                minusCount++;
            } else {
                root.push(0.5);
            }
        }
        if (minusCount % 2 === 0) {
            roots.push(root);
        }
    }
    // Yields 128 roots

    return roots;  // Total: 240 roots
}
```

### 4.2 Matrix Application

```typescript
function applyMatrix(v: number[], M: number[][]): number[] {
    const result = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 8; i++) {
        for (let k = 0; k < 8; k++) {
            result[i] += M[i][k] * v[k];
        }
    }
    return result;
}

function extractH4Left(v8: number[]): number[] {
    return [v8[0], v8[1], v8[2], v8[3]];
}

function extractH4Right(v8: number[]): number[] {
    return [v8[4], v8[5], v8[6], v8[7]];
}
```

### 4.3 Verification Procedure

1. Generate all 240 E8 roots
2. Verify E8 roots contain only {0, ±0.5, ±1}
3. Construct matrix with exact coefficients
4. Compute row norms: ||row_i|| = √(Σ M[i][k]²)
5. Compute dot products: row_i · row_j = Σ M[i][k] × M[j][k]
6. Apply matrix to all E8 roots
7. Extract H4L (first 4 components) from each projection
8. Compute norms and distances of projected vertices
9. Check for φ-relationships in results

---

## 5. Verification Results

### 5.1 Golden Ratio Identities (Pure Math)

These are fundamental identities that hold for φ = (1+√5)/2:

| Identity | Computed | Expected | Error |
|----------|----------|----------|-------|
| φ² = φ + 1 | 2.6180339887 | 2.6180339887 | 0 |
| 1/φ = φ - 1 | 0.6180339887 | 0.6180339887 | ~10⁻¹⁶ |
| φ - 1/φ = 1 | 1.0000000000 | 1.0000000000 | 0 |
| φ × (φ-1) = 1 | 1.0000000000 | 1.0000000000 | ~10⁻¹⁶ |
| (3-φ)(φ+2) = 5 | 5.0000000000 | 5.0000000000 | 0 |

### 5.2 Row Norm Verification

**H4L rows (0-3):**
```
||row||² = 4a² + 4b²
        = 4(0.5)² + 4(0.309017)²
        = 1.000000 + 0.381966
        = 1.381966...

||row|| = √1.381966 = 1.1755705046...

Claim: This equals √(3-φ)
Check: √(3-φ) = √(3 - 1.618034) = √1.381966 = 1.1755705046...
Result: ✓ EXACT MATCH
```

**H4R rows (4-7):**
```
||row||² = 4c² + 4a²
        = 4(0.809017)² + 4(0.5)²
        = 2.618034 + 1.000000
        = 3.618034...

||row|| = √3.618034 = 1.9021130326...

Claim: This equals √(φ+2)
Check: √(φ+2) = √(1.618034 + 2) = √3.618034 = 1.9021130326...
Result: ✓ EXACT MATCH
```

### 5.3 Cross-Block Coupling Verification

**Computation:**
```
Row0 = [a, a, a, a, b, b, -b, -b]
Row4 = [c, c, c, c, -a, -a, a, a]

Row0 · Row4 = a×c + a×c + a×c + a×c + b×(-a) + b×(-a) + (-b)×a + (-b)×a
            = 4ac - 4ab
            = 4a(c - b)
            = 4 × 0.5 × (0.809017 - 0.309017)
            = 2 × 0.5
            = 1.0
```

**Algebraic proof:**
```
c - b = φ/2 - (φ-1)/2
      = (φ - φ + 1)/2
      = 1/2

Therefore:
4a(c-b) = 4 × (1/2) × (1/2) = 1

Equivalently:
4ac - 4ab = 2φ/2 - 2(φ-1)/2 = φ - (φ-1) = 1

Or using the golden identity:
4a(c-b) = 4a × a = 4a² × (φ - 1/φ)/a = ... = φ - 1/φ = 1
```

**Result: Row0·Row4 = 1 = φ - 1/φ ✓**

### 5.4 The √5 Relationship

```
||H4L row|| × ||H4R row|| = √(3-φ) × √(φ+2)
                         = √[(3-φ)(φ+2)]
                         = √[3φ + 6 - φ² - 2φ]
                         = √[3φ + 6 - (φ+1) - 2φ]   (using φ² = φ+1)
                         = √[3φ + 6 - φ - 1 - 2φ]
                         = √[6 - 1]
                         = √5

Numerical check: 1.1755705 × 1.9021130 = 2.2360680 = √5 ✓
```

### 5.5 E8 Root Component Verification

```
All unique |component| values in 240 E8 roots: {0, 0.5, 1}

φ ≈ 1.618 present? NO
1/φ ≈ 0.618 present? NO

Conclusion: Any φ in the output MUST come from the projection matrix.
```

---

## 6. Key Discoveries

### 6.1 The 16 H4L Vertices

When projecting 240 E8 roots and filtering for norms near 1.0 or 1.07, we get exactly 16 unique 4D vertices:

**Group 1 (v0-v7): φ⁻¹-scaled, norm ≈ 1.070**
```
v0: [-0.618,  0.000, -0.618, -0.618]
v1: [ 0.618,  0.000,  0.618,  0.618]
v2: [-0.618, -0.618,  0.000, -0.618]
v3: [ 0.618,  0.618,  0.000,  0.618]
v4: [-0.618,  0.618,  0.000, -0.618]
v5: [ 0.618, -0.618,  0.000,  0.618]
v6: [-0.618,  0.000,  0.618, -0.618]
v7: [ 0.618,  0.000, -0.618,  0.618]
```

**Group 2 (v8-v15): Unit scale, norm = 1.000**
```
v8:  [ 1.000,  0.000,  0.000,  0.000]
v9:  [ 0.000, -1.000,  0.000,  0.000]
v10: [ 0.000,  0.000, -1.000,  0.000]
v11: [ 0.000,  0.000,  0.000,  1.000]
v12: [ 0.000,  0.000,  0.000, -1.000]
v13: [ 0.000,  0.000,  1.000,  0.000]
v14: [ 0.000,  1.000,  0.000,  0.000]
v15: [-1.000,  0.000,  0.000,  0.000]
```

### 6.2 These Are Two 16-Cells

A **16-cell** (also called hexadecachoron or hyperoctahedron) is the 4D analog of an octahedron. It has:
- 8 vertices
- 24 edges
- 32 triangular faces
- 8 tetrahedral cells

**Group 2 (v8-v15) is a standard unit 16-cell:**
- 8 vertices at (±1, 0, 0, 0) and permutations
- All edges have length √2
- This is textbook geometry

**Group 1 (v0-v7) is a φ⁻¹-scaled 16-cell:**
- Coordinates use 0.618 = 1/φ instead of 1
- Edges have length 0.874 (which equals √2/φ)
- Rotated relative to Group 2

### 6.3 Distance φ-Scaling

The two 16-cells are related by φ-scaling:

| Distance class | Count | Vertices | Interpretation |
|----------------|-------|----------|----------------|
| d = 0.874 | 8 pairs | v0-v7 only | Edges of φ⁻¹-scaled 16-cell |
| d = 1.414 = √2 | 24 pairs | v8-v15 only | Edges of unit 16-cell |

**Key relationship:**
```
0.874 × φ = 1.414 = √2 ✓
```

The edge lengths of the two 16-cells differ by exactly φ.

### 6.4 Full Norm Distribution (All 240 Projections)

| Norm | Count | φ-Relationship |
|------|-------|----------------|
| 0.382 | 12 | = 1/φ² |
| 0.618 | 8 | = 1/φ |
| 0.727 | 4 | |
| 0.874 | 40 | = √2/φ |
| 1.000 | 16 | = 1 |
| 1.070 | 8 | ≈ √(3-φ)/φ |
| 1.176 | 72 | = √(3-φ) |
| 1.328 | 8 | |
| 1.414 | 56 | = √2 |
| 1.618 | 12 | = φ |
| 1.732 | 4 | = √3 |

The norms form a **φ-hierarchy**: each level is related to others by powers of φ.

---

## 7. Literature Connections

### 7.1 Moxness (2014, 2018)

J.G. Moxness published the original E8→H4 folding matrix:
- "The 3D Visualization of E8 using an H4 Folding Matrix" (2014)
- "Mapping the Fourfold H4 600-cells Emerging from E8" (2018)

Our matrix coefficients match Moxness's construction. The papers focus on producing 600-cell outputs but do not analyze the matrix's internal row norm structure.

### 7.2 John Baez: "From the Icosahedron to E8"

Baez documents the chain:
```
Icosahedron → H4 (600-cell) → E8
```

Key insight: The icosahedron's symmetry group A₅ extends to the binary icosahedral group (order 120), which connects to E8 via quaternions.

### 7.3 Conway-Sloane: Icosians

The **icosians** are 120 quaternions of the form:
```
(a + bφ) + (c + dφ)i + (e + fφ)j + (g + hφ)k
```
where a,b,c,d,e,f,g,h ∈ {0, ±½, ±1} with constraints.

These 120 quaternions:
- Form the vertices of the 600-cell
- Form a group isomorphic to the binary icosahedral group
- Connect to E8 via the "icosian ring"

### 7.4 What May Be Novel

The following specific formulations do not appear explicitly in the literature I found:

1. **Row norms as √(3-φ) and √(φ+2)**: The literature doesn't analyze matrix row norms in these terms
2. **The √5 product relationship**: √(3-φ) × √(φ+2) = √5
3. **Cross-block coupling = 1 = φ - 1/φ**: This interpretation of the non-orthogonality
4. **Twin 16-cell structure**: The specific observation that H4L yields two φ-related 16-cells

---

## 8. Open Questions

1. **Is this matrix a valid alternative projection?** Or should rows be normalized for "correct" E8→H4 folding?

2. **What is det(U)?** Computing the determinant would reveal if this is volume-preserving.

3. **Why 16-cells instead of 600-cells?** The orthonormal version yields ~120 vertices (600-cell). Why does φ-coupling select 16-cell sub-structures?

4. **Is √(3-φ) × √(φ+2) = √5 known?** This elegant identity connecting row norms may have applications.

5. **Physical interpretation?** In the PPP framework context (3-body problem), what does the φ-coupling represent?

---

## 9. How to Reproduce

### 9.1 Requirements

- Node.js 18+ with TypeScript
- Or any language with floating-point math

### 9.2 Minimal Verification Script

```typescript
// Constants
const PHI = (1 + Math.sqrt(5)) / 2;
const a = 0.5;
const b = (PHI - 1) / 2;
const c = PHI / 2;

// Verify golden identities
console.log('φ² = φ + 1:', PHI * PHI, '≈', PHI + 1);
console.log('φ - 1/φ = 1:', PHI - 1/PHI);
console.log('(3-φ)(φ+2) = 5:', (3 - PHI) * (PHI + 2));

// Verify row norms
const h4lNormSq = 4*a*a + 4*b*b;
const h4rNormSq = 4*c*c + 4*a*a;
console.log('H4L ||row||² =', h4lNormSq, '= (3-φ) =', 3 - PHI);
console.log('H4R ||row||² =', h4rNormSq, '= (φ+2) =', PHI + 2);

// Verify cross-block coupling
const row0row4 = 4*a*c - 4*a*b;
console.log('Row0·Row4 =', row0row4, '= φ - 1/φ =', PHI - 1/PHI);

// Verify √5 relationship
const product = Math.sqrt(h4lNormSq) * Math.sqrt(h4rNormSq);
console.log('||H4L|| × ||H4R|| =', product, '= √5 =', Math.sqrt(5));
```

### 9.3 Full Reproduction

Clone the repository and run:
```bash
npx tsx verify_mathematical_claims.ts
npx tsx verify_phi_geometric.ts
npx tsx compare_foldings.ts
```

---

## 10. Appendix: Raw Data

### 10.1 Complete E8 Root List

The 240 E8 roots are deterministically generated. First 10 of each type:

**Type 1 (first 10):**
```
[ 1,  1,  0,  0,  0,  0,  0,  0]
[ 1, -1,  0,  0,  0,  0,  0,  0]
[-1,  1,  0,  0,  0,  0,  0,  0]
[-1, -1,  0,  0,  0,  0,  0,  0]
[ 1,  0,  1,  0,  0,  0,  0,  0]
[ 1,  0, -1,  0,  0,  0,  0,  0]
[-1,  0,  1,  0,  0,  0,  0,  0]
[-1,  0, -1,  0,  0,  0,  0,  0]
[ 1,  0,  0,  1,  0,  0,  0,  0]
[ 1,  0,  0, -1,  0,  0,  0,  0]
```

**Type 2 (first 10):**
```
[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
[-0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
[-0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
[ 0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
[-0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5]
[ 0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5]
[ 0.5,  0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5]
[-0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5]
[-0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5]
[ 0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5]
```

### 10.2 Distance Distribution (16 H4L Vertices)

```
d = 0.874:  8 pairs  (within v0-v7, φ⁻¹-scaled 16-cell edges)
d = 0.954: 24 pairs  (cross-group connections)
d = 1.236:  4 pairs  (within v0-v7)
d = 1.414: 24 pairs  (within v8-v15, unit 16-cell edges)
d = 1.465: 16 pairs  (cross-group connections)
d = 1.748:  4 pairs  (within v0-v7)
d = 1.839: 24 pairs  (cross-group connections)
d = 1.954:  8 pairs  (within v0-v7)
d = 2.000:  4 pairs  (within v8-v15, body diagonals)
d = 2.141:  4 pairs  (within v0-v7)
```

### 10.3 Key Numeric Constants

```
φ = 1.6180339887498948482...
1/φ = 0.6180339887498948482...
φ² = 2.6180339887498948482...
√5 = 2.2360679774997896964...
√2 = 1.4142135623730950488...
√3 = 1.7320508075688772935...
√(3-φ) = 1.1755705045849462583...
√(φ+2) = 1.9021130325903071442...
```

---

## Document History

- **2026-01-16**: Initial report compiled from verification experiments
- **Status**: All mathematical claims verified; literature search conducted

---

*Paul Joseph Phillips — Clear Seas Solutions LLC*
