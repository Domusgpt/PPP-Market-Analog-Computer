# The φ-Coupled E8 → H4 Projection Matrix

## Overview

This document describes a potentially novel 8×8 matrix discovered during the PPP framework development. Unlike the standard orthonormal Moxness matrix, this matrix exhibits a **golden ratio coupling** between its left and right 4D projection subspaces.

## The Matrix

### Coefficients

```
φ = (1 + √5) / 2 ≈ 1.6180339887...  (golden ratio)

a = 1/2 = 0.5
b = (φ - 1) / 2 = 1/(2φ) ≈ 0.30901699...
c = φ / 2 ≈ 0.80901699...
```

### Key Relationships

```
c / b = φ²  ≈ 2.618
b × φ = a   (golden identity)
c = a × φ   (golden scaling)
```

### Full 8×8 Matrix U

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

### Numeric Values

```
U =
┌                                                                            ┐
│  0.500   0.500   0.500   0.500   0.309   0.309  -0.309  -0.309  │
│  0.500   0.500  -0.500  -0.500   0.309  -0.309   0.309  -0.309  │
│  0.500  -0.500   0.500  -0.500   0.309  -0.309  -0.309   0.309  │
│  0.500  -0.500  -0.500   0.500   0.309   0.309  -0.309  -0.309  │
│  0.809   0.809   0.809   0.809  -0.500  -0.500   0.500   0.500  │
│  0.809   0.809  -0.809  -0.809  -0.500   0.500  -0.500   0.500  │
│  0.809  -0.809   0.809  -0.809  -0.500   0.500   0.500  -0.500  │
│  0.809  -0.809  -0.809   0.809  -0.500  -0.500   0.500   0.500  │
└                                                                            ┘
```

## Structural Properties

### 1. Hadamard-like Sign Pattern

The first 4 columns of each block follow a 4×4 Hadamard matrix sign pattern:

```
H4L block (cols 0-3):        H4R block (cols 0-3):
[+ + + +]  (sum = 4a)        [+ + + +]  (sum = 4c)
[+ + - -]  (sum = 0)         [+ + - -]  (sum = 0)
[+ - + -]  (sum = 0)         [+ - + -]  (sum = 0)
[+ - - +]  (sum = 0)         [+ - - +]  (sum = 0)
```

This is exactly the 4×4 Hadamard matrix H₄, scaled by `a` and `c` respectively.

### 2. Row Norms (NOT Unit)

```
||rows 0-3|| = √(4a² + 4b²) = √(1 + 4b²) = √(3 - φ) ≈ 1.176
||rows 4-7|| = √(4c² + 4a²) = √(φ² + 1) = √(φ + 2) ≈ 1.902
```

The rows are **not normalized** to unit length. This is intentional.

### 3. Within-Block Orthogonality

**H4L block (rows 0-3):**
```
row0 · row1 = 0  ✓
row0 · row2 = 0  ✓
row0 · row3 = 0.382 ≈ 1/φ²  ✗ (not zero!)
row1 · row2 = 0  ✓
row1 · row3 = 0  ✓
row2 · row3 = 0  ✓
```

**H4R block (rows 4-7):** Similar pattern.

### 4. Cross-Block φ-Coupling (The Key Discovery)

```
row0 · row4 = 4(a × c) + 4(b × -a)
            = 4ac - 4ab
            = 4a(c - b)
            = 4 × 0.5 × (0.809 - 0.309)
            = 2 × 0.5
            = 1.0 exactly
```

Algebraically:
```
row0 · row4 = 4a(c - b)
            = 4 × (1/2) × (φ/2 - (φ-1)/2)
            = 2 × ((φ - φ + 1)/2)
            = 2 × (1/2)
            = 1

Equivalently:
            = 4ac - 4ab
            = 2(φ/2) - 2((φ-1)/2)
            = φ - (φ - 1)
            = φ - φ⁻¹
            = 1  (fundamental golden identity!)
```

**This is not noise.** The L and R subspaces are coupled by exactly φ - φ⁻¹ = 1.

## Projection Behavior

### Input: 240 E8 Root Vectors

The E8 root system has 240 vectors in 8D:
- 112 roots: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
- 128 roots: (±½)⁸ with even number of minus signs

### Output: Norm Distribution

When projecting all 240 E8 roots through this matrix:

**Left projection (rows 0-3):**
```
Unique norms: 0.382, 0.618, 0.727, 0.874, 1.0, 1.07, 1.176, 1.328, 1.414, 1.618, 1.732
```

**Right projection (rows 4-7):**
```
Unique norms: 0.618, 1.0, 1.176, 1.414, 1.618, 1.732, 1.902, 2.149, 2.288, 2.618, 2.803
```

Note the appearance of golden ratio multiples: 0.618 (1/φ), 1.618 (φ), 2.618 (φ²).

### Output: Vertex Count

With current norm-based filtering (|norm - 1| < 0.1 or |norm - φ| < 0.1):
```
H4L unit scale:  16 vertices
H4L φ-scale:     12 vertices
H4R unit scale:   4 vertices
H4R φ-scale:      8 vertices
Total:           40 unique vertices
```

**Observation:** 16 vertices = exactly a tesseract (8-cell) or two 16-cells.

## Geometric Interpretation

### Hypothesis: Inscribed Tesseract Projection

The 600-cell contains inscribed sub-polytopes:
- 5 disjoint 24-cells (24 vertices each)
- 3 inscribed tesseracts per 24-cell (16 vertices each)
- 75 tesseracts total in the 600-cell

This matrix may be projecting E8 roots to the **tesseracts inscribed in the 600-cell** rather than to the full 600-cell vertices.

### Why This Could Be Significant

1. **Dimensional reduction**: E8 (240 roots) → Tesseract (16 vertices) is a 15:1 compression
2. **Golden spiral structure**: The φ-coupling preserves icosahedral relationships
3. **Sub-lattice selection**: May be selecting a physically meaningful subset

## Comparison with Orthonormal Moxness Matrix

| Property | This Matrix | Orthonormal Moxness |
|----------|-------------|---------------------|
| det(U) | Unknown (likely ≠ 1) | = 1 (unimodular) |
| U × Uᵀ | ≠ I₈ | = I₈ |
| Row norms | 1.176, 1.902 | 1.0 |
| Cross-block coupling | = 1 (φ - φ⁻¹) | = 0 |
| Output vertices | ~40 (tesseract?) | ~120 (600-cell) |
| Preserves | φ-relationships | Volume, angles |

## Mathematical Questions

1. **What is det(U)?** Need LU decomposition to compute.
2. **Is this a known projection type?** Golden spiral? Conformal?
3. **Do the 16 L-vertices form a valid tesseract?** Check edge distances.
4. **What is the physical/geometric meaning of φ-coupling?**

## Code Reference

Implementation: `lib/topology/E8H4Folding.ts`

```typescript
export function createMoxnessMatrix(): Matrix8x8 {
    const a = 0.5;
    const b = 0.5 * (PHI - 1);  // 1/(2φ)
    const c = 0.5 * PHI;         // φ/2
    // ... matrix construction
}
```

## References

- Moxness, J.G. "The 3D Visualization of E8 using an H4 Folding Matrix" (2014)
- Moxness, J.G. "Mapping the Fourfold H4 600-cells Emerging from E8" (2018)

## Status

**Under Investigation** - This matrix produces interesting but unexpected results. Parallel testing with a corrected orthonormal matrix is needed to determine if this is:
- A bug (wrong coefficients)
- A feature (valid alternative projection)
- A discovery (novel φ-coupled projection type)

---
*Document created: 2026-01-16*
*PPP Framework - Clear Seas Solutions LLC*
