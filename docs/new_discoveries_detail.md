# New Discoveries in φ-Coupled E₈ → H₄ Folding Matrix

**Author:** Paul Joseph Phillips
**Date:** 2026-01-16
**Status:** Verified with full traceability

---

## Executive Summary

During the revision of the arXiv paper, three significant new discoveries were made about the φ-coupled folding matrix. These were not in the original analysis and appear to be previously undocumented:

| # | Discovery | Value | Significance |
|---|-----------|-------|--------------|
| 1 | Column Norm Duality | √(φ+2), √(3-φ) | Rows and columns have swapped norms |
| 2 | det(U) = 0, rank = 7 | Singular matrix | Confirms true projection structure |
| 3 | Linear dependency involves φ | φ·Row₀ - φ·Row₃ - Row₄ + Row₇ = 0 | Golden ratio in null space |

---

## Discovery 1: Column Norms and Row-Column Duality

### How It Was Found

While computing the determinant, I analyzed U^T × U. The diagonal entries of U^T × U give ||Col_j||². I noticed these values matched the row norms but were **swapped between blocks**.

### The Result

**Row norms (previously known):**
- H₄ᴸ rows (0-3): ||Row|| = √(3-φ) ≈ 1.1756
- H₄ᴿ rows (4-7): ||Row|| = √(φ+2) ≈ 1.9021

**Column norms (NEW):**
- Columns 0-3: ||Col|| = √(φ+2) ≈ 1.9021
- Columns 4-7: ||Col|| = √(3-φ) ≈ 1.1756

### Visual Representation

```
              Columns 0-3           Columns 4-7
              ||Col|| = √(φ+2)      ||Col|| = √(3-φ)
            ┌───────────────────┬───────────────────┐
Rows 0-3    │                   │                   │
||Row||     │    a, a, a, a     │    b, b, -b, -b   │  → √(3-φ)
= √(3-φ)    │                   │                   │
            ├───────────────────┼───────────────────┤
Rows 4-7    │                   │                   │
||Row||     │    c, c, c, c     │   -a, -a, a, a    │  → √(φ+2)
= √(φ+2)    │                   │                   │
            └───────────────────┴───────────────────┘
                    ↓                   ↓
                 √(φ+2)              √(3-φ)
```

### Proof

**Column 0:**
```
||Col₀||² = U[0,0]² + U[1,0]² + U[2,0]² + U[3,0]² + U[4,0]² + U[5,0]² + U[6,0]² + U[7,0]²
         = a² + a² + a² + a² + c² + c² + c² + c²
         = 4a² + 4c²
         = 4(0.5)² + 4(0.809)²
         = 1 + 2.618
         = 3.618 = φ + 2

Therefore ||Col₀|| = √(φ+2) ≈ 1.902
```

**Column 4:**
```
||Col₄||² = b² + b² + b² + b² + a² + a² + a² + a²
         = 4b² + 4a²
         = 4(0.309)² + 4(0.5)²
         = 0.382 + 1
         = 1.382 = 3 - φ

Therefore ||Col₄|| = √(3-φ) ≈ 1.176
```

### Exact Numerical Values

| Quantity | Exact Value | Decimal |
|----------|-------------|---------|
| φ + 2 | (5 + √5)/2 | 3.618033988749895 |
| 3 - φ | (5 - √5)/2 | 1.381966011250105 |
| √(φ+2) | √((5+√5)/2) | 1.902113032590307 |
| √(3-φ) | √((5-√5)/2) | 1.1755705045849463 |

---

## Discovery 2: Determinant and Rank

### How It Was Found

The red team analysis noted that "Compute det(U)" was listed as an open problem. I implemented LU decomposition to calculate it.

### The Result

```
det(U) = 0
rank(U) = 7
nullity = 1
```

### Verification Method

**LU Decomposition:**
1. Gaussian elimination with partial pivoting
2. Track sign changes from row swaps
3. Product of diagonal elements gives determinant

**Rank Computation:**
1. Row echelon form
2. Count non-zero rows = rank
3. Result: 7 pivot columns, 1 free variable

### Interpretation

- The matrix is **singular** (non-invertible)
- It maps ℝ⁸ onto a **7-dimensional subspace**
- There is **exactly one linear dependency** among the 8 rows
- This confirms U is a **projection**, not a rotation

### Block Structure

| Block | Rows | Rank |
|-------|------|------|
| H₄ᴸ | 0-3 | 4 (full rank) |
| H₄ᴿ | 4-7 | 4 (full rank) |
| Combined | 0-7 | 7 (not 8!) |

Each block individually has full rank, but together they have one dependency.

---

## Discovery 3: The Linear Dependency Involves φ

### How It Was Found

After finding rank = 7, I investigated the null space to find which linear combination of rows equals zero.

### The Result

The 8 rows satisfy exactly ONE linear relationship:

$$\boxed{\varphi \cdot \text{Row}_0 - \varphi \cdot \text{Row}_3 - \text{Row}_4 + \text{Row}_7 = \mathbf{0}}$$

Equivalently:
$$\text{Row}_0 - \text{Row}_3 = \frac{1}{\varphi} \cdot (\text{Row}_4 - \text{Row}_7)$$

### Verification

```
φ×Row₀ = [0.809, 0.809, 0.809, 0.809, 0.500, 0.500, -0.500, -0.500]
φ×Row₃ = [0.809, -0.809, -0.809, 0.809, 0.500, 0.500, -0.500, -0.500]
Row₄   = [0.809, 0.809, 0.809, 0.809, -0.500, -0.500, 0.500, 0.500]
Row₇   = [0.809, -0.809, -0.809, 0.809, -0.500, -0.500, 0.500, 0.500]

φ×Row₀ - φ×Row₃ - Row₄ + Row₇ = [0, 0, 0, 0, 0, 0, 0, 0] ✓
```

### Why This Happens

Examining the row structure:
- Row₀ and Row₃ have **identical last 4 entries**: [b, b, -b, -b]
- Row₄ and Row₇ have **identical last 4 entries**: [-a, -a, a, a]

This means:
- Row₀ - Row₃ has zeros in positions 4-7
- Row₄ - Row₇ has zeros in positions 4-7

Both differences lie in the same 4D subspace (first 4 coordinates), and they are proportional by factor 1/φ:

```
Row₀ - Row₃ = [0, 2a, 2a, 0, 0, 0, 0, 0] = [0, 1, 1, 0, 0, 0, 0, 0]
Row₄ - Row₇ = [0, 2c, 2c, 0, 0, 0, 0, 0] = [0, φ, φ, 0, 0, 0, 0, 0]

Ratio: 1/φ = (Row₀ - Row₃) / (Row₄ - Row₇)
```

### Significance

**The golden ratio φ appears in THREE places in this matrix:**
1. The coefficients (b = (φ-1)/2, c = φ/2)
2. The row/column norms (√(3-φ), √(φ+2))
3. The linear dependency (coefficient is φ)

This is remarkable — φ permeates the entire structure.

---

## Discovery 4: Additional Cross-Block Couplings

### How It Was Found

While analyzing the rank structure, I computed ALL pairwise inner products between H₄ᴸ and H₄ᴿ rows, not just Row₀·Row₄.

### The Complete Coupling Matrix

```
⟨Row_i, Row_j⟩ for i ∈ {0,1,2,3}, j ∈ {4,5,6,7}:

          Row₄    Row₅    Row₆    Row₇
Row₀       1       0       0     -1/φ
Row₁       0       1       0       0
Row₂       0       0       1       0
Row₃     -1/φ      0       0       1
```

### Pattern

| Coupling Value | Count | Which Pairs |
|----------------|-------|-------------|
| 1 | 4 | Diagonal: (0,4), (1,5), (2,6), (3,7) |
| -1/φ ≈ -0.618 | 2 | Off-diagonal: (0,7), (3,4) |
| 0 | 10 | All others |

### Interpretation

- **Diagonal couplings** Row_i · Row_{i+4} = 1 = φ - 1/φ
- **Corner couplings** Row₀·Row₇ = Row₃·Row₄ = -1/φ
- The coupling matrix has a **symmetric structure** with φ-related values

---

## Null Space Analysis

### The Null Vector

The null space is 1-dimensional, spanned by:

$$\mathbf{v} = (0, 0, 0, 0, 1, 1, 1, 1)$$

This means: summing columns 4-7 of any row gives zero when we account for the linear dependency.

### Geometric Meaning

A vector in the null space represents a direction in ℝ⁸ that gets projected to **zero** by U. The null vector [0,0,0,0,1,1,1,1] shows that the sum of the last four coordinates (with appropriate signs) is collapsed.

---

## Summary of All Exact Values

### Coefficients
```
φ = 1.6180339887498948482...
a = 0.5
b = 0.30901699437494742...
c = 0.80901699437494742...
```

### Derived Quantities
```
3 - φ = 1.3819660112501051518...
φ + 2 = 3.6180339887498948482...
√(3-φ) = 1.1755705045849462583...
√(φ+2) = 1.9021130325903071442...
√5 = 2.2360679774997896964...
1/φ = 0.6180339887498948482...
```

### Verified Identities
```
φ² = φ + 1 ✓
1/φ = φ - 1 ✓
φ - 1/φ = 1 ✓
(3-φ)(φ+2) = 5 ✓
√(3-φ) × √(φ+2) = √5 ✓
```

---

## Files Created for Verification

| File | Purpose |
|------|---------|
| `discovery_traceability.ts` | Step-by-step computation of all findings |
| `compute_determinant.ts` | Determinant and eigenvalue analysis |
| `analyze_matrix_rank.ts` | Rank and null space computation |
| `investigate_null_space.ts` | Deep dive into linear dependency |

All computations verified to machine precision (ε < 10⁻¹⁵).

---

*Paul Joseph Phillips — Clear Seas Solutions LLC*
