# Track A: Pure Mathematics of E₈ → H₄ Folding

**Status:** Research Investigation
**Date:** January 2026
**Focus:** Classification, Uniqueness, and Theoretical Connections

---

## Research Question

**Is the Moxness E₈ → H₄ folding matrix unique, or is it one member of a family? What determines its structure?**

---

## 1. The Classification Problem

### 1.1 What We're Classifying

We seek all 8×8 real matrices **U** such that:

1. **E₈ roots map to H₄-symmetric structures**
   - Input: 240 vectors in ℝ⁸ with entries in {0, ±½, ±1}
   - Output: Vectors with H₄ symmetry in ℝ⁴ × ℝ⁴

2. **The matrix preserves some geometric structure**
   - Not necessarily orthonormal (Moxness has det = 0)
   - But preserves "H₄-ness" of the output

### 1.2 The Known Example: Moxness Matrix

```
Coefficients: a = ½, b = (φ-1)/2, c = φ/2

         col0   col1   col2   col3   col4   col5   col6   col7
       ┌────────────────────────────────────────────────────────┐
row 0  │   a      a      a      a      b      b     -b     -b   │
row 1  │   a      a     -a     -a      b     -b      b     -b   │
row 2  │   a     -a      a     -a      b     -b     -b      b   │
row 3  │   a     -a     -a      a      b      b     -b     -b   │
row 4  │   c      c      c      c     -a     -a      a      a   │
row 5  │   c      c     -c     -c     -a      a     -a      a   │
row 6  │   c     -c      c     -c     -a      a      a     -a   │
row 7  │   c     -c     -c      c     -a     -a      a      a   │
       └────────────────────────────────────────────────────────┘
```

**Properties:**
- det(U) = 0
- rank(U) = 7
- Right null space: (0,0,0,0,1,1,1,1)ᵀ
- Row norms: √(3-φ) and √(φ+2)
- Product: √(3-φ) × √(φ+2) = √5

---

## 2. First Principles Derivation

### 2.1 Why Must φ Appear?

**Theorem (Geometric Necessity):** Any linear map T: ℝ⁸ → ℝ⁴ that sends E₈ roots to H₄-symmetric configurations must have matrix entries in ℚ(√5).

**Proof Sketch:**
1. E₈ roots span ℚ⁸ (all coordinates are rational or half-integer)
2. H₄ vertices require φ = (1+√5)/2 (96 of 120 vertices contain φ)
3. If T had purely rational entries, T(E₈) ⊂ ℚ⁴
4. But H₄ configurations require ℚ(√5) coordinates
5. Therefore T must have entries in ℚ(√5) \ ℚ

**Corollary:** The coefficients a, b, c must satisfy algebraic relations in ℚ(√5).

### 2.2 The Hadamard-Like Structure

The sign pattern of the Moxness matrix resembles a Hadamard structure:
- Each 4×4 block has ±1 patterns (scaled by a, b, or c)
- The upper-left and lower-right are "positive" heavy
- The upper-right and lower-left are "mixed" sign

**Observation:** The 4×4 blocks appear related to the Kronecker product H₂ ⊗ H₂ where H₂ is the 2×2 Hadamard matrix.

### 2.3 The φ-Geometric Progression

The coefficients form a geometric progression with ratio φ:

$$b \cdot \varphi = a, \quad a \cdot \varphi = c$$

This is equivalent to:
- b = a/φ = a(φ-1) = a/φ
- c = aφ

**Constraint:** Given a = ½, the other coefficients are determined by φ-scaling.

---

## 3. Uniqueness Investigation

### 3.1 Degrees of Freedom

Starting from constraints:
1. **Hadamard-like sign pattern** (fixed)
2. **φ-based field extension** (required by H₄ geometry)
3. **Geometric progression** (a, b, c with ratio φ)

This leaves only **one degree of freedom**: the base coefficient a (or equivalently, overall scale).

Setting a = ½ normalizes the output norms to √(3-φ) and √(φ+2).

### 3.2 Symmetry Equivalence Classes

Two matrices U and U' are **equivalent** if:
- U' = PUQ where P ∈ F₄ (24-cell symmetry) and Q ∈ W(E₈) (Weyl group)
- This represents a change of basis in both input and output spaces

**Conjecture:** The Moxness matrix is unique up to:
1. Overall scale (choosing a)
2. Left multiplication by F₄ elements
3. Right multiplication by W(E₈) elements
4. Sign permutations that preserve the Hadamard structure

### 3.3 Alternative Folding Matrices

**Question:** Are there other coefficient choices that preserve H₄ symmetry?

**Test:** Replace (a, b, c) = (½, (φ-1)/2, φ/2) with other ℚ(√5) triplets.

**Candidate Family:**
$$a_n = \frac{\varphi^n}{2}, \quad b_n = \frac{\varphi^{n-1}}{2}, \quad c_n = \frac{\varphi^{n+1}}{2}$$

For n = 0: (a, b, c) = (½, (φ-1)/2, φ/2) — the Moxness matrix
For n = 1: (a, b, c) = (φ/2, ½, φ²/2) — a φ-scaled variant

**Prediction:** All members of this family should produce valid E₈→H₄ foldings with different norm structures.

---

## 4. Connection to McKay Correspondence

### 4.1 The McKay Graph

The McKay correspondence relates:
- Finite subgroups Γ ⊂ SU(2)
- Affine Dynkin diagrams
- Resolution of singularities ℂ²/Γ

**Key Relationship:**
- Binary icosahedral group 2I (order 120) ↔ Affine E₈ diagram
- |2I| = 120 = number of 600-cell vertices

### 4.2 E₈ Roots as Graph Edges

The 240 E₈ roots can be viewed as:
- Directed edges of the affine E₈ Dynkin diagram
- 240 = 2 × 120 (edges in both directions)

**Hypothesis:** The Moxness folding implements the McKay correspondence geometrically:
- E₈ roots (240 edges) → H₄ configurations (120 vertices of 600-cell)
- The 2:1 ratio suggests a quotient by the center of 2I

### 4.3 The Null Space and McKay Kernel

The null space of the Moxness matrix is (0,0,0,0,1,1,1,1)ᵀ.

**Research Question:** Does this correspond to the kernel of the incidence matrix of the affine E₈ diagram?

**Investigation:** The affine E₈ diagram has 9 nodes. Its incidence matrix has a 1-dimensional null space corresponding to the "imaginary root" δ.

**Potential Connection:** The Moxness null space might encode the "affine direction" that distinguishes E₈ from affine Ê₈.

---

## 5. Connection to Dechant's Clifford Algebra

### 5.1 Dechant's Result

Pierre-Philippe Dechant (2013) showed that H₄ can be constructed from H₃ using Clifford algebra Cl(3):
- Start with icosahedral group H₃ in 3D
- Embed in Cl(3) as spinors
- The spinor space is 4D with H₄ symmetry

This is "bottom-up" induction: H₃ → H₄

### 5.2 Moxness as "Top-Down" Projection

The Moxness matrix performs the reverse:
- Start with E₈ in 8D
- Project to H₄ in 4D

**Conjecture:** The Moxness matrix is the linear operator representation of Dechant's spinor induction, viewed in the reverse direction.

### 5.3 Spinor Interpretation

In Clifford algebra, a vector v ∈ ℝ⁴ transforms under a rotor R as:
$$v' = RvR^{\dagger}$$

The Moxness matrix might decompose as:
$$U = L \otimes R$$

where L and R are left and right spinor multiplications.

**Investigation Required:** Decompose U into Clifford algebra elements and verify.

---

## 6. The 24-Cell Connection

### 6.1 F₄ Root System

The 24-cell is the root polytope of F₄. Its 24 vertices are:
- Permutations of (±1, ±1, 0, 0) — 24 vertices

This is a subset of the 600-cell (120 vertices).

### 6.2 600-Cell = 25 × 24-Cell

The 600-cell contains 25 inscribed 24-cells. Under H₄, these form a single orbit.

**Connection to E₈:** E₈ has 240 roots. Under Moxness projection:
- 240 roots → configurations in H₄
- The 240:120 ratio (2:1) suggests folding by a ℤ₂ quotient

### 6.3 Trinity Decomposition

Each 24-cell decomposes into 3 inscribed 16-cells.
- 24 = 3 × 8 vertices
- The three 16-cells are rotated by 60° isoclinically

**Application:** This provides the geometric basis for the "Trinity" architecture in the CPE.

---

## 7. Computational Verification Plan

### 7.1 Uniqueness Test

```typescript
// Test alternative coefficient families
function testCoefficientFamily(n: number): boolean {
  const phi = (1 + Math.sqrt(5)) / 2;
  const a = Math.pow(phi, n) / 2;
  const b = Math.pow(phi, n - 1) / 2;
  const c = Math.pow(phi, n + 1) / 2;

  const U = buildMoxnessMatrix(a, b, c);
  const e8Roots = generateE8Roots();

  // Check if all projections have H4 symmetry
  return verifyH4Symmetry(U, e8Roots);
}
```

### 7.2 McKay Correspondence Test

```typescript
// Test if null space corresponds to affine root
function testMcKayNullSpace(): boolean {
  const moxnessNull = [0, 0, 0, 0, 1, 1, 1, 1];
  const affineE8Null = computeAffineE8Null();

  return isProportional(moxnessNull, affineE8Null);
}
```

### 7.3 Clifford Decomposition Test

```typescript
// Test if U decomposes into spinor products
function testCliffordDecomposition(): {left: Spinor, right: Spinor} | null {
  const U = buildMoxnessMatrix();

  // Attempt to find L, R such that U = L ⊗ R
  return findSpinorDecomposition(U);
}
```

---

## 8. Open Problems

### 8.1 Classification

**Problem 1:** Classify all 8×8 matrices over ℚ(√5) that project E₈ roots to H₄-symmetric configurations.

**Expected Result:** A finite family parameterized by:
- Scale factor
- Discrete symmetry choices

### 8.2 McKay Correspondence

**Problem 2:** Prove or disprove that the Moxness null space corresponds to the affine direction in the McKay correspondence.

**Expected Result:** Either a rigorous proof connecting the null space to the imaginary root δ of affine E₈, or a counterexample showing they're unrelated.

### 8.3 Clifford Algebra

**Problem 3:** Derive the Moxness matrix from first principles using Dechant's Clifford algebra construction.

**Expected Result:** Express U as a composition of Clifford algebra operations, providing a "why" for the specific coefficient values.

### 8.4 Uniqueness

**Problem 4:** Prove that the Moxness matrix is unique up to symmetry equivalence.

**Expected Result:** A uniqueness theorem stating that any E₈→H₄ folding with the same properties is equivalent to the Moxness matrix under F₄ × W(E₈) action.

---

## 9. Preliminary Results

### 9.1 Verified Properties

| Property | Value | Proof Status |
|----------|-------|--------------|
| det(U) = 0 | 0 | Proven |
| rank(U) = 7 | 7 | Proven |
| √5-Coupling | √(3-φ)×√(φ+2) = √5 | Proven |
| Null space | (0,0,0,0,1,1,1,1)ᵀ | Proven |
| Row dependency | φ·R₀ - φ·R₃ - R₄ + R₇ = 0 | Proven |

### 9.2 Conjectured Properties

| Conjecture | Status | Evidence |
|------------|--------|----------|
| Uniqueness up to symmetry | Unproven | Sign pattern constraints |
| McKay correspondence connection | Unproven | Dimensional matching |
| Clifford algebra origin | Unproven | Dechant's related work |
| φ-family generalization | Unproven | Algebraic structure |

---

## 10. Next Steps

1. **Implement uniqueness test** — Scan coefficient space for alternatives
2. **Compute affine E₈ incidence kernel** — Compare to Moxness null space
3. **Study Dechant's papers in detail** — Find explicit Clifford algebra connection
4. **Formalize the classification theorem** — State precise conditions for equivalence

---

## References

[1] J. C. Baez, "From the icosahedron to E₈," *London Math. Soc. Newsletter*, 2018. arXiv:1712.06436

[2] P.-P. Dechant, "Clifford algebra unveils a surprising geometric significance of quaternionic root systems of Coxeter groups," *Adv. Appl. Clifford Algebras*, 23, 301–321, 2013.

[3] J. G. Moxness, "The 3D visualization of E₈ using an H₄ folding matrix," 2014.

[4] M. Koca, R. Koç, M. Al-Barwani, "Quaternionic roots of E₈ related Coxeter graphs and quasicrystals," *J. Math. Phys.*, 44, 3123–3140, 2003.

[5] J. McKay, "Graphs, singularities, and finite groups," *Proc. Symp. Pure Math.*, 37, 183–186, 1980.

---

*Track A Research Document — January 2026*
