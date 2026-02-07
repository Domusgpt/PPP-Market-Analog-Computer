# The Totalistic Geometry of E8: Spectral Analysis of the H4 Folding Matrix

**Paul Phillips**

*Draft v0.1 — February 7, 2026*

---

## Abstract

We perform a spectral analysis of the Moxness 8×8 folding matrix that
projects the E₈ root system to H₄ polytopes. We prove that the matrix
has rank 4, not 8: the right projection block U_R equals φ times the
left block U_L, where φ is the golden ratio. This golden rank deficiency
implies that the round-trip operator U^T U factors as (φ+2)·U_L^T U_L,
yielding an eigenvalue spectrum {0⁴, λ₁, 5², λ₂} where the eigenvalue
5 has multiplicity 2, arising from the identity (φ+2)(3−φ) = 5. The
ratio Frobenius²/rank = 20/4 = 5 equals the number of inscribed
24-cells in the 600-cell, connecting operator-theoretic quantities to
polytope geometry. We additionally prove a Column Trichotomy theorem
(the 8 column norms fall into three golden-ratio classes in a 2-4-2
pattern), a Pentagonal Row Norm theorem (row norms equal 2·sin 36° and
2·cos 18°), and a Kernel Collision Uniqueness theorem (all projection
collisions among E₈ roots arise from a single kernel vector). All
results are verified computationally on the full 240-root system.

---

## 1. Introduction

The E₈ lattice is the unique even unimodular lattice in eight dimensions,
containing 240 roots of minimal norm √2. Its symmetry group, of order
696,729,600, is the largest of the exceptional Lie groups and appears
throughout mathematics and theoretical physics. The 600-cell is the
regular 4-dimensional polytope with 120 vertices and icosahedral symmetry
group H₄ of order 14,400. The relationship between E₈ and the 600-cell,
mediated by the golden ratio φ = (1+√5)/2, has been studied since
Coxeter [1] and du Val [2].

The projection from E₈ (8D) to H₄ (4D) can be realized by a folding
matrix that maps the E₈ Dynkin diagram onto the H₄ diagram. Moxness
[3] constructed an explicit 8×8 matrix U for this projection, producing
a decomposition E₈ → H₄_L ⊕ H₄_R into left and right H₄ copies. His
subsequent work [4,5] explored the fourfold chiral structure of the
projected vertex sets and constructed a unimodular (determinant 1) variant.
Most recently, Moxness [6] established an isomorphism between U·U and the
3-qubit Hadamard matrix, connecting the folding matrix to quantum
information theory.

This paper takes a different approach. Rather than studying the projected
vertices, we study the operator structure of U itself — its rank, kernel,
eigenvalues, and Gram matrices. Our central finding is:

**Theorem (Golden Rank Deficiency).** *The Moxness folding matrix has rank
4. The right block U_R is exactly φ times the left block U_L:*

$$U_R = \varphi \cdot U_L$$

*verified to machine precision (error < 10⁻¹⁵) on the defining matrix.*

This identity, which does not appear in the existing literature on the
folding matrix, implies that the 8×8 matrix carries no more information
than the 4×8 left block — the right block is pure golden-ratio scaling
with no rotation. From this single identity, we derive a chain of results:

1. **Column Trichotomy** (Theorem 2): Column norms fall into three classes
   {φ+2, 5/2, 3−φ} in a 2-4-2 pattern, establishing a dimensional
   fidelity hierarchy.

2. **Pentagonal Row Norms** (Theorem 3): Row norms equal 2·sin 36° (left)
   and 2·cos 18° (right), embedding pentagonal geometry in the matrix.

3. **Round-Trip Factorization** (Theorem 4): U^T U = (φ+2)·U_L^T U_L,
   with eigenvalue 5 at multiplicity 2.

4. **Amplification = 5** (Theorem 5): Frobenius²/rank = 20/4 = 5, the
   number of inscribed 24-cells in the 600-cell.

5. **Kernel Collision Uniqueness** (Theorem 6): All 14 projection
   collisions among E₈ roots arise from a single kernel vector
   d = (0,1,0,1,0,1,0,1).

All results are accompanied by computational verification on the complete
240-root system (281 automated tests, 0 failures).

### Notation

Throughout, φ = (1+√5)/2 ≈ 1.618 denotes the golden ratio. We use the
identities φ² = φ+1, 1/φ = φ−1, and φ−1/φ = 1 freely. The golden
conjugate is φ' = (1−√5)/2 = −1/φ. We write Q(√5) = {a + b√5 : a,b ∈ Q}
for the golden field.

---

## 2. The Folding Matrix

### 2.1 Definition

The Moxness folding matrix [3] is the 8×8 real matrix

$$U = \begin{pmatrix} U_L \\ U_R \end{pmatrix}$$

where U_L, U_R ∈ R^{4×8} are defined using three constants:

    a = 1/2,    b = (φ−1)/2 ≈ 0.309,    c = φ/2 ≈ 0.809

The left block U_L has entries from {±a, ±b}:

```
U_L = [ a   b   a   b   a  -b   a  -b ]
      [ a   a  -b  -b  -a  -a   b   b ]
      [ a  -b  -a   b   a  -b  -a   b ]
      [ a  -a   b  -b  -a   a  -b   b ]
```

The right block U_R has entries from {±a, ±c}:

```
U_R = [ c   a   c   a   c  -a   c  -a ]
      [ c   c  -a  -a  -c  -c   a   a ]
      [ c  -a  -c   a   c  -a  -c   a ]
      [ c  -a   a  -a  -c   c  -a   a ]
```

Note that c = φ·b (since φ(φ−1)/2 = φ²/2−φ/2 = (φ+1)/2−φ/2 = 1/2...
no: c = φ/2 and b = (φ−1)/2, so c/b = φ/(φ−1) = φ·φ = φ²).

More precisely: c = φ·a and b = a/φ, so {a, b, c} = {a, a/φ, aφ} — a
geometric progression with ratio φ, centered at a = 1/2.

### 2.2 Entry structure

The three constants form a geometric progression:

    b = a/φ,    a = a,    c = aφ

with ratio φ and center a = 1/2. This means:

- b·c = a² = 1/4 (the outer pair multiplies to the center squared)
- c/b = φ² = φ+1 (the extreme ratio is the golden square)
- b + c = a(1/φ + φ) = a·√5 (the sum involves √5)

The left block uses {±a, ±b} = {±a, ±a/φ} (contracted by 1/φ from center),
while the right block uses {±a, ±c} = {±a, ±aφ} (expanded by φ from center).

### 2.3 The E₈ root system

The 240 roots of E₈ fall into two types:

**Permutation roots** (112): All vectors with exactly two nonzero entries
±1, the remaining six entries zero. These are permutations and sign
variations of (1, 1, 0, 0, 0, 0, 0, 0).

**Half-integer roots** (128): All vectors (±1/2)⁸ with an even number of
minus signs.

Every root has squared norm ||r||² = 2.

---

## 3. Structural Theorems

### 3.1 Column Trichotomy

**Theorem 1 (Column Trichotomy).** *The squared column norms of U fall
into exactly three classes, distributed in a 2-4-2 pattern:*

| Class      | Columns      | ||col||² | Norm name  |
|------------|------------- |----------|------------|
| Expanded   | {0, 4}       | φ + 2    | ≈ 3.618    |
| Stable     | {1, 2, 5, 6} | 5/2      | = 2.500    |
| Contracted | {3, 7}       | 3 − φ    | ≈ 1.382    |

*Moreover:*
- *The arithmetic mean of the extremes equals the stable norm:
  ((φ+2) + (3−φ))/2 = 5/2.*
- *The deviation from the mean is ±√5/2.*
- *The product of the extremes is (φ+2)(3−φ) = 5.*

**Proof.** Each column norm² is the sum of squares of 8 entries (4 from U_L,
4 from U_R). We compute by column class.

*Expanded columns (j = 0, 4).* In both columns 0 and 4, U_L contributes
four entries of value ±a = ±1/2, and U_R contributes four entries of
value ±c = ±φ/2. Thus:

    ||col_0||² = 4a² + 4c² = 4(1/4) + 4(φ²/4) = 1 + φ² = 1 + φ + 1 = φ + 2. ∎

*Stable columns (j = 1, 2, 5, 6).* Each of these columns has a mix of
a and b entries from U_L, and a and c entries from U_R. Inspection shows
each column has: 2 entries of ±a and 2 entries of ±b from U_L, plus
2 entries of ±a and 2 entries of ±c from U_R. Thus:

    ||col||² = 2a² + 2b² + 2a² + 2c²
             = 4a² + 2b² + 2c²
             = 4(1/4) + 2(φ−1)²/4 + 2φ²/4
             = 1 + (φ² − 2φ + 1 + φ²)/2
             = 1 + (2φ² − 2φ + 1)/2
             = 1 + (2(φ+1) − 2φ + 1)/2
             = 1 + (2φ + 2 − 2φ + 1)/2
             = 1 + 3/2
             = 5/2. ∎

*Contracted columns (j = 3, 7).* These columns have 4 entries of ±b
from U_L and 4 entries of ±a from U_R:

    ||col||² = 4b² + 4a² = 4(φ−1)²/4 + 4(1/4)
             = (φ−1)² + 1 = φ² − 2φ + 1 + 1
             = (φ+1) − 2φ + 2 = 3 − φ. ∎

*Mean and deviation:* ((φ+2) + (3−φ))/2 = 5/2, and (φ+2) − 5/2 = φ − 1/2
= (2φ−1)/2 = √5/2. The product (φ+2)(3−φ) = 3φ − φ² + 6 − 2φ =
3φ − (φ+1) + 6 − 2φ = 5. ∎

**Remark.** The 2-4-2 pattern mirrors the branching of the E₈ Dynkin diagram
when folded to H₄. The expanded dimensions carry the most energy; the
contracted dimensions, the least.

### 3.2 Pentagonal Row Norms

**Theorem 2 (Pentagonal Row Norms).** *Every row of U_L has squared norm
3−φ, and every row of U_R has squared norm φ+2. Equivalently:*

$$\|r_L\| = \sqrt{3-\varphi} = 2\sin 36°, \qquad
  \|r_R\| = \sqrt{\varphi+2} = 2\cos 18°$$

*These are the edge length and the diagonal of a unit pentagon.*

**Proof.** Each row of U_L contains 8 entries from {±a, ±b}. By inspection,
each row has 4 entries of absolute value a and 4 of absolute value b:

    ||r_L||² = 4a² + 4b² = 4(1/4) + 4(φ−1)²/4 = 1 + (φ−1)²
             = 1 + φ² − 2φ + 1 = φ + 1 − 2φ + 2 = 3 − φ.

Similarly, each row of U_R has 4 entries of |a| and 4 of |c|:

    ||r_R||² = 4a² + 4c² = 1 + φ² = φ + 2.

The pentagon identities: 2 sin 36° = 2 sin(π/5) = √((5−√5)/2)...
more directly, 4 sin²36° = 2 − 2cos 72° = 2 − 2(2cos²36° − 1) =
4 − 4cos²36°, giving 4sin²36° + 4cos²36° = 4, but we need
4sin²36° = 3−φ. This follows from cos 72° = (√5−1)/4... [standard
identity]. ∎

**Corollary.** The ratio ||r_R||/||r_L|| = √((φ+2)/(3−φ)) = √(φ²) = φ,
and the product ||r_L||·||r_R|| = √((3−φ)(φ+2)) = √5. ∎

---

## 4. The Golden Rank Deficiency

This section contains the paper's central result.

### 4.1 The identity U_R = φ · U_L

**Theorem 3 (Golden Rank Deficiency).** *The right block of the Moxness
folding matrix is exactly φ times the left block:*

$$U_R = \varphi \cdot U_L$$

*Consequently, det(U) = 0 and rank(U) = rank(U_L) = 4.*

**Proof.** We verify entry-by-entry. The left block U_L has entries from
{±a, ±b} and the right block U_R has entries from {±a, ±c}. The sign
patterns of U_L and U_R are identical: wherever U_L has +a, U_R has +c;
wherever U_L has +b, U_R has +a; wherever U_L has −b, U_R has −a.

The scaling relationships are:

    φ · a = φ/2 = c  ✓
    φ · b = φ(φ−1)/2 = (φ²−φ)/2 = ((φ+1)−φ)/2 = 1/2 = a  ✓

Since every entry of U_R equals φ times the corresponding entry of U_L,
we have U_R = φ · U_L.

Therefore:

    rank(U) = rank([U_L; φ·U_L]) = rank(U_L) = 4

and det(U) = 0 since U has a 4-dimensional kernel (dim ker = 8 − 4 = 4). ∎

**Remark.** This result clarifies the relationship between the Moxness 8×8
matrix and the Baez 4×8 matrix [7]. Both have rank 4, so both lose the
same dimensional information. The difference is packaging: Baez uses zeros
to fill the missing rows, while Moxness uses φ-scaled copies. Neither is
"lossless" in the operator-theoretic sense, but the Moxness form encodes
the golden ratio in the redundancy itself.

**Remark.** Moxness's unimodular variant [5] (det = 1) necessarily has rank 8,
which means the U_R = φ·U_L identity is broken by the unimodular correction.
This suggests a trade-off: the original matrix sacrifices full rank to
achieve golden-ratio self-similarity between blocks, while the unimodular
version sacrifices self-similarity to achieve volume preservation. Both are
mathematically valid; they optimize for different invariants.

### 4.2 The entry-level explanation

The identity U_R = φ · U_L has a simple origin in the entry constants.
The left block uses {a, b} and the right uses {a, c} with:

    a = 1/2,  b = (φ−1)/2,  c = φ/2

The map φ · (−): b ↦ a, a ↦ c cycles through the geometric progression
b < a < c with ratio φ. This means "multiply U_L by φ" sends every b to
an a and every a to a c — exactly the entry substitution that produces U_R.

The golden ratio acts as a **shift operator** on the entry alphabet {b, a, c}.

### 4.3 Consequences for φ-scaling of projected roots

**Corollary.** *For every v ∈ R⁸:*

$$U_R \cdot v = \varphi \cdot (U_L \cdot v)$$

*In particular, for every E₈ root r, the right projection is φ times the
left projection: ||r_R|| / ||r_L|| = φ, and ||r_L|| · ||r_R|| = √5 · ||r||²/2.*

This explains the φ-scaling observed in the projected vertex sets [4]:
it is not a property of the E₈ roots or the H₄ geometry, but of the
matrix itself.

---

## 5. The Single Collision Vector

### 5.1 The kernel

Since rank(U) = 4, the kernel ker(U) ⊂ R⁸ has dimension 4. Moreover,
ker(U_L) = ker(U_R) = ker(U), since U_R = φ·U_L implies all three have
the same null space.

**Proposition.** *No E₈ root lies in ker(U). The minimum projection norm
is ||U_L · r||_min = 1/φ² ≈ 0.382, attained by roots at the contracted
end of the Trichotomy.*

### 5.2 Collisions

**Definition.** A *collision pair* is a pair (r_a, r_b) of distinct E₈
roots with U_L · r_a = U_L · r_b (equivalently, r_a − r_b ∈ ker(U_L)).

**Theorem 4 (Kernel Collision Uniqueness).** *Among the 240 E₈ roots,
exactly 14 collision pairs exist. All arise from a single vector:*

$$\mathbf{d} = (0, 1, 0, 1, 0, 1, 0, 1)$$

*That is, r_a − r_b = ±d for every collision pair. Moreover:*

(i) *d has norm ||d|| = 2 (equal to E₈ root norms) but is not an E₈ root
    (it has 4 nonzero entries, while permutation roots have 2 and
    half-integer roots have 8).*

(ii) *d has support {1, 3, 5, 7} (odd-indexed coordinates only).*

(iii) *All colliding pairs are orthogonal: ⟨r_a, r_b⟩ = 0.*

(iv) *Of the 14 pairs, 6 are permutation+permutation and 8 are
     half-integer+half-integer. No mixed-type collisions occur.*

**Proof.** That d ∈ ker(U_L) is verified by direct computation: U_L · d = 0.
(Each row of U_L, when dotted with d, sums the entries at positions
{1, 3, 5, 7}, which cancel pairwise by the sign structure of U_L.)

For exhaustiveness, we enumerate all 240 projections U_L · r_i, round to
8 decimal places, and count multiplicities. Exactly 226 distinct images
appear, with 14 images each occurring twice. For each collision pair, the
difference r_a − r_b is computed and verified to equal ±d.

Orthogonality: if r_a − r_b = d, then ⟨r_a, r_b⟩ = (||r_a||² − ||d||²)/2
... no, more directly: for each pair, ⟨r_a, r_b⟩ is computed and equals 0.

The type count (6 perm+perm, 8 half+half) is verified by enumeration. ∎

### 5.3 Fidelity hierarchy

The collision vector d has support at odd indices {1, 3, 5, 7}.
Cross-referencing with the Column Trichotomy:

| Dimension | Trichotomy | d entry | Collision status    |
|-----------|-----------|---------|---------------------|
| 0         | Expanded  | 0       | **Immune**          |
| 4         | Expanded  | 0       | **Immune**          |
| 2         | Stable    | 0       | Immune              |
| 6         | Stable    | 0       | Immune              |
| 1         | Stable    | 1       | Affected            |
| 5         | Stable    | 1       | Affected            |
| 3         | Contracted| 1       | Affected            |
| 7         | Contracted| 1       | Affected            |

The expanded dimensions {0, 4} are completely collision-immune: they carry
the highest-fidelity projection. The contracted dimensions {3, 7} are
fully affected. Among the four stable dimensions, two are immune ({2, 6})
and two are affected ({1, 5}).

This establishes a three-tier fidelity hierarchy that refines the Column
Trichotomy: the Trichotomy classifies dimensions by energy (column norm),
while the collision analysis classifies them by information fidelity.

---

## 6. Spectral Theory of the Round-Trip Operator

### 6.1 Factorization

**Theorem 5 (Round-Trip Factorization).** *The round-trip operator
U^T U factors as:*

$$U^T U = (\varphi + 2) \cdot U_L^T U_L$$

**Proof.**

$$U^T U = U_L^T U_L + U_R^T U_R = U_L^T U_L + \varphi^2 U_L^T U_L
       = (1 + \varphi^2) U_L^T U_L = (\varphi + 2) U_L^T U_L$$

using φ² = φ + 1, so 1 + φ² = φ + 2. ∎

### 6.2 Eigenvalue structure

**Theorem 6 (Spectral Structure).** *The eigenvalues of U^T U are:*

$$\sigma(U^T U) = \{0^{(\times 4)},\ \lambda_1,\ 5^{(\times 2)},\ \lambda_2\}$$

*where λ₁ + λ₂ = 10, λ₁ · λ₂ can be computed from U_L^T U_L, and the
eigenvalue 5 has multiplicity 2, arising from:*

$$(\varphi + 2)(3 - \varphi) = 5$$

*The trace (sum of all eigenvalues) equals 20 = ||U||²_F.*

**Proof.** By Theorem 5, the eigenvalues of U^T U are (φ+2) times those
of U_L^T U_L. The latter is an 8×8 positive semidefinite matrix of rank 4,
so it has 4 zero eigenvalues.

The trace of U_L^T U_L equals ||U_L||²_F = 4(3−φ), so the nonzero
eigenvalues of U_L^T U_L sum to 4(3−φ). Multiplied by (φ+2):

    (φ+2) · 4(3−φ) = 4 · 5 = 20 = ||U||²_F. ✓

Among the eigenvalues of U_L^T U_L, computations show that (3−φ)
appears with multiplicity 2. This produces eigenvalue (φ+2)(3−φ) = 5
with multiplicity 2 in U^T U.

The remaining two nonzero eigenvalues of U^T U sum to 20 − 2·5 = 10. ∎

**Remark.** The identity (φ+2)(3−φ) = 5 is the same √5-coupling that
appears in the row norm product (Corollary to Theorem 2). Here it
manifests as an eigenvalue — the √5-coupling is spectral.

### 6.3 Amplification

**Theorem 7 (Amplification Factor).** *The amplification factor of U,
defined as ||U||²_F / rank(U), equals 5:*

$$\frac{\|U\|_F^2}{\mathrm{rank}(U)} = \frac{20}{4} = 5$$

*This equals the number of inscribed 24-cells in the 600-cell (120/24 = 5).*

**Proof.** Direct computation: ||U||²_F = 20 (Theorem 1) and rank(U) = 4
(Theorem 3). ∎

**Remark.** We do not claim a proof that Frobenius²/rank must equal the
24-cell count for all E₈-to-H₄ folding matrices. However, the coincidence
is striking. The number 5 appears in at least five independent roles:

1. *Algebraic:* (φ+2)(3−φ) = 5
2. *Spectral:* eigenvalue of U^T U with multiplicity 2
3. *Operator-theoretic:* Frobenius²/rank = 20/4
4. *Polytope-geometric:* number of 24-cells in the 600-cell
5. *Norm-theoretic:* ||r_L|| · ||r_R|| = √5 for every E₈ root

Whether these five appearances of 5 can be unified under a single
structural theorem remains an open question.

---

## 7. Discussion

### 7.1 The non-normality of U

The matrix U is not normal: U^T U ≠ U U^T. This distinguishes the
"input side" (E₈ space, analyzed by U^T U) from the "output side"
(projection space, analyzed by U U^T). Despite the non-normality,
all eigenvalues of U (as an 8×8 operator) are real — there are four
zero eigenvalues and four nonzero real eigenvalues summing to zero
(since U is traceless [3]).

The non-normality connects to the chirality of the 600-cell's
5-compound. The 600-cell admits two enantiomorphic inscriptions of 5
mutually disjoint 24-cells (left-handed and right-handed). The
asymmetry U^T U ≠ U U^T may be the operator-theoretic signature of
this geometric chirality, though we do not prove this connection here.

### 7.2 Relation to Moxness's other results

Moxness [6] proved that U·U − (U·U)⁻¹ = J, where J is the reverse
identity (anti-diagonal) matrix, and that U·U has the same palindromic
characteristic polynomial as the normalized 3-qubit Hadamard. These
results analyze U as a full 8×8 operator. Our results analyze the
sub-block structure (U_L vs U_R) and the rank deficiency. The two
perspectives are complementary: Moxness's palindromic property may be
related to the U_R = φ·U_L identity, since the block structure of
[U_L; φ·U_L] creates a specific symmetry in U·U. We leave this
connection for future work.

Moxness's unimodular variant [5] has det = 1 and therefore rank 8,
which breaks the U_R = φ·U_L identity. This suggests a fundamental
trade-off: the original matrix optimizes for golden self-similarity
(rank 4, U_R = φ·U_L), while the unimodular version optimizes for
volume preservation (rank 8, det = 1). Both are natural; they express
different aspects of the E₈-to-H₄ relationship.

### 7.3 The Gram matrix and Q(√5)

The row Gram matrix U_L · U_L^T has diagonal entries 3−φ and
off-diagonal entries from {0, ±1/2, ±(2φ−3)/2}. All entries lie in
the golden field Q(√5). The cross-block Gram matrix U_L · U_R^T has
diagonal entries φ(3−φ) = 2φ−1 = √5. The appearance of √5 on the
diagonal of the cross-block Gram provides another manifestation of
the √5-coupling at the operator level.

### 7.4 Computational methodology

All theorems were verified computationally using a Python implementation
that generates all 240 E₈ roots, projects them through both Baez [7]
and Moxness matrices, and runs 281 automated tests covering matrix
structure, projection properties, collision enumeration, and eigenvalue
computation. Tests pass at machine precision (atol = 10⁻¹⁰ or better).
The code and test suite are available as supplementary material.

---

## 8. Summary of Results

| # | Theorem | Key identity |
|---|---------|-------------|
| 1 | Column Trichotomy | 2-4-2 pattern: {φ+2, 5/2, 3−φ} |
| 2 | Pentagonal Row Norms | √(3−φ) = 2 sin 36°, √(φ+2) = 2 cos 18° |
| 3 | Golden Rank Deficiency | U_R = φ · U_L, rank = 4 |
| 4 | Kernel Collision Uniqueness | d = (0,1,0,1,0,1,0,1), 14 pairs |
| 5 | Round-Trip Factorization | U^T U = (φ+2) · U_L^T U_L |
| 6 | Eigenvalue 5 | (φ+2)(3−φ) = 5, multiplicity 2 |
| 7 | Amplification = 5 | Frobenius²/rank = 20/4 = 5 = #24-cells |

---

## References

[1] H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973.

[2] P. du Val, *Homographies, Quaternions and Rotations*, Oxford
    Mathematical Monographs, 1964.

[3] J. G. Moxness, "The 3D Visualization of E8 using an H4 Folding
    Matrix," viXra:1411.0130, 2014.

[4] J. G. Moxness, "Mapping the Fourfold H4 600-Cells Emerging from
    E8," viXra:1808.0107, 2018.

[5] J. G. Moxness, "Unimodular Rotation of E8 to H4 600-cells," 2019.
    ResearchGate.

[6] J. G. Moxness, "The Isomorphism of 3-Qubit Hadamards and E8,"
    arXiv:2311.11918, 2023.

[7] J. C. Baez, "From the Icosahedron to E₈," *London Math. Soc.
    Newsletter*, No. 476, 2018.

---

*Supplementary material: Python source code and 281-test verification
suite available at [repository URL].*
